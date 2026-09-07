"""The only module that calls a model.

Everything else takes a `ModelClient`. Tests pass in a fake, so the test suite
never makes a network call. The cost of a run is counted here, in one place,
instead of being estimated afterwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, NoReturn, Protocol

from pydantic import BaseModel

from halo.platform.budget import BudgetTracker

DEFAULT_MODEL = "global.anthropic.claude-sonnet-4-6"
"""The newest Sonnet this account can invoke, using the cheaper routing.

We use `global.` instead of `us.`. The Bedrock rate card prices the global
profile at 3.00/15.00 per million tokens. The regional profile costs 3.30/16.50
for the same model. The difference is that a global request is served wherever
capacity is available, instead of being restricted to one geography. That does
not matter for synthetic practice data. It would matter for real customer data.

Sonnet 5 is the model this project was planned around, and the account is
*authorized* for it. But Bedrock refuses the call with "not available for this
account", in every region tried, and after the Anthropic use case form cleared.
That is a tier the account is not offered, so 4.6 it is until that changes.

Bedrock has two API surfaces. They accept different model id formats. An account
can be entitled to one and not the other:

- **Mantle** (the Messages API on Bedrock, preferred for new code) takes bare ids
  like `anthropic.claude-sonnet-5`.
- **InvokeModel** (the older bedrock-runtime path) takes a cross-region inference
  profile id like `us.anthropic.claude-sonnet-4-6`, or a dated foundation-model id.

`BedrockClient` chooses the surface from the id format. Switching between them
requires only a `--model` change.
"""

PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.")

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Dollars per million tokens. These come from the Bedrock offer rate cards for
# us-east-1, read with `list-foundation-model-agreement-offers` under
# `usageBasedPricingTerm`. They are not the first-party Anthropic prices. The two
# are different. The earlier first-party figures under-reported this project's
# spend by about 10%.
#
# Each model has two rates, and the model id format selects one:
#   `us.` and other regional profiles pay the higher "Geo" rate.
#   `global.` profiles pay 10% less. In exchange, the request is served wherever
#   capacity is available instead of being restricted to one geography.
PRICE_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-sonnet-4-6": (Decimal("3.30"), Decimal("16.50")),
    "claude-opus-4-5": (Decimal("16.50"), Decimal("82.50")),
    "claude-sonnet-4-5": (Decimal("3.30"), Decimal("16.50")),
    "claude-haiku-4-5": (Decimal("1.10"), Decimal("5.50")),
}

GLOBAL_DISCOUNT = Decimal("0.909091")
"""`global.` profiles bill 3.00/15.00 where regional bills 3.30/16.50."""

# Cache rates are multiples of whatever the input rate is, and the same
# multiples apply to regional and global profiles. Read from the same rate card:
# on Sonnet 4.6 regional, input is 3.30 and cache read is 0.33; on global, input
# is 3.00 and cache read is 0.30. Both are a tenth. So the global discount is
# applied to the base rate first, and the multiplier after.
CACHE_READ_MULTIPLIER = Decimal("0.10")
"""A cached input token costs a tenth of a fresh one."""

CACHE_WRITE_5M_MULTIPLIER = Decimal("1.25")
"""Writing to the 5-minute cache costs a quarter more than a fresh input token.

Caching only pays off if the prefix is read back. One write plus one read costs
1.35x a single uncached call; the second read is where it turns profitable."""

CACHE_WRITE_1H_MULTIPLIER = Decimal("2.00")
"""The 1-hour cache costs twice a fresh input token to write."""

MIN_OUTPUT_TOKENS = 256
"""The smallest output cap worth paying for.

A call sized to the remaining budget can be clamped so low that it produces a
report truncated mid-sentence — which costs money and yields nothing usable.
Below this the run stops on the dollar budget instead, which is what it is
actually short of."""


class UnpricedModel(RuntimeError):
    """This model has no entry in the rate card, so its spend cannot be counted.

    Raised when a client is constructed, not when a call is made. The dollar
    limit on a budget is only a limit if every call can be priced: an unpriced
    call adds zero to `usage.usd`, `max_usd` is never reached, and the run
    proceeds under a cap that has quietly stopped existing. Nothing looks wrong
    at any point, including in the ledger afterwards.

    The fix is a line in `PRICE_PER_MTOK` read off the account's own rate card,
    which is what the message says.

    A `RuntimeError` so that the CLI's existing handlers catch it: this is a
    setup problem, and it should read like the other setup problems rather than
    arriving as a traceback.
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            f"no rate card entry for {model!r}, so its cost cannot be counted and a "
            "dollar budget could not stop a run using it. Add the model to "
            "PRICE_PER_MTOK in platform/bedrock.py, reading the rates from "
            "`aws bedrock list-foundation-model-agreement-offers` for your account."
        )
        self.model = model


class Truncated(RuntimeError):
    """The model hit its output ceiling before finishing a structured answer.

    `max_tokens` is enforced by the service, and the model is not told about it:
    generation stops mid-token and the call returns 200 with whatever was
    produced. For a tool-use turn that partial content is still worth a decision,
    so `converse` hands it back and the loop decides. For a report there is
    nothing to decide — `parsed_output` comes back `None`, and half a structured
    answer is not an answer — so this is raised instead, and `ModelResult.parsed`
    keeps its promise of holding a complete one.

    Kept separate from `BudgetExceeded` because the fix differs. A clamped
    ceiling means the run is short of money; the default ceiling means the report
    is longer than the ceiling allows, and raising `max_usd` would change
    nothing.
    """

    def __init__(self, model: str, max_tokens: int) -> None:
        super().__init__(
            f"{model} stopped at its {max_tokens:,}-token output ceiling with the "
            "answer unfinished, so no complete report was produced"
        )
        self.max_tokens = max_tokens


def is_priced(model: str) -> bool:
    """Whether this model's spend can be counted."""
    return _price_key(model) is not None


def _price_key(model: str) -> str | None:
    """Reduce any Bedrock id format to the model family used as the price key.

    `us.anthropic.claude-sonnet-4-6` and
    `anthropic.claude-sonnet-4-6-20260101-v1:0` are the same model at the same
    price. If the price table used the full id as its key, one of these would be
    priced at zero without any error.
    """
    trimmed = model
    for prefix in PROFILE_PREFIXES:
        trimmed = trimmed.removeprefix(prefix)
    trimmed = trimmed.removeprefix("anthropic.")
    return next((key for key in PRICE_PER_MTOK if trimmed.startswith(key)), None)


@dataclass(frozen=True)
class TokenCounts:
    """The four billable token categories in one response.

    They are separate counts, not subsets. `input_tokens` from the API excludes
    anything served from cache or written to it, so the real input size is the
    sum of all three input categories. Reading only `input_tokens` on a cached
    call under-reports both the spend and the context size.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def counts_from(usage: Any) -> TokenCounts:
    """Read the token categories out of an SDK usage object.

    The 5-minute and 1-hour caches are priced differently, so the breakdown in
    `cache_creation` is used when the SDK provides it. When it does not, the
    total falls back to the 5-minute rate, which is the default TTL and the
    cheaper of the two — so a missing breakdown under-reports rather than
    over-reports, and the run is not stopped by a budget it did not spend.
    """
    write_5m = write_1h = 0
    breakdown = getattr(usage, "cache_creation", None)
    if breakdown is not None:
        write_5m = getattr(breakdown, "ephemeral_5m_input_tokens", 0) or 0
        write_1h = getattr(breakdown, "ephemeral_1h_input_tokens", 0) or 0
    if not (write_5m or write_1h):
        write_5m = getattr(usage, "cache_creation_input_tokens", 0) or 0

    return TokenCounts(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
    )


def estimate_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> Decimal:
    """Cost of one call.

    An unknown model returns zero rather than raising: a pricing lookup is not
    the place to take a run down, and callers that only want a number should get
    one. The safety property lives one level up instead — `BedrockClient` refuses
    to be constructed for a model it cannot price, so an unpriced model cannot
    reach this function through a real run. See `UnpricedModel`.
    """
    key = _price_key(model)
    if key is None:
        return Decimal("0.00")

    price_in, price_out = PRICE_PER_MTOK[key]
    if model.startswith("global."):
        price_in *= GLOBAL_DISCOUNT
        price_out *= GLOBAL_DISCOUNT

    million = Decimal(1_000_000)
    cost = (
        Decimal(input_tokens) * price_in
        + Decimal(output_tokens) * price_out
        + Decimal(cache_read_tokens) * price_in * CACHE_READ_MULTIPLIER
        + Decimal(cache_write_5m_tokens) * price_in * CACHE_WRITE_5M_MULTIPLIER
        + Decimal(cache_write_1h_tokens) * price_in * CACHE_WRITE_1H_MULTIPLIER
    ) / million
    return cost.quantize(Decimal("0.000001"))


def affordable_output_tokens(model: str, remaining_usd: Decimal) -> int:
    """How many output tokens `remaining_usd` still buys on this model.

    Output is the only part of a call's cost the caller controls once the
    messages are built, so this is what `max_tokens` gets clamped to. Sizing the
    call to the budget is the only thing that stops a single turn from passing
    `max_usd`: the loop checks between steps, and a non-streaming request cannot
    be stopped part-way, so without this the overshoot is already paid for by
    the time anything notices.

    The rate is read for a thousand tokens rather than one. `estimate_usd`
    quantizes to six decimal places, which at single-token granularity rounds a
    cheap model's output rate by a few percent.

    An unpriced model returns zero, matching `estimate_usd`. That cannot reach a
    real run — `BedrockClient` refuses to be constructed for one — and a caller
    that clamps to zero would raise `UnpricedModel`'s problem as a confusing
    `max_tokens` error instead.
    """
    per_1k = estimate_usd(model, 0, 1000)
    if per_1k <= 0 or remaining_usd <= 0:
        return 0
    return int(remaining_usd * 1000 / per_1k)


def input_usd_for(model: str, counts: TokenCounts) -> Decimal:
    """What the input half of a call cost, at the cache rates it actually paid.

    Read back from a completed call so the next one can reserve room for its own
    input. Counting the tokens ahead of time instead would mean a
    `count_tokens` round trip per turn, and that endpoint cannot see what the
    cache will serve — against a deliberately cached system prefix it would
    price reads at ten times what they cost and stop runs that had budget left.
    """
    return estimate_usd(
        model,
        counts.input_tokens,
        0,
        cache_read_tokens=counts.cache_read_tokens,
        cache_write_5m_tokens=counts.cache_write_5m_tokens,
        cache_write_1h_tokens=counts.cache_write_1h_tokens,
    )


def estimate_usd_for(model: str, counts: TokenCounts) -> Decimal:
    """`estimate_usd` for a whole `TokenCounts`."""
    return estimate_usd(
        model,
        counts.input_tokens,
        counts.output_tokens,
        cache_read_tokens=counts.cache_read_tokens,
        cache_write_5m_tokens=counts.cache_write_5m_tokens,
        cache_write_1h_tokens=counts.cache_write_1h_tokens,
    )


@dataclass(frozen=True)
class ModelResult[T: BaseModel]:
    """A parsed model response and what the call cost."""

    parsed: T
    input_tokens: int
    output_tokens: int
    usd: Decimal
    model: str
    stop_reason: str | None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ModelTurn:
    """One assistant turn in a tool-use loop.

    `content` keeps the raw block list. That list must go back into the next
    request unchanged. Rebuilding it from extracted text loses the `tool_use`
    ids.
    """

    content: list[Any]
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    usd: Decimal
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def tool_uses(self) -> list[Any]:
        return [block for block in self.content if getattr(block, "type", None) == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(
            block.text for block in self.content if getattr(block, "type", None) == "text"
        )


def system_blocks(system: str) -> list[dict[str, Any]]:
    """The system prompt as one cacheable block.

    M6 runs four loops per request, and each loop resends its system prompt on
    every turn. That prefix is identical across turns and across specialists of
    the same kind, which is exactly the shape prompt caching is for: a cached
    input token costs a tenth of a fresh one.

    The breakpoint goes here and not on the messages, because the messages grow
    every turn and a breakpoint inside them would be invalidated by the growth.
    Anything volatile — today's date, the brief — stays after it.

    Two honest caveats. The minimum cacheable prefix is model-dependent and these
    prompts are shorter than it today, so `cache_read_input_tokens` will stay at
    zero until they grow; the accounting for it exists either way, and
    `Usage.cache_hit_rate` is the number to watch. And nothing in the offline
    suite can prove Bedrock accepted the breakpoint: the tests assert the shape
    of the request, and only a live run shows a cache read.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


class ModelClient(Protocol):
    """The interface every agent depends on.

    `BedrockClient` implements it for real. The tests implement it with a fake.
    """

    def parse[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output_format: type[T],
        max_tokens: int = ...,
    ) -> ModelResult[T]: ...

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = ...,
    ) -> ModelTurn: ...


class BedrockClient:
    """Claude on Amazon Bedrock, returning validated Pydantic objects.

    This uses structured output instead of asking the model to reply with JSON.
    The response is parsed directly into a domain model. A malformed answer
    becomes a validation error at the boundary, instead of a `KeyError` three
    layers deeper in the code.
    """

    def __init__(
        self,
        *,
        region: str = DEFAULT_REGION,
        model: str = DEFAULT_MODEL,
        tracker: BudgetTracker | None = None,
    ) -> None:
        from anthropic import AnthropicBedrock, AnthropicBedrockMantle, APITimeoutError

        # Before anything else. A model with no rate card entry makes every
        # dollar budget in the process unenforceable, and the failure is silent
        # in both directions: nothing errors, and the ledger reports $0.00 for a
        # run that really spent money.
        if not is_priced(model):
            raise UnpricedModel(model)

        # An inference-profile id works only with InvokeModel. A bare id works
        # only with Mantle. Choosing the client from the id format keeps the two
        # surfaces one flag apart, instead of two separate code paths.
        uses_profile = model.startswith(PROFILE_PREFIXES)
        self._client = (
            AnthropicBedrock(aws_region=region)
            if uses_profile
            else AnthropicBedrockMantle(aws_region=region)
        )
        self.surface = "invoke_model" if uses_profile else "mantle"
        self.model = model
        self.region = region
        self._tracker = tracker
        self._last_input_usd = Decimal("0.00")
        # Held rather than imported at module scope so the SDK stays behind the
        # same lazy import as the clients above.
        self._timeout_error = APITimeoutError

    def _prepared(self, max_tokens: int) -> tuple[Any, int]:
        """The client and output cap to use for one call, against the budget.

        Three things happen here, and only the first is what `check` alone did:

        1. The budget is checked, so a run already over its limit never calls.
        2. `max_tokens` is clamped to what the remaining dollars buy. A
           non-streaming request cannot be stopped part-way, so a call that
           starts is paid for in full; sizing it to fit is what keeps a single
           turn from passing `max_usd` and being noticed only afterwards.
        3. The request gets the remaining wall clock as its timeout, so a call
           that would outlast the budget is cut off in flight.

        `max_retries=0` is not a preference. The SDK retries timeouts, so at the
        default of 2 a call given the remaining wall clock could take three
        times it — a budget quietly worth triple what it says.
        """
        if self._tracker is None:
            return self._client, max_tokens

        self._tracker.check()
        # Output is not the whole bill. The transcript and its tool results are
        # resent every turn, and on a long loop that input outgrows the answer.
        # Sizing the output cap against the *whole* remainder would authorize a
        # call whose input alone could pass the limit, so the last call's input
        # is held back first.
        spendable = self._tracker.remaining_usd - self._last_input_usd
        affordable = affordable_output_tokens(self.model, spendable)
        if affordable < MIN_OUTPUT_TOKENS:
            # Enough budget left to pass `check`, not enough to buy an answer
            # worth having. Stopping here produces the same clean escalation as
            # any other breach, instead of a deliberately truncated report that
            # reads like a complete one.
            raise self._tracker.exceeded(
                "max_usd",
                f"{self._tracker.usage.usd}, which after reserving {self._last_input_usd} for "
                f"input leaves room for only {affordable} output tokens",
            )

        client = self._client.with_options(
            timeout=self._tracker.remaining_seconds,
            max_retries=0,
        )
        return client, min(max_tokens, affordable)

    def _cut_off(self, exc: Exception) -> NoReturn:
        """Re-raise a timed-out request as whatever actually ended it.

        When a tracker set the timeout, the timeout *is* the budget, so the
        breach is reported as the budget dimension that set it. Leaking
        `APITimeoutError` upward would make the loop import the SDK to catch it,
        and would report an infrastructure fault for a limit doing its job.

        An untracked client has no such timeout — what it hit is the SDK's own
        ten-minute default, which is a real fault and is left alone.
        """
        if self._tracker is None:
            raise exc
        raise self._tracker.exceeded(
            "wall_clock_seconds",
            f"{self._tracker.elapsed_seconds:.2f} when the request was cut off in flight",
        ) from exc

    def parse[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output_format: type[T],
        max_tokens: int = 16_000,
    ) -> ModelResult[T]:
        client, max_tokens = self._prepared(max_tokens)

        try:
            response = client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system_blocks(system),
                messages=[{"role": "user", "content": user}],
                output_format=output_format,
                thinking={"type": "adaptive"},
            )
        except self._timeout_error as exc:
            self._cut_off(exc)

        counts = counts_from(response.usage)
        usd = estimate_usd_for(self.model, counts)
        # What this call's input cost is the estimate for the next one's. Within
        # a loop the transcript only grows, so this is a floor rather than a
        # guarantee — it does not yet include the turn just appended or the tool
        # results still to come. It turns an unbounded overshoot into a bounded
        # one, which is the part that was missing.
        self._last_input_usd = input_usd_for(self.model, counts)
        if self._tracker is not None:
            self._tracker.record_model_call(
                counts.input_tokens,
                counts.output_tokens,
                usd,
                cache_read_tokens=counts.cache_read_tokens,
                cache_write_tokens=counts.cache_write_tokens,
            )

        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            # Charged above before raising: the tokens were generated and billed
            # whether or not they added up to an answer. Dropping the cost here
            # would let a run retry its way past a budget it had already spent.
            raise Truncated(self.model, max_tokens)

        return ModelResult(
            parsed=response.parsed_output,
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            usd=usd,
            model=self.model,
            stop_reason=response.stop_reason,
            cache_read_tokens=counts.cache_read_tokens,
            cache_write_tokens=counts.cache_write_tokens,
        )

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 8_000,
    ) -> ModelTurn:
        """One turn of a tool-use loop. The caller owns the loop.

        This is deliberately not a `while stop_reason == "tool_use"` helper. The
        loop is where budgets are checked and tool results are audited. Putting
        the loop here would move both of those outside the agent's control.
        """
        client, max_tokens = self._prepared(max_tokens)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_blocks(system),
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
            )
        except self._timeout_error as exc:
            self._cut_off(exc)

        counts = counts_from(response.usage)
        usd = estimate_usd_for(self.model, counts)
        # What this call's input cost is the estimate for the next one's. Within
        # a loop the transcript only grows, so this is a floor rather than a
        # guarantee — it does not yet include the turn just appended or the tool
        # results still to come. It turns an unbounded overshoot into a bounded
        # one, which is the part that was missing.
        self._last_input_usd = input_usd_for(self.model, counts)
        if self._tracker is not None:
            self._tracker.record_model_call(
                counts.input_tokens,
                counts.output_tokens,
                usd,
                cache_read_tokens=counts.cache_read_tokens,
                cache_write_tokens=counts.cache_write_tokens,
            )

        return ModelTurn(
            content=list(response.content),
            stop_reason=response.stop_reason,
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            usd=usd,
            cache_read_tokens=counts.cache_read_tokens,
            cache_write_tokens=counts.cache_write_tokens,
        )
