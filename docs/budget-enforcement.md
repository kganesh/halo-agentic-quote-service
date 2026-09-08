# Budget enforcement — the code path

`halo-agentic-quote-service` · commit `9d4e71c` · extracted 2026-09-08

Every block below is the real source, pulled out of the repository with `ast`, not
retyped. Read top to bottom and you have the whole enforcement path.

> **This file quotes source and will drift.** It is a snapshot at the commit above,
> not a live view. When the enforcement path changes, re-extract rather than
> hand-editing the blocks — the symbols quoted are `Budget`, `Usage`,
> `BudgetExceeded`, `BudgetTracker`, `UnpricedModel`, `affordable_output_tokens`,
> `input_usd_for`, `Truncated`, `BedrockClient._prepared` / `._cut_off` /
> `.converse` / `.parse`, `run_specialist` and `_charge`.

---

## The constraint everything follows from

The expensive operation is a non-streaming call to Claude on Bedrock. Three
properties decide the design:

1. **There is no stop request.** The Messages API has no server-side cancel.
   Once a request is in flight the only abort is closing the HTTP connection,
   and you are billed for whatever was generated first.
2. **The calls are non-streaming.** `messages.create` / `messages.parse` hold one
   request open until the whole response arrives. No partial state, nowhere to
   intervene.
3. **The client blocks the event loop.** `BedrockClient.converse` is synchronous
   and called from inside `async def`. While a request is open nothing else runs
   — no Ctrl-C, no `CancelledError`.

So all the leverage is *before* the request goes out. Everything here is a
consequence of that.

---

## 1 · The primitive — `platform/budget.py`

Four limits, all hard stops. No prompt is ever asked to be brief.

```python
class Budget(BaseModel):
    """Limits for one agent run. All four are hard stops."""

    wall_clock_seconds: float = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_usd: Decimal = Field(gt=0)
```


What a run has spent so far. Cache tokens are counted because the API reports
them separately from `input_tokens`, not inside it — leaving them out would let a
cached run pass a `max_tokens` limit it had actually exceeded.

```python
class Usage(BaseModel):
    """What a run has spent so far."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: int = 0
    usd: Decimal = Decimal("0.00")

    @property
    def total_tokens(self) -> int:
        """Every token the run was billed for.

        Cache tokens are counted here because the API reports them separately
        from `input_tokens`, not inside it. Leaving them out would let a cached
        run pass a `max_tokens` limit it had actually exceeded.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. Zero when nothing is cached.

        The number to watch when turning caching on: if it stays at zero across
        repeated runs, something is invalidating the prefix.
        """
        billed_input = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / billed_input if billed_input else 0.0
```


The breach carries `owner`, and that matters more than it looks. Several budgets
are live at once: each specialist holds one, and the run holds another that the
shared model client and gateway count against.

```python
class BudgetExceeded(Exception):
    """Records which limit was reached, and whose it was.

    `owner` matters because several budgets are in play at once: each specialist
    holds one, and the run holds another that the shared model client and gateway
    count against. Without it, a run-level trip inside the pricing specialist
    reads as "pricing exhausted its budget", and raising pricing's allowance
    changes nothing — the reason names the wrong thing to fix.
    """

    def __init__(self, dimension: str, limit: object, spent: object, owner: str = "run") -> None:
        super().__init__(f"{owner} budget exceeded on {dimension}: spent {spent}, limit {limit}")
        self.dimension = dimension
        self.owner = owner
```


The tracker: a budget, an injectable clock, and accumulated usage. `check()` is
the backstop; `remaining_usd` / `remaining_seconds` are what let a caller size the
*next* call to fit; `exceeded()` mints a breach for a limit the caller detected
with information the tracker does not have.

```python
class BudgetTracker:
    """Wraps a budget with a clock. One tracker per agent run.

    `owner` names whose allowance this is, so a breach can say which budget to
    raise. It defaults to "run" because that is the one a caller who does not
    care about the distinction is holding.
    """

    def __init__(
        self, budget: Budget, now: callable = time.monotonic, *, owner: str = "run"
    ) -> None:
        self._budget = budget
        self._owner = owner
        self._now = now
        self._started = now()
        self.usage = Usage()

    @property
    def elapsed_seconds(self) -> float:
        return self._now() - self._started

    @property
    def remaining_seconds(self) -> float:
        """Wall clock left before this budget trips. Never negative.

        This is what a caller passes as a request timeout, so that a call which
        would outlast the budget is cut off by the HTTP layer instead of running
        to completion and being noticed afterwards. Zero means the budget is
        already spent; `check` is what turns that into a `BudgetExceeded`.
        """
        return max(0.0, self._budget.wall_clock_seconds - self.elapsed_seconds)

    @property
    def remaining_usd(self) -> Decimal:
        """Dollars left before this budget trips. Never negative.

        `check` runs between steps, so it can only catch an overspend after the
        call that caused it has been paid for. This is the number that lets a
        caller size the next call to fit, which is the only way a per-call
        overshoot is prevented rather than reported.
        """
        return max(Decimal("0.00"), self._budget.max_usd - self.usage.usd)

    def check(self) -> None:
        """Raise if any limit has been passed. Call this before each step."""
        if self.elapsed_seconds > self._budget.wall_clock_seconds:
            raise BudgetExceeded(
                "wall_clock_seconds",
                self._budget.wall_clock_seconds,
                round(self.elapsed_seconds, 2),
                self._owner,
            )
        if self.usage.total_tokens > self._budget.max_tokens:
            raise BudgetExceeded(
                "max_tokens", self._budget.max_tokens, self.usage.total_tokens, self._owner
            )
        if self.usage.tool_calls > self._budget.max_tool_calls:
            raise BudgetExceeded(
                "max_tool_calls", self._budget.max_tool_calls, self.usage.tool_calls, self._owner
            )
        if self.usage.usd > self._budget.max_usd:
            raise BudgetExceeded("max_usd", self._budget.max_usd, self.usage.usd, self._owner)

    def exceeded(self, dimension: str, spent: object) -> BudgetExceeded:
        """The breach for a limit the caller detected on its own.

        `check` covers what the tracker can see by itself. Two things it cannot:
        sizing a call to the remaining dollars needs a rate card, and a request
        cut off in flight is known to the HTTP layer. Both are found outside,
        and both are the same kind of failure — so they are raised as the same
        exception, carrying this tracker's owner and limit rather than a second
        error type the loop would have to learn.
        """
        limits: dict[str, object] = {
            "wall_clock_seconds": self._budget.wall_clock_seconds,
            "max_tokens": self._budget.max_tokens,
            "max_tool_calls": self._budget.max_tool_calls,
            "max_usd": self._budget.max_usd,
        }
        return BudgetExceeded(dimension, limits[dimension], spent, self._owner)

    def record_model_call(
        self,
        input_tokens: int,
        output_tokens: int,
        usd: Decimal,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.cache_read_tokens += cache_read_tokens
        self.usage.cache_write_tokens += cache_write_tokens
        self.usage.usd += usd

    def record_tool_call(self) -> None:
        self.usage.tool_calls += 1
```


---

## 2 · Pricing — `platform/bedrock.py`

A model with no rate-card entry makes every dollar budget in the process
unenforceable, silently in both directions. So the client refuses to exist for one.

```python
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
```


Dollars to output tokens. This is the conversion the whole clamp rests on — and
note it is the **output** rate, because `max_tokens` caps output only.

```python
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
```


The input half of a completed call, so the next one can reserve room for its own.

```python
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
```


Truncation is its own failure, deliberately not a `BudgetExceeded`.

```python
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
```


---

## 3 · The single authorisation point — `BedrockClient`

Every model call in the service goes through `_prepared`. This is the mechanism.

```python
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
```


A request the wall-clock timeout ended mid-flight, translated back into the budget
dimension that set it — so the agent loop never imports the SDK to catch it.

```python
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
```


One turn of a tool-use loop. Note the ordering: `_prepared` first, the call, then
`_last_input_usd` recorded from the real counts, then the tracker charged.

```python
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
```


The structured-report path. Same shape, plus the truncation guard — and the
charge happens **before** the raise, because those tokens were billed.

```python
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
```


---

## 4 · The consumer — `agents/loop.py`

The budget lives on the specialist's own tracker rather than on the client,
because the client is shared by every specialist in a run.

```python
def _charge(tracker: BudgetTracker, response: Any) -> None:
    """Record one model call against the specialist's own budget.

    The budget lives here rather than on the client because the client is shared
    by every specialist in a run. A `BedrockClient` may also hold a tracker of
    its own for whole-run accounting; this one is the one that stops the loop,
    and `tracker.check()` runs before each call so the stop happens before the
    money is spent rather than after.
    """
    tracker.record_model_call(
        response.input_tokens,
        response.output_tokens,
        response.usd,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
    )
```


One specialist to a typed report, or to a reason it could not. The enforcement
here: `tracker.check()` before each turn and after each tool call, the
`max_tokens` branch, and two handlers that turn a breach into a structured
outcome carrying no partial answer.

```python
async def run_specialist(
    specialist: Specialist,
    brief: str,
    *,
    principal: Principal,
    client: ModelClient,
    gateway: ToolGateway,
    guardrail: Guardrail | None = None,
    today: date | None = None,
) -> tuple[SpecialistRun, BaseModel | None]:
    """Run one specialist to a typed report, or to a reason it could not.

    The tracker is created here, from the specialist's own budget, so that a
    runaway pricing loop cannot spend the supply specialist's allowance. The
    supervisor sums what they each used; it does not hand out one pool.

    The whole run is one `state` span, with a `model` span per turn, a `tool`
    span per call from the gateway underneath, and a `decision` span wherever the
    harness concluded something. Every exit from `finish` records one, because a
    trace that only spans the successful path cannot explain a run that produced
    nothing.
    """
    tracker = BudgetTracker(specialist.budget, owner=specialist.name)
    system = (
        f"{specialist.system}\n\nToday is {today or date.today():%A %d %B %Y}.\n\n{EVIDENCE_RULE}"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
    before = len(gateway.audit)

    def finish(status: OutcomeStatus, **fields: Any) -> tuple[SpecialistRun, None]:
        outcome = Outcome(status=status, agent=specialist.name, usage=tracker.usage, **fields)
        with telemetry.span(telemetry.DECISION, f"{specialist.name}.stopped") as decision:
            telemetry.record_outcome(decision, outcome)
        return SpecialistRun(specialist.name, outcome, gateway.audit[before:]), None

    with telemetry.span(telemetry.STATE, specialist.name, max_turns=specialist.max_turns):
        try:
            for _ in range(specialist.max_turns):
                tracker.check()
                with telemetry.span(telemetry.MODEL, f"{specialist.name}.turn") as model_span:
                    turn = client.converse(system=system, messages=messages, tools=specialist.tools)
                    _charge(tracker, turn)
                    model_span.set_attribute("halo.stop_reason", str(turn.stop_reason))
                    telemetry.record_usage(model_span, tracker.usage)
                messages.append({"role": "assistant", "content": turn.content})

                if turn.stop_reason == "max_tokens":
                    # The service enforced the output ceiling and the model was
                    # never told about it, so this turn stops mid-token. Treated
                    # as an end_turn it would be indistinguishable from a
                    # specialist that had finished speaking, and the truncation
                    # would travel on into the report as though it were an
                    # answer. The partial content stays in `messages` for the
                    # trace and goes no further.
                    return finish(
                        OutcomeStatus.ESCALATED,
                        escalation_reason=(
                            f"{specialist.name} was cut off at its output ceiling mid-turn, "
                            "so the work it was describing is incomplete"
                        ),
                        next_state="await_budget_increase",
                    )

                if turn.stop_reason != "tool_use":
                    break

                results = []
                for block in turn.tool_uses:
                    route = specialist.routes.get(block.name, block.name)
                    call = await gateway.call(route, dict(block.input))
                    # Counted here as well as on the gateway. The gateway's tracker,
                    # when it has one, is the whole run; `max_tool_calls` on a
                    # specialist's budget is only a limit if the specialist's own
                    # tracker is the one counting.
                    tracker.record_tool_call()
                    tracker.check()

                    # M5: a refusal is an answer. It ends this specialist rather than
                    # becoming an error the model works around with what it holds.
                    if is_denial(call.error):
                        return finish(
                            OutcomeStatus.REFUSED,
                            escalation_reason=f"{call.name} ({call.id}) {call.error}",
                            next_state="denied_by_scope",
                        )

                    body = json.dumps(
                        {
                            "tool_call_id": call.id,
                            "result" if call.ok else "error": call.result
                            if call.ok
                            else call.error,
                        },
                        default=str,
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": wrap(Evidence(id=call.id, source=call.name, body=body)),
                            "is_error": not call.ok,
                        }
                    )
                messages.append({"role": "user", "content": results})
            else:
                return finish(
                    OutcomeStatus.ESCALATED,
                    escalation_reason=f"{specialist.name} used {specialist.max_turns} turns "
                    "without reaching an answer",
                    next_state="needs_human_sourcing",
                )

            succeeded = {call.name for call in gateway.audit[before:] if call.ok}
            if missing := specialist.required_routes - succeeded:
                return finish(
                    OutcomeStatus.ESCALATED,
                    escalation_reason=(
                        f"{specialist.name} is incomplete — these tools were never called "
                        f"successfully: {', '.join(sorted(missing))}"
                    ),
                    next_state="needs_human_sourcing",
                )

            tracker.check()
            with telemetry.span(telemetry.MODEL, f"{specialist.name}.report") as model_span:
                result = client.parse(
                    system=specialist.report_instruction,
                    user=_transcript(messages),
                    output_format=specialist.output,
                )
                _charge(tracker, result)
                telemetry.record_usage(model_span, tracker.usage)
        except Truncated as exc:
            # Distinct from the breach below on purpose. Running out of money is
            # fixed by raising a limit; an answer that outgrew its ceiling is
            # not, and an escalation that names the wrong one sends someone to
            # change a number that was never reached.
            return finish(
                OutcomeStatus.ESCALATED,
                escalation_reason=f"{specialist.name} could not finish its report: {exc}",
                next_state="await_budget_increase",
            )
        except BudgetExceeded as exc:
            # The done-when for M6. The reason names the dimension and no partial
            # answer travels with it: a truncated report reads like a complete one
            # two layers up.
            #
            # Which budget ran out is not always this specialist's. The shared model
            # client and gateway count against the run's budget, and that one can
            # trip while any specialist happens to be working. Saying "pricing
            # exhausted its budget" then sends someone to raise a limit that was
            # never reached.
            reason = (
                f"{specialist.name} exhausted its budget: {exc}"
                if exc.owner == specialist.name
                else f"the {exc.owner} budget ran out while {specialist.name} was working: {exc}"
            )
            return finish(
                OutcomeStatus.ESCALATED,
                escalation_reason=reason,
                next_state="await_budget_increase",
            )

        report = result.parsed
        calls = gateway.audit[before:]

        if problems := verify_figures(report.figure_checks(), calls):
            return finish(
                OutcomeStatus.ESCALATED,
                payload=report.model_dump(mode="json"),
                escalation_reason=(
                    f"{specialist.name} reported figures that could not be traced to the tools "
                    f"that supposedly produced them: {'; '.join(problems)}"
                ),
                next_state="needs_regrounding",
            )

        if guardrail is not None:
            verdict = guardrail.inspect(_prose(report), surface=Surface.OUTPUT)
            if verdict.blocked:
                return finish(
                    OutcomeStatus.REFUSED,
                    payload=report.model_dump(mode="json"),
                    escalation_reason=f"{specialist.name} was blocked: {verdict.summary()}",
                    next_state="blocked_by_guardrail",
                )

        outcome = Outcome(
            status=OutcomeStatus.COMPLETED,
            agent=specialist.name,
            payload=report.model_dump(mode="json"),
            next_state="reported",
            usage=tracker.usage,
        )
        # The successful path gets a decision span too, and it carries the count
        # of figures that were checked rather than the figures. "Verified" with
        # nothing behind it is the same sentence whether three figures were
        # traced or none were reported at all.
        with telemetry.span(
            telemetry.DECISION,
            f"{specialist.name}.verified",
            figures_checked=len(report.figure_checks()),
            tool_calls=len(calls),
        ) as decision:
            telemetry.record_outcome(decision, outcome)

        return SpecialistRun(specialist.name, outcome, calls), report
```


---

## Enforcement points

| Where | Catches |
|---|---|
| `loop.py` — before each turn, after each tool call, before the report | accumulated spend, between steps |
| `gateway.py:167` | tool-call count, before dispatch |
| `bedrock.py` `_prepared` | the call about to be made — clamp + timeout |
| `bedrock.py` `_cut_off` | a request killed in flight |
| `loop.py` stop_reason branch, and `parse` | a ceiling the service enforced |

The same two handlers exist in `sourcing.py` (both call paths), `advisor.py` and
`drafter.py` — five call sites, all catching `Truncated` before `BudgetExceeded`.

## Two properties that hold everywhere

- **No partial payload travels with a breach.** A truncated report reads exactly
  like a complete one two layers up.
- **The reason names the owner.** With five trackers live in a quote run, a breach
  that does not say whose it was sends someone to raise the wrong limit.

## Three things the arithmetic is not

| Intuition | Actually |
|---|---|
| A dollar budget converts to a token budget | Input and output are priced differently. On global Sonnet 4.6, $1.00 is **333,333 input tokens or 66,666 output tokens**, or any mix. |
| `max_tokens` bounds what a call costs | It bounds **output only**. `max_tokens=100_000` authorises 100k output tokens — **$1.50** — before any input is counted. |
| Setting 80,000 means 80,000 remain | `max_tokens` is **per request and has no memory**. All accumulation lives in `BudgetTracker`. |

## Tests

`tests/test_budget_inflight.py` — 31 cases covering the clamp arithmetic, the
input reservation, the timeout translation, the truncation exits, and the
loop-level escalation. Full suite: 401 passing.
