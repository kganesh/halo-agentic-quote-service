"""A call is sized to the budget before it is made, and cut off if it overruns.

`check` runs between steps, so on its own it can only report an overspend after
the call that caused it has been paid for. A non-streaming request cannot be
stopped part-way and there is no server-side stop, so the two things that can
actually hold a limit are both decided before the request goes out:

  - `max_tokens` clamped to what the remaining dollars buy, so the call cannot
    cost more than is left;
  - the remaining wall clock as the request timeout, so a call that would
    outlast the budget is dropped in flight.

The rate figures here come from the same account rate card as
`test_cache_pricing`: global Sonnet 4.6 bills 3.00/15.00 per million.
"""

from decimal import Decimal

import pytest

from halo.platform.bedrock import (
    MIN_OUTPUT_TOKENS,
    BedrockClient,
    TokenCounts,
    Truncated,
    affordable_output_tokens,
    input_usd_for,
)
from halo.platform.budget import Budget, BudgetExceeded, BudgetTracker

GLOBAL = "global.anthropic.claude-sonnet-4-6"
UNPRICED = "fake.specialist"


class Clock:
    """A hand-wound clock. `_prepared` reads it more than once per call, so a
    fixed sequence would run out mid-assertion."""

    def __init__(self, at: float = 0.0):
        self.at = at

    def __call__(self) -> float:
        return self.at


def budget(*, usd="1.00", seconds=300.0) -> Budget:
    return Budget(
        wall_clock_seconds=seconds,
        max_tokens=200_000,
        max_tool_calls=20,
        max_usd=Decimal(usd),
    )


class FakeUsage:
    input_tokens = 1_000
    output_tokens = 4_000
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    cache_creation = None


class FakeResponse:
    """An answer that stopped before it was finished.

    Both shapes the SDK reports for it: `stop_reason == "max_tokens"`, and a
    `parsed_output` of `None` from structured output that did not close.
    """

    content: list = []

    def __init__(self, *, stop_reason: str, parsed_output=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output
        self.usage = FakeUsage()


def truncated_response(*, stop_reason: str) -> FakeResponse:
    return FakeResponse(stop_reason=stop_reason)


class FakeMessages:
    """Stands in for `client.messages`, recording how it was called."""

    def __init__(self, outcome):
        self._outcome = outcome
        self.max_tokens: int | None = None

    def _answer(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    create = _answer
    parse = _answer


class FakeSdkClient:
    """Stands in for the Anthropic SDK client, recording `with_options`."""

    def __init__(self, outcome=None):
        self.messages = FakeMessages(outcome)
        self.options: dict | None = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def client_with(tracker: BudgetTracker, outcome=None) -> tuple[BedrockClient, FakeSdkClient]:
    """A real `BedrockClient` with the SDK underneath replaced.

    Built through the real constructor so the rate-card guard and the surface
    choice are the ones under test, not a reimplementation of them.
    """
    client = BedrockClient(model=GLOBAL, tracker=tracker)
    sdk = FakeSdkClient(outcome)
    client._client = sdk
    return client, sdk


class TestWhatTheRemainingBudgetBuys:
    def test_a_dollar_buys_what_the_rate_card_says(self):
        """Global Sonnet 4.6 output is 15.00/M, so $1.00 is 66,666 tokens."""
        assert affordable_output_tokens(GLOBAL, Decimal("1.00")) == 66_666

    def test_it_shrinks_as_the_budget_is_spent(self):
        half = affordable_output_tokens(GLOBAL, Decimal("0.50"))
        assert half == pytest.approx(33_333, abs=1)

    def test_an_exhausted_budget_buys_nothing(self):
        assert affordable_output_tokens(GLOBAL, Decimal("0.00")) == 0

    def test_an_unpriced_model_is_not_clamped(self):
        """Zero means "no rate card", the same convention `estimate_usd` uses.

        A caller cannot tell the difference from the number alone, which is why
        `BedrockClient` refuses to be constructed for an unpriced model rather
        than relying on this."""
        assert affordable_output_tokens(UNPRICED, Decimal("1.00")) == 0


class TestWhatIsLeft:
    def test_remaining_dollars_come_off_the_limit(self):
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(1000, 1000, Decimal("0.30"))
        assert tracker.remaining_usd == Decimal("0.70")

    def test_remaining_dollars_never_go_negative(self):
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(1000, 1000, Decimal("1.50"))
        assert tracker.remaining_usd == Decimal("0.00")

    def test_remaining_seconds_come_off_the_clock(self):
        clock = Clock()
        tracker = BudgetTracker(budget(seconds=60.0), now=clock)
        clock.at = 20.0
        assert tracker.remaining_seconds == 40.0

    def test_remaining_seconds_never_go_negative(self):
        clock = Clock()
        tracker = BudgetTracker(budget(seconds=60.0), now=clock)
        clock.at = 90.0
        assert tracker.remaining_seconds == 0.0


class TestTheCallIsSizedToTheBudget:
    def test_a_full_budget_leaves_the_requested_cap_alone(self):
        client, _ = client_with(BudgetTracker(budget(usd="1.00")))
        assert client._prepared(8_000)[1] == 8_000

    def test_a_nearly_spent_budget_clamps_the_cap(self):
        """$0.10 left buys 6,666 output tokens, which is under the 8,000 asked
        for. The call is made small enough to fit rather than being made at full
        size and found to have overspent afterwards."""
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(0, 0, Decimal("0.90"))
        client, _ = client_with(tracker)
        assert client._prepared(8_000)[1] == 6_666

    def test_the_clamp_reaches_the_sdk(self):
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(0, 0, Decimal("0.90"))
        client, sdk = client_with(tracker, outcome=RuntimeError("unused"))
        with pytest.raises(RuntimeError):
            client.converse(system="s", messages=[], tools=[], max_tokens=8_000)
        assert sdk.messages.max_tokens == 6_666

    def test_a_budget_too_small_to_answer_stops_the_run(self):
        """Enough left to pass `check`, not enough to buy a usable answer.

        Spending it on a report truncated mid-sentence costs money and produces
        nothing, so this ends as the budget breach it is."""
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(0, 0, Decimal("0.999"))
        client, _ = client_with(tracker)
        with pytest.raises(BudgetExceeded) as breach:
            client._prepared(8_000)
        assert breach.value.dimension == "max_usd"

    def test_that_floor_is_where_it_says_it_is(self):
        tracker = BudgetTracker(budget(usd="1.00"))
        client, _ = client_with(tracker)
        assert client._prepared(8_000)[1] >= MIN_OUTPUT_TOKENS

    def test_an_untracked_client_is_not_clamped(self):
        """A client with no tracker is doing something other than a budgeted
        run — `doctor`, a one-off — and has no limit to be sized against."""
        client, sdk = client_with(BudgetTracker(budget()))
        client._tracker = None
        assert client._prepared(8_000) == (sdk, 8_000)
        assert sdk.options is None


class TestTheCallIsBoundedByTheClock:
    def test_the_request_gets_the_remaining_wall_clock_as_its_timeout(self):
        clock = Clock()
        tracker = BudgetTracker(budget(seconds=60.0), now=clock)
        client, sdk = client_with(tracker)
        clock.at = 20.0
        client._prepared(8_000)
        assert sdk.options["timeout"] == 40.0

    def test_the_request_is_not_retried(self):
        """The SDK retries timeouts, and at its default of 2 a call given the
        remaining wall clock could run for three times it. A budget that can be
        overrun threefold is not a budget."""
        client, sdk = client_with(BudgetTracker(budget()))
        client._prepared(8_000)
        assert sdk.options["max_retries"] == 0


class TestACallCutOffInFlight:
    def raises_timeout(self, tracker):
        import anthropic

        return client_with(tracker, outcome=anthropic.APITimeoutError(request=None))

    def test_converse_reports_it_as_the_wall_clock_budget(self):
        client, _ = self.raises_timeout(BudgetTracker(budget(seconds=60.0)))
        with pytest.raises(BudgetExceeded) as breach:
            client.converse(system="s", messages=[], tools=[])
        assert breach.value.dimension == "wall_clock_seconds"

    def test_parse_reports_it_the_same_way(self):
        client, _ = self.raises_timeout(BudgetTracker(budget(seconds=60.0)))
        with pytest.raises(BudgetExceeded) as breach:
            client.parse(system="s", user="u", output_format=Budget)
        assert breach.value.dimension == "wall_clock_seconds"

    def test_the_breach_names_the_budget_that_set_the_timeout(self):
        """The loop decides what to say from `owner`. A timeout arriving as an
        SDK error would lose that, and read as an infrastructure fault rather
        than a limit doing its job."""
        tracker = BudgetTracker(budget(seconds=60.0), owner="pricing")
        client, _ = self.raises_timeout(tracker)
        with pytest.raises(BudgetExceeded) as breach:
            client.converse(system="s", messages=[], tools=[])
        assert breach.value.owner == "pricing"


class TestInputIsReservedBeforeOutputIsSized:
    """The transcript and its tool results are resent every turn. A cap sized
    against the whole remaining budget ignores that, and authorizes a call whose
    input alone could pass the limit."""

    def test_the_first_call_reserves_nothing(self):
        """No call has been measured yet, so there is nothing to reserve from."""
        client, _ = client_with(BudgetTracker(budget(usd="1.00")))
        assert client._prepared(100_000)[1] == 66_666

    def test_a_measured_input_is_held_back_from_the_next_call(self):
        """30k input tokens cost $0.09 at 3.00/M. With $0.10 left, sizing output
        against the full remainder would allow another $0.10 of output on top of
        that input — $0.19 against $0.10."""
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(0, 0, Decimal("0.90"))
        client, _ = client_with(tracker)
        client._last_input_usd = Decimal("0.09")

        # $0.10 - $0.09 = $0.01, which buys 666 output tokens, not 6,666.
        assert client._prepared(8_000)[1] == 666

    def test_a_reservation_larger_than_the_remainder_stops_the_run(self):
        tracker = BudgetTracker(budget(usd="1.00"))
        tracker.record_model_call(0, 0, Decimal("0.95"))
        client, _ = client_with(tracker)
        client._last_input_usd = Decimal("0.20")
        with pytest.raises(BudgetExceeded) as breach:
            client._prepared(8_000)
        assert breach.value.dimension == "max_usd"

    def test_the_reservation_is_read_off_the_rate_the_call_actually_paid(self):
        """Cached input costs a tenth. Reserving at the uncached rate — which is
        what a `count_tokens` round trip would report — would hold back ten times
        too much against a prefix this project caches on purpose."""
        cached = TokenCounts(input_tokens=0, cache_read_tokens=100_000)
        fresh = TokenCounts(input_tokens=100_000)
        assert input_usd_for(GLOBAL, cached) == input_usd_for(GLOBAL, fresh) / 10


class TestAnAnswerCutOffAtTheCeiling:
    def test_a_truncated_report_is_not_returned_as_one(self):
        """`max_tokens` is enforced by the service and the model is not told
        about it, so the call succeeds with a partial answer. Handing that back
        as a `ModelResult` would put half a report where a whole one belongs."""
        client, _ = client_with(
            BudgetTracker(budget()), outcome=truncated_response(stop_reason="max_tokens")
        )
        with pytest.raises(Truncated):
            client.parse(system="s", user="u", output_format=Budget)

    def test_an_unparseable_answer_is_treated_the_same(self):
        """The SDK types `parsed_output` optional and returns `None` when the
        structured output did not finish."""
        client, _ = client_with(
            BudgetTracker(budget()), outcome=truncated_response(stop_reason="end_turn")
        )
        with pytest.raises(Truncated):
            client.parse(system="s", user="u", output_format=Budget)

    def test_the_tokens_it_burned_are_still_charged(self):
        """They were generated and billed. Dropping the cost with the answer
        would let a run retry its way past a budget it had already spent."""
        tracker = BudgetTracker(budget())
        client, _ = client_with(tracker, outcome=truncated_response(stop_reason="max_tokens"))
        with pytest.raises(Truncated):
            client.parse(system="s", user="u", output_format=Budget)
        assert tracker.usage.output_tokens == 4_000
        assert tracker.usage.usd > 0

    def test_it_is_not_reported_as_running_out_of_money(self):
        """Raising `max_usd` fixes a breach. It does not fix a report that
        outgrew its ceiling, so the two must not arrive as the same failure."""
        client, _ = client_with(
            BudgetTracker(budget()), outcome=truncated_response(stop_reason="max_tokens")
        )
        with pytest.raises(Truncated) as cut:
            client.parse(system="s", user="u", output_format=Budget)
        assert not isinstance(cut.value, BudgetExceeded)


class StubGateway:
    """A gateway that is never reached. The turns under test stop before tools."""

    async def call(self, name, arguments):
        raise AssertionError("no tool should be called on a truncated turn")

    @property
    def audit(self):
        return []


class OneTurnModel:
    """Returns a single turn with a chosen `stop_reason`, then refuses to be
    called again — so a test that expects the loop to stop fails loudly if it
    carries on instead."""

    model = GLOBAL

    def __init__(self, stop_reason: str):
        self._stop_reason = stop_reason
        self.turns = 0
        self.reports = 0

    def converse(self, *, system, messages, tools, max_tokens=8_000):
        from halo.platform.bedrock import ModelTurn

        self.turns += 1
        return ModelTurn(
            content=[{"type": "text", "text": "half a sen"}],
            stop_reason=self._stop_reason,
            input_tokens=200,
            output_tokens=60,
            usd=Decimal("0.01"),
        )

    def parse(self, *, system, user, output_format, max_tokens=16_000):
        self.reports += 1
        raise AssertionError("a truncated turn must not go on to be reported")


def a_specialist():
    from halo.agents.loop import Specialist

    class Report(Budget):
        def figure_checks(self):
            return []

    return Specialist(
        name="pricing",
        system="price it",
        tools=[{"name": "get_price"}],
        routes={"get_price": "pim_oms.get_price"},
        output=Report,
        budget=budget(),
    )


class TestASpecialistCutOffMidTurn:
    """`max_tokens` is enforced by the service and the model is never told about
    it, so a turn that hits it stops mid-token and still returns 200. Treated as
    an `end_turn` it is indistinguishable from a specialist that had finished
    speaking."""

    async def run(self, stop_reason: str):
        from halo.agents.loop import run_specialist
        from halo.platform.identity import Principal, Role

        model = OneTurnModel(stop_reason)
        run, report = await run_specialist(
            a_specialist(),
            "brief",
            principal=Principal(
                user_id="usr-mwest01",
                tenant_id="tnt-mwest1",
                role=Role.SELLER,
                account_ids=("acct-mwest02",),
            ),
            client=model,
            gateway=StubGateway(),
        )
        return run, report, model

    async def test_it_escalates_rather_than_reporting(self):
        from halo.platform.outcome import OutcomeStatus

        run, report, model = await self.run("max_tokens")
        assert run.outcome.status is OutcomeStatus.ESCALATED
        assert report is None

    async def test_no_partial_answer_travels_with_it(self):
        run, _, _ = await self.run("max_tokens")
        assert run.outcome.payload is None

    async def test_the_reason_says_it_was_cut_off(self):
        run, _, _ = await self.run("max_tokens")
        assert "cut off at its output ceiling" in run.outcome.escalation_reason

    async def test_it_never_reaches_the_report_call(self):
        """The failure this replaces: breaking on a truncated turn as though it
        were an `end_turn`, then paying for a report written from it."""
        _, _, model = await self.run("max_tokens")
        assert model.reports == 0
