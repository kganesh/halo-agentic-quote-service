"""One bounded tool-use loop, run four times with different equipment.

M2 was a single loop holding pricing, supply, policy and logistics at once. It
worked and it was fragile in a specific way: the model spent turns re-reading its
own transcript, and when one part went wrong the whole run went wrong with it.

A `Specialist` is that loop with the scope narrowed — a tool subset, a system
prompt about one job, a typed report, and its own budget. The narrowing is the
point. A specialist that runs out of budget takes its own question down and
nothing else, and the supervisor is told which question that was.

Everything the earlier milestones added still applies here, in one place instead
of four: tool results arrive inside an evidence envelope (M4), a scope denial
ends the run (M5), and a reported figure has to appear in the call it cites (M2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel

from halo.agents.provenance import FigureCheck, verify_figures
from halo.platform import telemetry
from halo.platform.bedrock import ModelClient, Truncated
from halo.platform.budget import Budget, BudgetExceeded, BudgetTracker
from halo.platform.envelope import EVIDENCE_RULE, Evidence, wrap
from halo.platform.gateway import ToolCall, ToolGateway
from halo.platform.guardrails import Guardrail, Surface
from halo.platform.identity import Principal, is_denial
from halo.platform.outcome import Outcome, OutcomeStatus


class Report(Protocol):
    """What a specialist's structured answer must be able to tell the supervisor."""

    def figure_checks(self) -> list[FigureCheck]: ...


@dataclass(frozen=True)
class Specialist:
    """One job, one tool subset, one budget.

    `required` is enforced by the loop rather than asked for in the prompt. A
    sequence the answer depends on belongs in the harness, where it either
    happened or it did not — three live M2 runs failed three different ways on
    exactly this, and each time the obvious fix was another sentence in the
    prompt.
    """

    name: str
    system: str
    tools: list[dict[str, Any]]
    routes: dict[str, str]
    output: type[BaseModel]
    budget: Budget
    required: frozenset[str] = frozenset()
    max_turns: int = 8
    report_instruction: str = "Report what you found, with a tool_call_id per figure."

    @property
    def required_routes(self) -> set[str]:
        return {self.routes[name] for name in self.required}


@dataclass
class SpecialistRun:
    """A finished specialist: its outcome, and what it spent getting there."""

    specialist: str
    outcome: Outcome
    calls: list[ToolCall] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome.status is OutcomeStatus.COMPLETED


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


def _prose(report: BaseModel) -> str:
    """Every free-text field of a report, for the output guardrail.

    Numbers are checked by provenance and do not need reading. Sentences are
    where a commitment gets made, so they are what gets inspected.
    """
    return "\n".join(
        str(value)
        for value in report.model_dump().values()
        if isinstance(value, str)
        or (isinstance(value, list) and all(isinstance(item, str) for item in value) and value)
    )


def _transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            lines.append(f"BRIEF: {content}")
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                lines.append(f"TOOL RESULT: {block['content']}")
            elif getattr(block, "type", None) == "text":
                lines.append(f"ASSISTANT: {block.text}")
    return "\n".join(lines)
