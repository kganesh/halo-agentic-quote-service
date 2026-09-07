"""M2: build a quote from tools, and prove every figure came from one.

The M1 drafter invented its numbers and reported that it had done so. This agent
has a catalogue, a supplier and a carrier behind an MCP gateway. The only figures
it may use are figures a tool returned.

Two mechanisms do this work. Neither is a prompt instruction:

- For every money figure, the model reports the `tool_call_id` it came from.
- `verify` then checks two things. First, that the cited call exists in the
  audit. Second, that the value actually appears in that call's result.

The second check is the important one. A model can cite a real call id for a
number that call never returned. Checking only the id would accept it.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from halo.agents.provenance import FigureCheck, SourcedFigure, verify_figures
from halo.domain.catalog import DecorationMethod
from halo.domain.quote import Citation, CitationKind, DecorationCharge, Quote, QuoteLine
from halo.domain.request import QuoteRequest
from halo.platform.bedrock import ModelClient, Truncated
from halo.platform.budget import BudgetExceeded, BudgetTracker
from halo.platform.envelope import EVIDENCE_RULE, Evidence, wrap
from halo.platform.gateway import ToolCall, ToolGateway
from halo.platform.guardrails import Guardrail, GuardrailVerdict, Surface
from halo.platform.identity import Principal, is_denial
from halo.platform.outcome import Outcome, OutcomeStatus

AGENT_NAME = "sourcing"
MAX_TURNS = 12
MAX_NUDGES = 2

REQUIRED_TOOLS = {
    "pim_oms.get_price",
    "supplier.get_decoration_charges",
    "supplier.earliest_ship_date",
    "shipping.estimate_freight",
}
"""Tools that a complete sourcing must have called.

The loop enforces this. The prompt does not ask for it. Three live runs failed in
three different ways. In one, the model calculated a ship date backwards from the
customer's deadline. In another, it skipped `earliest_ship_date` completely and
cited an id that did not exist. Each time, the obvious fix was another sentence
in the prompt. A sequence that the answer depends on belongs in the harness,
where it either happened or it did not.
"""

SYSTEM_PROMPT = """\
You are sourcing a quote for a HALO sales representative, using the tools
provided. HALO distributes branded merchandise.

Today's date is {today:%A %d %B %Y}.

Every money figure and every date in your answer must come from a tool result.
You have no pricing, stock, capacity or freight knowledge of your own, and a
figure you did not fetch is not usable. If a tool cannot give you something,
record it in `unresolved` instead of supplying it yourself.

`unresolved` is only for figures no tool could give you. Anything a person has to
decide before the order is entered goes in `open_questions`. Examples: a garment
colour, a size breakdown, an imprint placement. A quote with open questions is
still a quote. A quote with an unsourced figure is not.

Work in this order:

1. `search_products` to find a SKU matching the request.
2. `get_price` for that SKU at the requested quantity.
3. `check_inventory` with the `method` argument set, so only suppliers that can
   both hold the goods and decorate them come back.
4. `get_decoration_charges` for the decoration method and colour count.
5. `earliest_ship_date` for the supplier that can actually do the run.
6. `estimate_freight` to the destination state.

If the request names an account, call `get_account` for it first. A refusal is
an answer: report it and stop. Do not continue with details you were given
earlier in this conversation, and do not quote the account from memory. You are
not being asked to be careful here — the run is checked, and a quote assembled
after a refusal is discarded.

`promised_ship_date` is the `ship_date` string that `earliest_ship_date` returned,
copied exactly. Do not compute it, adjust it, or work backwards from the date the
customer asked for. If the supplier's date is later than the customer's date,
record that in `unresolved`. Do not change the date.

For each figure you report, give the `tool_call_id` of the call that produced it.
Every tool result is labelled with its id. Do not guess an id. Do not reuse an id
from a different figure. Both are checked.

{evidence_rule}"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_account",
        "description": (
            "The customer record for the account this quote is for. Returns a 403 "
            "if the account is not one you may read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
    {
        "name": "search_products",
        "description": "Find catalogue SKUs matching a plain-English product description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "outerwear",
                        "knits",
                        "headwear",
                        "drinkware",
                        "bags",
                        "tech",
                        "writing",
                    ],
                },
                "limit": {"type": "integer"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "get_price",
        "description": "Unit price for a SKU at a quantity, from the quantity-break table.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
            "required": ["sku", "quantity"],
        },
    },
    {
        "name": "check_inventory",
        "description": "Which suppliers hold enough of a SKU to cover a quantity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "quantity": {"type": "integer"},
                "color": {"type": "string"},
                "method": {
                    "type": "string",
                    "description": "decoration method; restricts to suppliers that offer it",
                },
            },
            "required": ["sku", "quantity", "method"],
        },
    },
    {
        "name": "get_decoration_charges",
        "description": "Setup and per-unit run charges for a decoration job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "colors": {"type": "integer"},
                "units": {"type": "integer"},
            },
            "required": ["method", "colors", "units"],
        },
    },
    {
        "name": "earliest_ship_date",
        "description": "First day a supplier can finish the run, capacity and lead time both.",
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "method": {"type": "string"},
                "units": {"type": "integer"},
                "from_day": {"type": "string", "description": "ISO date"},
            },
            "required": ["supplier_id", "method", "units", "from_day"],
        },
    },
    {
        "name": "estimate_freight",
        "description": "Ground transit days and freight cost to a destination state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_state": {"type": "string"},
                "units": {"type": "integer"},
                "residential": {"type": "boolean"},
            },
            "required": ["to_state", "units"],
        },
    },
]

TOOL_ROUTES = {
    "get_account": "accounts.get_account",
    "search_products": "pim_oms.search_products",
    "get_price": "pim_oms.get_price",
    "check_inventory": "supplier.check_inventory",
    "get_decoration_charges": "supplier.get_decoration_charges",
    "earliest_ship_date": "supplier.earliest_ship_date",
    "estimate_freight": "shipping.estimate_freight",
}


class SourcingDecision(BaseModel):
    """What the model decided, with a source for every figure."""

    sku: str
    product_name: str
    quantity: int = Field(ge=1)
    unit_price: SourcedFigure
    decoration_method: DecorationMethod
    imprint_colors: int = Field(ge=1)
    decoration_setup_fee: SourcedFigure
    decoration_run_charge_per_unit: SourcedFigure
    supplier_id: str
    supplier_name: str
    promised_ship_date: date
    ship_date_tool_call_id: str = Field(pattern=r"^tc-\d{4}$")
    freight_cost: SourcedFigure
    unresolved: list[str] = Field(default_factory=list)
    """Figures no tool could supply. Any entry here means there is no quote."""
    open_questions: list[str] = Field(default_factory=list)
    """Things a person must decide before the order is entered. For example a
    colour choice, a size breakdown, or an imprint placement.

    This is separate from `unresolved` because they are different problems. A
    quote with a colour still to be chosen is a normal quote. A quote with an
    unsourced price is not a quote at all. Putting both in `unresolved` made
    every realistic request escalate, which made `completed` unreachable in
    practice."""


def verify(decision: SourcingDecision, audit: list[ToolCall]) -> list[str]:
    """Problems with the decision's provenance. Empty means every figure checks out.

    Each figure names the field it could legitimately have come from. Without
    this, any number in the result would match. See `provenance.values_under`.
    """
    return verify_figures(
        [
            FigureCheck(
                "unit_price",
                decision.unit_price.value,
                decision.unit_price.tool_call_id,
                ("price",),
            ),
            FigureCheck(
                "decoration_setup_fee",
                decision.decoration_setup_fee.value,
                decision.decoration_setup_fee.tool_call_id,
                ("setup",),
            ),
            FigureCheck(
                "decoration_run_charge_per_unit",
                decision.decoration_run_charge_per_unit.value,
                decision.decoration_run_charge_per_unit.tool_call_id,
                ("run_charge",),
            ),
            FigureCheck(
                "freight_cost",
                decision.freight_cost.value,
                decision.freight_cost.tool_call_id,
                ("freight", "cost"),
            ),
            FigureCheck(
                "promised_ship_date",
                decision.promised_ship_date,
                decision.ship_date_tool_call_id,
                ("date",),
            ),
        ],
        audit,
    )


def _citation(field_name: str, call_id: str, audit: list[ToolCall]) -> Citation:
    call = next(c for c in audit if c.id == call_id)
    return Citation(
        kind=CitationKind.TOOL_CALL,
        ref=call_id,
        supporting_text=f"{field_name} from {call.name}({json.dumps(call.arguments, default=str)})",
    )


def assemble(decision: SourcingDecision, audit: list[ToolCall], *, account_id: str) -> Quote:
    """Turn a verified decision into a Quote. Call `verify` first.

    `account_id` is passed in rather than read off the decision. The model does
    not get to say who the quote is for: that was settled at admission, and a
    quote addressed to an account the principal cannot read is the thing M5
    exists to prevent.
    """
    return Quote(
        request_id=f"req-{len(audit):04d}",
        account_id=account_id,
        lines=[
            QuoteLine(
                sku=decision.sku,
                description=decision.product_name,
                quantity=decision.quantity,
                unit_price=decision.unit_price.value,
                citations=[_citation("unit price", decision.unit_price.tool_call_id, audit)],
            )
        ],
        decoration=DecorationCharge(
            method=decision.decoration_method,
            colors=decision.imprint_colors,
            setup_fee=decision.decoration_setup_fee.value,
            run_charge_per_unit=decision.decoration_run_charge_per_unit.value,
            citations=[
                _citation("setup fee", decision.decoration_setup_fee.tool_call_id, audit),
                _citation(
                    "run charge", decision.decoration_run_charge_per_unit.tool_call_id, audit
                ),
            ],
        ),
        shipping_cost=decision.freight_cost.value,
        shipping_citations=[_citation("freight", decision.freight_cost.tool_call_id, audit)],
        promised_ship_date=decision.promised_ship_date,
        margin_pct=Decimal("0.0"),
    )


async def source_quote(
    request: QuoteRequest,
    *,
    principal: Principal,
    client: ModelClient,
    gateway: ToolGateway,
    tracker: BudgetTracker,
    today: date | None = None,
    guardrail: Guardrail | None = None,
) -> Outcome:
    """Run the tool loop, verify what comes back, and assemble or escalate."""
    quarantined: list[tuple[str, GuardrailVerdict]] = []
    system = SYSTEM_PROMPT.format(today=today or date.today(), evidence_rule=EVIDENCE_RULE)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": request.model_dump_json(indent=2)}
    ]

    nudges = 0
    try:
        for _ in range(MAX_TURNS):
            turn = client.converse(system=system, messages=messages, tools=TOOLS)
            messages.append({"role": "assistant", "content": turn.content})

            if turn.stop_reason == "max_tokens":
                # Cut off at the output ceiling, not finished. Falling through
                # would nudge it for the tools it still owes, on top of a turn
                # that stopped mid-token — spending more budget to extend an
                # answer that was already truncated once.
                return Outcome(
                    status=OutcomeStatus.ESCALATED,
                    agent=AGENT_NAME,
                    escalation_reason=(
                        "sourcing was cut off at its output ceiling mid-turn, so the work "
                        "it was describing is incomplete"
                    ),
                    next_state="await_budget_increase",
                    usage=tracker.usage,
                )

            if turn.stop_reason != "tool_use":
                missing = REQUIRED_TOOLS - {c.name for c in gateway.audit if c.ok}
                if not missing or nudges >= MAX_NUDGES:
                    break
                nudges += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not called these yet, and the quote cannot be "
                            f"completed without them: {', '.join(sorted(missing))}. "
                            "Call them now."
                        ),
                    }
                )
                continue

            results = []
            for block in turn.tool_uses:
                route = TOOL_ROUTES.get(block.name, block.name)
                call = await gateway.call(route, dict(block.input))
                # The id goes back with the result so the model can cite it.
                # Nothing else in the loop tells the model what a tool_call_id
                # looks like.
                payload = {
                    "tool_call_id": call.id,
                    "result" if call.ok else "error": call.result if call.ok else call.error,
                }
                body = json.dumps(payload, default=str)
                # A supplier's own text reaches the model here and nowhere else.
                # It goes in wrapped, and anything addressed to the model is
                # recorded on the way past. The run continues: a hostile
                # production note is not a reason to abandon a legitimate
                # quote, it is a reason not to obey the note.
                if guardrail is not None:
                    verdict = guardrail.inspect(body, surface=Surface.INPUT)
                    if verdict.categories:
                        quarantined.append((call.id, verdict))
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
            return Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                escalation_reason=f"gave up after {MAX_TURNS} turns without a decision",
                next_state="needs_human_sourcing",
                usage=tracker.usage,
            )

        if missing := REQUIRED_TOOLS - {c.name for c in gateway.audit if c.ok}:
            return Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                escalation_reason=(
                    "sourcing is incomplete — these tools were never called "
                    f"successfully: {', '.join(sorted(missing))}"
                ),
                next_state="needs_human_sourcing",
                usage=tracker.usage,
            )

        decision_result = client.parse(
            system="Report the sourcing you just completed, with a tool_call_id per figure.",
            user=_transcript(messages),
            output_format=SourcingDecision,
        )
    except Truncated as exc:
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            escalation_reason=f"the sourcing report could not be finished: {exc}",
            next_state="await_budget_increase",
            usage=tracker.usage,
        )
    except BudgetExceeded as exc:
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            escalation_reason=f"budget exhausted while sourcing: {exc}",
            next_state="await_budget_increase",
            usage=tracker.usage,
        )

    decision = decision_result.parsed
    audit = gateway.audit

    # Design rule 03, enforced rather than requested. The prompt asks the model
    # to report a refusal; this makes it so whether or not the model complied.
    # The failure being prevented is specific: the model already has the account
    # in its context from an earlier turn, the tool says no, and it answers from
    # what it is still holding. Nothing in the answer looks wrong.
    if denied := [call for call in audit if is_denial(call.error)]:
        return Outcome(
            status=OutcomeStatus.REFUSED,
            agent=AGENT_NAME,
            escalation_reason=(
                "a tool refused this principal and the run cannot continue: "
                + "; ".join(f"{call.name} ({call.id}) {call.error}" for call in denied)
            ),
            next_state="denied_by_scope",
            usage=tracker.usage,
        )

    # What the seller would read, checked before anything is assembled. The
    # figures are already tied to tool calls by `verify`; this catches the other
    # half, where the prose around them promises something no tool can give.
    if guardrail is not None:
        verdict = guardrail.inspect(_decision_text(decision), surface=Surface.OUTPUT)
        if verdict.blocked:
            return Outcome(
                status=OutcomeStatus.REFUSED,
                agent=AGENT_NAME,
                payload=decision.model_dump(mode="json"),
                escalation_reason=f"the draft was blocked before assembly: {verdict.summary()}",
                next_state="blocked_by_guardrail",
                usage=tracker.usage,
            )

    if problems := verify(decision, audit):
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            payload=decision.model_dump(mode="json"),
            escalation_reason="figures could not be traced to the tools that supposedly "
            f"produced them: {'; '.join(problems)}",
            next_state="needs_regrounding",
            usage=tracker.usage,
        )

    if decision.unresolved:
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            payload=decision.model_dump(mode="json"),
            evidence=[_citation("unit price", decision.unit_price.tool_call_id, audit)],
            escalation_reason="sourced, but incomplete: " + "; ".join(decision.unresolved),
            next_state="needs_human_sourcing",
            usage=tracker.usage,
        )

    quote = assemble(decision, audit, account_id=request.account_id or principal.account_ids[0])
    payload = quote.model_dump(mode="json")
    payload["open_questions"] = decision.open_questions
    # A quote can be correct and still have been quoted at while something tried
    # to steer it. That attempt belongs in the record either way, so that a note
    # which got as far as the model is never invisible just because it failed.
    if quarantined:
        payload["quarantined"] = [
            {"tool_call_id": call_id, "finding": verdict.summary()}
            for call_id, verdict in quarantined
        ]
    return Outcome(
        status=OutcomeStatus.COMPLETED,
        agent=AGENT_NAME,
        payload=payload,
        evidence=quote.all_citations,
        next_state="ready_for_margin_review",
        usage=tracker.usage,
    )


def _decision_text(decision: SourcingDecision) -> str:
    """The free text a seller would read. Numbers are checked elsewhere.

    `unresolved` and `open_questions` are included because they are the fields a
    model writes sentences into, which makes them where a commitment ends up
    when one is made at all.
    """
    return "\n".join(
        [
            decision.product_name,
            decision.supplier_name,
            *decision.unresolved,
            *decision.open_questions,
        ]
    )


def _transcript(messages: list[dict[str, Any]]) -> str:
    """A plain-text rendering of the loop, used by the reporting call.

    This keeps only tool results and assistant text. The reporting call needs the
    ids and the figures. It does not need the model's reasoning about them.
    """
    lines: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            lines.append(f"REQUEST: {content}")
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                lines.append(f"TOOL RESULT: {block['content']}")
            elif getattr(block, "type", None) == "text":
                lines.append(f"ASSISTANT: {block.text}")
    return "\n".join(lines)
