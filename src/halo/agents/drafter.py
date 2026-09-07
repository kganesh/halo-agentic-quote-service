"""M1: draft a quote with no catalogue, no supplier and no policy corpus.

This agent is meant to be wrong. It exists so that the failure has a name and a
shape before we build the machinery that fixes it. Every entry in the draft's
`assumptions` list becomes either a tool call added in M2 or a retrieval added
in M3.

It returns `escalated` and never `completed`. Design rule 02 says a figure
without a citation is not an answer. This agent cannot produce a citation, so it
has no path that can legitimately complete.
"""

from __future__ import annotations

from datetime import date

from halo.domain.request import UngroundedDraft
from halo.platform.bedrock import ModelClient, Truncated
from halo.platform.budget import BudgetExceeded, BudgetTracker
from halo.platform.identity import Principal
from halo.platform.outcome import Outcome, OutcomeStatus

AGENT_NAME = "drafter"

SYSTEM_PROMPT = """\
You are drafting a preliminary quote for a HALO sales representative. HALO is a \
distributor of branded merchandise: apparel, headwear, drinkware, bags and \
promotional hard goods.

Today's date is {today:%A %d %B %Y}. Sellers write dates without a year. Resolve \
a bare month and day to the next such date on or after today. Never resolve it to \
a date in the past.

Read the representative's request and produce a complete draft quote: line items \
with quantities and unit prices, the decoration setup and per-unit run charges, a \
shipping cost, and a ship date.

`product_description` names only the product, for example "mid-weight fleece \
hoodie". It is not the whole request. Quantity, decoration, destination, date and \
budget each have their own field. Do not repeat them in the description.

You have no access to HALO's catalogue, pricing system, supplier capacity or \
policy documents in this draft.

List every figure you could not look up in `assumptions`, one entry per figure, \
each naming what you assumed and what would have to be checked to confirm it. Be \
specific: "assumed $11.50 unit cost for a mid-weight fleece hoodie at 500 units" \
is useful; "assumed pricing" is not.
"""


def draft_quote(
    request_text: str,
    *,
    principal: Principal,
    client: ModelClient,
    tracker: BudgetTracker,
    today: date | None = None,
) -> Outcome:
    """Turn a seller's sentence into an unsourced draft and an honest escalation.

    We pass `today` in instead of letting the model decide it. Without it, the
    first real run read "by Oct 15" as October 2025, a date that had already
    passed. A model has no clock, so it uses whatever its training data suggests.
    Later milestones calculate lead times from this date. A wrong year would
    produce wrong capacity checks and wrong promised ship dates, with no error.
    """
    try:
        result = client.parse(
            system=SYSTEM_PROMPT.format(today=today or date.today()),
            user=request_text,
            output_format=UngroundedDraft,
        )
    except Truncated as exc:
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            escalation_reason=f"the draft was cut off before it was finished: {exc}",
            next_state="await_budget_increase",
            usage=tracker.usage,
        )
    except BudgetExceeded as exc:
        return Outcome(
            status=OutcomeStatus.ESCALATED,
            agent=AGENT_NAME,
            escalation_reason=f"budget exhausted before drafting: {exc}",
            next_state="await_budget_increase",
            usage=tracker.usage,
        )

    draft = result.parsed
    return Outcome(
        status=OutcomeStatus.ESCALATED,
        agent=AGENT_NAME,
        payload=draft.model_dump(mode="json"),
        # Deliberately empty. There is nothing to cite. Inventing a citation to
        # make the outcome look complete is exactly the failure this project is
        # built to catch.
        evidence=[],
        escalation_reason=(
            f"draft is ungrounded: {len(draft.assumptions)} figure(s) were assumed "
            f"rather than looked up, and no line carries a citation. "
            f"Principal {principal.user_id} must not send this to a customer."
        ),
        next_state="needs_grounding",
        usage=tracker.usage,
    )
