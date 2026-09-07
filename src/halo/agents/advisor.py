"""M3: answer a policy question from the Atlas corpus, and prove the answer is in it.

This is the M2 mechanism applied to text instead of numbers. In M2, a figure had
to appear in the tool result it cited. Here, a claim has to appear in the chunk
it cites. The model must return the exact text, word for word, so the check is a
substring test and not a judgement.

The word-for-word requirement is the important part. If you ask a model which
chunk a claim came from, it returns a confident id and a paraphrase. If you ask
it to quote the sentence, and then check that the sentence is in the chunk, you
can tell the two apart.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from halo.domain.quote import Citation, CitationKind
from halo.platform.bedrock import ModelClient, Truncated
from halo.platform.budget import BudgetExceeded, BudgetTracker
from halo.platform.envelope import EVIDENCE_RULE, Evidence, wrap_all
from halo.platform.guardrails import Guardrail, Surface
from halo.platform.identity import Principal
from halo.platform.outcome import Outcome, OutcomeStatus
from halo.rag.retrieve import AtlasRetriever, ScoredChunk

AGENT_NAME = "advisor"

SYSTEM_PROMPT = """\
You answer questions about HALO's decoration standards, quoting policy and
logistics, using only the numbered excerpts supplied with the question.

For every factual claim in your answer, produce a finding containing:
  - the claim itself, in your own words;
  - the `chunk_id` of the excerpt it came from;
  - `quote`: the exact sentence or bullet from that excerpt, copied character
    for character. Do not paraphrase, tidy, join or shorten it. It is checked
    against the excerpt verbatim.

If the excerpts do not answer part of the question, put that part in
`unsupported` and leave it out of the answer. An excerpt that is merely related
is not an answer, and no excerpt at all is a better outcome than a plausible
guess.

{evidence_rule}"""


class Finding(BaseModel):
    """One claim and the text that supports it."""

    claim: str = Field(min_length=1)
    chunk_id: str = Field(pattern=r"^atl-[a-z0-9-]+#[a-z0-9-]+$")
    quote: str = Field(min_length=1)


class PolicyAnswer(BaseModel):
    answer: str = Field(min_length=1)
    findings: list[Finding] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


def _normalise(text: str) -> str:
    """Collapse whitespace, so a quote that crosses a line break still matches.

    Nothing else is normalised, deliberately. Lowercasing or removing punctuation
    would start accepting quotes that are close but not exact. Those are what
    this check exists to catch.
    """
    return re.sub(r"\s+", " ", text).strip()


def verify(answer: PolicyAnswer, retrieved: list[ScoredChunk]) -> list[str]:
    """Problems with the answer's grounding. Empty means every claim checks out."""
    by_id = {hit.chunk.id: hit.chunk for hit in retrieved}
    problems: list[str] = []

    for finding in answer.findings:
        chunk = by_id.get(finding.chunk_id)
        if chunk is None:
            problems.append(f"cites {finding.chunk_id}, which was not among the excerpts supplied")
        elif _normalise(finding.quote) not in _normalise(chunk.text):
            problems.append(
                f"the quote attributed to {finding.chunk_id} is not in it: {finding.quote[:70]!r}"
            )
    return problems


def _excerpts(retrieved: list[ScoredChunk]) -> str:
    """The retrieved chunks, each inside an evidence envelope.

    An Atlas document is written by HALO, so this corpus is less hostile than a
    supplier feed. It is wrapped anyway. The boundary is defined by where the
    text was fetched from, not by how much we trust today's contents of the
    place we fetched it from.
    """
    return wrap_all(
        [
            Evidence(
                id=hit.chunk.id,
                source=f"atlas/{hit.chunk.doc_title} — {hit.chunk.heading}",
                body=hit.chunk.text,
            )
            for hit in retrieved
        ]
    )


def _answer_text(answer: PolicyAnswer) -> str:
    """Everything the seller would read. The claims are included because a
    commitment can be made in a finding as easily as in the summary."""
    return "\n".join([answer.answer, *(finding.claim for finding in answer.findings)])


def _prompt(question: str, retrieved: list[ScoredChunk]) -> str:
    return f"QUESTION\n{question}\n\nEXCERPTS\n{_excerpts(retrieved)}"


def _refusal(reason: str, usage, payload: dict | None = None) -> Outcome:
    """A guardrail stop. `refused`, not `escalated`: there is nothing for a human
    to approve here, and the approval queue should not fill up with attacks."""
    return Outcome(
        status=OutcomeStatus.REFUSED,
        agent=AGENT_NAME,
        payload=payload,
        escalation_reason=reason,
        next_state="blocked_by_guardrail",
        usage=usage,
    )


def answer_policy_question(
    question: str,
    *,
    principal: Principal,
    client: ModelClient,
    retriever: AtlasRetriever,
    tracker: BudgetTracker,
    limit: int = 6,
    guardrail: Guardrail | None = None,
) -> tuple[Outcome, list[ScoredChunk]]:
    """Retrieve, answer, verify. Returns the outcome and what was retrieved."""
    if guardrail is not None:
        verdict = guardrail.inspect(question, surface=Surface.INPUT)
        if verdict.blocked:
            return (
                _refusal(f"the question was blocked: {verdict.summary()}", tracker.usage),
                [],
            )

    retrieved = retriever.search(question, limit=limit)

    if not retrieved:
        return (
            Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                escalation_reason="the Atlas corpus returned nothing for this question",
                next_state="needs_human_answer",
                usage=tracker.usage,
            ),
            retrieved,
        )

    try:
        result = client.parse(
            system=SYSTEM_PROMPT.format(evidence_rule=EVIDENCE_RULE),
            user=_prompt(question, retrieved),
            output_format=PolicyAnswer,
        )
    except Truncated as exc:
        return (
            Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                escalation_reason=f"the answer was cut off before it was finished: {exc}",
                next_state="await_budget_increase",
                usage=tracker.usage,
            ),
            retrieved,
        )
    except BudgetExceeded as exc:
        return (
            Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                escalation_reason=f"budget exhausted before answering: {exc}",
                next_state="await_budget_increase",
                usage=tracker.usage,
            ),
            retrieved,
        )

    parsed = result.parsed

    if problems := verify(parsed, retrieved):
        return (
            Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                payload=parsed.model_dump(mode="json"),
                escalation_reason="the answer could not be traced to the excerpts it "
                f"cites: {'; '.join(problems)}",
                next_state="needs_regrounding",
                usage=tracker.usage,
            ),
            retrieved,
        )

    if not parsed.findings:
        return (
            Outcome(
                status=OutcomeStatus.ESCALATED,
                agent=AGENT_NAME,
                payload=parsed.model_dump(mode="json"),
                escalation_reason="the corpus does not cover this question: "
                + ("; ".join(parsed.unsupported) or "no supporting excerpt found"),
                next_state="needs_human_answer",
                usage=tracker.usage,
            ),
            retrieved,
        )

    # The output check runs after grounding, not instead of it. A verified
    # answer can still be one that should not be sent: the excerpts genuinely
    # say what it quotes, and the sentence it built around them commits HALO to
    # something. Grounded and allowed are different questions.
    if guardrail is not None:
        verdict = guardrail.inspect(
            _answer_text(parsed),
            surface=Surface.OUTPUT,
            grounding_source=_excerpts(retrieved),
        )
        if verdict.blocked:
            return (
                _refusal(
                    f"the answer was blocked before it reached the seller: {verdict.summary()}",
                    tracker.usage,
                    parsed.model_dump(mode="json"),
                ),
                retrieved,
            )

    citations = [
        Citation(
            kind=CitationKind.CHUNK,
            ref=finding.chunk_id,
            supporting_text=finding.quote,
        )
        for finding in parsed.findings
    ]
    payload = parsed.model_dump(mode="json")
    payload["retrieved"] = [hit.chunk.id for hit in retrieved]

    return (
        Outcome(
            status=OutcomeStatus.COMPLETED,
            agent=AGENT_NAME,
            payload=payload,
            evidence=citations,
            next_state="answered",
            usage=tracker.usage,
        ),
        retrieved,
    )
