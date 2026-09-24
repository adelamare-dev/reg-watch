"""Verify the answer's grounding, and decide whether to try again."""

from __future__ import annotations

from graph.observability import extract_usage, generation_span
from graph.prompts import CRITIC_SYSTEM, render_chunks
from graph.schemas import CriticVerdict
from graph.state import MAX_ITERATIONS, GraphState


def critic_node(state: GraphState, *, llm, client=None) -> dict:
    """Score the answer, count the pass, and hand back a reformulated query.

    The reformulated query is written to `search_query`, never to `question`:
    what the run answers must stay the original question across retries, and
    only what it searches for may change.
    """
    messages = [
        ("system", CRITIC_SYSTEM),
        (
            "human",
            f"Question: {state['question']}\n\n"
            f"Answer to verify:\n{state['answer']}\n\n"
            f"Corpus chunks:\n\n{render_chunks(state['hits'])}",
        ),
    ]
    # `model` is not part of the fakes used in tests: they stand for a chat
    # model without carrying its identity, so the span degrades gracefully.
    model = getattr(llm, "model", "unknown")

    try:
        # The span sits inside this try: it must observe and mark a failure
        # (ERROR level, re-raised) before the existing except below turns it
        # into `critic_error` — the outer catch is what keeps this failure
        # from escaping the node, not the span.
        with generation_span(client, name="critic", model=model, messages=messages) as span:
            verdict: CriticVerdict = llm.with_structured_output(CriticVerdict).invoke(messages)
            usage = extract_usage(verdict)
            if usage is not None:
                span.update(usage_details=usage)
    except Exception as error:  # noqa: BLE001 - any provider/parsing failure is the same case here
        # No retry here: the graph already owns the retry budget, and a second
        # loop layered on top would make runs non-deterministic and traces
        # unreadable. Leaving `iteration` untouched matters too: this pass
        # never judged grounding, so it must not spend one of the two retries
        # as if it had.
        return {"critic_error": str(error)}

    update: dict = {
        "verified_claims": [claim.model_dump() for claim in verdict.claims],
        "divergences": [divergence.model_dump() for divergence in verdict.divergences],
        "is_grounded": verdict.is_grounded,
        "iteration": state["iteration"] + 1,
    }

    if not verdict.is_grounded and verdict.reformulated_query:
        update["search_query"] = verdict.reformulated_query

    return update


def route_after_critic(state: GraphState) -> str:
    """End, retry, or degrade into a refusal."""
    if state["critic_error"]:
        return "refusal"
    if state["is_grounded"]:
        return "end"
    if state["iteration"] < MAX_ITERATIONS:
        return "retrieval"
    return "refusal"
