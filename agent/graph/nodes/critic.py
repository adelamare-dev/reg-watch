"""Verify the answer's grounding, and decide whether to try again."""

from __future__ import annotations

from graph.prompts import CRITIC_SYSTEM, render_chunks
from graph.schemas import CriticVerdict
from graph.state import MAX_ITERATIONS, GraphState


def critic_node(state: GraphState, *, llm) -> dict:
    """Score the answer, count the pass, and hand back a reformulated query.

    The reformulated query overwrites `question` so the retrieval node needs
    no second search path. What the run answers is unchanged - only what it
    searches for is.
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

    verdict: CriticVerdict = llm.with_structured_output(CriticVerdict).invoke(messages)

    update: dict = {
        "verified_claims": [claim.model_dump() for claim in verdict.claims],
        "divergences": [divergence.model_dump() for divergence in verdict.divergences],
        "is_grounded": verdict.is_grounded,
        "iteration": state["iteration"] + 1,
    }

    if not verdict.is_grounded and verdict.reformulated_query:
        update["question"] = verdict.reformulated_query

    return update


def route_after_critic(state: GraphState) -> str:
    """End, retry, or degrade into a partial refusal."""
    if state["is_grounded"]:
        return "end"
    if state["iteration"] < MAX_ITERATIONS:
        return "retrieval"
    return "refusal"
