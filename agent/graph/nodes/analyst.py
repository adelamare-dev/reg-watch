"""Write the answer from the retrieved chunks."""

from __future__ import annotations

from graph.prompts import ANALYST_SYSTEM, language_name, render_chunks
from graph.state import GraphState


def analyst_messages(state: GraphState) -> list[tuple[str, str]]:
    """Build the two turns: the standing instructions, then the material."""
    system = ANALYST_SYSTEM.format(language_name=language_name(state["language"]))
    human = (
        f"Question: {state['question']}\n\n"
        f"Corpus chunks:\n\n{render_chunks(state['hits'])}"
    )
    return [("system", system), ("human", human)]


def analyst_node(state: GraphState, *, llm) -> dict:
    """Produce the answer, leaving verification to the critic."""
    response = llm.invoke(analyst_messages(state))
    return {"answer": response.content}
