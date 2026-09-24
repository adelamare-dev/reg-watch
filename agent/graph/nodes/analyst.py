"""Write the answer from the retrieved chunks."""

from __future__ import annotations

from graph.observability import extract_usage, generation_span
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


def analyst_node(state: GraphState, *, llm, client=None) -> dict:
    """Produce the answer, leaving verification to the critic."""
    messages = analyst_messages(state)
    # `model` is not part of the fakes used in tests: they stand for a chat
    # model without carrying its identity, so the span degrades gracefully.
    model = getattr(llm, "model", "unknown")

    with generation_span(client, name="analyst", model=model, messages=messages) as span:
        response = llm.invoke(messages)
        usage = extract_usage(response)
        if usage is not None:
            span.update(usage_details=usage)

    return {"answer": response.content}
