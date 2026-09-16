"""The shape every node reads and writes.

A flat TypedDict rather than a domain class: the state has to survive
checkpointing today and event streaming later, and both want something a
serialiser can walk without help.

Hits are kept in their serialised dict form — the one `mcp_server.tools`
already produces — rather than as `SearchHit` objects, for the same reason.
"""

from __future__ import annotations

from typing import TypedDict

# Two passes, then the graph degrades into a partial refusal. A third pass
# costs two more LLM calls and, in practice, rarely reformulates its way to
# better hits.
MAX_ITERATIONS = 2


class GraphState(TypedDict):
    """Everything the three nodes exchange."""

    question: str
    language: str
    regulation_filter: str | None
    hits: list[dict]
    is_grounded: bool
    used_exact_filter: bool
    answer: str | None
    verified_claims: list[dict]
    divergences: list[dict]
    iteration: int
    refusal_reason: str | None
    llm_provider: str | None
    corpus_version: str | None


def initial_state(
    question: str, *, regulation_filter: str | None = None
) -> GraphState:
    """Build the entry state, with every field explicitly set.

    Nodes read keys directly rather than with `.get()`, so a missing key is a
    KeyError at the first node instead of a None threading its way through the
    whole run.
    """
    return GraphState(
        question=question,
        language="fr",
        regulation_filter=regulation_filter,
        hits=[],
        is_grounded=False,
        used_exact_filter=False,
        answer=None,
        verified_claims=[],
        divergences=[],
        iteration=0,
        refusal_reason=None,
        llm_provider=None,
        corpus_version=None,
    )
