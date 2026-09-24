"""Search the corpus, and decide whether an answer is possible at all.

The tools are imported directly rather than reached over MCP: they are plain
synchronous functions taking their retriever as an argument, so there is one
tool definition, consumed two ways — over the wire by external clients, in
process by this graph.
"""

from __future__ import annotations

from graph.language import detect_language
from graph.observability import retriever_span
from graph.state import GraphState
from mcp_server.tools import canonical_regulation, search_regulatory_corpus


def retrieval_node(state: GraphState, *, retriever, client=None) -> dict:
    """Fill hits, grounding and corpus version from one search.

    A regulation outside the corpus raises rather than refuses: an unknown
    filter is a caller bug, and answering "no regulatory basis" would hide it.

    Language is detected on `question` alone, never on `search_query`: a
    critic's reformulation is keywords with no function words, and running
    the detector on it would silently flip the answer's language on a retry.
    """
    language = detect_language(state["question"])
    regulation = canonical_regulation(state["regulation_filter"])
    query = state["search_query"] or state["question"]

    # `top_k` and `embedder` are not part of `RetrieverLike`: fakes used across
    # the test suite do not carry them, and the span must not force that
    # requirement on them.
    top_k = getattr(retriever, "top_k", None)

    with retriever_span(client, name="retrieval", query=query, top_k=top_k) as span:
        result = search_regulatory_corpus(
            retriever,
            query=query,
            regulation=regulation,
            language=language,
        )
        hits = result["hits"]

        update: dict = {"hit_count": len(hits), "exact_filter_used": result["used_exact_filter"]}
        embedder = getattr(retriever, "embedder", None)
        if embedder is not None:
            # Makes the fastembed/mistral switch visible in the trace without
            # a separate embedding span: the embedding call happens inside
            # `retriever.search`, one layer below where the client is held.
            update["metadata"] = {
                "embedding_model": embedder.model_id,
                "embedding_dimension": embedder.dimension,
            }
        span.update(**update)

    return {
        "language": language,
        "hits": hits,
        "is_grounded": result["is_grounded"],
        "used_exact_filter": result["used_exact_filter"],
        "corpus_version": hits[0]["consolidation_date"] if hits else None,
    }


def route_after_retrieval(state: GraphState) -> str:
    """Send ungrounded questions straight out, without spending an LLM call."""
    return "analyst" if state["is_grounded"] else "refusal"
