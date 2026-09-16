"""End-to-end against the real corpus. Requires a reachable Qdrant.

Marked `integration`, so it stays out of the default run - same convention as
the MCP suite.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from graph.nodes.retrieval import retrieval_node
from graph.state import initial_state
from rag.config import get_settings
from rag.embedder import build_embedder
from rag.retriever import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    settings = get_settings()
    embedder = build_embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return Retriever(client=client, embedder=embedder, settings=settings)


def test_retrieval_grounds_a_real_dora_question(retriever: Retriever):
    state = initial_state("Que dit l'article 28 de DORA sur les prestataires tiers ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["is_grounded"] is True
    assert update["hits"]
    assert update["corpus_version"] is not None


def test_a_named_regulation_keeps_the_answer_inside_it(retriever: Retriever):
    # The retriever drops the filter and retries when a filtered search comes
    # back empty, and says so through `used_exact_filter`. Both outcomes are
    # correct; what must never happen is a filtered search silently returning
    # another regulation's articles.
    state = initial_state("Que dit l'article 28 ?", regulation_filter="DORA")

    update = retrieval_node(state, retriever=retriever)

    assert update["is_grounded"] is True
    if update["used_exact_filter"]:
        assert all(hit["regulation"] == "DORA" for hit in update["hits"])


def test_an_out_of_corpus_question_is_not_grounded(retriever: Retriever):
    state = initial_state("Quelle est la recette de la tarte Tatin ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["is_grounded"] is False
