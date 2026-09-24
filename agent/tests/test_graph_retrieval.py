"""Retrieval fills the state, and decides whether the answer can exist at all."""

from __future__ import annotations

import pytest

from graph.nodes.refusal import refusal_node
from graph.nodes.retrieval import retrieval_node, route_after_retrieval
from graph.state import initial_state
from mcp_server.tools import _serialise_hit
from tests.fakes import FakeLangfuseClient
from tests.test_mcp_tools import FakeRetriever, make_hit


def test_retrieval_records_hits_and_grounding():
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("Que dit l'article 28 de DORA ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["is_grounded"] is True
    assert len(update["hits"]) == 1
    assert update["hits"][0]["regulation"] == "DORA"
    assert update["hits"][0]["text"] == "Full text of DORA article 28."


def test_retrieval_detects_the_question_language():
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("What does article 28 of DORA say?")

    update = retrieval_node(state, retriever=retriever)

    assert update["language"] == "en"
    assert retriever.search_calls[0]["language"] == "en"


def test_retrieval_passes_the_canonical_regulation_filter():
    # "AI Act" must reach the retriever as "AI_ACT" or the payload filter
    # silently matches nothing.
    retriever = FakeRetriever(hits=[make_hit(regulation="AI_ACT")])
    state = initial_state("Gestion des risques", regulation_filter="AI Act")

    retrieval_node(state, retriever=retriever)

    assert retriever.search_calls[0]["regulation"] == "AI_ACT"


def test_retrieval_rejects_a_regulation_outside_the_corpus():
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("Obligations", regulation_filter="MiCA")

    with pytest.raises(ValueError):
        retrieval_node(state, retriever=retriever)


def test_retrieval_records_the_corpus_version_from_the_hits():
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("Que dit l'article 28 de DORA ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["corpus_version"] == "2025-01-17"


def test_retrieval_leaves_the_corpus_version_unset_without_hits():
    retriever = FakeRetriever(hits=[])
    state = initial_state("Quelle est la capitale de la Suisse ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["corpus_version"] is None


def test_retrieval_works_identically_without_a_tracing_client():
    # Non-regression: omitting `client` must behave exactly as before.
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("Que dit l'article 28 de DORA ?")

    update = retrieval_node(state, retriever=retriever)

    assert update["is_grounded"] is True


def test_retrieval_opens_a_retriever_span_when_traced():
    client = FakeLangfuseClient()
    retriever = FakeRetriever(hits=[make_hit()])
    state = initial_state("Que dit l'article 28 de DORA ?")

    retrieval_node(state, retriever=retriever, client=client)

    assert len(client.spans) == 1
    span = client.spans[0]
    assert span.as_type == "retriever"


def test_retrieval_span_reports_hit_count_and_filter_usage():
    client = FakeLangfuseClient()
    retriever = FakeRetriever(hits=[make_hit(), make_hit(article_number="30")], used_exact_filter=True)
    state = initial_state("Que dit l'article 28 de DORA ?")

    retrieval_node(state, retriever=retriever, client=client)

    span = client.spans[0]
    assert span.updates[-1]["hit_count"] == 2
    assert span.updates[-1]["exact_filter_used"] is True


def test_retrieval_span_carries_the_embedding_backend_in_use():
    # The fastembed/mistral switch must be visible in the trace even though
    # the embedding call happens inside `retriever.search`, not in this node.
    client = FakeLangfuseClient()
    retriever = FakeRetriever(hits=[make_hit()])
    retriever.embedder = FakeEmbedderInfo(model_id="fastembed/paraphrase-multilingual", dimension=384)
    state = initial_state("Que dit l'article 28 de DORA ?")

    retrieval_node(state, retriever=retriever, client=client)

    span = client.spans[0]
    assert span.updates[-1]["metadata"]["embedding_model"] == "fastembed/paraphrase-multilingual"
    assert span.updates[-1]["metadata"]["embedding_dimension"] == 384


def test_retrieval_span_declares_no_cost_for_a_free_local_embedder():
    # fastembed runs locally at zero cost: nothing here must declare a cost,
    # unlike a paid provider such as mistral-embed would.
    client = FakeLangfuseClient()
    retriever = FakeRetriever(hits=[make_hit()])
    retriever.embedder = FakeEmbedderInfo(model_id="fastembed/paraphrase-multilingual", dimension=384)
    state = initial_state("Que dit l'article 28 de DORA ?")

    retrieval_node(state, retriever=retriever, client=client)

    span = client.spans[0]
    assert "usage_details" not in span.updates[-1]
    assert "cost_details" not in span.updates[-1]


class FakeEmbedderInfo:
    def __init__(self, *, model_id: str, dimension: int) -> None:
        self.model_id = model_id
        self.dimension = dimension


def test_ungrounded_retrieval_routes_to_refusal():
    state = initial_state("Quelle est la capitale de la Suisse ?")
    state["is_grounded"] = False

    assert route_after_retrieval(state) == "refusal"


def test_grounded_retrieval_routes_to_the_analyst():
    state = initial_state("Que dit l'article 28 de DORA ?")
    state["is_grounded"] = True

    assert route_after_retrieval(state) == "analyst"


def test_refusal_states_that_no_regulatory_basis_was_found():
    state = initial_state("Quelle est la capitale de la Suisse ?")
    state["is_grounded"] = False

    update = refusal_node(state)

    assert update["answer"] is None
    assert "aucune base réglementaire" in update["refusal_reason"].lower()


def test_refusal_speaks_the_language_of_the_question():
    state = initial_state("What is the capital of Switzerland?")
    state["language"] = "en"
    state["is_grounded"] = False

    update = refusal_node(state)

    assert "no regulatory basis" in update["refusal_reason"].lower()


def test_refusal_after_exhausted_retries_is_a_partial_one():
    # Hits exist, but the critic never found them sufficient. That is a
    # different failure from "nothing was found" and must read differently.
    # Note the critic has already set is_grounded to False by this point, so
    # the retrieved hits — not the flag — are what tells the two apart.
    state = initial_state("Question ambiguë")
    state["hits"] = [_serialise_hit(make_hit())]
    state["is_grounded"] = False
    state["iteration"] = 2

    update = refusal_node(state)

    assert "insuffisant" in update["refusal_reason"].lower()


def test_refusal_without_hits_is_never_a_partial_one():
    state = initial_state("Quelle est la capitale de la Suisse ?")
    state["hits"] = []
    state["is_grounded"] = False

    update = refusal_node(state)

    assert "aucune base réglementaire" in update["refusal_reason"].lower()


def test_a_critic_error_produces_a_technical_refusal_before_anything_else():
    # Even a state that also looks like "nothing found" (iteration 0, no
    # hits) must read as technical when `critic_error` is set: the priority
    # is not incidental, it is the point.
    state = initial_state("Question ambiguë")
    state["critic_error"] = "HTTP 429: rate limited"

    update = refusal_node(state)

    assert update["answer"] is None
    reason = update["refusal_reason"].lower()
    assert "aucune base réglementaire" not in reason
    assert "insuffisant" not in reason


def test_the_technical_refusal_speaks_english_too():
    state = initial_state("Ambiguous question")
    state["language"] = "en"
    state["critic_error"] = "HTTP 429: rate limited"

    update = refusal_node(state)

    reason = update["refusal_reason"].lower()
    assert "no regulatory basis" not in reason
    assert "insufficient" not in reason


def test_the_analyst_answer_is_not_published_on_a_technical_refusal():
    state = initial_state("Question ambiguë")
    state["answer"] = "Une réponse que le critique n'a jamais validée."
    state["critic_error"] = "HTTP 429: rate limited"

    update = refusal_node(state)

    assert update["answer"] is None
