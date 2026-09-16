"""Retrieval fills the state, and decides whether the answer can exist at all."""

from __future__ import annotations

import pytest

from graph.nodes.refusal import refusal_node
from graph.nodes.retrieval import retrieval_node, route_after_retrieval
from graph.state import initial_state
from mcp_server.tools import _serialise_hit
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
