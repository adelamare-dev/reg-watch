"""The graph state is the contract every node reads and writes."""

from __future__ import annotations

from graph.state import MAX_ITERATIONS, GraphState, initial_state


def test_initial_state_carries_the_question():
    state = initial_state("Que dit l'article 28 de DORA ?")

    assert state["question"] == "Que dit l'article 28 de DORA ?"
    assert state["iteration"] == 0
    assert state["hits"] == []
    assert state["answer"] is None
    assert state["refusal_reason"] is None


def test_initial_state_accepts_a_regulation_filter():
    state = initial_state("Gestion du risque tiers", regulation_filter="DORA")

    assert state["regulation_filter"] == "DORA"


def test_initial_state_defaults_the_regulation_filter_to_none():
    state = initial_state("Gestion du risque tiers")

    assert state["regulation_filter"] is None


def test_state_reserves_no_copilotkit_key():
    # The UI layer owns that key; colliding with it would break the V6 wiring.
    assert "copilotkit" not in GraphState.__annotations__


def test_state_exposes_the_fields_the_ui_displays():
    # Each answer must be able to say which model produced it and which
    # corpus version it rests on.
    annotations = GraphState.__annotations__

    assert "llm_provider" in annotations
    assert "corpus_version" in annotations
    assert "divergences" in annotations
    assert "verified_claims" in annotations


def test_retry_budget_is_two_iterations():
    assert MAX_ITERATIONS == 2
