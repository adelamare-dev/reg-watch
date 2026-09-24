"""Deterministic assertions over a graph's final state.

No LLM call is involved: these check structural facts the graph itself
already computed (`answer`, `refusal_reason`, `divergences`), so a golden
entry tagged `refus` or `divergence` can be scored without an evaluator model.
"""

from __future__ import annotations

from eval.metrics import assert_divergences_non_vides, assert_refus_aucune_base


def _base_state(**overrides: object) -> dict:
    state = {
        "question": "test",
        "answer": None,
        "refusal_reason": None,
        "divergences": [],
    }
    state.update(overrides)
    return state


# --- refus_aucune_base -------------------------------------------------


def test_assert_refus_aucune_base_passes_on_a_real_refusal() -> None:
    state = _base_state(answer=None, refusal_reason="hors du corpus DORA/AI Act")

    result = assert_refus_aucune_base(state)

    assert result.passed is True


def test_assert_refus_aucune_base_fails_when_an_answer_was_produced() -> None:
    state = _base_state(answer="Voici la reponse.", refusal_reason=None)

    result = assert_refus_aucune_base(state)

    assert result.passed is False
    assert result.reason  # a human-readable explanation, not just False


def test_assert_refus_aucune_base_fails_when_refusal_reason_is_missing() -> None:
    state = _base_state(answer=None, refusal_reason=None)

    result = assert_refus_aucune_base(state)

    assert result.passed is False
    assert "refusal_reason" in result.reason


# --- divergences_non_vides ----------------------------------------------


def test_assert_divergences_non_vides_passes_when_divergences_are_present() -> None:
    state = _base_state(divergences=[{"theme": "notification d'incident"}])

    result = assert_divergences_non_vides(state)

    assert result.passed is True


def test_assert_divergences_non_vides_fails_when_divergences_are_empty() -> None:
    state = _base_state(divergences=[])

    result = assert_divergences_non_vides(state)

    assert result.passed is False
    assert result.reason
