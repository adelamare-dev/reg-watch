"""The runner: execute the graph over a validated dataset, one entry at a time.

Uses a fake `run_question` (never the real graph) so these tests exercise only
the runner's own responsibilities: looping over entries, forcing `eval_top_k`
onto retrieval, and shaping the result into `EvalSample`.
"""

from __future__ import annotations

from eval.dataset import GoldenEntry
from eval.metrics import EvalSample
from eval.runner import run_dataset


def _entry(**overrides: object) -> GoldenEntry:
    base: dict = {
        "id": "test-001",
        "categorie": "factuel_mono",
        "langue": "fr",
        "comportement_attendu": "repondre",
        "question": "question de test",
        "draft": False,
        "reference": "réponse de référence de test",
        "reference_context_ids": ["DORA-Art1"],
    }
    base.update(overrides)
    return GoldenEntry.model_validate(base)


class FakeRetriever:
    """Records the `top_k` it was constructed/called with, nothing else."""

    def __init__(self, *, top_k: int = 8) -> None:
        self.top_k = top_k


class _FakeState(dict):
    """A plain dict stands in for `GraphState` — the runner reads it by key."""


def fake_run_question(question, *, retriever, llm, provider, regulation_filter=None, client=None):
    """Records the `retriever` (and its `top_k`) it was actually called with."""
    return _FakeState(
        {
            "question": question,
            "answer": "réponse produite de test",
            "hits": [
                {"text": "contexte de test 1", "citation": "DORA, article 1"},
                {"text": "contexte de test 2", "citation": "DORA, article 2"},
            ],
            "refusal_reason": None,
            "divergences": [],
            "retriever_top_k_seen": getattr(retriever, "top_k", None),
        }
    )


def test_run_dataset_produces_one_sample_per_entry() -> None:
    entries = [_entry(id="a"), _entry(id="b")]

    samples = run_dataset(
        entries,
        retriever=FakeRetriever(),
        llm=object(),
        provider=object(),
        eval_top_k=3,
        run_question=fake_run_question,
    )

    assert [sample.entry_id for sample in samples] == ["a", "b"]
    assert all(isinstance(sample, EvalSample) for sample in samples)


def test_run_dataset_collects_the_produced_answer_and_contexts() -> None:
    entries = [_entry(id="a")]

    [sample] = run_dataset(
        entries,
        retriever=FakeRetriever(),
        llm=object(),
        provider=object(),
        eval_top_k=3,
        run_question=fake_run_question,
    )

    assert sample.answer == "réponse produite de test"
    assert sample.retrieved_contexts == ["contexte de test 1", "contexte de test 2"]
    assert sample.question == "question de test"
    assert sample.reference == "réponse de référence de test"


def test_run_dataset_keeps_the_full_final_state_on_the_sample() -> None:
    entries = [_entry(id="a")]

    [sample] = run_dataset(
        entries,
        retriever=FakeRetriever(),
        llm=object(),
        provider=object(),
        eval_top_k=3,
        run_question=fake_run_question,
    )

    assert sample.state["answer"] == "réponse produite de test"


def test_run_dataset_forces_eval_top_k_onto_the_retriever_used_for_search() -> None:
    # Production `top_k` is 8; evaluation must never use it — a divergence
    # that is intentional (see `runner.py`), and this is what protects it.
    entries = [_entry(id="a")]
    retriever = FakeRetriever(top_k=8)

    [sample] = run_dataset(
        entries,
        retriever=retriever,
        llm=object(),
        provider=object(),
        eval_top_k=3,
        run_question=fake_run_question,
    )

    assert sample.state["retriever_top_k_seen"] == 3
    # The caller's own retriever object is untouched: only the copy handed to
    # the graph run is capped, so a shared retriever isn't mutated for
    # anything else that might use it concurrently.
    assert retriever.top_k == 8


def test_run_dataset_passes_the_regulation_filter_through_when_the_entry_has_none() -> None:
    entries = [_entry(id="a")]
    seen: dict = {}

    def spying_run_question(question, *, retriever, llm, provider, regulation_filter=None, client=None):
        seen["regulation_filter"] = regulation_filter
        return fake_run_question(
            question, retriever=retriever, llm=llm, provider=provider, client=client
        )

    run_dataset(
        entries,
        retriever=FakeRetriever(),
        llm=object(),
        provider=object(),
        eval_top_k=3,
        run_question=spying_run_question,
    )

    assert seen["regulation_filter"] is None
