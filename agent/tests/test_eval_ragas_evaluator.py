"""`RagasEvaluator`: the RAGAS-backed implementation of the `Evaluator` protocol.

`ragas` is an optional dependency (see `pyproject.toml`'s `eval` extra) and is
NOT installed in the default test environment. Every test here must therefore
pass without it: importing `eval.metrics` must never require `ragas`, and
constructing a `RagasEvaluator` without it installed must fail with an
actionable message rather than a bare `ImportError`.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import EvalSample, RagasEvaluator, read_metric_columns, to_raw_scores
from eval.report import EvaluationInvalid, build_report


def test_metrics_module_imports_without_ragas_installed() -> None:
    # If this test file collects at all, `from eval.metrics import ...` above
    # already succeeded without ragas — this assertion just documents intent.
    assert RagasEvaluator is not None


def test_scores_are_extracted_per_question_not_pre_averaged() -> None:
    """The gate needs one score per question, `nan`s included.

    Handing it an already-averaged number would destroy the very information
    the 20% rejection rule is computed from — the count of questions that
    failed to produce a score at all.
    """
    raw = to_raw_scores(
        {
            "faithfulness": [0.9, 0.8, float("nan")],
            "context_precision": [1.0, 0.5, 0.75],
        }
    )

    assert raw["faithfulness"] == pytest.approx([0.9, 0.8, math.nan], nan_ok=True)
    assert len(raw["context_precision"]) == 3


def test_a_ragas_run_past_the_failure_threshold_publishes_no_score() -> None:
    """A 429-riddled run must reach the gate, not slip past it as a report."""
    raw = to_raw_scores({"faithfulness": [0.9, float("nan"), float("nan")]})

    with pytest.raises(EvaluationInvalid):
        build_report(raw)


class _ResultWithoutContains:
    """Mirrors ragas's `EvaluationResult`: `__getitem__` by name, nothing else.

    With no `__contains__` and no `__iter__`, `"name" in result` falls back to
    the old sequence protocol and probes `result[0]`, which raises on a
    string-keyed store. Any column reader that tests membership breaks here.
    """

    def __init__(self, scores: dict[str, list[float]]) -> None:
        self._scores = scores

    def __getitem__(self, key: str) -> list[float]:
        return self._scores[key]


def test_columns_are_read_without_membership_testing() -> None:
    result = _ResultWithoutContains({"faithfulness": [0.9, 0.8]})

    columns = read_metric_columns(result, ("faithfulness", "context_recall"))

    assert columns == {"faithfulness": [0.9, 0.8]}


def test_a_metric_ragas_did_not_compute_is_skipped_not_fatal() -> None:
    result = _ResultWithoutContains({"faithfulness": [1.0]})

    columns = read_metric_columns(result, ("faithfulness", "answer_relevancy"))

    assert "answer_relevancy" not in columns


def test_ragas_evaluator_raises_actionable_error_without_ragas() -> None:
    try:
        import ragas  # noqa: F401

        pytest.skip("ragas is installed; this test only applies to its absence")
    except ImportError:
        pass

    with pytest.raises(ImportError) as excinfo:
        RagasEvaluator(llm=object())

    message = str(excinfo.value)
    assert "ragas" in message.lower()
    assert "uv sync" in message
    assert "eval" in message


def test_eval_sample_carries_what_ragas_needs() -> None:
    sample = EvalSample(
        entry_id="test-001",
        state={"answer": "réponse de test"},
        question="question de test",
        answer="réponse de test",
        retrieved_contexts=["contexte de test"],
        reference="référence de test",
    )

    assert sample.question == "question de test"
    assert sample.answer == "réponse de test"
    assert sample.retrieved_contexts == ["contexte de test"]
    assert sample.reference == "référence de test"


def test_eval_sample_ragas_fields_default_to_none_or_empty() -> None:
    # Deterministic-only samples (refus/divergence entries) never populate
    # these — the assertions in this module read `state` alone.
    sample = EvalSample(entry_id="test-001", state={})

    assert sample.question is None
    assert sample.answer is None
    assert sample.retrieved_contexts == []
    assert sample.reference is None
