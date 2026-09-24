"""The report: aggregate raw per-metric scores into a publishable result, or refuse to.

Three protections, tested here:
  1. `nan`s are counted per metric, explicitly, before any averaging.
  2. Above 20% failures on any metric, the run is invalid: no score is published.
  3. The report states which metrics are independent of the graph's own critic
     (context_precision, context_recall — they score retrieval) and which are
     circular (faithfulness — the critic already checks the same thing).

No ragas import here: `build_report` takes raw per-metric score lists (what a
ragas `EvaluationResult` would hand back column-wise), so this whole module is
testable without ragas installed.
"""

from __future__ import annotations

import math

import pytest

from eval.report import EvaluationInvalid, build_report

NAN = float("nan")


def test_a_run_with_zero_percent_failures_publishes_scores() -> None:
    scores = {"faithfulness": [0.9, 0.8, 1.0, 0.7, 0.85]}

    report = build_report(scores)

    assert report.published is True
    assert report.scores["faithfulness"] == pytest.approx(sum([0.9, 0.8, 1.0, 0.7, 0.85]) / 5)


def test_a_run_with_ten_percent_failures_still_publishes() -> None:
    # 1 nan out of 10 = 10%, under the 20% rejection threshold.
    values = [0.9] * 9 + [NAN]

    report = build_report({"faithfulness": values})

    assert report.published is True
    assert report.nan_counts["faithfulness"] == 1
    assert report.scores["faithfulness"] == pytest.approx(0.9)


def test_a_run_with_twenty_percent_or_more_failures_raises_and_publishes_nothing() -> None:
    # 2 nan out of 10 = 20%: at the threshold, the run must be refused, not
    # merely flagged — "AUCUN score publié" is the requirement, not a warning.
    values = [0.9] * 8 + [NAN, NAN]

    with pytest.raises(EvaluationInvalid) as excinfo:
        build_report({"faithfulness": values})

    assert "faithfulness" in str(excinfo.value)
    assert "20" in str(excinfo.value) or "2" in str(excinfo.value)


def test_nan_counts_are_exact_per_metric() -> None:
    # 1 nan out of 6 = ~16.7%, under the 20% rejection threshold.
    scores = {
        "faithfulness": [0.9, NAN, 0.8, 0.7, 0.6, 0.6],
        "context_precision": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }

    report = build_report(scores)

    assert report.nan_counts == {"faithfulness": 1, "context_precision": 0}


def test_a_result_with_nan_never_produces_a_silently_averaged_score() -> None:
    # The regression this guards against: `EvaluationResult.__repr__` in ragas
    # uses `safe_nanmean`, which drops `nan`s from the mean without saying so.
    # A naive `sum(values) / len(values)` on a list containing `nan` always
    # comes out `nan` itself — so if this ever silently reproduced
    # `safe_nanmean`'s behaviour, the average below would stop being `nan`
    # and this assertion would catch it.
    values = [0.9, 0.8, NAN, 0.7, 0.6]  # 1/5 = 20% -> invalid, must raise

    with pytest.raises(EvaluationInvalid):
        build_report({"faithfulness": values})

    # And even a batch just under the threshold must report the true mean
    # over ALL non-nan values, not a value that silently matches what
    # `safe_nanmean` would have produced without disclosing the drop.
    under_threshold = [0.9, 0.8, NAN, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]  # 1/10 = 10%
    report = build_report({"faithfulness": under_threshold})
    non_nan = [v for v in under_threshold if not math.isnan(v)]
    assert report.scores["faithfulness"] == pytest.approx(sum(non_nan) / len(non_nan))
    assert report.nan_counts["faithfulness"] == 1


def test_multiple_metrics_each_checked_against_the_threshold_independently() -> None:
    # faithfulness is fine, context_precision is over threshold: the whole
    # run must still be rejected, because one bad metric invalidates the run.
    scores = {
        "faithfulness": [0.9, 0.9, 0.9, 0.9, 0.9],
        "context_precision": [0.9, NAN, NAN, 0.9, 0.9],
    }

    with pytest.raises(EvaluationInvalid) as excinfo:
        build_report(scores)

    assert "context_precision" in str(excinfo.value)


def test_report_states_the_circularity_limitation_in_its_rendered_output() -> None:
    scores = {
        "faithfulness": [0.9, 0.9],
        "context_precision": [0.8, 0.8],
        "context_recall": [0.7, 0.7],
    }

    report = build_report(scores)
    rendered = report.render()

    assert "faithfulness" in rendered
    assert "critic" in rendered.lower() or "critique" in rendered.lower()
    assert "context_precision" in rendered
    assert "context_recall" in rendered
    assert "independ" in rendered.lower() or "indépend" in rendered.lower()


def test_render_of_an_empty_report_does_not_crash() -> None:
    report = build_report({})

    assert report.render()
