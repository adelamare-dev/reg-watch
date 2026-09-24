"""Turn raw per-metric RAGAS scores into a publishable report, or refuse to.

A RAGAS `evaluate()` call can return `nan` for a metric on a given question
(a timed-out or rate-limited judge call) without raising — and its own
`EvaluationResult.__repr__` uses `safe_nanmean`, which drops those `nan`s from
the average silently. A report built the same way would show, say,
"faithfulness = 0.87" when in truth only 3 of 11 questions produced a score at
all. This module exists to make that failure mode structurally impossible:
`nan`s are counted per metric before anything is averaged, and past 20%
failures on any one metric the whole run is refused — no score is published,
an `EvaluationInvalid` is raised instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Beyond this fraction of `nan` results on any single metric, the run cannot
# be trusted to represent the dataset it claims to score.
MAX_FAILURE_RATE = 0.20

# `faithfulness` re-checks answer/context grounding, which the graph's own
# Critic node already verifies before an answer ever reaches the user — so it
# measures internal consistency with the Critic, not an independent signal.
# `context_precision` and `context_recall` score retrieval (Qdrant +
# embeddings), which the Critic never touches, so they are the metrics that
# actually test something the graph doesn't already grade itself on.
CIRCULAR_METRICS = {"faithfulness", "answer_relevancy"}
INDEPENDENT_METRICS = {"context_precision", "context_recall"}


class EvaluationInvalid(Exception):
    """Raised instead of publishing a report: too many `nan` results to trust it."""


@dataclass(frozen=True)
class EvaluationReport:
    """A run whose failure rate stayed under the threshold on every metric."""

    scores: dict[str, float]
    nan_counts: dict[str, int]
    sample_counts: dict[str, int]
    published: bool = True

    def render(self) -> str:
        """Human-readable summary, including the circularity disclosure.

        The limitation is stated here, in the report's own output, rather
        than only in this module's docstring: a reader who never opens the
        source must still see it.
        """
        lines = ["=== RAPPORT D'ÉVALUATION RAGAS ==="]
        if not self.scores:
            lines.append("(aucune métrique)")
            return "\n".join(lines)

        for metric, score in self.scores.items():
            nan_count = self.nan_counts.get(metric, 0)
            total = self.sample_counts.get(metric, 0)
            lines.append(f"{metric}: {score:.3f} ({nan_count}/{total} échecs 'nan')")

        lines.append("")
        lines.append("--- Limite de circularité ---")
        circular = sorted(m for m in self.scores if m in CIRCULAR_METRICS)
        independent = sorted(m for m in self.scores if m in INDEPENDENT_METRICS)
        if circular:
            lines.append(
                f"{', '.join(circular)} : le nœud Critique du graphe fait déjà une "
                "vérification d'ancrage équivalente — ces métriques mesurent une "
                "cohérence interne avec le Critique, pas une performance indépendante."
            )
        if independent:
            lines.append(
                f"{', '.join(independent)} : évaluent le retrieval (Qdrant + "
                "embeddings), que le Critique ne touche pas — ce sont les mesures "
                "réellement indépendantes de ce rapport."
            )
        return "\n".join(lines)


def build_report(raw_scores: dict[str, list[float]]) -> EvaluationReport:
    """Aggregate raw per-metric score lists, enforcing the 20% failure gate.

    `raw_scores` mirrors a ragas `EvaluationResult` column-wise: one list of
    per-question scores per metric name, `nan` marking a judge call that
    failed instead of scoring. Raises `EvaluationInvalid` — publishing
    nothing — the moment any single metric's failure rate reaches 20%.
    """
    nan_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    scores: dict[str, float] = {}
    invalid_metrics: list[str] = []

    for metric, values in raw_scores.items():
        total = len(values)
        nan_values = [value for value in values if math.isnan(value)]
        nan_count = len(nan_values)
        nan_counts[metric] = nan_count
        sample_counts[metric] = total

        failure_rate = nan_count / total if total else 0.0
        if failure_rate >= MAX_FAILURE_RATE:
            invalid_metrics.append(
                f"{metric}: {nan_count}/{total} échecs 'nan' "
                f"({failure_rate:.0%} >= seuil de {MAX_FAILURE_RATE:.0%})"
            )
            continue

        valid_values = [value for value in values if not math.isnan(value)]
        scores[metric] = sum(valid_values) / len(valid_values) if valid_values else math.nan

    if invalid_metrics:
        raise EvaluationInvalid(
            "run invalide, aucun score publié — seuil d'échec dépassé pour : "
            + "; ".join(invalid_metrics)
        )

    return EvaluationReport(
        scores=scores, nan_counts=nan_counts, sample_counts=sample_counts
    )
