"""Evaluator interface and the deterministic, zero-LLM-call assertions.

`refus`/`divergence` golden entries can be scored from the graph's final
state alone — no judge model needed. `repondre` entries cannot (grading
whether a free-text answer matches a reference needs RAGAS or similar),
which is why only these two assertions live here; the rest is out of scope
for this change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AssertionResult:
    """A pass/fail plus a human-readable reason, so a failure is diagnosable
    from the report alone."""

    passed: bool
    reason: str = ""


def assert_refus_aucune_base(state: dict[str, Any]) -> AssertionResult:
    """The graph refused: no answer was produced, and it recorded why."""
    if state.get("answer") is not None:
        return AssertionResult(
            passed=False,
            reason=f"expected a refusal but an answer was produced: {state['answer']!r}",
        )
    if not state.get("refusal_reason"):
        return AssertionResult(
            passed=False,
            reason="expected 'refusal_reason' to be set, but it is empty",
        )
    return AssertionResult(passed=True)


def assert_divergences_non_vides(state: dict[str, Any]) -> AssertionResult:
    """The graph surfaced at least one divergence between the two regulations."""
    divergences = state.get("divergences") or []
    if not divergences:
        return AssertionResult(
            passed=False,
            reason="expected at least one divergence, but 'divergences' is empty",
        )
    return AssertionResult(passed=True)


@dataclass(frozen=True)
class EvalSample:
    """One golden entry paired with the graph's final state for that question.

    The four `question`/`answer`/`retrieved_contexts`/`reference` fields exist
    only for RAGAS's benefit (its `SingleTurnSample` needs exactly this
    shape); `refus`/`divergence` entries never populate them and are scored
    from `state` alone, so they all default to empty/`None`.
    """

    entry_id: str
    state: dict[str, Any]
    question: str | None = None
    answer: str | None = None
    retrieved_contexts: list[str] = field(default_factory=list)
    reference: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """The outcome of running an `Evaluator` over a batch of samples."""

    results: dict[str, AssertionResult]


class Evaluator(Protocol):
    """What a scoring backend must implement, deterministic or LLM-judged.

    The runner and CLI that call this are out of scope here — this is only
    the socket they will plug into.
    """

    def evaluate(self, samples: list[EvalSample]) -> EvaluationReport: ...


# RAGAS metric names, in the deprecated-but-Mistral-compatible lowercase form
# (`from ragas import evaluate`, not `ragas.metrics.collections` — the modern
# path is closed to Mistral: `instructor` expects the v1 SDK). See the eval
# extra's comment in `pyproject.toml` for how that was established.
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _import_ragas():
    """Import `ragas` lazily, failing with a fixable instruction instead of
    a bare `ImportError` — this module (and the deterministic assertions
    above) must stay usable in the default environment, where `ragas` is not
    installed on purpose (see `pyproject.toml`'s `eval` extra).
    """
    try:
        import ragas
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed; run `uv sync --extra eval` to enable "
            "RagasEvaluator (agent/pyproject.toml, optional 'eval' extra)"
        ) from exc
    return ragas, evaluate, LangchainLLMWrapper


def read_metric_columns(result: Any, metric_names: tuple[str, ...]) -> dict[str, list[float]]:
    """Pull the per-question score list for each metric that was computed.

    Membership testing is deliberately avoided: the RAGAS result object
    defines `__getitem__` for string keys but neither `__contains__` nor
    `__iter__`, so `name in result` silently falls back to the sequence
    protocol and probes integer indices, raising on a string-keyed store.
    Asking for the key and catching its absence is the only safe read.
    """
    columns: dict[str, list[float]] = {}
    for metric in metric_names:
        try:
            columns[metric] = list(result[metric])
        except (KeyError, TypeError, IndexError):
            continue
    return columns


def to_raw_scores(columns: dict[str, list[float]]) -> dict[str, list[float]]:
    """Normalise a RAGAS result into per-question score lists, `nan`s kept.

    Keeping the individual values — rather than a mean — is what lets the
    report count how many questions failed to score at all. RAGAS returns
    `nan` for a judge call that errored, without raising, so an average taken
    here would quietly describe a different, smaller dataset than the one
    that was submitted.
    """
    return {metric: [float(value) for value in values] for metric, values in columns.items()}


class RagasEvaluator:
    """Scores `repondre`/`croise`/`reference_exacte` samples with RAGAS.

    Only exercised end-to-end when `ragas` is installed; the constructor's
    import failure path is what the default (ragas-less) test environment
    actually runs.
    """

    def __init__(self, *, llm: Any, strict: bool = False, run_config: Any = None) -> None:
        _ragas, _evaluate, LangchainLLMWrapper = _import_ragas()
        self._evaluate = _evaluate
        self._judge = LangchainLLMWrapper(llm)
        self._strict = strict
        self._run_config = run_config

    def evaluate(self, samples: list[EvalSample]):
        from datasets import Dataset  # ragas's own dependency, see the eval extra

        from eval.report import build_report

        rows = [
            {
                "question": sample.question,
                "answer": sample.answer,
                "contexts": sample.retrieved_contexts,
                "ground_truth": sample.reference,
            }
            for sample in samples
        ]
        dataset = Dataset.from_list(rows)

        kwargs: dict[str, Any] = {"llm": self._judge, "raise_exceptions": self._strict}
        if self._run_config is not None:
            kwargs["run_config"] = self._run_config

        result = self._evaluate(dataset, **kwargs)

        # Read the per-question columns rather than the aggregate: the result
        # object's own repr averages them with the `nan`s dropped.
        columns = read_metric_columns(result, RAGAS_METRIC_NAMES)
        return build_report(to_raw_scores(columns))
