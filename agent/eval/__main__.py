"""Run the evaluation pipeline from the command line.

Deterministic-only by default: with no `ragas` installed (the default
environment, see `pyproject.toml`'s `eval` extra) or with `--deterministic-only`
passed explicitly, this validates every `refus`/`divergence` golden entry
against the graph's own final state — no judge model, no LLM call at all. A
full RAGAS run is opt-in via `--ragas`, because it costs `6 + eval_top_k` LLM
calls per `repondre`/`croise`/`reference_exacte` question (see
`agent/eval/runner.py`) and requires the optional dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from eval.dataset import DatasetValidationError, load_dataset, validate_for_evaluation
from eval.metrics import (
    AssertionResult,
    EvalSample,
    EvaluationReport,
    assert_divergences_non_vides,
    assert_refus_aucune_base,
)
from eval.report import EvaluationInvalid
from eval.report import EvaluationReport as RagasOutcome
from eval.runner import run_dataset as _default_run_dataset
from graph.llm import build_llm
from graph.observability import flush, init_observability
from rag.config import get_settings
from rag.embedder import build_embedder
from rag.retriever import Retriever

# One deterministic assertion per `comportement_attendu` that can be checked
# without a judge model. `repondre` entries have no entry here on purpose:
# grading free text against a reference needs RAGAS, which is exactly what
# deterministic-only mode exists to run without.
_DETERMINISTIC_ASSERTIONS = {
    "refus": assert_refus_aucune_base,
    "divergence": assert_divergences_non_vides,
}


def _run_deterministic(samples: list[EvalSample], entries_by_id: dict[str, Any]) -> EvaluationReport:
    """Score every sample whose entry has a deterministic assertion."""
    results: dict[str, AssertionResult] = {}
    for sample in samples:
        entry = entries_by_id[sample.entry_id]
        assertion = _DETERMINISTIC_ASSERTIONS.get(entry.comportement_attendu)
        if assertion is None:
            continue
        results[sample.entry_id] = assertion(sample.state)
    return EvaluationReport(results=results)


def render_outcome(outcome: Any) -> str:
    """Render either kind of outcome: assertions, RAGAS scores, or both."""
    lines: list[str] = []

    deterministic = getattr(outcome, "results", None)
    if deterministic:
        lines.append("=== ASSERTIONS DÉTERMINISTES ===")
        for entry_id, result in deterministic.items():
            mark = "OK" if result.passed else "ÉCHEC"
            suffix = f" — {result.reason}" if result.reason else ""
            lines.append(f"[{mark}] {entry_id}{suffix}")

    if isinstance(outcome, RagasOutcome):
        if lines:
            lines.append("")
        lines.append(outcome.render())

    return "\n".join(lines)


def run_cli(
    *,
    dataset_path: Path,
    deterministic_only: bool,
    run_dataset: Any = _default_run_dataset,
    retriever: Any,
    llm: Any,
    provider: Any,
    eval_top_k: int = 3,
    client: Any = None,
    evaluator_factory: Any = None,
    strict: bool = False,
) -> tuple[int, Any]:
    """The injectable core of `main()`: takes every collaborator as a keyword
    argument so tests can run it with fakes, no network, no LLM.

    Returns `(exit_code, outcome)`. `outcome` is an `eval.metrics.EvaluationReport`
    in deterministic-only mode, or an `eval.report.EvaluationReport` when RAGAS
    ran. A non-zero exit code means either the dataset failed validation or
    the run was invalid (see `eval.report.EvaluationInvalid`).
    """
    try:
        entries = load_dataset(dataset_path)
        validate_for_evaluation(entries)
    except DatasetValidationError as exc:
        print("=== DATASET INVALIDE ===", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1, None

    entries_by_id = {entry.id: entry for entry in entries}

    samples = run_dataset(
        entries,
        retriever=retriever,
        llm=llm,
        provider=provider,
        eval_top_k=eval_top_k,
        client=client,
    )

    deterministic_report = _run_deterministic(samples, entries_by_id)
    deterministic_failed = any(not r.passed for r in deterministic_report.results.values())

    if deterministic_only:
        return (1 if deterministic_failed else 0), deterministic_report

    # RAGAS only scores the entries a deterministic assertion cannot: grading
    # free text against a reference is precisely what needs a judge model.
    graded = [
        sample
        for sample in samples
        if entries_by_id[sample.entry_id].comportement_attendu not in _DETERMINISTIC_ASSERTIONS
    ]
    evaluator = evaluator_factory(llm=llm, strict=strict)
    ragas_report = evaluator.evaluate(graded)

    publish_scores(client, ragas_report)

    return (1 if deterministic_failed else 0), ragas_report


def publish_scores(client: Any, report: Any) -> None:
    """Send the run's aggregate scores, when tracing is on.

    No `trace_id` is attached on purpose. A run spans one trace per question,
    so there is no single trace these aggregates belong to — and by the time
    they exist every span is closed, which would make `get_current_trace_id()`
    return `None` and drop the scores without a word.
    """
    if client is None:
        return
    for metric, score in report.scores.items():
        client.create_score(
            name=f"ragas_{metric}",
            value=score,
            data_type="NUMERIC",
            comment=f"{report.nan_counts.get(metric, 0)} échec(s) 'nan' "
            f"sur {report.sample_counts.get(metric, 0)} question(s)",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="Évalue le graphe RegWatch sur le jeu de données golden.",
    )
    parser.add_argument("dataset", type=Path, help="Chemin du fichier JSONL golden")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Passe raise_exceptions=True à ragas.evaluate : utile en dev pour "
            "voir la première erreur, contre-productif sur un run complet où "
            "un premier 429 tuerait tous les appels restants."
        ),
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        default=False,
        help=(
            "Active la notation RAGAS (repondre/croise/reference_exacte). "
            "Nécessite `uv sync --extra eval` et un modèle juge. Sans cette "
            "option, seules les assertions déterministes (refus/divergence) "
            "sont vérifiées, sans aucun appel LLM."
        ),
    )
    args = parser.parse_args()

    # Fail before building anything expensive: an absent optional dependency
    # is a setup mistake, not a run to spend a Qdrant connection on.
    evaluator_factory = None
    if args.ragas:
        try:
            from eval.metrics import RagasEvaluator
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        evaluator_factory = RagasEvaluator

    settings = get_settings()
    embedder = build_embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    retriever = Retriever(client=qdrant_client, embedder=embedder, settings=settings)
    llm, provider = build_llm(settings)
    tracing_client = init_observability(settings)

    try:
        exit_code, outcome = run_cli(
            dataset_path=args.dataset,
            deterministic_only=evaluator_factory is None,
            retriever=retriever,
            llm=llm,
            provider=provider,
            eval_top_k=settings.eval_top_k,
            client=tracing_client,
            evaluator_factory=evaluator_factory,
            strict=args.strict,
        )
    except EvaluationInvalid as exc:
        print("=== RUN INVALIDE ===", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        # A run that raised is exactly the one whose trace is worth keeping.
        flush(tracing_client)

    if outcome is not None:
        print(render_outcome(outcome))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
