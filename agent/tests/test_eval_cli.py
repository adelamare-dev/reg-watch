"""The eval CLI: deterministic-only by default, ragas opt-in, non-zero exit on an invalid run.

Most tests here call `run_cli` with fakes, since it takes every collaborator
as a keyword argument. `main()` gets its own coverage further down: it is the
seam where real collaborators are assembled, and an injectable core proves
nothing about an entry point that forgets to build them.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import eval.__main__ as eval_main
from eval.__main__ import publish_scores, run_cli
from eval.metrics import EvalSample
from eval.metrics import EvaluationReport as DeterministicOutcome
from eval.report import EvaluationReport as RagasOutcome


def _write_dataset(tmp_path: Path, entries: list[dict]) -> Path:
    import json

    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return path


def _refus_entry(entry_id: str) -> dict:
    return {
        "id": entry_id,
        "categorie": "hors_corpus",
        "langue": "fr",
        "comportement_attendu": "refus",
        "question": "question de test hors corpus",
        "assertion": "refus_aucune_base",
        "draft": False,
    }


def _divergence_entry(entry_id: str) -> dict:
    return {
        "id": entry_id,
        "categorie": "divergence",
        "langue": "fr",
        "comportement_attendu": "divergence",
        "question": "question de test croisée",
        "reference_context_ids": ["DORA-Art1", "AI_ACT-Art1"],
        "assertion": "divergences_non_vides",
        "draft": False,
    }


def test_main_assembles_real_collaborators_instead_of_passing_none() -> None:
    """`main()` must build a retriever, an LLM and a provider before running.

    Handing `run_cli` a `None` retriever type-checks and passes every
    fake-driven test above, then dies on the first real attribute access
    inside the graph. Reading the source is a blunt instrument, but running
    `main()` for real would need Qdrant and an API key, which the suite
    deliberately does without.
    """
    source = inspect.getsource(eval_main.main)

    for constructor in ("get_settings(", "build_embedder(", "Retriever(", "build_llm("):
        assert constructor in source, f"main() never calls {constructor}"

    for unwired in ("retriever=None", "llm=None", "provider=None"):
        assert unwired not in source, f"main() still hands {unwired} to the runner"


class _ClientWithoutActiveSpan:
    """A tracing client as it is once every graph run has finished.

    `get_current_trace_id()` reads the *active* span, and by scoring time
    every span the run opened has been closed — so a publisher that asks for
    it there gets `None` and silently writes nothing.
    """

    def __init__(self) -> None:
        self.scores: list[dict] = []

    def get_current_trace_id(self) -> str | None:
        return None

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)


def test_scores_are_published_even_though_no_span_is_active_any_more() -> None:
    client = _ClientWithoutActiveSpan()
    report = RagasOutcome(
        scores={"faithfulness": 0.87},
        nan_counts={"faithfulness": 0},
        sample_counts={"faithfulness": 4},
    )

    publish_scores(client, report)

    assert [s["name"] for s in client.scores] == ["ragas_faithfulness"]
    assert client.scores[0]["value"] == 0.87


def test_publishing_without_a_client_is_a_no_op() -> None:
    report = RagasOutcome(scores={"faithfulness": 0.9}, nan_counts={}, sample_counts={})

    publish_scores(None, report)  # must not raise


def test_main_supports_the_ragas_mode_it_advertises() -> None:
    """`--ragas` must reach `RagasEvaluator`, not a placeholder that gives up.

    An option documented in `--help` that always returns an error is worse
    than no option: it reads as a supported path right up to the moment
    someone depends on it.
    """
    source = inspect.getsource(eval_main.main)

    assert "non câblé" not in source, "--ragas is still a stub"
    assert "RagasEvaluator" in source


def fake_run_dataset(entries, *, retriever, llm, provider, eval_top_k, client=None, run_question=None):
    """Stands in for `eval.runner.run_dataset`: no graph, no LLM call."""
    samples = []
    for entry in entries:
        if entry.comportement_attendu == "refus":
            state = {"answer": None, "refusal_reason": "hors du corpus"}
        else:
            state = {"answer": "réponse", "divergences": [{"theme": "x"}]}
        samples.append(EvalSample(entry_id=entry.id, state=state))
    return samples


def test_deterministic_only_mode_needs_no_llm_and_validates_refus_and_divergence(tmp_path) -> None:
    dataset_path = _write_dataset(
        tmp_path, [_refus_entry("r1"), _divergence_entry("d1")]
    )

    exit_code, outcome = run_cli(
        dataset_path=dataset_path,
        deterministic_only=True,
        run_dataset=fake_run_dataset,
        retriever=None,
        llm=None,
        provider=None,
        client=None,
    )

    assert exit_code == 0
    assert isinstance(outcome, DeterministicOutcome)
    assert outcome.results["r1"].passed is True
    assert outcome.results["d1"].passed is True


def test_deterministic_only_mode_never_imports_or_constructs_a_ragas_evaluator(tmp_path) -> None:
    # The graph itself still runs (it needs to, to produce a `state` the
    # assertions can read) — what deterministic-only mode guarantees is that
    # no judge-model call is made on top of it. Asserting `ragas` is never
    # imported by `run_cli` is the strongest structural guarantee available:
    # `eval.metrics` imports cleanly without ragas installed (see
    # `test_eval_ragas_evaluator.py`), so any accidental judge-model use would
    # show up as an attempt to import a package this environment lacks.
    import sys

    dataset_path = _write_dataset(tmp_path, [_refus_entry("r1")])
    ragas_was_imported_before = "ragas" in sys.modules

    exit_code, outcome = run_cli(
        dataset_path=dataset_path,
        deterministic_only=True,
        run_dataset=fake_run_dataset,
        retriever=None,
        llm=None,
        provider=None,
        client=None,
    )

    assert exit_code == 0
    assert outcome.results["r1"].passed is True
    assert ("ragas" in sys.modules) == ragas_was_imported_before


def test_deterministic_only_mode_fails_a_refus_entry_whose_state_answered_instead(tmp_path) -> None:
    dataset_path = _write_dataset(tmp_path, [_refus_entry("r1")])

    def wrong_answer_run_dataset(entries, **kwargs):
        return [EvalSample(entry_id="r1", state={"answer": "oups", "refusal_reason": None})]

    exit_code, outcome = run_cli(
        dataset_path=dataset_path,
        deterministic_only=True,
        run_dataset=wrong_answer_run_dataset,
        retriever=None,
        llm=None,
        provider=None,
        client=None,
    )

    assert exit_code != 0
    assert outcome.results["r1"].passed is False


def test_cli_exits_non_zero_when_the_dataset_fails_validation(tmp_path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "id": "draft-1",
                "categorie": "hors_corpus",
                "langue": "fr",
                "comportement_attendu": "refus",
                "question": "<à rédiger>",
                "assertion": "refus_aucune_base",
                "draft": True,
            }
        ],
    )

    exit_code, outcome = run_cli(
        dataset_path=dataset_path,
        deterministic_only=True,
        run_dataset=fake_run_dataset,
        retriever=None,
        llm=None,
        provider=None,
        client=None,
    )

    assert exit_code != 0
