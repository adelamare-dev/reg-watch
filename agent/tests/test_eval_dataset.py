"""Loading and validating the golden dataset.

Two concerns are kept apart on purpose:
- `load_dataset` reads the JSONL as-is, drafts included, so the templates can
  be inspected and edited before anyone has written a real answer.
- `validate_for_evaluation` is the gate a run through the graph must pass;
  it is the one piece of code that stands between a real evaluation report
  and a dataset nobody has actually written by hand yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dataset import (
    DatasetValidationError,
    GoldenEntry,
    load_dataset,
    validate_for_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = REPO_ROOT / "data" / "golden-dataset.jsonl"


def _write_jsonl(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _repondre_entry(**overrides: object) -> dict:
    base = {
        "id": "fact-dora-001",
        "categorie": "factuel_mono",
        "langue": "fr",
        "comportement_attendu": "repondre",
        "question": "<a rediger>",
        "reference": "<a rediger>",
        "reference_context_ids": ["DORA-Art11"],
        "draft": True,
    }
    base.update(overrides)
    return base


def _refus_entry(**overrides: object) -> dict:
    base = {
        "id": "hors-001",
        "categorie": "hors_corpus",
        "langue": "fr",
        "comportement_attendu": "refus",
        "question": "<a rediger>",
        "assertion": "refus_aucune_base",
        "draft": True,
    }
    base.update(overrides)
    return base


def _divergence_entry(**overrides: object) -> dict:
    base = {
        "id": "div-001",
        "categorie": "divergence",
        "langue": "en",
        "comportement_attendu": "divergence",
        "question": "<a rediger>",
        "reference_context_ids": ["DORA-Art11", "AI_ACT-Art9"],
        "assertion": "divergences_non_vides",
        "draft": True,
    }
    base.update(overrides)
    return base


# --- load_dataset: reads drafts as-is --------------------------------------


def test_load_dataset_reads_draft_entries_without_raising(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [_repondre_entry(), _refus_entry()])

    entries = load_dataset(path)

    assert len(entries) == 2
    assert all(isinstance(e, GoldenEntry) for e in entries)
    assert entries[0].draft is True


def test_load_dataset_parses_each_declared_field(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [_divergence_entry()])

    [entry] = load_dataset(path)

    assert entry.id == "div-001"
    assert entry.categorie == "divergence"
    assert entry.langue == "en"
    assert entry.comportement_attendu == "divergence"
    assert entry.assertion == "divergences_non_vides"
    assert entry.reference_context_ids == ["DORA-Art11", "AI_ACT-Art9"]


# --- validate_for_evaluation: the methodological gate ----------------------


def test_validate_for_evaluation_rejects_draft_entries(tmp_path: Path) -> None:
    """The one non-negotiable rule: a draft can never silently pass as ready."""
    path = _write_jsonl(tmp_path, [_repondre_entry(draft=True)])
    entries = load_dataset(path)

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "fact-dora-001" in message
    assert "draft" in message


def test_validate_for_evaluation_accepts_a_fully_written_repondre_entry() -> None:
    entries = [
        GoldenEntry.model_validate(
            _repondre_entry(
                draft=False,
                question="Quelles obligations DORA impose-t-elle en matiere de tests de resilience ?",
                reference="Un texte de reference ecrit a la main.",
            )
        )
    ]

    validate_for_evaluation(entries)  # must not raise


def test_validate_for_evaluation_rejects_unknown_categorie() -> None:
    entries = [GoldenEntry.model_validate(_repondre_entry(draft=False, categorie="inconnue"))]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "fact-dora-001" in message
    assert "categorie" in message


def test_validate_for_evaluation_rejects_unknown_comportement() -> None:
    entries = [
        GoldenEntry.model_validate(
            _repondre_entry(draft=False, comportement_attendu="inconnu")
        )
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    assert "comportement_attendu" in str(excinfo.value)


def test_validate_for_evaluation_rejects_unknown_langue() -> None:
    entries = [GoldenEntry.model_validate(_repondre_entry(draft=False, langue="de"))]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    assert "langue" in str(excinfo.value)


def test_validate_for_evaluation_rejects_duplicate_ids() -> None:
    entries = [
        GoldenEntry.model_validate(_repondre_entry(draft=False, id="dup-1")),
        GoldenEntry.model_validate(_refus_entry(draft=False, id="dup-1")),
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "dup-1" in message
    assert "dupli" in message.lower()


def test_validate_for_evaluation_rejects_repondre_entry_missing_reference() -> None:
    entries = [
        GoldenEntry.model_validate(
            _repondre_entry(draft=False, reference=None, question="Une vraie question.")
        )
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "fact-dora-001" in message
    assert "reference" in message


def test_validate_for_evaluation_rejects_repondre_entry_missing_context_ids() -> None:
    entries = [
        GoldenEntry.model_validate(
            _repondre_entry(
                draft=False,
                reference_context_ids=[],
                question="Une vraie question.",
                reference="Une vraie reference.",
            )
        )
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "fact-dora-001" in message
    assert "reference_context_ids" in message


def test_validate_for_evaluation_rejects_refus_entry_missing_assertion() -> None:
    entries = [
        GoldenEntry.model_validate(
            _refus_entry(draft=False, assertion=None, question="Une vraie question.")
        )
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "hors-001" in message
    assert "assertion" in message


def test_validate_for_evaluation_rejects_divergence_entry_missing_assertion() -> None:
    entries = [
        GoldenEntry.model_validate(
            _divergence_entry(draft=False, assertion=None, question="Une vraie question.")
        )
    ]

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    message = str(excinfo.value)
    assert "div-001" in message
    assert "assertion" in message


def test_validate_for_evaluation_accepts_a_fully_written_refus_entry() -> None:
    entries = [
        GoldenEntry.model_validate(
            _refus_entry(draft=False, question="Une question hors du corpus DORA/AI Act.")
        )
    ]

    validate_for_evaluation(entries)  # must not raise


# --- the delivered golden-dataset.jsonl -------------------------------------


def test_golden_dataset_file_exists_and_is_versioned() -> None:
    assert GOLDEN_DATASET_PATH.exists(), "data/golden-dataset.jsonl is missing"


def test_golden_dataset_has_exactly_fifteen_entries() -> None:
    entries = load_dataset(GOLDEN_DATASET_PATH)

    assert len(entries) == 15


def test_golden_dataset_category_distribution_matches_the_spec() -> None:
    entries = load_dataset(GOLDEN_DATASET_PATH)

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.categorie] = counts.get(entry.categorie, 0) + 1

    assert counts == {
        "hors_corpus": 4,
        "divergence": 3,
        "factuel_mono": 4,
        "croise": 2,
        "reference_exacte": 2,
    }


def test_golden_dataset_is_entirely_draft() -> None:
    """Every delivered template is a placeholder, never a written answer."""
    entries = load_dataset(GOLDEN_DATASET_PATH)

    assert all(entry.draft is True for entry in entries)


def test_golden_dataset_is_rejected_by_the_evaluation_gate() -> None:
    """Proof the mechanism works: the shipped file cannot be used to evaluate yet."""
    entries = load_dataset(GOLDEN_DATASET_PATH)

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_for_evaluation(entries)

    assert "draft" in str(excinfo.value)


def test_golden_dataset_ids_are_unique() -> None:
    entries = load_dataset(GOLDEN_DATASET_PATH)

    ids = [entry.id for entry in entries]
    assert len(ids) == len(set(ids))
