"""The golden dataset: one JSONL entry per manually-written evaluation case.

Loading and validating are kept as two separate functions rather than one
function with a `strict` flag: a flag invites a caller to pass `strict=False`
out of habit and silently defeat the one protection that matters here — that
a `draft` entry can never be scored as if a human had written it. Two names
make the permissive path (`load_dataset`, for inspecting or editing templates)
a deliberate, visible choice distinct from the evaluation gate
(`validate_for_evaluation`).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

KNOWN_CATEGORIES = {
    "hors_corpus",
    "divergence",
    "factuel_mono",
    "croise",
    "reference_exacte",
}
KNOWN_BEHAVIOURS = {"refus", "divergence", "repondre"}
KNOWN_LANGUAGES = {"fr", "en"}


class DatasetValidationError(Exception):
    """Raised by `validate_for_evaluation` with one actionable message per problem.

    Every message names the offending entry's `id`, the field at fault, and
    what was expected, so a failure can be fixed without re-reading this
    module's source.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


class GoldenEntry(BaseModel):
    """One evaluation case, in either its `repondre`/`croise`/`reference_exacte`
    shape (with `reference` + `reference_context_ids`) or its `refus`/`divergence`
    shape (with `assertion`). Both shapes share this one model; which fields are
    actually required is a `validate_for_evaluation` concern, not a parsing one,
    because a draft entry is legitimately allowed to leave them as placeholders.
    """

    id: str
    categorie: str
    langue: str
    comportement_attendu: str
    question: str
    draft: bool
    reference: str | None = None
    reference_context_ids: list[str] = []
    assertion: str | None = None


def load_dataset(path: Path) -> list[GoldenEntry]:
    """Read the JSONL file as-is, drafts included.

    Used to inspect or edit templates before a human has written the real
    question/reference — never gates anything.
    """
    entries: list[GoldenEntry] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    [f"line {line_number}: invalid JSON ({exc})"]
                ) from exc
            try:
                entries.append(GoldenEntry.model_validate(raw))
            except ValidationError as exc:
                entry_id = raw.get("id", f"line {line_number}")
                raise DatasetValidationError(
                    [f"entry '{entry_id}': {exc}"]
                ) from exc
    return entries


def validate_for_evaluation(entries: list[GoldenEntry]) -> None:
    """The gate a dataset must pass before it can be used to score the graph.

    Raises `DatasetValidationError` listing every problem found (not just the
    first), each naming the entry `id`, the field, and what was expected.
    """
    problems: list[str] = []
    seen_ids: dict[str, int] = {}

    for entry in entries:
        if entry.id in seen_ids:
            problems.append(
                f"entry '{entry.id}': duplicated id (already used earlier in the file)"
            )
        else:
            seen_ids[entry.id] = 1

        if entry.draft:
            problems.append(
                f"entry '{entry.id}': field 'draft' is true — a human has not "
                "written this question/reference yet, it cannot be used to evaluate"
            )

        if entry.categorie not in KNOWN_CATEGORIES:
            problems.append(
                f"entry '{entry.id}': field 'categorie' has unknown value "
                f"'{entry.categorie}', expected one of {sorted(KNOWN_CATEGORIES)}"
            )

        if entry.comportement_attendu not in KNOWN_BEHAVIOURS:
            problems.append(
                f"entry '{entry.id}': field 'comportement_attendu' has unknown value "
                f"'{entry.comportement_attendu}', expected one of {sorted(KNOWN_BEHAVIOURS)}"
            )

        if entry.langue not in KNOWN_LANGUAGES:
            problems.append(
                f"entry '{entry.id}': field 'langue' has unknown value "
                f"'{entry.langue}', expected one of {sorted(KNOWN_LANGUAGES)}"
            )

        if entry.comportement_attendu == "repondre":
            if not entry.reference:
                problems.append(
                    f"entry '{entry.id}': comportement_attendu is 'repondre' but "
                    "field 'reference' is empty, expected a hand-written reference answer"
                )
            if not entry.reference_context_ids:
                problems.append(
                    f"entry '{entry.id}': comportement_attendu is 'repondre' but "
                    "field 'reference_context_ids' is empty, expected at least one "
                    "article id (e.g. 'DORA-Art11')"
                )

        if entry.comportement_attendu in {"refus", "divergence"} and not entry.assertion:
            problems.append(
                f"entry '{entry.id}': comportement_attendu is '{entry.comportement_attendu}' "
                "but field 'assertion' is empty, expected a deterministic assertion name "
                "(e.g. 'refus_aucune_base', 'divergences_non_vides')"
            )

    if problems:
        raise DatasetValidationError(problems)
