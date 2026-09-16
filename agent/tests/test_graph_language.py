"""Language detection decides the answer's language, never the retrieval scope."""

from __future__ import annotations

import pytest

from graph.language import detect_language


@pytest.mark.parametrize(
    "question",
    [
        "Que dit l'article 28 de DORA ?",
        "Quelles sont les obligations de notification d'incidents ?",
        "Comment gérer le risque lié aux prestataires tiers ?",
    ],
)
def test_detects_french(question: str):
    assert detect_language(question) == "fr"


@pytest.mark.parametrize(
    "question",
    [
        "What does article 28 of DORA say?",
        "Which incident reporting obligations apply?",
        "How should third-party risk be managed?",
    ],
)
def test_detects_english(question: str):
    assert detect_language(question) == "en"


def test_falls_back_to_french_when_nothing_is_decisive():
    # Shared legal vocabulary carries no signal either way. French is the
    # project's default language, so an undecidable question answers in French.
    assert detect_language("Article 28 ?") == "fr"


def test_ignores_case_and_accents():
    assert detect_language("QUELLES SONT LES OBLIGATIONS ?") == "fr"
    assert detect_language("WHICH OBLIGATIONS APPLY?") == "en"
