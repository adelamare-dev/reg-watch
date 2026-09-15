"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def dora_fr_html() -> str:
    """Excerpt of the French DORA HTML as served by EUR-Lex."""
    return (FIXTURES / "dora_fr_excerpt.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def aiact_en_html() -> str:
    """Excerpt of the English AI Act HTML as served by EUR-Lex."""
    return (FIXTURES / "aiact_en_excerpt.html").read_text(encoding="utf-8")
