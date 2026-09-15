"""Exact regulatory reference extraction.

The Qdrant payload filter is meant to be the *deterministic* part of the system,
where vector search is probabilistic. A false positive is therefore worse than a
false negative: it applies a wrong exact filter and discards the right chunks.
Hence the emphasis on negative cases below.
"""

from __future__ import annotations

import pytest

from rag.references import extract_references


class TestArticle:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Que dit l'article 28 de DORA ?", ["28"]),
            ("Article 9 de l'AI Act", ["9"]),
            ("article 73", ["73"]),
            ("What does Article 17 require?", ["17"]),
            ("ARTICLE 30", ["30"]),
            ("les articles 28 et 30", ["28", "30"]),
        ],
    )
    def test_extracts_article_numbers(self, query: str, expected: list[str]) -> None:
        assert extract_references(query).articles == expected

    def test_dedupes_preserving_order(self) -> None:
        refs = extract_references("article 28, puis article 5, puis article 28")
        assert refs.articles == ["28", "5"]

    def test_ignores_text_without_reference(self) -> None:
        refs = extract_references("Quelles sont les obligations de notification ?")
        assert refs.articles == []
        assert not refs.has_any


class TestRegulation:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("le règlement (UE) 2022/2554", ["2022/2554"]),
            ("Regulation (EU) 2024/1689", ["2024/1689"]),
            ("règlement UE 2024/1774", ["2024/1774"]),
            ("règlement n° 2022/2554", ["2022/2554"]),
            ("Regulation 2024/1689", ["2024/1689"]),
        ],
    )
    def test_extracts_context_anchored_reference(
        self, query: str, expected: list[str]
    ) -> None:
        assert extract_references(query).regulations == expected

    @pytest.mark.parametrize(
        "query",
        [
            # A bare r"(\d{4})/(\d{3,4})" pattern would match each of these —
            # hence the regulatory context anchoring.
            "applicable depuis le 2025/0117",
            "la période 2024/2025 a été retenue",
            "voir la page 2022/1024 du recueil",
            "consolidé au 17/01/2025",
            "exercice 2026/2027",
        ],
    )
    def test_does_not_match_bare_number_pair(self, query: str) -> None:
        assert extract_references(query).regulations == []


class TestAnnex:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("annexe III de l'AI Act", ["III"]),
            ("Annex IV", ["IV"]),
            ("l'annexe I", ["I"]),
            ("ANNEXE VIII", ["VIII"]),
        ],
    )
    def test_extracts_annex_numbers(self, query: str, expected: list[str]) -> None:
        assert extract_references(query).annexes == expected

    def test_ignores_annex_word_without_roman_numeral(self) -> None:
        assert extract_references("cette annexe est importante").annexes == []


class TestCombinations:
    def test_extracts_several_kinds_at_once(self) -> None:
        refs = extract_references(
            "Comparer l'article 28 du règlement (UE) 2022/2554 et l'annexe III"
        )
        assert refs.articles == ["28"]
        assert refs.regulations == ["2022/2554"]
        assert refs.annexes == ["III"]
        assert refs.has_any

    def test_has_any_is_false_on_purely_semantic_query(self) -> None:
        refs = extract_references(
            "Comment DORA et l'AI Act traitent-ils le risque lié aux tiers ?"
        )
        assert not refs.has_any
