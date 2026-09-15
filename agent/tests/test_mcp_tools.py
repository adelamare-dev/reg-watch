"""The four regulatory tools exposed over MCP.

Every test here runs against a fake retriever: the tool layer is pure
translation between the MCP wire format and the RAG read facade, so it can be
exercised without Qdrant, without a model download, and without a running
server.
"""

from __future__ import annotations

import json

import pytest

from mcp_server.tools import (
    canonical_regulation,
    compare_regulations,
    get_article,
    get_corpus_manifest,
    search_regulatory_corpus,
)
from rag.references import References
from rag.retriever import RetrievalResult
from rag.store import SearchHit


def make_hit(
    *,
    regulation: str = "DORA",
    article_number: str = "28",
    language: str = "fr",
    score: float = 0.9,
    paragraph: str = "1",
) -> SearchHit:
    return SearchHit(
        score=score,
        chunk_text=f"Chunk of {regulation} article {article_number}.",
        parent_text=f"Full text of {regulation} article {article_number}.",
        citation=f"{regulation}, article {article_number}",
        regulation=regulation,
        article_number=article_number,
        paragraph=paragraph,
        language=language,
        consolidation_date="2025-01-17",
    )


class FakeRetriever:
    """Records the arguments it was called with and replays canned results."""

    def __init__(
        self,
        *,
        hits: list[SearchHit] | None = None,
        article: SearchHit | None = None,
        used_exact_filter: bool = False,
    ) -> None:
        self._hits = hits if hits is not None else []
        self._article = article
        self._used_exact_filter = used_exact_filter
        self.search_calls: list[dict] = []
        self.get_article_calls: list[dict] = []

    def search(
        self,
        question: str,
        *,
        language: str | None = None,
        regulation: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        self.search_calls.append(
            {
                "question": question,
                "language": language,
                "regulation": regulation,
                "top_k": top_k,
            }
        )
        return RetrievalResult(
            hits=self._hits,
            references=References(),
            used_exact_filter=self._used_exact_filter,
            is_grounded=bool(self._hits),
        )

    def get_article(
        self, regulation: str, article_number: str, language: str | None = None
    ) -> SearchHit | None:
        self.get_article_calls.append(
            {
                "regulation": regulation,
                "article_number": article_number,
                "language": language,
            }
        )
        return self._article


# --- canonical_regulation -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DORA", "DORA"),
        ("dora", "DORA"),
        ("  Dora  ", "DORA"),
        ("2022/2554", "DORA"),
        ("AI_ACT", "AI_ACT"),
        ("AI Act", "AI_ACT"),
        ("ai act", "AI_ACT"),
        ("AI-Act", "AI_ACT"),
        ("règlement IA", "AI_ACT"),
        ("2024/1689", "AI_ACT"),
    ],
)
def test_canonical_regulation_accepts_usual_names(raw: str, expected: str) -> None:
    assert canonical_regulation(raw) == expected


def test_canonical_regulation_passes_none_through() -> None:
    assert canonical_regulation(None) is None


def test_canonical_regulation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown regulation"):
        canonical_regulation("MiCA")


# --- search_regulatory_corpus ---------------------------------------------


def test_search_returns_one_entry_per_hit() -> None:
    retriever = FakeRetriever(
        hits=[make_hit(article_number="28"), make_hit(article_number="30")]
    )

    result = search_regulatory_corpus(retriever, query="risque tiers")

    assert [hit["article_number"] for hit in result["hits"]] == ["28", "30"]
    assert result["hits"][0]["regulation"] == "DORA"
    assert result["hits"][0]["score"] == pytest.approx(0.9)
    assert result["hits"][0]["text"] == "Full text of DORA article 28."


def test_search_reports_grounding_and_filter_usage() -> None:
    retriever = FakeRetriever(hits=[make_hit()], used_exact_filter=True)

    result = search_regulatory_corpus(retriever, query="article 28")

    assert result["is_grounded"] is True
    assert result["used_exact_filter"] is True


def test_search_without_hits_is_not_grounded() -> None:
    result = search_regulatory_corpus(FakeRetriever(hits=[]), query="MiCA staking")

    assert result["hits"] == []
    assert result["is_grounded"] is False


def test_search_normalises_regulation_name_before_filtering() -> None:
    """A caller saying "AI Act" must filter as if it had said "AI_ACT"."""
    retriever = FakeRetriever(hits=[make_hit()])

    search_regulatory_corpus(retriever, query="gestion des risques", regulation="AI Act")

    assert retriever.search_calls[0]["regulation"] == "AI_ACT"


def test_search_forwards_language_and_top_k() -> None:
    retriever = FakeRetriever(hits=[make_hit()])

    search_regulatory_corpus(retriever, query="incident", language="en", top_k=3)

    call = retriever.search_calls[0]
    assert call["language"] == "en"
    assert call["top_k"] == 3


def test_search_derives_regulation_from_article_number_argument() -> None:
    """`article_number` alone must not leak across regulations.

    Filtering on an article number without naming a regulation matches the
    same number in both texts, so a question about "article 28" can answer
    with the other regulation's article 28.
    """
    retriever = FakeRetriever(hits=[make_hit()])

    search_regulatory_corpus(
        retriever, query="obligations", regulation="DORA", article_number="28"
    )

    assert "28" in retriever.search_calls[0]["question"]
    assert retriever.search_calls[0]["regulation"] == "DORA"


def test_search_rejects_unknown_regulation() -> None:
    with pytest.raises(ValueError, match="unknown regulation"):
        search_regulatory_corpus(FakeRetriever(), query="x", regulation="RGPD")


# --- get_article ----------------------------------------------------------


def test_get_article_returns_full_text() -> None:
    retriever = FakeRetriever(article=make_hit(article_number="28"))

    result = get_article(retriever, regulation="DORA", article_number="28")

    assert result["found"] is True
    assert result["article_number"] == "28"
    assert result["text"] == "Full text of DORA article 28."
    assert result["consolidation_date"] == "2025-01-17"


def test_get_article_reports_a_missing_article_without_raising() -> None:
    result = get_article(FakeRetriever(article=None), regulation="DORA", article_number="999")

    assert result["found"] is False
    assert result["text"] is None


def test_get_article_normalises_regulation_name() -> None:
    retriever = FakeRetriever(article=make_hit(regulation="AI_ACT", article_number="9"))

    get_article(retriever, regulation="AI Act", article_number="9")

    assert retriever.get_article_calls[0]["regulation"] == "AI_ACT"


def test_get_article_never_runs_a_vector_search() -> None:
    """Determinism is the whole point of this tool."""
    retriever = FakeRetriever(article=make_hit())

    get_article(retriever, regulation="DORA", article_number="28")

    assert retriever.search_calls == []


# --- compare_regulations --------------------------------------------------


def test_compare_queries_each_regulation_separately() -> None:
    retriever = FakeRetriever(hits=[make_hit()])

    compare_regulations(retriever, theme="gestion des risques")

    assert [call["regulation"] for call in retriever.search_calls] == ["DORA", "AI_ACT"]


def test_compare_returns_both_sides_side_by_side() -> None:
    retriever = FakeRetriever(hits=[make_hit()])

    result = compare_regulations(retriever, theme="notification d'incidents")

    assert result["theme"] == "notification d'incidents"
    assert "dora" in result
    assert "ai_act" in result


def test_compare_forwards_language_to_both_sides() -> None:
    retriever = FakeRetriever(hits=[make_hit()])

    compare_regulations(retriever, theme="incident reporting", language="en")

    assert [call["language"] for call in retriever.search_calls] == ["en", "en"]


def test_compare_does_not_arbitrate_between_regulations() -> None:
    """The tool juxtaposes; judging divergence belongs to the Critic node."""
    retriever = FakeRetriever(hits=[make_hit()])

    result = compare_regulations(retriever, theme="risk management")

    assert "verdict" not in result
    assert "conclusion" not in result


# --- get_corpus_manifest --------------------------------------------------


def test_manifest_exposes_document_versions(tmp_path) -> None:
    manifest_path = tmp_path / "corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-09-15",
                "documents": [
                    {
                        "id": "dora-reg-fr",
                        "celex": "32022R2554",
                        "regulation": "DORA",
                        "language": "fr",
                        "consolidation_date": "2025-01-17",
                        "sha256": "abc",
                        "content_sha256": "def",
                        "url": "https://eur-lex.europa.eu/x",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = get_corpus_manifest(manifest_path)

    assert result["generated_at"] == "2026-09-15"
    document = result["documents"][0]
    assert document["id"] == "dora-reg-fr"
    assert document["consolidation_date"] == "2025-01-17"
    assert document["content_sha256"] == "def"


def test_manifest_reports_a_corpus_that_was_never_ingested(tmp_path) -> None:
    """Null hashes mean the manifest has not been through an ingestion yet."""
    manifest_path = tmp_path / "corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-09-15",
                "documents": [
                    {
                        "id": "dora-reg-fr",
                        "celex": "32022R2554",
                        "regulation": "DORA",
                        "language": "fr",
                        "consolidation_date": "2025-01-17",
                        "sha256": None,
                        "content_sha256": None,
                        "url": "https://eur-lex.europa.eu/x",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = get_corpus_manifest(manifest_path)

    assert result["is_ingested"] is False
