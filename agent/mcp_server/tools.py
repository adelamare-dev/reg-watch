"""The four regulatory tools, as plain functions.

Each one takes its retriever as an argument rather than reaching for a global,
so the whole tool surface is exercisable without Qdrant, without a model, and
without a server. `server.py` is the only place that wires the real ones.

Return values are plain dictionaries: the MCP layer turns them into structured
content, and the UI consumes those fields directly instead of re-parsing prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from rag.retriever import RetrievalResult
from rag.store import SearchHit

# Canonical values as stored in the Qdrant payload.
DORA = "DORA"
AI_ACT = "AI_ACT"

# Retrieval filters on the canonical value, but a question reaches us with
# whatever the user typed. Without this mapping, "AI Act" filters on nothing
# and an `article_number` lookup can answer from the wrong regulation.
_REGULATION_ALIASES: dict[str, str] = {
    "dora": DORA,
    "2022/2554": DORA,
    "reglement dora": DORA,
    "ai_act": AI_ACT,
    "ai act": AI_ACT,
    "aiact": AI_ACT,
    "ai-act": AI_ACT,
    "eu ai act": AI_ACT,
    "reglement ia": AI_ACT,
    "ia act": AI_ACT,
    "2024/1689": AI_ACT,
}

_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")


class RetrieverLike(Protocol):
    """The slice of the RAG read facade these tools depend on."""

    def search(
        self,
        question: str,
        *,
        language: str | None = ...,
        regulation: str | None = ...,
        top_k: int | None = ...,
    ) -> RetrievalResult: ...

    def get_article(
        self, regulation: str, article_number: str, language: str | None = ...
    ) -> SearchHit | None: ...


def canonical_regulation(name: str | None) -> str | None:
    """Map a regulation as a user writes it to the value stored in the payload.

    >>> canonical_regulation("AI Act")
    'AI_ACT'
    """
    if name is None:
        return None
    key = name.strip().lower().translate(_ACCENTS)
    key = " ".join(key.split())
    if key in _REGULATION_ALIASES:
        return _REGULATION_ALIASES[key]
    raise ValueError(f"unknown regulation {name!r}; expected one of DORA, AI_ACT")


def _serialise_hit(hit: SearchHit) -> dict[str, Any]:
    """Flatten a hit into the citation fields a caller needs.

    `text` carries the parent article rather than the matched sub-chunk: the
    sub-chunk is what made the match precise, the article is what makes the
    answer citable.
    """
    return {
        "regulation": hit.regulation,
        "article_number": hit.article_number,
        "paragraph": hit.paragraph,
        "language": hit.language,
        "citation": hit.citation,
        "text": hit.parent_text,
        "consolidation_date": hit.consolidation_date,
        "score": hit.score,
    }


def search_regulatory_corpus(
    retriever: RetrieverLike,
    *,
    query: str,
    regulation: str | None = None,
    language: str | None = None,
    article_number: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Search the corpus, with an exact payload filter when one applies."""
    canonical = canonical_regulation(regulation)

    question = query
    if article_number is not None:
        # The reference extractor reads article numbers out of the question
        # text, so an article passed as a separate argument has to reach it
        # through the question to take part in the exact filter.
        question = f"{query} article {article_number}"

    result = retriever.search(
        question,
        language=language,
        regulation=canonical,
        top_k=top_k,
    )

    return {
        "hits": [_serialise_hit(hit) for hit in result.hits],
        "is_grounded": result.is_grounded,
        "used_exact_filter": result.used_exact_filter,
    }


def get_article(
    retriever: RetrieverLike,
    *,
    regulation: str,
    article_number: str,
    language: str | None = None,
) -> dict[str, Any]:
    """Fetch one article by reference, with no vector search involved."""
    canonical = canonical_regulation(regulation)
    assert canonical is not None  # regulation is required here

    hit = retriever.get_article(canonical, article_number, language)
    if hit is None:
        return {
            "found": False,
            "regulation": canonical,
            "article_number": article_number,
            "text": None,
        }

    return {
        "found": True,
        "regulation": hit.regulation,
        "article_number": hit.article_number,
        "language": hit.language,
        "citation": hit.citation,
        "text": hit.parent_text,
        "consolidation_date": hit.consolidation_date,
    }


def compare_regulations(
    retriever: RetrieverLike,
    *,
    theme: str,
    language: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Put what each regulation says on a theme side by side.

    Deliberately does not judge: it retrieves each side independently and
    returns both. Deciding whether the two diverge, and saying so without
    arbitrating, belongs to the orchestration layer.
    """
    sides: dict[str, Any] = {"theme": theme}
    for key, regulation in (("dora", DORA), ("ai_act", AI_ACT)):
        result = retriever.search(
            theme, language=language, regulation=regulation, top_k=top_k
        )
        sides[key] = {
            "regulation": regulation,
            "hits": [_serialise_hit(hit) for hit in result.hits],
            "is_grounded": result.is_grounded,
        }
    return sides


def get_corpus_manifest(manifest_path: Path) -> dict[str, Any]:
    """Report which corpus version the answers are grounded in."""
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    documents = raw.get("documents", [])

    return {
        "generated_at": raw.get("generated_at"),
        # Hashes stay null until an ingestion has run, so their presence is
        # what tells a caller the corpus is actually queryable.
        "is_ingested": bool(documents)
        and all(document.get("content_sha256") for document in documents),
        "documents": [
            {
                "id": document.get("id"),
                "celex": document.get("celex"),
                "regulation": document.get("regulation"),
                "language": document.get("language"),
                "consolidation_date": document.get("consolidation_date"),
                "sha256": document.get("sha256"),
                "content_sha256": document.get("content_sha256"),
                "url": document.get("url"),
            }
            for document in documents
        ],
    }
