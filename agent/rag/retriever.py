"""Read facade of the RAG layer: turns a free-text question into grounded hits.

This is the only module the orchestration layer and the tool server are meant
to call for retrieval. It never calls an LLM: given a question, it returns
either evidence (`SearchHit`s from the corpus) or an explicit "not grounded"
signal, and leaves the decision of what to do with that to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.config import Settings
from rag.embedder import Embedder
from rag.references import References, extract_references
from rag.store import SearchHit, build_filter, search


@dataclass(frozen=True)
class RetrievalResult:
    """Everything the caller needs to answer, refuse, or cite its sources."""

    hits: list[SearchHit]
    references: References
    used_exact_filter: bool
    is_grounded: bool


def _dedupe_by_article(hits: list[SearchHit]) -> list[SearchHit]:
    """Keep one hit per (regulation, article_number, language).

    A single article is split into several child chunks at ingestion time, so
    a vector search routinely returns more than one sub-chunk of the same
    article. All of them share the same `parent_text`, so keeping more than
    one just spends the LLM's context window on a repeated article instead of
    additional distinct evidence. Hits arrive already sorted by descending
    score, so the first hit seen for a key is its best-scoring sub-chunk.
    """
    best_by_key: dict[tuple[str, str, str], SearchHit] = {}
    order: list[tuple[str, str, str]] = []
    for hit in hits:
        key = (hit.regulation, hit.article_number, hit.language)
        if key not in best_by_key:
            best_by_key[key] = hit
            order.append(key)
    return [best_by_key[key] for key in order]


class Retriever:
    """Vector + exact-filter search over the regulatory corpus."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        embedder: Embedder,
        settings: Settings,
        collection_name: str | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = settings
        # Overridable so tests do not need a Settings-derived collection name
        # that also matches the embedder's actual dimension.
        self._collection_name = collection_name or settings.collection_name

    def search(
        self,
        question: str,
        *,
        language: str | None = None,
        regulation: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Answer a question with corpus evidence, exact filter first.

        When the question cites a specific article or regulation, an exact
        payload filter is applied on top of the vector search. If that filter
        turns out to match nothing — the user misquoted the article number,
        or cited it under the wrong regulation — a filtered empty result
        would be worse than no filter at all: the vector search alone might
        still have found the right context. So an empty filtered result
        triggers one retry without the filter, and `used_exact_filter`
        reports what actually produced the returned hits.
        """
        references = extract_references(question)
        vector = self._embedder.embed_query(question)
        limit = top_k if top_k is not None else self._settings.top_k

        query_filter = build_filter(references, language=language, regulation=regulation)
        used_exact_filter = query_filter is not None

        hits = search(
            self._client,
            self._collection_name,
            vector,
            limit=limit,
            score_threshold=self._settings.score_threshold,
            query_filter=query_filter,
        )

        if used_exact_filter and not hits:
            hits = search(
                self._client,
                self._collection_name,
                vector,
                limit=limit,
                score_threshold=self._settings.score_threshold,
                query_filter=None,
            )
            used_exact_filter = False

        hits = _dedupe_by_article(hits)

        return RetrievalResult(
            hits=hits,
            references=references,
            used_exact_filter=used_exact_filter,
            is_grounded=bool(hits),
        )

    def get_article(
        self, regulation: str, article_number: str, language: str | None = None
    ) -> SearchHit | None:
        """Fetch one article deterministically by its reference, no vector search.

        Used when the caller already knows exactly what it wants (a citation
        follow-up, a cross-reference) rather than approximating it through
        similarity search.
        """
        must = [
            FieldCondition(key="regulation", match=MatchValue(value=regulation)),
            FieldCondition(key="article_number", match=MatchValue(value=article_number)),
        ]
        if language is not None:
            must.append(FieldCondition(key="language", match=MatchValue(value=language)))

        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(must=must),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None

        payload = records[0].payload or {}
        return SearchHit(
            score=1.0,
            chunk_text=payload["text"],
            parent_text=payload["parent_text"],
            citation=payload["citation"],
            regulation=payload["regulation"],
            article_number=payload["article_number"],
            paragraph=payload["paragraph"],
            language=payload["language"],
            consolidation_date=payload.get("consolidation_date"),
        )
