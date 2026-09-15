"""Qdrant client wrapper: collection lifecycle, batched upsert, filtering, search.

Payload indexes are created on every field the retriever ever filters on
(`regulation`, `article_number`, `language`, `annex`). Without them, a filtered
query still runs the full HNSW graph traversal and discards non-matching
points afterward; with them, Qdrant prunes the traversal itself. On a corpus
of two regulations this barely matters, but the cost of adding the index is
zero and the alternative silently degrades as the corpus grows.

`query_points` (not the deprecated `search`) is the entry point for vector
search in this client version.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.chunker import Chunk
from rag.references import References

# Large enough to amortize per-request overhead, small enough to stay well
# under Qdrant's default payload size limit for a single upsert call.
UPSERT_BATCH_SIZE = 128

# Fields the retriever needs to filter on. Kept in one place so a new
# filterable field only needs to be added here.
_INDEXED_PAYLOAD_FIELDS = ("regulation", "article_number", "language", "annex")


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk, with the metadata a citation and a prompt need."""

    score: float
    chunk_text: str
    parent_text: str
    citation: str
    regulation: str
    article_number: str
    paragraph: str
    language: str
    consolidation_date: str | None = None


def _collection_dimension(client: QdrantClient, name: str) -> int:
    """Read back the vector size of an existing collection."""
    info = client.get_collection(name)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, VectorParams):
        return vectors_config.size
    # Named-vectors config: this store always creates a single anonymous
    # vector, so any other shape means the collection was made by other code.
    raise ValueError(f"collection {name!r} does not use a single anonymous vector")


def ensure_collection(
    client: QdrantClient, name: str, dimension: int, recreate: bool = False
) -> None:
    """Create the collection if it does not exist yet.

    A dimension mismatch against an existing collection means the embedding
    model changed since the last ingestion; that is a re-ingestion problem for
    a human to resolve, not something to paper over automatically.
    """
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if client.collection_exists(name):
        existing_dimension = _collection_dimension(client, name)
        if existing_dimension != dimension:
            raise ValueError(
                f"collection {name!r} already exists with dimension "
                f"{existing_dimension}, but the active embedding model produces "
                f"dimension {dimension}; re-ingest into a new collection instead "
                "of mixing incompatible vectors"
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    for field_name in _INDEXED_PAYLOAD_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def _chunk_payload(chunk: Chunk) -> dict[str, object]:
    """Everything a hit needs to render a citation, without re-reading the corpus."""
    return {
        "text": chunk.text,
        "parent_text": chunk.parent_text,
        "regulation": chunk.regulation,
        "regulation_ref": chunk.regulation_ref,
        "article_number": chunk.article_number,
        "paragraph": chunk.paragraph,
        "language": chunk.language,
        "chapter": chunk.chapter,
        "annex": chunk.annex,
        "article_title": chunk.article_title,
        "citation": chunk.citation,
    }


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Write chunks in batches, keyed by their deterministic point id.

    Re-running ingestion over an unchanged source document upserts the same
    ids with the same payload, so it is a no-op in effect rather than a source
    of duplicate points.
    """
    points = [
        PointStruct(id=chunk.point_id, vector=vector, payload=_chunk_payload(chunk))
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=name, points=batch)


def build_filter(
    references: References,
    language: str | None = None,
    regulation: str | None = None,
) -> Filter | None:
    """Turn extracted references into a Qdrant payload filter.

    An empty `Filter()` is not the same as `None` to Qdrant's query planner: it
    still forces a filtered search path instead of the unfiltered fast path, so
    the no-criteria case must return `None` rather than an empty filter object.
    """
    should: list[FieldCondition] = []
    must: list[FieldCondition] = []

    if references.articles:
        should.append(
            FieldCondition(
                key="article_number", match=MatchAny(any=references.articles)
            )
        )
    if references.annexes:
        should.append(
            FieldCondition(key="annex", match=MatchAny(any=references.annexes))
        )
    if references.regulations:
        must.append(
            FieldCondition(
                key="regulation_ref", match=MatchAny(any=references.regulations)
            )
        )
    if regulation is not None:
        must.append(FieldCondition(key="regulation", match=MatchValue(value=regulation)))
    if language is not None:
        must.append(FieldCondition(key="language", match=MatchValue(value=language)))

    if not should and not must:
        return None
    return Filter(must=must or None, should=should or None)


def search(
    client: QdrantClient,
    name: str,
    vector: list[float],
    limit: int,
    score_threshold: float,
    query_filter: Filter | None = None,
) -> list[SearchHit]:
    """Vector search, returning only hits at or above `score_threshold`."""
    response = client.query_points(
        collection_name=name,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
    )
    hits: list[SearchHit] = []
    for point in response.points:
        payload = point.payload or {}
        hits.append(
            SearchHit(
                score=point.score,
                chunk_text=payload["text"],
                parent_text=payload["parent_text"],
                citation=payload["citation"],
                regulation=payload["regulation"],
                article_number=payload["article_number"],
                paragraph=payload["paragraph"],
                language=payload["language"],
                consolidation_date=payload.get("consolidation_date"),
            )
        )
    return hits


def count(client: QdrantClient, name: str) -> int:
    """Number of points currently stored, used to verify an ingestion run."""
    return client.count(collection_name=name).count
