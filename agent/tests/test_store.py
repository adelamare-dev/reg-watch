"""Qdrant client wrapper: collection lifecycle, upsert, filtering, search.

Runs entirely against `QdrantClient(":memory:")`, the embedded in-process mode
qdrant-client ships for exactly this purpose, so none of this needs Docker or a
running server. Vectors are tiny hand-built unit vectors (dimension 4) rather
than real embeddings, since what is under test is Qdrant plumbing, not
embedding quality.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag.chunker import Chunk
from rag.references import References
from rag.store import build_filter, count, ensure_collection, search, upsert_chunks

DIMENSION = 4
COLLECTION = "regwatch_test"


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def make_chunk(
    point_id: str,
    *,
    article_number: str = "28",
    regulation: str = "DORA",
    regulation_ref: str = "2022/2554",
    paragraph: str = "1",
    language: str = "fr",
    annex: str | None = None,
    text: str = "Some paragraph text.",
) -> Chunk:
    return Chunk(
        point_id=point_id,
        text=text,
        parent_text=f"Full article text for {article_number}. {text}",
        regulation=regulation,
        regulation_ref=regulation_ref,
        article_number=article_number,
        paragraph=paragraph,
        language=language,
        annex=annex,
        article_title="Some title",
    )


class TestEnsureCollection:
    def test_creates_the_collection_when_absent(self, client: QdrantClient) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)

        assert client.collection_exists(COLLECTION)

    def test_is_idempotent_when_dimension_matches(self, client: QdrantClient) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        ensure_collection(client, COLLECTION, DIMENSION)

        assert client.collection_exists(COLLECTION)

    def test_raises_on_dimension_mismatch(self, client: QdrantClient) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)

        with pytest.raises(ValueError, match="dimension"):
            ensure_collection(client, COLLECTION, DIMENSION + 1)

    def test_creates_payload_indexes_for_filterable_fields(
        self, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The embedded ":memory:" mode ignores payload indexes outright (it
        # only warns), so exercise the real call surface instead: every
        # filterable field must trigger one create_payload_index call.
        indexed_fields: list[str] = []
        original = client.create_payload_index

        def spy(*, collection_name: str, field_name: str, **kwargs: object) -> object:
            indexed_fields.append(field_name)
            return original(collection_name=collection_name, field_name=field_name, **kwargs)

        monkeypatch.setattr(client, "create_payload_index", spy)

        ensure_collection(client, COLLECTION, DIMENSION)

        assert {"regulation", "article_number", "language", "annex"} <= set(
            indexed_fields
        )


class TestUpsertAndCount:
    def test_upsert_then_count(self, client: QdrantClient) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        chunks = [make_chunk("11111111-1111-1111-1111-111111111111")]
        vectors = [[1.0, 0.0, 0.0, 0.0]]

        upsert_chunks(client, COLLECTION, chunks, vectors)

        assert count(client, COLLECTION) == 1

    def test_reupserting_the_same_chunks_does_not_duplicate(
        self, client: QdrantClient
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        chunks = [make_chunk("11111111-1111-1111-1111-111111111111")]
        vectors = [[1.0, 0.0, 0.0, 0.0]]

        upsert_chunks(client, COLLECTION, chunks, vectors)
        upsert_chunks(client, COLLECTION, chunks, vectors)

        assert count(client, COLLECTION) == 1


class TestBuildFilter:
    def test_returns_none_without_any_criteria(self) -> None:
        assert build_filter(References()) is None

    def test_filters_on_a_single_article(self) -> None:
        query_filter = build_filter(References(articles=["28"]))

        assert query_filter is not None
        assert query_filter.should is not None
        assert len(query_filter.should) == 1

    def test_filters_on_article_and_language(self) -> None:
        query_filter = build_filter(References(articles=["28"]), language="fr")

        assert query_filter is not None
        assert query_filter.must is not None
        assert len(query_filter.must) == 1
        assert query_filter.should is not None

    def test_filters_on_an_annex(self) -> None:
        query_filter = build_filter(References(annexes=["III"]))

        assert query_filter is not None
        assert query_filter.should is not None

    def test_filters_on_regulation_ref(self) -> None:
        query_filter = build_filter(References(regulations=["2022/2554"]))

        assert query_filter is not None
        assert query_filter.must is not None
        assert len(query_filter.must) == 1

    def test_regulation_argument_adds_a_must_clause(self) -> None:
        query_filter = build_filter(References(), regulation="DORA")

        assert query_filter is not None
        assert query_filter.must is not None
        assert len(query_filter.must) == 1


class TestSearch:
    def _seed(self, client: QdrantClient) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        chunks = [
            make_chunk(
                "11111111-1111-1111-1111-111111111111",
                article_number="28",
                text="Text about article 28.",
            ),
            make_chunk(
                "22222222-2222-2222-2222-222222222222",
                article_number="30",
                text="Text about article 30.",
            ),
        ]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        upsert_chunks(client, COLLECTION, chunks, vectors)

    def test_search_with_article_filter_returns_only_that_article(
        self, client: QdrantClient
    ) -> None:
        self._seed(client)
        query_filter = build_filter(References(articles=["28"]))

        hits = search(
            client,
            COLLECTION,
            [1.0, 0.0, 0.0, 0.0],
            limit=10,
            score_threshold=0.0,
            query_filter=query_filter,
        )

        assert len(hits) == 1
        assert hits[0].article_number == "28"

    def test_search_respects_the_score_threshold(self, client: QdrantClient) -> None:
        self._seed(client)

        # The second vector is orthogonal to the query: cosine similarity 0.0,
        # well below a threshold that should keep only the exact match.
        hits = search(
            client,
            COLLECTION,
            [1.0, 0.0, 0.0, 0.0],
            limit=10,
            score_threshold=0.9,
        )

        assert len(hits) == 1
        assert hits[0].article_number == "28"

    def test_hits_carry_the_full_parent_text(self, client: QdrantClient) -> None:
        self._seed(client)

        hits = search(
            client, COLLECTION, [1.0, 0.0, 0.0, 0.0], limit=1, score_threshold=0.0
        )

        assert hits[0].parent_text.startswith("Full article text for 28.")

    def test_hits_carry_score_and_citation(self, client: QdrantClient) -> None:
        self._seed(client)

        hits = search(
            client, COLLECTION, [1.0, 0.0, 0.0, 0.0], limit=1, score_threshold=0.0
        )

        assert hits[0].score == pytest.approx(1.0)
        assert "28" in hits[0].citation
