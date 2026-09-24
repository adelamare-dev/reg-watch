"""Retriever: the read facade of the RAG layer.

Runs entirely against `QdrantClient(":memory:")` and a deterministic fake
embedder (dimension 4), so nothing here touches Docker, a running Qdrant
server, or fastembed's model download.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from rag.chunker import Chunk
from rag.config import Settings
from rag.retriever import Retriever
from rag.store import ensure_collection, upsert_chunks

DIMENSION = 4
COLLECTION = "regwatch_test"


class FakeEmbedder:
    """Deterministic stand-in for the Embedder protocol.

    Maps a fixed set of known query strings to hand-picked unit vectors so
    that similarity to the seeded corpus vectors is fully predictable, and
    falls back to a neutral vector for anything else.
    """

    model_id = "fake-embedder"
    dimension = DIMENSION

    _KNOWN: dict[str, list[float]] = {
        "article 28": [1.0, 0.0, 0.0, 0.0],
        "article 999": [1.0, 0.0, 0.0, 0.0],
        "article 30": [0.0, 1.0, 0.0, 0.0],
        "semantic question": [0.0, 0.0, 1.0, 0.0],
        "nothing relevant": [0.0, 0.0, 0.0, 1.0],
    }

    def embed_passages(self, texts):  # noqa: ANN001, ANN201 - test double
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._KNOWN.get(text, [0.5, 0.5, 0.5, 0.5]))


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def settings() -> Settings:
    return Settings(top_k=10, score_threshold=0.5)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_chunk(
    point_id: str,
    *,
    article_number: str = "28",
    regulation: str = "DORA",
    regulation_ref: str = "2022/2554",
    paragraph: str = "1",
    language: str = "fr",
    text: str = "Some paragraph text.",
    parent_text: str | None = None,
) -> Chunk:
    return Chunk(
        point_id=point_id,
        text=text,
        parent_text=parent_text or f"Full article text for {article_number}. {text}",
        regulation=regulation,
        regulation_ref=regulation_ref,
        article_number=article_number,
        paragraph=paragraph,
        language=language,
        article_title="Some title",
    )


def make_retriever(
    client: QdrantClient, embedder: FakeEmbedder, settings: Settings
) -> Retriever:
    return Retriever(
        client=client, embedder=embedder, settings=settings, collection_name=COLLECTION
    )


class TestSemanticSearchWithoutReferences:
    def test_no_exact_filter_is_used(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [make_chunk("11111111-1111-1111-1111-111111111111", article_number="28")],
            [[0.0, 0.0, 1.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("semantic question")

        assert result.used_exact_filter is False
        assert result.references.has_any is False
        assert len(result.hits) == 1


class TestExactArticleFilter:
    def test_filter_applied_and_only_that_article_returned(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
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
            ],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("article 28")

        assert result.used_exact_filter is True
        assert result.references.articles == ["28"]
        assert len(result.hits) == 1
        assert result.hits[0].article_number == "28"


class TestExactFilterFallback:
    def test_falls_back_to_unfiltered_search_when_filter_finds_nothing(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                    text="Text about article 28.",
                )
            ],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        # "article 999" does not exist in the corpus: the exact filter on
        # article_number=999 matches nothing, so the retriever must retry
        # without it rather than returning an empty result.
        result = retriever.search("article 999")

        assert result.references.articles == ["999"]
        assert result.used_exact_filter is False
        assert len(result.hits) == 1
        assert result.hits[0].article_number == "28"


class TestDeduplicationByArticle:
    def test_keeps_only_the_best_scoring_subchunk_per_article(
        self, client: QdrantClient, embedder: FakeEmbedder
    ) -> None:
        # score_threshold is relaxed here because the point is to observe
        # both same-article candidates before dedup collapses them, not to
        # exercise the threshold cutoff (covered separately below).
        settings = Settings(top_k=10, score_threshold=-1.0)
        ensure_collection(client, COLLECTION, DIMENSION)
        # Two sub-chunks of the same article: a weaker match and a near-exact
        # one. Both carry the same parent_text, so surfacing both would just
        # duplicate the article in the prompt window.
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                    paragraph="1",
                    text="Weaker paragraph.",
                ),
                make_chunk(
                    "22222222-2222-2222-2222-222222222222",
                    article_number="28",
                    paragraph="2",
                    text="Stronger paragraph.",
                ),
            ],
            [[0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 1.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("semantic question")

        assert len(result.hits) == 1
        assert result.hits[0].paragraph == "2"

    def test_dedup_preserves_descending_score_order(
        self, client: QdrantClient, embedder: FakeEmbedder
    ) -> None:
        settings = Settings(top_k=10, score_threshold=-1.0)
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                    paragraph="1",
                    text="Article 28 weaker sub-chunk.",
                ),
                make_chunk(
                    "22222222-2222-2222-2222-222222222222",
                    article_number="28",
                    paragraph="2",
                    text="Article 28 stronger sub-chunk.",
                ),
                make_chunk(
                    "33333333-3333-3333-3333-333333333333",
                    article_number="30",
                    text="Article 30.",
                ),
            ],
            [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.8, 0.6],
            ],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("semantic question")

        # embed_query("semantic question") = [0,0,1,0]. Cosine similarity:
        # article 28's stronger sub-chunk (parallel, score 1.0) beats article
        # 30 (score 0.8), which beats article 28's weaker sub-chunk (score
        # ~0.577) — so dedup must keep article 28's *stronger* sub-chunk and
        # still rank it ahead of article 30.
        scores = [hit.score for hit in result.hits]
        assert scores == sorted(scores, reverse=True)
        assert result.hits[0].article_number == "28"
        assert result.hits[1].article_number == "30"


class TestGroundedness:
    def test_is_grounded_false_when_no_hit_clears_the_threshold(
        self, client: QdrantClient, embedder: FakeEmbedder
    ) -> None:
        settings = Settings(top_k=10, score_threshold=0.99)
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [make_chunk("11111111-1111-1111-1111-111111111111", article_number="28")],
            [[0.0, 0.0, 0.0, 1.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        # embed_query("article 30") = [0,1,0,0], orthogonal to the seeded
        # vector [0,0,0,1]: cosine similarity 0.0, well under the 0.99
        # threshold, so nothing should be considered grounded.
        result = retriever.search("article 30")

        assert result.is_grounded is False
        assert result.hits == []

    def test_is_grounded_true_when_a_hit_clears_the_threshold(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [make_chunk("11111111-1111-1111-1111-111111111111", article_number="28")],
            [[0.0, 0.0, 1.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("semantic question")

        assert result.is_grounded is True
        assert len(result.hits) == 1


class TestGetArticle:
    def test_returns_the_full_parent_text(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                    regulation="DORA",
                    parent_text="Full DORA article 28 text.",
                )
            ],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        hit = retriever.get_article("DORA", "28")

        assert hit is not None
        assert hit.parent_text == "Full DORA article 28 text."

    def test_returns_none_for_a_missing_article(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [make_chunk("11111111-1111-1111-1111-111111111111", article_number="28")],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        assert retriever.get_article("DORA", "999") is None

    def test_filters_by_language(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                    language="fr",
                    parent_text="Texte francais de l'article 28.",
                ),
                make_chunk(
                    "22222222-2222-2222-2222-222222222222",
                    article_number="28",
                    language="en",
                    parent_text="English text of article 28.",
                ),
            ],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        retriever = make_retriever(client, embedder, settings)

        hit = retriever.get_article("DORA", "28", language="en")

        assert hit is not None
        assert hit.parent_text == "English text of article 28."


class TestTopKOverride:
    def test_top_k_parameter_overrides_settings(
        self, client: QdrantClient, embedder: FakeEmbedder
    ) -> None:
        settings = Settings(top_k=10, score_threshold=-1.0)
        ensure_collection(client, COLLECTION, DIMENSION)
        upsert_chunks(
            client,
            COLLECTION,
            [
                make_chunk(
                    "11111111-1111-1111-1111-111111111111",
                    article_number="28",
                ),
                make_chunk(
                    "22222222-2222-2222-2222-222222222222",
                    article_number="30",
                ),
                make_chunk(
                    "33333333-3333-3333-3333-333333333333",
                    article_number="31",
                    regulation="DORA",
                    regulation_ref="2022/2554",
                    text="Third article text.",
                ),
            ],
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
            ],
        )
        retriever = make_retriever(client, embedder, settings)

        result = retriever.search("nothing relevant", top_k=1)

        assert len(result.hits) == 1


class TestEmbedderAccessor:
    """`Retriever.embedder` exists so callers (tracing) can report which
    embedding back-end is active without `rag` depending on them in turn.
    """

    def test_embedder_property_exposes_the_configured_embedder(
        self, client: QdrantClient, embedder: FakeEmbedder, settings: Settings
    ) -> None:
        retriever = Retriever(
            client=client, embedder=embedder, settings=settings, collection_name=COLLECTION
        )

        assert retriever.embedder is embedder
