"""Embedding back-ends, and the asymmetric query/passage prefixes.

The prefix handling is the point of most of these tests. Some retrieval models
are trained with `query: ` / `passage: ` markers and degrade *silently* without
them — no error, just worse results. fastembed does not apply them on its own:
its `query_embed` and `passage_embed` both delegate to `embed` untouched, so the
responsibility sits here.
"""

from __future__ import annotations

import pytest

from rag.embedder import (
    MODELS_REQUIRING_PREFIXES,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    apply_passage_prefix,
    apply_query_prefix,
    needs_prefixes,
)

E5 = "intfloat/multilingual-e5-large"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TestPrefixPolicy:
    def test_e5_requires_prefixes(self) -> None:
        assert needs_prefixes(E5)

    def test_minilm_does_not_require_prefixes(self) -> None:
        # A symmetric-similarity model; prefixes would be embedded as content.
        assert not needs_prefixes(MINILM)

    def test_mistral_does_not_require_prefixes(self) -> None:
        assert not needs_prefixes("mistral-embed")

    def test_an_unknown_model_defaults_to_no_prefix(self) -> None:
        assert not needs_prefixes("some/unreleased-model")

    def test_the_registry_is_not_empty(self) -> None:
        assert E5 in MODELS_REQUIRING_PREFIXES


class TestPrefixApplication:
    def test_prefixes_a_query_for_a_model_that_needs_it(self) -> None:
        assert apply_query_prefix("Article 28", E5) == f"{QUERY_PREFIX}Article 28"

    def test_prefixes_a_passage_for_a_model_that_needs_it(self) -> None:
        assert apply_passage_prefix("Le texte", E5) == f"{PASSAGE_PREFIX}Le texte"

    def test_leaves_text_untouched_for_a_model_that_does_not(self) -> None:
        assert apply_query_prefix("Article 28", MINILM) == "Article 28"
        assert apply_passage_prefix("Le texte", MINILM) == "Le texte"

    def test_query_and_passage_prefixes_differ(self) -> None:
        # Collapsing the two would defeat the asymmetry the model was trained on.
        assert QUERY_PREFIX != PASSAGE_PREFIX

    def test_does_not_prefix_twice(self) -> None:
        once = apply_query_prefix("Article 28", E5)
        assert apply_query_prefix(once, E5) == once


class TestFastEmbedEmbedder:
    """Exercised against the real registry, without downloading any weights."""

    def test_reports_the_dimension_declared_by_the_registry(self) -> None:
        from rag.embedder import FastEmbedEmbedder

        assert FastEmbedEmbedder(MINILM).dimension == 384
        assert FastEmbedEmbedder(E5).dimension == 1024

    def test_rejects_an_unknown_model_at_construction(self) -> None:
        from rag.embedder import FastEmbedEmbedder

        # Failing here beats failing after a multi-gigabyte download.
        with pytest.raises(ValueError, match="unknown"):
            FastEmbedEmbedder("not/a-real-model")

    def test_rejects_a_monolingual_model(self) -> None:
        from rag.embedder import FastEmbedEmbedder

        # The corpus is indexed FR and EN in one vector space; a monolingual
        # model would quietly wreck cross-language retrieval.
        with pytest.raises(ValueError, match="multilingual"):
            FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    def test_exposes_its_model_identifier(self) -> None:
        from rag.embedder import FastEmbedEmbedder

        assert FastEmbedEmbedder(MINILM).model_id == MINILM


class TestFactory:
    def test_builds_a_fastembed_backend(self) -> None:
        from rag.embedder import FastEmbedEmbedder, build_embedder

        embedder = build_embedder(provider="fastembed", model=MINILM)
        assert isinstance(embedder, FastEmbedEmbedder)

    def test_rejects_an_unknown_provider(self) -> None:
        from rag.embedder import build_embedder

        with pytest.raises(ValueError, match="provider"):
            build_embedder(provider="openai", model=MINILM)

    def test_mistral_backend_requires_an_api_key(self) -> None:
        from rag.embedder import build_embedder

        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            build_embedder(provider="mistral", model="mistral-embed", api_key=None)
