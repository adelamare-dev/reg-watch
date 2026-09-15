"""Parent-document chunking of articles."""

from __future__ import annotations

import pytest

from rag.chunker import CHILD_MAX_CHARS, CHILD_MIN_CHARS, chunk_article
from rag.parser import Article, Paragraph


def make_article(
    paragraphs: list[tuple[str, str]],
    *,
    number: str = "28",
    title: str = "Principes généraux",
    chapter: str | None = "V",
) -> Article:
    return Article(
        regulation="DORA",
        regulation_ref="2022/2554",
        article_number=number,
        title=title,
        paragraphs=tuple(Paragraph(number=n, text=t) for n, t in paragraphs),
        language="fr",
        chapter=chapter,
    )


class TestChildGranularity:
    def test_one_child_per_paragraph(self) -> None:
        article = make_article(
            [("1", "Premier paragraphe. " * 20), ("2", "Second paragraphe. " * 20)]
        )
        children = chunk_article(article)
        assert len(children) == 2
        assert [c.paragraph for c in children] == ["1", "2"]

    def test_merges_a_paragraph_too_short_to_carry_meaning(self) -> None:
        # An isolated "3. Idem." embeds to noise; merging keeps it retrievable.
        article = make_article(
            [("1", "Court."), ("2", "Un paragraphe de longueur normale. " * 10)]
        )
        children = chunk_article(article)
        assert len(children) == 1
        assert children[0].text.startswith("Court.")
        assert "longueur normale" in children[0].text

    def test_a_merged_child_keeps_the_first_paragraph_number(self) -> None:
        article = make_article(
            [("1", "Court."), ("2", "Un paragraphe de longueur normale. " * 10)]
        )
        assert chunk_article(article)[0].paragraph == "1"

    def test_a_trailing_short_paragraph_merges_backwards(self) -> None:
        article = make_article(
            [("1", "Un paragraphe de longueur normale. " * 10), ("2", "Idem.")]
        )
        children = chunk_article(article)
        assert len(children) == 1
        assert children[0].text.endswith("Idem.")

    def test_splits_a_paragraph_longer_than_the_model_window(self) -> None:
        # The "Definitions" article is a single 17k-character paragraph; left
        # whole it would be silently truncated at embedding time.
        long_text = " ".join(f"Définition numéro {i} du présent règlement." for i in range(400))
        children = chunk_article(make_article([("1", long_text)]))
        assert len(children) > 1
        assert all(len(c.text) <= CHILD_MAX_CHARS for c in children)

    def test_splits_on_a_sentence_boundary(self) -> None:
        long_text = " ".join(f"Phrase numéro {i} complète." for i in range(400))
        children = chunk_article(make_article([("1", long_text)]))
        assert all(c.text.rstrip().endswith(".") for c in children[:-1])

    def test_split_children_share_the_paragraph_number(self) -> None:
        long_text = " ".join(f"Phrase numéro {i} complète." for i in range(400))
        children = chunk_article(make_article([("1", long_text)]))
        assert {c.paragraph for c in children} == {"1"}


class TestParentPayload:
    def test_every_child_carries_the_whole_parent_article(self) -> None:
        # Parent-document retrieval: children are indexed for precision, but the
        # complete article is what reaches the LLM.
        article = make_article(
            [("1", "Premier paragraphe. " * 20), ("2", "Second paragraphe. " * 20)]
        )
        for child in chunk_article(article):
            assert child.parent_text == article.full_text

    def test_children_carry_the_article_metadata(self) -> None:
        article = make_article([("1", "Un paragraphe de longueur normale. " * 10)])
        child = chunk_article(article)[0]
        assert child.regulation == "DORA"
        assert child.regulation_ref == "2022/2554"
        assert child.article_number == "28"
        assert child.chapter == "V"
        assert child.language == "fr"

    def test_child_text_is_prefixed_with_its_citation(self) -> None:
        # Without this, a bare paragraph embeds with no clue of which article it
        # belongs to, and cross-article queries retrieve near-duplicates.
        article = make_article([("1", "Un paragraphe de longueur normale. " * 10)])
        child = chunk_article(article)[0]
        assert child.embedding_text.startswith("DORA article 28")
        assert "Principes généraux" in child.embedding_text


class TestIdentity:
    def test_the_identifier_is_stable_across_runs(self) -> None:
        article = make_article([("1", "Un paragraphe de longueur normale. " * 10)])
        assert chunk_article(article)[0].point_id == chunk_article(article)[0].point_id

    def test_identifiers_differ_across_languages(self) -> None:
        fr = make_article([("1", "Un paragraphe de longueur normale. " * 10)])
        en = Article(**{**fr.__dict__, "language": "en"})
        assert chunk_article(fr)[0].point_id != chunk_article(en)[0].point_id

    def test_identifiers_differ_across_articles(self) -> None:
        a = make_article([("1", "Un paragraphe de longueur normale. " * 10)], number="28")
        b = make_article([("1", "Un paragraphe de longueur normale. " * 10)], number="29")
        assert chunk_article(a)[0].point_id != chunk_article(b)[0].point_id


class TestEdgeCases:
    def test_an_article_with_only_short_paragraphs_still_yields_one_child(self) -> None:
        children = chunk_article(make_article([("1", "Court."), ("2", "Bref.")]))
        assert len(children) == 1
        assert "Court." in children[0].text and "Bref." in children[0].text

    def test_thresholds_are_ordered_consistently(self) -> None:
        assert CHILD_MIN_CHARS < CHILD_MAX_CHARS
