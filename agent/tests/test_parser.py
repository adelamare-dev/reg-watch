"""Structural parsing of EUR-Lex HTML.

Fixtures are verbatim excerpts of the real EUR-Lex output, not hand-written
markup: the point of these tests is to catch the day EUR-Lex changes its
structure, which hand-made HTML could never do.
"""

from __future__ import annotations

import pytest

from rag.parser import ParseError, parse_document


class TestArticleExtraction:
    def test_extracts_every_article_of_the_excerpt(self, dora_fr_html: str) -> None:
        articles = parse_document(dora_fr_html, regulation="DORA", language="fr")
        assert [a.article_number for a in articles] == ["1", "17", "28"]

    def test_reads_the_number_from_the_id_not_the_title(self, dora_fr_html: str) -> None:
        # The French text spells the first article "Article premier"; only the
        # element id carries a usable number.
        first = parse_document(dora_fr_html, regulation="DORA", language="fr")[0]
        assert first.article_number == "1"

    def test_captures_the_article_title(self, dora_fr_html: str) -> None:
        articles = parse_document(dora_fr_html, regulation="DORA", language="fr")
        by_number = {a.article_number: a for a in articles}
        assert by_number["28"].title == "Principes généraux"

    def test_parses_english_documents_identically(self, aiact_en_html: str) -> None:
        articles = parse_document(aiact_en_html, regulation="AI_ACT", language="en")
        assert [a.article_number for a in articles] == ["9", "73"]
        assert articles[0].title == "Risk management system"


class TestParagraphs:
    def test_splits_an_article_into_numbered_paragraphs(self, dora_fr_html: str) -> None:
        article = next(
            a
            for a in parse_document(dora_fr_html, regulation="DORA", language="fr")
            if a.article_number == "28"
        )
        assert [p.number for p in article.paragraphs] == [str(i) for i in range(1, 11)]

    def test_keeps_paragraph_text_free_of_its_own_number(self, dora_fr_html: str) -> None:
        article = next(
            a
            for a in parse_document(dora_fr_html, regulation="DORA", language="fr")
            if a.article_number == "28"
        )
        first = article.paragraphs[0]
        assert not first.text.startswith("1.")
        assert "prestataires tiers de services TIC" in first.text

    def test_inlines_lettered_enumerations_into_the_paragraph(
        self, dora_fr_html: str
    ) -> None:
        # Sub-points a), b), c) live in nested tables; dropping them would cut
        # the substance out of the article.
        article = next(
            a
            for a in parse_document(dora_fr_html, regulation="DORA", language="fr")
            if a.article_number == "28"
        )
        assert "a)" in article.paragraphs[0].text
        assert "pleinement responsables" in article.paragraphs[0].text

    def test_normalises_non_breaking_spaces(self, dora_fr_html: str) -> None:
        article = parse_document(dora_fr_html, regulation="DORA", language="fr")[0]
        assert "\xa0" not in article.full_text


class TestMetadata:
    def test_resolves_the_chapter_from_the_enclosing_element(
        self, dora_fr_html: str
    ) -> None:
        # Chapters are not siblings of articles: they wrap them, so the chapter
        # can only be recovered by walking up the ancestors.
        articles = parse_document(dora_fr_html, regulation="DORA", language="fr")
        by_number = {a.article_number: a for a in articles}
        assert by_number["1"].chapter == "I"
        assert by_number["17"].chapter == "III"
        assert by_number["28"].chapter == "V"

    def test_carries_regulation_and_language(self, dora_fr_html: str) -> None:
        article = parse_document(dora_fr_html, regulation="DORA", language="fr")[0]
        assert article.regulation == "DORA"
        assert article.regulation_ref == "2022/2554"
        assert article.language == "fr"

    def test_full_text_joins_title_and_paragraphs(self, dora_fr_html: str) -> None:
        article = next(
            a
            for a in parse_document(dora_fr_html, regulation="DORA", language="fr")
            if a.article_number == "28"
        )
        assert article.full_text.startswith("Article 28")
        assert "Principes généraux" in article.full_text


class TestSingleBlockArticles:
    """Articles with no numbered subdivision, e.g. "Definitions"."""

    def test_keeps_an_article_that_has_no_numbered_paragraph(self) -> None:
        # These articles put their content directly in `p`/`table` children
        # instead of per-paragraph `div`s. Scanning only `div`s drops them
        # entirely — silently losing ~20% of the corpus.
        articles = parse_document(
            single_block_html(), regulation="AI_ACT", language="en"
        )
        assert len(articles) == 1
        assert articles[0].article_number == "4"
        assert articles[0].title == "AI literacy"
        assert "sufficient level of AI literacy" in articles[0].full_text

    def test_folds_the_whole_block_under_paragraph_one(self) -> None:
        article = parse_document(
            single_block_html(), regulation="AI_ACT", language="en"
        )[0]
        assert [p.number for p in article.paragraphs] == ["1"]


def single_block_html() -> str:
    """An article carrying a single unnumbered block of text."""
    return """<html><body>
    <div class="eli-subdivision" id="cpt_I">
      <div class="eli-subdivision" id="art_4">
        <p class="oj-ti-art">Article&nbsp;4</p>
        <div class="eli-title" id="art_4.tit_1">
          <p class="oj-sti-art">AI literacy</p>
        </div>
        <p class="oj-normal">Providers and deployers of AI systems shall take
        measures to ensure a&nbsp;sufficient level of AI literacy of their
        staff.</p>
      </div>
    </div></body></html>"""


class TestParseFailures:
    def test_raises_when_the_document_yields_no_article(self) -> None:
        # Guards against a silent EUR-Lex layout change: an empty result must be
        # loud, never an empty corpus quietly indexed.
        with pytest.raises(ParseError, match="no article"):
            parse_document("<html><body><p>rien</p></body></html>",
                           regulation="DORA", language="fr")

    def test_raises_when_the_article_count_misses_the_expectation(
        self, dora_fr_html: str
    ) -> None:
        with pytest.raises(ParseError, match="expected 64"):
            parse_document(
                dora_fr_html, regulation="DORA", language="fr", expected_articles=64
            )

    def test_accepts_a_matching_expectation(self, dora_fr_html: str) -> None:
        articles = parse_document(
            dora_fr_html, regulation="DORA", language="fr", expected_articles=3
        )
        assert len(articles) == 3
