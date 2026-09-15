"""Structural parsing of EUR-Lex HTML into articles.

EUR-Lex serves regulations in the ELI/OJ layout, which is regular enough to be
parsed structurally rather than by text heuristics:

    div.eli-subdivision#cpt_<roman>        chapter, *wraps* its articles
      div.eli-subdivision#art_<n>          one article
        p.oj-ti-art                        "Article 28" (or "Article premier")
        div.eli-title > p.oj-sti-art       article title
        div#<nnn>.<ppp>                    one numbered paragraph
          table                            lettered sub-points a), b), c)

Splitting by fixed size is deliberately avoided: it would break articles apart
and destroy the traceability of citations, which is the whole point here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node

# Maps a regulation key to its official reference, used in citations.
REGULATION_REFS: dict[str, str] = {
    "DORA": "2022/2554",
    "AI_ACT": "2024/1689",
}

_ARTICLE_ID = re.compile(r"^art_(\d+)$")
_CHAPTER_ID = re.compile(r"^cpt_([IVXLC]+)$")
_ANNEX_ID = re.compile(r"^anx_([IVXLC]+)$")
# Paragraph openers: "1." in most texts, "1a." in amended ones.
_PARAGRAPH_NUMBER = re.compile(r"^(\d+[a-z]?)\.\s*")


class ParseError(RuntimeError):
    """Raised when the HTML does not match the expected EUR-Lex layout.

    Parsing failures must be loud: a silent change in the EUR-Lex layout would
    otherwise index an empty or truncated corpus without anyone noticing.
    """


@dataclass(frozen=True)
class Paragraph:
    """A numbered paragraph inside an article."""

    number: str
    text: str


@dataclass(frozen=True)
class Article:
    """One article, the unit of citation and the parent chunk of retrieval."""

    regulation: str
    regulation_ref: str
    article_number: str
    title: str
    paragraphs: tuple[Paragraph, ...]
    language: str
    chapter: str | None = None
    annex: str | None = None

    @property
    def full_text(self) -> str:
        """Article as handed to the LLM: heading, title, then every paragraph."""
        head = f"Article {self.article_number}"
        if self.title:
            head = f"{head} — {self.title}"
        body = "\n\n".join(f"{p.number}. {p.text}" for p in self.paragraphs)
        return f"{head}\n\n{body}".strip()


def _clean(text: str) -> str:
    """Normalise EUR-Lex whitespace.

    The source is dense in non-breaking spaces (inside "Article 28", around
    numbers). Left as-is they leak into citations and break naive matching.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def _ancestor_match(node: Node, pattern: re.Pattern[str]) -> str | None:
    """Walk up the ancestors and return the first id matching `pattern`."""
    current = node.parent
    while current is not None:
        match = pattern.match(current.attributes.get("id") or "")
        if match:
            return match.group(1)
        current = current.parent
    return None


def _paragraph_text(node: Node) -> str:
    """Flatten a paragraph node, keeping nested lettered sub-points inline.

    Sub-points live in nested tables; dropping them would strip the substance
    out of the article, and rendering them as a table would serve the embedder
    nothing useful.
    """
    return _clean(node.text(separator=" "))


def _extract_paragraphs(article_node: Node) -> tuple[Paragraph, ...]:
    """Split an article into its numbered paragraphs.

    Two layouts coexist and both must be handled. Articles subdivided into
    numbered paragraphs wrap each one in its own `div`. Articles that are a
    single block — "Definitions", "AI literacy" — instead drop their content
    straight into `p` and `table` children, with no wrapping `div` at all;
    scanning only `div` children silently loses those articles entirely.
    """
    paragraphs: list[Paragraph] = []
    fallback: list[str] = []

    for child in article_node.iter():
        classes = child.attributes.get("class") or ""
        # Skip the heading and the title block, already captured elsewhere.
        if "eli-title" in classes or "oj-ti-art" in classes:
            continue
        # Footnotes are editorial apparatus, not regulatory text.
        if "oj-note" in classes:
            continue

        text = _paragraph_text(child)
        if not text:
            continue

        if match := _PARAGRAPH_NUMBER.match(text):
            paragraphs.append(
                Paragraph(number=match.group(1), text=text[match.end():].strip())
            )
        else:
            fallback.append(text)

    if not paragraphs and fallback:
        # Single-block article: keep it whole under paragraph number "1".
        return (Paragraph(number="1", text=" ".join(fallback)),)
    return tuple(paragraphs)


def _parse_article(node: Node, regulation: str, language: str) -> Article | None:
    """Build an `Article` from an `div#art_<n>` node, or None if unusable."""
    match = _ARTICLE_ID.match(node.attributes.get("id") or "")
    if not match:
        return None

    # The number comes from the id, never from the heading: the French text
    # spells the first article "Article premier", which carries no digit.
    number = match.group(1)

    title_node = node.css_first("p.oj-sti-art")
    title = _clean(title_node.text()) if title_node else ""

    paragraphs = _extract_paragraphs(node)
    if not paragraphs:
        return None

    return Article(
        regulation=regulation,
        regulation_ref=REGULATION_REFS.get(regulation, ""),
        article_number=number,
        title=title,
        paragraphs=paragraphs,
        language=language,
        chapter=_ancestor_match(node, _CHAPTER_ID),
        annex=_ancestor_match(node, _ANNEX_ID),
    )


def parse_document(
    html: str,
    *,
    regulation: str,
    language: str,
    expected_articles: int | None = None,
) -> list[Article]:
    """Parse an EUR-Lex HTML document into articles, in document order.

    `expected_articles` turns a layout regression into an immediate failure
    rather than a quietly shrunken corpus.
    """
    tree = HTMLParser(html)

    articles: list[Article] = []
    for node in tree.css("div.eli-subdivision"):
        article = _parse_article(node, regulation=regulation, language=language)
        if article is not None:
            articles.append(article)

    if not articles:
        raise ParseError(
            f"{regulation}/{language}: no article found — the EUR-Lex layout "
            "has most likely changed"
        )

    if expected_articles is not None and len(articles) != expected_articles:
        raise ParseError(
            f"{regulation}/{language}: expected {expected_articles} articles, "
            f"parsed {len(articles)}"
        )

    articles.sort(key=lambda a: int(a.article_number.rstrip("abcdefgh") or 0))
    return articles
