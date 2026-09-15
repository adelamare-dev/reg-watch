"""Parent-document chunking: fine children for precision, whole article for the LLM.

One article is one parent. Children are its numbered paragraphs, which is the
natural structural subdivision of EUR-Lex texts and happens to be exactly the
granularity a citation needs. Fixed-size splitting is avoided on purpose: it cuts
across articles and destroys the traceability of citations.

Each child carries the full parent text in its payload, so retrieval hands the
complete article to the LLM without a second round-trip.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from rag.parser import Article

# Below this, a paragraph carries too little signal to embed on its own
# ("3. Idem.") and is merged with its neighbour.
CHILD_MIN_CHARS = 100

# The multilingual models in use truncate at 512 tokens. Measured on this corpus,
# French legal text runs ~4.6 chars/token on average but down to ~3.3 in the
# densest passages — i.e. as little as ~1670 chars for 512 tokens. Staying at
# 1400 keeps every child inside the window even in the worst case, so nothing is
# ever silently truncated at embedding time.
CHILD_MAX_CHARS = 1400

# Namespace for deterministic point identifiers. Fixed once: changing it would
# orphan every previously ingested point.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A child chunk: what gets embedded, plus the parent it belongs to."""

    point_id: str
    text: str
    parent_text: str
    regulation: str
    regulation_ref: str
    article_number: str
    paragraph: str
    language: str
    chapter: str | None = None
    annex: str | None = None
    article_title: str = ""

    @property
    def citation(self) -> str:
        """Human-readable provenance, reused in the UI and in prompts."""
        ref = f"{self.regulation} article {self.article_number}"
        if self.paragraph:
            ref = f"{ref}, paragraphe {self.paragraph}"
        return ref

    @property
    def embedding_text(self) -> str:
        """Text actually embedded.

        A bare paragraph gives the model no clue which article it belongs to,
        and paragraphs numbered "1." across 64 articles embed to near-duplicates.
        Prefixing with the citation and the article title disambiguates them.
        """
        head = f"{self.regulation} article {self.article_number}"
        if self.article_title:
            head = f"{head} — {self.article_title}"
        return f"{head}\n{self.text}"


def _point_id(
    regulation: str, article: str, paragraph: str, index: int, language: str
) -> str:
    """Derive a stable identifier from the chunk's coordinates.

    Deterministic so that re-ingesting is idempotent: the same paragraph always
    lands on the same point instead of piling up duplicates.
    """
    key = f"{regulation}|{article}|{paragraph}|{index}|{language}"
    return str(uuid.uuid5(_NAMESPACE, key))


def _split_long(text: str) -> list[str]:
    """Split an oversized paragraph on sentence boundaries.

    Long enumerations ("Definitions" runs past 15 000 characters as a single
    paragraph) must be cut, but never mid-sentence: a truncated legal clause
    reads as a different obligation.
    """
    sentences = _SENTENCE_END.split(text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= CHILD_MAX_CHARS:
            current = candidate
            continue
        if current:
            parts.append(current)
        # A single sentence longer than the window still has to be cut somewhere.
        while len(sentence) > CHILD_MAX_CHARS:
            parts.append(sentence[:CHILD_MAX_CHARS])
            sentence = sentence[CHILD_MAX_CHARS:]
        current = sentence

    if current:
        parts.append(current)
    return parts or [text]


def _merge_short(
    paragraphs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Merge paragraphs too short to embed meaningfully.

    A short paragraph joins the one that follows it; a short *trailing*
    paragraph joins the one before, since it has no successor to merge into.
    """
    merged: list[tuple[str, str]] = []
    pending: tuple[str, str] | None = None

    for number, text in paragraphs:
        if pending is not None:
            number, text = pending[0], f"{pending[1]} {text}"
            pending = None
        if len(text) < CHILD_MIN_CHARS:
            pending = (number, text)
            continue
        merged.append((number, text))

    if pending is not None:
        if merged:
            last_number, last_text = merged[-1]
            merged[-1] = (last_number, f"{last_text} {pending[1]}")
        else:
            merged.append(pending)

    return merged


def chunk_article(article: Article) -> list[Chunk]:
    """Split one article into its child chunks."""
    parent_text = article.full_text
    paragraphs = _merge_short([(p.number, p.text) for p in article.paragraphs])

    chunks: list[Chunk] = []
    for number, text in paragraphs:
        for index, part in enumerate(_split_long(text)):
            chunks.append(
                Chunk(
                    point_id=_point_id(
                        article.regulation,
                        article.article_number,
                        number,
                        index,
                        article.language,
                    ),
                    text=part,
                    parent_text=parent_text,
                    regulation=article.regulation,
                    regulation_ref=article.regulation_ref,
                    article_number=article.article_number,
                    paragraph=number,
                    language=article.language,
                    chapter=article.chapter,
                    annex=article.annex,
                    article_title=article.title,
                )
            )
    return chunks


def chunk_articles(articles: list[Article]) -> list[Chunk]:
    """Chunk a whole document."""
    return [chunk for article in articles for chunk in chunk_article(article)]
