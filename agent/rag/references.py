"""Exact regulatory reference extraction from a query.

Dense embeddings notoriously fail to tell "Article 28" from "Article 82". When a
numeric reference is detected here, the retriever adds an *exact* Qdrant payload
filter on top of the vector search — deterministic, where sparse/BM25 would stay
probabilistic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A bare r"(\d{4})/(\d{3,4})" would also match "2024/2025", "2025/0117" or a page
# range. A false positive here applies a wrong exact filter and discards the right
# chunks, so we require a regulatory marker before the number pair.
_REGULATION_CONTEXT = r"(?:r[eè]glement|regulation|directive)\s*(?:\(\s*(?:UE|EU)\s*\)\s*)?(?:(?:UE|EU)\s*)?(?:n[°o]\s*)?"

PATTERNS: dict[str, re.Pattern[str]] = {
    "article": re.compile(r"\barticles?\s+(\d{1,3})\b", re.IGNORECASE),
    "regulation": re.compile(
        _REGULATION_CONTEXT + r"(\d{4}/\d{3,4})\b", re.IGNORECASE
    ),
    "annex": re.compile(r"\bannexe?\s+([IVX]+)\b", re.IGNORECASE),
}

# "les articles 28 et 30": captures numbers trailing a first article match.
# Applied with `.match(text, pos)`, so already anchored at the previous match end.
_ARTICLE_TAIL = re.compile(r"\s*(?:,|et|and|&)\s*(\d{1,3})\b", re.IGNORECASE)


@dataclass(frozen=True)
class References:
    """Exact references detected in a query."""

    articles: list[str] = field(default_factory=list)
    regulations: list[str] = field(default_factory=list)
    annexes: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        """True when at least one exact reference was detected."""
        return bool(self.articles or self.regulations or self.annexes)


def _dedupe(values: list[str]) -> list[str]:
    """Deduplicate while preserving first-appearance order."""
    return list(dict.fromkeys(values))


def _extract_articles(text: str) -> list[str]:
    """Extract article numbers, enumerations included.

    "les articles 28 et 30" yields ["28", "30"]: the second number is not
    preceded by the word "article" and would escape the main pattern.
    """
    found: list[str] = []
    for match in PATTERNS["article"].finditer(text):
        found.append(match.group(1))
        pos = match.end()
        while tail := _ARTICLE_TAIL.match(text, pos):
            found.append(tail.group(1))
            pos = tail.end()
    return found


def extract_references(text: str) -> References:
    """Extract exact regulatory references from free text.

    >>> refs = extract_references("l'article 28 du règlement (UE) 2022/2554")
    >>> refs.articles, refs.regulations
    (['28'], ['2022/2554'])
    """
    return References(
        articles=_dedupe(_extract_articles(text)),
        regulations=_dedupe(PATTERNS["regulation"].findall(text)),
        annexes=_dedupe([m.upper() for m in PATTERNS["annex"].findall(text)]),
    )
