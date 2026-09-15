"""Corpus manifest: which texts, which versions, and proof they have not changed.

The manifest is committed; the corpus itself is not. It pins, for every
document, the CELEX URL, the consolidation date and two hashes, so that an
answer can state which version of a regulation it was grounded in, and so that
`ingest --verify` fails loudly when EUR-Lex publishes a different text.

Two hashes rather than one:

- `sha256` covers the raw HTML, so any byte-level change is visible.
- `content_sha256` covers the extracted, normalised text. EUR-Lex embeds
  timestamps and session identifiers in the page, so the raw hash drifts on
  every download; only the content hash tells an actual amendment apart from
  cosmetic markup churn.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from selectolax.parser import HTMLParser


class ManifestError(RuntimeError):
    """Raised when the manifest is malformed, or when a document has drifted."""


@dataclass
class DocumentEntry:
    """One source document, in one language."""

    id: str
    celex: str
    regulation: str
    url: str
    language: str
    consolidation_date: str
    expected_articles: int
    sha256: str | None = None
    content_sha256: str | None = None
    fetched_at: str | None = None
    article_count: int | None = None
    chunk_count: int | None = None


@dataclass
class Manifest:
    """The committed description of the corpus."""

    generated_at: str
    documents: list[DocumentEntry] = field(default_factory=list)

    def validate(self) -> None:
        """Check internal coherence before anything relies on it."""
        seen: set[str] = set()
        for document in self.documents:
            if document.id in seen:
                raise ManifestError(f"duplicate document id {document.id!r}")
            seen.add(document.id)

    def save(self, path: Path) -> None:
        """Write the manifest as reviewable JSON.

        It is committed and read in diffs, so it stays indented and key-ordered
        rather than compact.
        """
        self.validate()
        payload = {
            "generated_at": self.generated_at,
            "documents": [asdict(document) for document in self.documents],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def by_id(self, document_id: str) -> DocumentEntry:
        for document in self.documents:
            if document.id == document_id:
                return document
        raise ManifestError(f"unknown document id {document_id!r}")


def load_manifest(path: Path) -> Manifest:
    """Read a manifest from disk."""
    if not path.exists():
        raise ManifestError(f"manifest not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"malformed manifest at {path}: {exc}") from exc

    return Manifest(
        generated_at=payload.get("generated_at", ""),
        documents=[DocumentEntry(**item) for item in payload.get("documents", [])],
    )


def raw_hash(html: str) -> str:
    """SHA-256 of the document exactly as served."""
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def content_hash(html: str) -> str:
    """SHA-256 of the visible text, normalised.

    Markup, comments, whitespace and non-breaking spaces are stripped, so the
    digest tracks the regulatory text and nothing else.
    """
    text = HTMLParser(html).text(separator=" ")
    text = text.replace("\xa0", " ")
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def verify_entry(entry: DocumentEntry, html: str) -> None:
    """Check a freshly downloaded document against its recorded hashes."""
    if entry.content_sha256 is None:
        raise ManifestError(
            f"{entry.id}: never ingested — run `pnpm ingest` before verifying"
        )

    current = content_hash(html)
    if current != entry.content_sha256:
        raise ManifestError(
            f"{entry.id}: content changed upstream — expected "
            f"{entry.content_sha256[:12]}…, got {current[:12]}…. The EUR-Lex text "
            "no longer matches the ingested corpus."
        )


def today() -> str:
    return date.today().isoformat()
