"""Corpus manifest: versioning, hashing and drift detection.

The manifest is what lets an answer state which version of a regulation it was
grounded in, and what makes an ingested corpus reproducible. Its tests are
therefore about detecting change, not about happy paths.
"""

from __future__ import annotations

import json

import pytest

from rag.manifest import (
    DocumentEntry,
    Manifest,
    ManifestError,
    content_hash,
    load_manifest,
    raw_hash,
    verify_entry,
)


def make_entry(**overrides: object) -> DocumentEntry:
    defaults = dict(
        id="dora-reg-fr",
        celex="32022R2554",
        regulation="DORA",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32022R2554",
        language="fr",
        consolidation_date="2025-01-17",
        expected_articles=64,
    )
    return DocumentEntry(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestHashing:
    def test_raw_hash_is_stable(self) -> None:
        assert raw_hash("abc") == raw_hash("abc")

    def test_raw_hash_changes_with_content(self) -> None:
        assert raw_hash("abc") != raw_hash("abd")

    def test_raw_hash_is_hex_sha256(self) -> None:
        digest = raw_hash("abc")
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_content_hash_ignores_volatile_markup(self) -> None:
        # EUR-Lex embeds timestamps and session ids in the page. Hashing the raw
        # HTML alone would report drift on every download, for text that has not
        # changed by a single word.
        first = "<html><body><p>Article 1</p><!-- generated 2026-09-14 --></body></html>"
        second = "<html><body><p>Article 1</p><!-- generated 2026-09-15 --></body></html>"
        assert raw_hash(first) != raw_hash(second)
        assert content_hash(first) == content_hash(second)

    def test_content_hash_detects_a_real_text_change(self) -> None:
        first = "<html><body><p>Article 1</p></body></html>"
        second = "<html><body><p>Article 2</p></body></html>"
        assert content_hash(first) != content_hash(second)

    def test_content_hash_ignores_whitespace_reflow(self) -> None:
        assert content_hash("<p>Article  1</p>") == content_hash("<p>Article\n1</p>")


class TestVerification:
    def test_accepts_an_unchanged_document(self) -> None:
        html = "<html><body><p>Article 1</p></body></html>"
        entry = make_entry(sha256=raw_hash(html), content_sha256=content_hash(html))
        verify_entry(entry, html)  # must not raise

    def test_accepts_cosmetic_markup_drift(self) -> None:
        original = "<html><body><p>Article 1</p><!-- 2026-09-14 --></body></html>"
        entry = make_entry(
            sha256=raw_hash(original), content_sha256=content_hash(original)
        )
        verify_entry(entry, "<html><body><p>Article 1</p><!-- 2026-09-15 --></body></html>")

    def test_rejects_a_changed_regulatory_text(self) -> None:
        original = "<html><body><p>Article 1</p></body></html>"
        entry = make_entry(
            sha256=raw_hash(original), content_sha256=content_hash(original)
        )
        with pytest.raises(ManifestError, match="content changed"):
            verify_entry(entry, "<html><body><p>Article 2</p></body></html>")

    def test_rejects_an_entry_that_was_never_ingested(self) -> None:
        with pytest.raises(ManifestError, match="never ingested"):
            verify_entry(make_entry(), "<html></html>")


class TestManifestIO:
    def test_round_trips_through_json(self, tmp_path) -> None:
        manifest = Manifest(generated_at="2026-09-14", documents=[make_entry()])
        path = tmp_path / "corpus-manifest.json"
        manifest.save(path)
        assert load_manifest(path).documents[0].id == "dora-reg-fr"

    def test_writes_human_readable_json(self, tmp_path) -> None:
        # The manifest is committed and reviewed in diffs; it must stay readable.
        path = tmp_path / "corpus-manifest.json"
        Manifest(generated_at="2026-09-14", documents=[make_entry()]).save(path)
        raw = path.read_text(encoding="utf-8")
        assert raw.count("\n") > 5
        assert json.loads(raw)["documents"][0]["celex"] == "32022R2554"

    def test_rejects_duplicate_document_ids(self) -> None:
        with pytest.raises(ManifestError, match="duplicate"):
            Manifest(
                generated_at="2026-09-14",
                documents=[make_entry(), make_entry()],
            ).validate()

    def test_reports_a_missing_manifest_clearly(self, tmp_path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "absent.json")


class TestShippedManifest:
    """The manifest committed in the repository must stay coherent."""

    def test_is_loadable_and_valid(self) -> None:
        from rag.config import REPO_ROOT

        manifest = load_manifest(REPO_ROOT / "corpus-manifest.json")
        manifest.validate()
        assert len(manifest.documents) == 4

    def test_covers_both_regulations_in_both_languages(self) -> None:
        from rag.config import REPO_ROOT

        manifest = load_manifest(REPO_ROOT / "corpus-manifest.json")
        pairs = {(d.regulation, d.language) for d in manifest.documents}
        assert pairs == {
            ("DORA", "fr"),
            ("DORA", "en"),
            ("AI_ACT", "fr"),
            ("AI_ACT", "en"),
        }

    def test_declares_the_verified_article_counts(self) -> None:
        from rag.config import REPO_ROOT

        manifest = load_manifest(REPO_ROOT / "corpus-manifest.json")
        by_regulation = {d.regulation: d.expected_articles for d in manifest.documents}
        assert by_regulation["DORA"] == 64
        assert by_regulation["AI_ACT"] == 113
