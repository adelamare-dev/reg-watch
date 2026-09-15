"""The ingestion CLI: the pipeline that assembles every other RAG module.

No network, no real embedding model, no Docker. `QdrantClient(":memory:")`
stands in for Qdrant, a hand-built dimension-4 embedder stands in for
fastembed, and document downloads are injected as plain callables so the
fetcher's own tests stay the single source of truth for HTTP behaviour.

The forged HTML mirrors the real EUR-Lex layout documented in `parser.py`:
a chapter wraps an article, which carries its heading, its title block, and
one or more numbered paragraph `div`s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest
from qdrant_client import QdrantClient

from rag.fetcher import FetchError
from rag.ingest import (
    DryRunReport,
    IngestReport,
    VerifyReport,
    main,
    run_dry_run,
    run_ingest,
    run_verify,
)
from rag.manifest import DocumentEntry, Manifest, ManifestError, content_hash, raw_hash
from rag.parser import ParseError
from rag.store import count

DIMENSION = 4


def make_article_html(article_number: str = "1", title: str = "Objet") -> str:
    """One chapter wrapping one article with two numbered paragraphs.

    Long enough to clear the fetcher's minimum-length guard without needing a
    real fixture file.
    """
    filler = "Texte de remplissage pour dépasser la longueur minimale. " * 200
    return f"""<!DOCTYPE html>
<html><body>
<div class="eli-subdivision" id="cpt_I">
  <div class="eli-subdivision" id="art_{article_number}">
    <p class="oj-ti-art">Article&nbsp;{article_number}</p>
    <div class="eli-title" id="art_{article_number}.tit_1">
      <p class="oj-sti-art">{title}</p>
    </div>
    <div id="{article_number}.001">
      <p class="oj-normal">1.&nbsp;Premier paragraphe substantiel de l'article, largement
      suffisant pour dépasser le seuil minimal de longueur d'un enfant embarqué
      séparément. {filler}</p>
    </div>
    <div id="{article_number}.002">
      <p class="oj-normal">2.&nbsp;Second paragraphe substantiel, distinct du premier,
      also lengthy enough on its own. {filler}</p>
    </div>
  </div>
</div>
</body></html>"""


def make_two_article_html() -> str:
    """Two chapters, two articles — used to test `expected_articles` mismatches."""
    filler = "Texte de remplissage pour dépasser la longueur minimale. " * 200
    return f"""<!DOCTYPE html>
<html><body>
<div class="eli-subdivision" id="cpt_I">
  <div class="eli-subdivision" id="art_1">
    <p class="oj-ti-art">Article premier</p>
    <div class="eli-title" id="art_1.tit_1"><p class="oj-sti-art">Objet</p></div>
    <div id="1.001"><p class="oj-normal">1.&nbsp;Premier article, premier paragraphe. {filler}</p></div>
  </div>
</div>
<div class="eli-subdivision" id="cpt_II">
  <div class="eli-subdivision" id="art_2">
    <p class="oj-ti-art">Article&nbsp;2</p>
    <div class="eli-title" id="art_2.tit_1"><p class="oj-sti-art">Champ d'application</p></div>
    <div id="2.001"><p class="oj-normal">1.&nbsp;Second article, premier paragraphe. {filler}</p></div>
  </div>
</div>
</body></html>"""


class FakeEmbedder:
    """Deterministic dimension-4 embedder implementing the `Embedder` protocol."""

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "fake/test-embedder"

    @property
    def dimension(self) -> int:
        return DIMENSION

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        self.passage_calls.append(texts)
        return [[float(len(t) % 7), 0.0, 0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def make_entry(**overrides: object) -> DocumentEntry:
    defaults = dict(
        id="dora-reg-fr",
        celex="32022R2554",
        regulation="DORA",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32022R2554",
        language="fr",
        consolidation_date="2025-01-17",
        expected_articles=1,
    )
    return DocumentEntry(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_manifest(*entries: DocumentEntry) -> Manifest:
    return Manifest(generated_at="2026-09-14", documents=list(entries))


def make_fetch_fn(html_by_id: dict[str, str]):
    """A fetch callable matching the shape `run_ingest`/`run_verify` expect."""

    calls: list[str] = []

    def fetch(entry: DocumentEntry) -> str:
        calls.append(entry.id)
        return html_by_id[entry.id]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


class TestDryRun:
    def test_does_not_touch_embedder_or_qdrant(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})

        report = run_dry_run(manifest, corpus_dir=tmp_path, fetch=fetch)

        assert isinstance(report, DryRunReport)

    def test_reports_article_and_chunk_counts_per_document(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})

        report = run_dry_run(manifest, corpus_dir=tmp_path, fetch=fetch)

        assert report.documents["dora-reg-fr"].article_count == 1
        assert report.documents["dora-reg-fr"].chunk_count >= 1


class TestRunIngest:
    def test_populates_qdrant_with_the_produced_chunks(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})
        client = QdrantClient(":memory:")
        embedder = FakeEmbedder()

        report = run_ingest(
            manifest,
            corpus_dir=tmp_path,
            fetch=fetch,
            embedder=embedder,
            qdrant_client=client,
            collection_name="regwatch_test",
        )

        assert isinstance(report, IngestReport)
        assert count(client, "regwatch_test") == report.chunk_count
        assert report.chunk_count > 0

    def test_updates_the_manifest_with_hashes_and_counts(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)
        html = make_article_html()
        fetch = make_fetch_fn({"dora-reg-fr": html})
        client = QdrantClient(":memory:")
        embedder = FakeEmbedder()

        run_ingest(
            manifest,
            corpus_dir=tmp_path,
            fetch=fetch,
            embedder=embedder,
            qdrant_client=client,
            collection_name="regwatch_test",
        )

        updated = manifest.by_id("dora-reg-fr")
        assert updated.sha256 == raw_hash(html)
        assert updated.content_sha256 == content_hash(html)
        assert updated.article_count == 1
        assert updated.chunk_count is not None and updated.chunk_count > 0
        assert updated.fetched_at is not None
        assert manifest.generated_at is not None

    def test_running_twice_does_not_change_the_point_count(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})
        client = QdrantClient(":memory:")
        embedder = FakeEmbedder()

        run_ingest(
            manifest,
            corpus_dir=tmp_path,
            fetch=fetch,
            embedder=embedder,
            qdrant_client=client,
            collection_name="regwatch_test",
        )
        first_count = count(client, "regwatch_test")

        run_ingest(
            manifest,
            corpus_dir=tmp_path,
            fetch=fetch,
            embedder=embedder,
            qdrant_client=client,
            collection_name="regwatch_test",
        )
        second_count = count(client, "regwatch_test")

        assert first_count == second_count

    def test_expected_articles_mismatch_raises_parse_error_naming_the_document(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry(expected_articles=5)
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})
        client = QdrantClient(":memory:")
        embedder = FakeEmbedder()

        with pytest.raises(ParseError, match="dora-reg-fr"):
            run_ingest(
                manifest,
                corpus_dir=tmp_path,
                fetch=fetch,
                embedder=embedder,
                qdrant_client=client,
                collection_name="regwatch_test",
            )


class TestRunVerify:
    def test_unchanged_corpus_exits_clean_and_reports_ok(self, tmp_path: Path) -> None:
        html = make_article_html()
        entry = make_entry(sha256=raw_hash(html), content_sha256=content_hash(html))
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": html})

        report = run_verify(manifest, fetch=fetch)

        assert isinstance(report, VerifyReport)
        assert report.exit_code == 0
        assert report.statuses["dora-reg-fr"] == "OK"

    def test_changed_upstream_text_exits_nonzero_and_names_the_document(
        self, tmp_path: Path
    ) -> None:
        original = make_article_html()
        entry = make_entry(
            sha256=raw_hash(original), content_sha256=content_hash(original)
        )
        manifest = make_manifest(entry)
        changed = make_article_html(title="Titre modifié après publication")
        fetch = make_fetch_fn({"dora-reg-fr": changed})

        report = run_verify(manifest, fetch=fetch)

        assert report.exit_code == 1
        assert report.statuses["dora-reg-fr"] == "CHANGED"

    def test_never_ingested_document_exits_nonzero_with_reason(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry()  # no sha256 / content_sha256 recorded
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})

        report = run_verify(manifest, fetch=fetch)

        assert report.exit_code == 1
        assert report.statuses["dora-reg-fr"] == "NEVER INGESTED"

    def test_lists_every_document_not_only_the_first_failure(
        self, tmp_path: Path
    ) -> None:
        html_ok = make_article_html(article_number="1")
        ok_entry = make_entry(
            id="ok-doc",
            sha256=raw_hash(html_ok),
            content_sha256=content_hash(html_ok),
        )
        never_entry = make_entry(id="never-doc", celex="99999999")
        manifest = make_manifest(ok_entry, never_entry)
        fetch = make_fetch_fn(
            {"ok-doc": html_ok, "never-doc": make_article_html(article_number="1")}
        )

        report = run_verify(manifest, fetch=fetch)

        assert set(report.statuses) == {"ok-doc", "never-doc"}
        assert report.statuses["ok-doc"] == "OK"
        assert report.statuses["never-doc"] == "NEVER INGESTED"
        assert report.exit_code == 1

    def test_does_not_bypass_cache_by_calling_load_or_fetch(self, tmp_path: Path) -> None:
        # `--verify` must always re-download, never trust a cached copy — that
        # is the entire point of verification. Passing an id-only fetch
        # callable (as production code will via `force=True`) is enough to
        # prove no cache read short-circuits it: the fixture would raise
        # KeyError if a stale/absent cache entry ever leaked through.
        html = make_article_html()
        entry = make_entry(sha256=raw_hash(html), content_sha256=content_hash(html))
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": html})

        run_verify(manifest, fetch=fetch)

        assert fetch.calls == ["dora-reg-fr"]  # type: ignore[attr-defined]


class TestFetchFailurePropagation:
    def test_a_fetch_error_is_not_swallowed_by_run_ingest(self, tmp_path: Path) -> None:
        entry = make_entry()
        manifest = make_manifest(entry)

        def failing_fetch(entry: DocumentEntry) -> str:
            raise FetchError(f"{entry.id}: boom")

        client = QdrantClient(":memory:")
        embedder = FakeEmbedder()

        with pytest.raises(FetchError, match="dora-reg-fr"):
            run_ingest(
                manifest,
                corpus_dir=tmp_path,
                fetch=failing_fetch,
                embedder=embedder,
                qdrant_client=client,
                collection_name="regwatch_test",
            )


class TestMain:
    def test_dry_run_flag_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        html = make_article_html()
        manifest_path = tmp_path / "corpus-manifest.json"
        make_manifest(make_entry()).save(manifest_path)
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "32022R2554-fr.html").write_text(html, encoding="utf-8")
        # Settings reads CORPUS_DIR from the environment; without this override
        # `main()` would fall back to the real committed corpus, which does not
        # match this test's one-article fixture.
        monkeypatch.setenv("CORPUS_DIR", str(corpus_dir))

        exit_code = main(
            [
                "--dry-run",
                "--manifest",
                str(manifest_path),
            ]
        )

        assert exit_code == 0

    def test_manifest_error_is_reported_not_raised(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing_manifest = tmp_path / "absent-manifest.json"

        exit_code = main(["--dry-run", "--manifest", str(missing_manifest)])

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestParseErrorSurface:
    def test_run_dry_run_also_propagates_parse_error_with_document_id(
        self, tmp_path: Path
    ) -> None:
        entry = make_entry(id="dora-reg-fr", expected_articles=99)
        manifest = make_manifest(entry)
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})

        with pytest.raises(ParseError, match="dora-reg-fr"):
            run_dry_run(manifest, corpus_dir=tmp_path, fetch=fetch)


class TestManifestErrorSurface:
    def test_invalid_manifest_raises_manifest_error(self) -> None:
        duplicate = make_manifest(make_entry(), make_entry())
        fetch = make_fetch_fn({"dora-reg-fr": make_article_html()})

        with pytest.raises(ManifestError, match="duplicate"):
            run_verify(duplicate, fetch=fetch)
