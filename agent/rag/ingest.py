"""Ingestion CLI: the module that assembles fetch, parse, chunk, embed and store.

Three modes share one pipeline shape — download, parse, chunk — and diverge
after that:

- `ingest` (default) goes all the way to Qdrant and rewrites the manifest.
- `--dry-run` stops right after chunking, for fast iteration on parsing and
  chunking without paying for embeddings or a Qdrant round-trip.
- `--verify` skips chunking entirely and only re-hashes freshly downloaded
  documents against the manifest, to catch upstream drift.

Every mode is exposed as a plain function taking already-built collaborators
(a manifest, an embedder, a Qdrant client, a fetch callable) so the pipeline
can be tested without argparse, subprocesses, or the network. `main()` is the
only place that touches `sys.argv`, environment-backed settings, or exit
codes.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.chunker import Chunk, chunk_articles
from rag.config import Settings, get_settings
from rag.embedder import Embedder, build_embedder
from rag.fetcher import FetchError, load_or_fetch
from rag.manifest import (
    Manifest,
    ManifestError,
    content_hash,
    load_manifest,
    raw_hash,
    today,
    verify_entry,
)
from rag.parser import Article, ParseError, parse_document
from rag.store import count, ensure_collection, upsert_chunks
from qdrant_client import QdrantClient

# A fetch callable takes a manifest entry and returns its HTML. Production
# code wires this to `load_or_fetch` (ingest/dry-run, cache allowed) or to
# `fetch_document` with `force=True` (verify, cache never trusted); tests
# inject a plain dict-backed stub instead of touching HTTP.
FetchFn = Callable[["object"], str]


@dataclass
class DocumentStats:
    """Per-document numbers, shared by the dry-run and ingest reports."""

    article_count: int
    chunk_count: int
    median_chunk_chars: float


@dataclass
class DryRunReport:
    """Result of `--dry-run`: what would be indexed, without indexing it."""

    documents: dict[str, DocumentStats] = field(default_factory=dict)

    @property
    def total_articles(self) -> int:
        return sum(d.article_count for d in self.documents.values())

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.documents.values())


@dataclass
class IngestReport:
    """Result of a full ingestion run."""

    document_count: int
    article_count: int
    chunk_count: int
    model_id: str
    dimension: int
    collection_name: str


@dataclass
class VerifyReport:
    """Result of `--verify`: one status per document, never just the first failure."""

    statuses: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if all(status == "OK" for status in self.statuses.values()) else 1


def _parse_and_chunk(
    document_id: str, html: str, *, regulation: str, language: str, expected_articles: int | None
) -> tuple[list[Article], list[Chunk]]:
    """Parse then chunk one document, tagging a `ParseError` with its document id.

    `parse_document` only knows the regulation and language, not which
    manifest entry it came from; the id is what a human needs to fix the
    right cache entry or manifest row.
    """
    try:
        articles = parse_document(
            html,
            regulation=regulation,
            language=language,
            expected_articles=expected_articles,
        )
    except ParseError as exc:
        raise ParseError(f"{document_id}: {exc}") from exc
    return articles, chunk_articles(articles)


def run_dry_run(
    manifest: Manifest,
    *,
    corpus_dir: Path,
    fetch: FetchFn,
) -> DryRunReport:
    """Fetch, parse and chunk every document — no embedding, no Qdrant.

    The fast loop for iterating on the parser and the chunker: it exercises
    the exact same document-to-chunk path `run_ingest` uses, minus the two
    expensive steps.
    """
    manifest.validate()
    report = DryRunReport()

    for entry in manifest.documents:
        html = fetch(entry)
        _articles, chunks = _parse_and_chunk(
            entry.id,
            html,
            regulation=entry.regulation,
            language=entry.language,
            expected_articles=entry.expected_articles,
        )
        lengths = [len(c.text) for c in chunks]
        report.documents[entry.id] = DocumentStats(
            article_count=len(_articles),
            chunk_count=len(chunks),
            median_chunk_chars=statistics.median(lengths) if lengths else 0.0,
        )

    return report


def run_ingest(
    manifest: Manifest,
    *,
    corpus_dir: Path,
    fetch: FetchFn,
    embedder: Embedder,
    qdrant_client: QdrantClient,
    collection_name: str,
    batch_size: int = 64,
    recreate: bool = False,
) -> IngestReport:
    """Fetch, parse, chunk, embed and store every document, then update the manifest.

    Chunks are embedded in batches so progress can be reported and so a
    corpus too large to hold every vector in memory at once still works.
    Every document's manifest entry is refreshed with the hashes and counts
    computed along the way, since that is the only point in the pipeline
    where the raw HTML, the parsed articles and the chunks are all in hand at
    once.
    """
    manifest.validate()

    all_chunks: list[Chunk] = []
    total_articles = 0

    for entry in manifest.documents:
        html = fetch(entry)
        articles, chunks = _parse_and_chunk(
            entry.id,
            html,
            regulation=entry.regulation,
            language=entry.language,
            expected_articles=entry.expected_articles,
        )
        total_articles += len(articles)
        all_chunks.extend(chunks)

        entry.sha256 = raw_hash(html)
        entry.content_sha256 = content_hash(html)
        entry.fetched_at = today()
        entry.article_count = len(articles)
        entry.chunk_count = len(chunks)

    ensure_collection(qdrant_client, collection_name, embedder.dimension, recreate=recreate)

    vectors: list[list[float]] = []
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        print(
            f"[{min(start + batch_size, len(all_chunks)):>4}/{len(all_chunks)}] embedding batch…"
        )
        vectors.extend(embedder.embed_passages(c.embedding_text for c in batch))

    upsert_chunks(qdrant_client, collection_name, all_chunks, vectors)

    manifest.generated_at = today()

    return IngestReport(
        document_count=len(manifest.documents),
        article_count=total_articles,
        chunk_count=len(all_chunks),
        model_id=embedder.model_id,
        dimension=embedder.dimension,
        collection_name=collection_name,
    )


def run_verify(manifest: Manifest, *, fetch: FetchFn) -> VerifyReport:
    """Re-download every document and check it against its recorded hashes.

    Every document is attempted regardless of earlier failures, and the
    report lists all of them: a human deciding whether to re-ingest needs the
    full picture, not just the first regulation that drifted.
    """
    manifest.validate()
    report = VerifyReport()

    for entry in manifest.documents:
        html = fetch(entry)
        try:
            verify_entry(entry, html)
        except ManifestError as exc:
            message = str(exc)
            if "never ingested" in message:
                report.statuses[entry.id] = "NEVER INGESTED"
            else:
                report.statuses[entry.id] = "CHANGED"
            report.reasons[entry.id] = message
        else:
            report.statuses[entry.id] = "OK"

    return report


def _print_dry_run(report: DryRunReport) -> None:
    for document_id, stats in report.documents.items():
        print(
            f"{document_id}: {stats.article_count} articles, "
            f"{stats.chunk_count} chunks, "
            f"median chunk size {stats.median_chunk_chars:.0f} chars"
        )
    print(f"total: {report.total_articles} articles, {report.total_chunks} chunks")


def _print_ingest(report: IngestReport) -> None:
    print(
        f"ingested {report.document_count} document(s), "
        f"{report.article_count} article(s), {report.chunk_count} chunk(s)\n"
        f"model: {report.model_id} (dim {report.dimension})\n"
        f"collection: {report.collection_name}"
    )


def _print_verify(report: VerifyReport) -> None:
    for document_id, status in report.statuses.items():
        print(f"{document_id}: {status}")
    if report.exit_code != 0:
        print("verification failed — see reasons above", file=sys.stderr)
        for document_id, reason in report.reasons.items():
            print(f"  {document_id}: {reason}", file=sys.stderr)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest", description="Ingest the DORA / EU AI Act corpus into Qdrant."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-download every document and check it against the manifest, "
        "without touching Qdrant or the manifest file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch, parse and chunk only — no embedding, no Qdrant",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore the local cache and re-download"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="recreate the Qdrant collection from scratch"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="embedding batch size (default: 64)"
    )
    parser.add_argument("--manifest", type=Path, default=None, help="path to the manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected mode, and map the outcome to an exit code.

    All the actual work happens in `run_ingest` / `run_verify` / `run_dry_run`;
    this function only wires configuration together and turns exceptions that
    are expected operational failures into a clear stderr message instead of a
    raw traceback.
    """
    args = _build_argparser().parse_args(argv)
    settings: Settings = get_settings()
    manifest_path = args.manifest or settings.manifest_path

    try:
        manifest = load_manifest(manifest_path)

        if args.verify:
            from rag.fetcher import fetch_document

            def verify_fetch(entry: object) -> str:
                return fetch_document(entry)  # type: ignore[arg-type]

            report = run_verify(manifest, fetch=verify_fetch)
            _print_verify(report)
            return report.exit_code

        def cached_fetch(entry: object) -> str:
            html, _from_cache = load_or_fetch(
                entry, settings.corpus_dir, force=args.force  # type: ignore[arg-type]
            )
            return html

        if args.dry_run:
            dry_run_report = run_dry_run(
                manifest, corpus_dir=settings.corpus_dir, fetch=cached_fetch
            )
            _print_dry_run(dry_run_report)
            return 0

        embedder = build_embedder(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            api_key=settings.mistral_api_key,
        )
        qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

        ingest_report = run_ingest(
            manifest,
            corpus_dir=settings.corpus_dir,
            fetch=cached_fetch,
            embedder=embedder,
            qdrant_client=qdrant_client,
            collection_name=settings.collection_name,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )
        manifest.save(manifest_path)
        _print_ingest(ingest_report)
        return 0

    except (FetchError, ParseError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
