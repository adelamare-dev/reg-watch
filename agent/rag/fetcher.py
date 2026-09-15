"""Downloading EUR-Lex documents and caching them locally.

EUR-Lex pages are multi-megabyte, served gzip-compressed, and the server
rejects httpx's default User-Agent as non-browser traffic. A local cache is
kept alongside the corpus so that iterating on chunking or parsing does not
re-download several megabytes of HTML on every run.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from rag.manifest import DocumentEntry

# EUR-Lex serves a generic error page to clients it does not recognise as a
# browser; a realistic User-Agent avoids that without doing anything more
# invasive than what any browser sends.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# A consolidated regulation is hundreds of thousands of characters long. A
# response shorter than this is not a regulatory text — it is an error page,
# a redirect stub, or a truncated download — so it is rejected rather than
# silently ingested.
_MIN_DOCUMENT_LENGTH = 10_000


class FetchError(RuntimeError):
    """Raised when a document cannot be retrieved, or looks wrong once retrieved."""


def _default_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": _USER_AGENT})


def fetch_document(
    entry: DocumentEntry,
    *,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> str:
    """Download one document's HTML from its EUR-Lex URL.

    `client` is accepted so tests can inject an `httpx.MockTransport`-backed
    client instead of touching the network; when omitted, a real client with
    a browser User-Agent and redirect-following is used.
    """
    owns_client = client is None
    http_client = client or _default_client()
    try:
        try:
            response = http_client.get(entry.url, timeout=timeout, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise FetchError(f"{entry.id}: timed out fetching {entry.url}") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"{entry.id}: failed to fetch {entry.url}: {exc}") from exc

        if response.status_code != 200:
            raise FetchError(
                f"{entry.id}: HTTP {response.status_code} fetching {entry.url}"
            )

        # The declared encoding is not trusted: EUR-Lex sometimes mislabels
        # it, and a wrong guess corrupts every accented character in French
        # and German legal text without raising an error. Decoding as utf-8
        # with replacement is the only way to fail loudly instead of quietly.
        html = response.content.decode("utf-8", errors="replace")

        if len(html) < _MIN_DOCUMENT_LENGTH:
            raise FetchError(
                f"{entry.id}: response from {entry.url} is only {len(html)} "
                f"characters — too short to be a regulatory text"
            )

        return html
    finally:
        if owns_client:
            http_client.close()


def cache_path(entry: DocumentEntry, corpus_dir: Path) -> Path:
    """Local path where a document's HTML is cached."""
    return corpus_dir / f"{entry.celex}-{entry.language}.html"


def load_or_fetch(
    entry: DocumentEntry,
    corpus_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> tuple[str, bool]:
    """Read a document from the local cache, downloading it if needed.

    Returns `(html, from_cache)`. Re-downloading a multi-megabyte page on
    every iteration of chunker development would make that loop unworkable,
    so the cache is used unless `force` asks to bypass it.
    """
    path = cache_path(entry, corpus_dir)
    if not force and path.exists():
        return path.read_text(encoding="utf-8"), True

    html = fetch_document(entry, client=client)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html, False


def fetch_all(
    entries: list[DocumentEntry],
    corpus_dir: Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, str]:
    """Fetch or load every entry, mapping document id to HTML.

    One failing document must not hide the others: every entry is attempted,
    and failures are collected into a single aggregated `FetchError` naming
    every document that failed, rather than stopping at the first one.
    """
    results: dict[str, str] = {}
    failures: list[str] = []
    for entry in entries:
        try:
            html, _ = load_or_fetch(entry, corpus_dir, force=force, client=client)
            results[entry.id] = html
        except FetchError as exc:
            failures.append(str(exc))

    if failures:
        raise FetchError(
            f"{len(failures)} document(s) failed to fetch:\n" + "\n".join(failures)
        )

    return results
