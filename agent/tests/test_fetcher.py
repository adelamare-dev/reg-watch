"""Downloading EUR-Lex documents.

EUR-Lex is a slow, occasionally flaky upstream: pages are multi-megabyte,
served gzipped, and picky about the User-Agent. These tests exercise the
failure modes that would otherwise corrupt the corpus silently (bad encoding,
truncated pages) using `httpx.MockTransport`, so the suite never touches the
network by default.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from rag.fetcher import FetchError, cache_path, fetch_all, fetch_document, load_or_fetch
from rag.manifest import DocumentEntry

# A real regulation page is hundreds of thousands of characters; anything
# under the length check is padded to simulate that without shipping a fixture.
LONG_ENOUGH = "x" * 10_000


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


def mock_client(handler) -> httpx.Client:
    """An httpx.Client wired to a MockTransport instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestFetchDocument:
    def test_returns_decoded_html_on_success(self) -> None:
        body = f"<html><body>Article 1 {LONG_ENOUGH}</body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        html = fetch_document(make_entry(), client=mock_client(handler))
        assert html == body

    def test_raises_fetch_error_with_id_on_404(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        with pytest.raises(FetchError, match="dora-reg-fr"):
            fetch_document(make_entry(), client=mock_client(handler))

    def test_raises_fetch_error_with_id_on_500(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        with pytest.raises(FetchError, match="dora-reg-fr"):
            fetch_document(make_entry(), client=mock_client(handler))

    def test_raises_fetch_error_on_suspiciously_short_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>too short</html>")

        with pytest.raises(FetchError, match="dora-reg-fr"):
            fetch_document(make_entry(), client=mock_client(handler))

    def test_decodes_french_accents_correctly(self) -> None:
        # A wrong encoding guess silently mangles accented characters instead
        # of raising — this is the corpus-corruption scenario the fetcher must
        # rule out by forcing utf-8 decoding.
        body = f"<html><body>résilience opérationnelle, santé {LONG_ENOUGH}</body></html>"
        content = body.encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "text/html; charset=utf-8"},
            )

        html = fetch_document(make_entry(), client=mock_client(handler))
        assert "résilience opérationnelle" in html
        assert "santé" in html

    def test_follows_redirects(self) -> None:
        body = f"<html>{LONG_ENOUGH}</html>"
        target = "https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:redirected"

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == target:
                return httpx.Response(200, text=body)
            return httpx.Response(302, headers={"location": target})

        html = fetch_document(make_entry(), client=mock_client(handler))
        assert html == body


class TestCachePath:
    def test_builds_path_from_celex_and_language(self, tmp_path: Path) -> None:
        entry = make_entry(celex="32022R2554", language="fr")
        assert cache_path(entry, tmp_path) == tmp_path / "32022R2554-fr.html"

    def test_varies_by_language(self, tmp_path: Path) -> None:
        fr = cache_path(make_entry(language="fr"), tmp_path)
        en = cache_path(make_entry(language="en"), tmp_path)
        assert fr != en


class TestLoadOrFetch:
    def test_writes_cache_on_first_call(self, tmp_path: Path) -> None:
        body = f"<html>{LONG_ENOUGH}</html>"
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text=body)

        entry = make_entry()
        html, from_cache = load_or_fetch(entry, tmp_path, client=mock_client(handler))

        assert html == body
        assert from_cache is False
        assert calls["count"] == 1
        assert cache_path(entry, tmp_path).read_text(encoding="utf-8") == body

    def test_reads_cache_on_second_call_without_network(self, tmp_path: Path) -> None:
        body = f"<html>{LONG_ENOUGH}</html>"
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text=body)

        entry = make_entry()
        client = mock_client(handler)

        load_or_fetch(entry, tmp_path, client=client)
        html, from_cache = load_or_fetch(entry, tmp_path, client=client)

        assert html == body
        assert from_cache is True
        assert calls["count"] == 1  # second call must not hit the network

    def test_force_redownloads_even_when_cached(self, tmp_path: Path) -> None:
        body = f"<html>{LONG_ENOUGH}</html>"
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text=body)

        entry = make_entry()
        client = mock_client(handler)

        load_or_fetch(entry, tmp_path, client=client)
        html, from_cache = load_or_fetch(entry, tmp_path, client=client, force=True)

        assert from_cache is False
        assert calls["count"] == 2


class TestFetchAll:
    def test_aggregates_multiple_documents(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=f"<html>{request.url}{LONG_ENOUGH}</html>")

        entries = [
            make_entry(id="dora-reg-fr", celex="32022R2554", language="fr"),
            make_entry(
                id="aiact-reg-en",
                celex="32024R1689",
                regulation="AI_ACT",
                language="en",
                url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
                expected_articles=113,
            ),
        ]
        result = fetch_all(entries, tmp_path, client=mock_client(handler))
        assert set(result) == {"dora-reg-fr", "aiact-reg-en"}
        assert "32022R2554" in result["dora-reg-fr"]
        assert "32024R1689" in result["aiact-reg-en"]

    def test_lists_all_failing_ids_when_several_documents_fail(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        entries = [
            make_entry(id="dora-reg-fr", celex="32022R2554", language="fr"),
            make_entry(
                id="aiact-reg-en",
                celex="32024R1689",
                regulation="AI_ACT",
                language="en",
                url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
                expected_articles=113,
            ),
        ]
        with pytest.raises(FetchError) as excinfo:
            fetch_all(entries, tmp_path, client=mock_client(handler))
        assert "dora-reg-fr" in str(excinfo.value)
        assert "aiact-reg-en" in str(excinfo.value)


@pytest.mark.network
class TestRealEurLex:
    def test_downloads_the_full_dora_french_text(self) -> None:
        html = fetch_document(make_entry())
        assert len(html) > 500_000
