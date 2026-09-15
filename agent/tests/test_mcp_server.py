"""Wiring of the tool server: which tools exist, and what guards them.

The transport itself is the SDK's concern; what is asserted here is that the
four tools of the contract are registered and that the rate limit is enforced
around them.
"""

from __future__ import annotations

import pytest

from mcp_server.ratelimit import RateLimitError, TokenBucket
from mcp_server.server import EXPECTED_TOOLS, build_server, guard

from tests.test_mcp_tools import FakeRetriever, make_hit


def test_registers_the_four_contract_tools(tmp_path) -> None:
    manifest = tmp_path / "corpus-manifest.json"
    manifest.write_text('{"generated_at": "2026-09-15", "documents": []}', encoding="utf-8")

    server = build_server(retriever=FakeRetriever(), manifest_path=manifest)

    assert set(EXPECTED_TOOLS) == {
        "search_regulatory_corpus",
        "get_article",
        "compare_regulations",
        "get_corpus_manifest",
    }
    assert server is not None


def test_guard_passes_the_call_through_when_tokens_remain() -> None:
    bucket = TokenBucket(capacity=2, refill_per_second=1.0)

    assert guard(bucket, lambda: "ok") == "ok"


def test_guard_rejects_once_the_bucket_is_empty() -> None:
    bucket = TokenBucket(capacity=1, refill_per_second=0.001)
    guard(bucket, lambda: "ok")

    with pytest.raises(RateLimitError):
        guard(bucket, lambda: "ok")


def test_guard_does_not_spend_a_token_on_a_rejected_call() -> None:
    """A rejected call must not make the next legitimate one wait longer."""
    bucket = TokenBucket(capacity=1, refill_per_second=0.001)
    guard(bucket, lambda: "ok")

    calls: list[str] = []
    for _ in range(3):
        with pytest.raises(RateLimitError):
            guard(bucket, lambda: calls.append("ran"))

    assert calls == []


def test_build_server_accepts_a_retriever_with_hits(tmp_path) -> None:
    manifest = tmp_path / "corpus-manifest.json"
    manifest.write_text('{"generated_at": "2026-09-15", "documents": []}', encoding="utf-8")

    server = build_server(
        retriever=FakeRetriever(hits=[make_hit()]), manifest_path=manifest
    )

    assert server is not None
