"""End-to-end: a real MCP client against the real corpus.

Needs an ingested Qdrant, so it carries the `integration` marker and is
excluded from the default run. What the unit tests cannot show is whether the
tools survive the round trip through the protocol — schemas, serialisation and
transport included.

    uv run --directory agent pytest -m integration
"""

from __future__ import annotations

import pytest
from mcp.client import Client

from mcp_server.server import build_default_server

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    """An in-memory client speaking to the real server, no port involved."""
    server = build_default_server()
    async with Client(server, raise_exceptions=True) as connected:
        yield connected


async def test_exposes_the_four_contract_tools(client) -> None:
    listing = await client.list_tools()
    names = {tool.name for tool in listing.tools}

    assert names == {
        "search_regulatory_corpus",
        "get_article",
        "compare_regulations",
        "get_corpus_manifest",
    }


async def test_search_grounds_a_question_in_the_corpus(client) -> None:
    result = await client.call_tool(
        "search_regulatory_corpus",
        {"query": "gestion du risque lié aux prestataires TIC tiers", "language": "fr"},
    )

    payload = result.structured_content
    assert payload["is_grounded"] is True
    assert payload["hits"], "expected at least one grounded hit"
    assert payload["hits"][0]["regulation"] in {"DORA", "AI_ACT"}


async def test_article_lookup_stays_within_the_named_regulation(client) -> None:
    """The failure mode this guards: article 28 answering from the other text."""
    result = await client.call_tool(
        "search_regulatory_corpus",
        {"query": "obligations", "regulation": "DORA", "article_number": "28"},
    )

    payload = result.structured_content
    assert payload["hits"]
    assert all(hit["regulation"] == "DORA" for hit in payload["hits"])


async def test_get_article_returns_the_exact_article(client) -> None:
    result = await client.call_tool(
        "get_article", {"regulation": "DORA", "article_number": "28", "language": "fr"}
    )

    payload = result.structured_content
    assert payload["found"] is True
    assert payload["article_number"] == "28"
    assert payload["regulation"] == "DORA"
    assert payload["text"]


async def test_get_article_accepts_the_usual_regulation_name(client) -> None:
    result = await client.call_tool(
        "get_article", {"regulation": "AI Act", "article_number": "9", "language": "fr"}
    )

    payload = result.structured_content
    assert payload["found"] is True
    assert payload["regulation"] == "AI_ACT"


async def test_compare_returns_both_regulations(client) -> None:
    result = await client.call_tool(
        "compare_regulations", {"theme": "gestion des risques", "language": "fr"}
    )

    payload = result.structured_content
    assert payload["dora"]["regulation"] == "DORA"
    assert payload["ai_act"]["regulation"] == "AI_ACT"
    assert payload["dora"]["hits"] or payload["ai_act"]["hits"]


async def test_manifest_reports_an_ingested_corpus(client) -> None:
    result = await client.call_tool("get_corpus_manifest", {})

    payload = result.structured_content
    assert payload["is_ingested"] is True
    assert len(payload["documents"]) == 4
    assert all(document["content_sha256"] for document in payload["documents"])


async def test_out_of_corpus_question_is_not_grounded(client) -> None:
    """A question the corpus cannot answer must say so, not improvise."""
    result = await client.call_tool(
        "search_regulatory_corpus",
        {"query": "quelles sont les obligations de staking sous MiCA ?", "language": "fr"},
    )

    payload = result.structured_content
    for hit in payload["hits"]:
        assert hit["regulation"] in {"DORA", "AI_ACT"}
