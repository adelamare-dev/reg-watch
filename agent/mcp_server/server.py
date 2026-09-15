"""Assembles the tool server: real dependencies in, MCP server out.

This is the only module that knows about Qdrant, the embedder and the
filesystem. `tools.py` stays free of them, which is what lets the tool
surface be tested without any of them running.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from mcp.server import MCPServer

from mcp_server import tools
from mcp_server.ratelimit import TokenBucket
from mcp_server.tools import RetrieverLike

EXPECTED_TOOLS = (
    "search_regulatory_corpus",
    "get_article",
    "compare_regulations",
    "get_corpus_manifest",
)

# Retrieval embeds the query and hits Qdrant on every call, so the ceiling is
# there to bound what a runaway agent loop can cost. Generous enough that no
# legitimate multi-tool reasoning turn ever reaches it.
DEFAULT_BURST = 30
DEFAULT_REFILL_PER_SECOND = 5.0

T = TypeVar("T")


def guard(bucket: TokenBucket, call: Callable[[], T]) -> T:
    """Spend a token, then run the call.

    Acquiring first means a rejected call never reaches Qdrant; it also means
    a rejection costs no token, so being throttled does not push the next
    legitimate call further away.
    """
    bucket.acquire()
    return call()


def build_server(
    *,
    retriever: RetrieverLike,
    manifest_path: Path,
    bucket: TokenBucket | None = None,
) -> MCPServer:
    """Register the four regulatory tools against a live retriever."""
    limiter = bucket or TokenBucket(
        capacity=DEFAULT_BURST, refill_per_second=DEFAULT_REFILL_PER_SECOND
    )
    mcp = MCPServer("regwatch")

    @mcp.tool()
    def search_regulatory_corpus(
        query: str,
        regulation: str | None = None,
        language: str | None = None,
        article_number: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Search the DORA and EU AI Act corpus for passages answering a question.

        Returns the matching articles with their citation and score. An empty
        `hits` list with `is_grounded` false means the corpus has nothing
        relevant: answer that no regulatory basis was found rather than
        answering from general knowledge.
        """
        return guard(
            limiter,
            lambda: tools.search_regulatory_corpus(
                retriever,
                query=query,
                regulation=regulation,
                language=language,
                article_number=article_number,
                top_k=top_k,
            ),
        )

    @mcp.tool()
    def get_article(
        regulation: str, article_number: str, language: str | None = None
    ) -> dict[str, Any]:
        """Fetch the full text of one article by its exact reference.

        Deterministic: no similarity search involved. Use it when the article
        is already known, rather than searching for it.
        """
        return guard(
            limiter,
            lambda: tools.get_article(
                retriever,
                regulation=regulation,
                article_number=article_number,
                language=language,
            ),
        )

    @mcp.tool()
    def compare_regulations(
        theme: str, language: str | None = None, top_k: int | None = None
    ) -> dict[str, Any]:
        """Put DORA and the EU AI Act side by side on a given theme.

        Returns both sets of provisions without ranking or reconciling them.
        Where the two texts diverge, present the divergence; do not arbitrate.
        """
        return guard(
            limiter,
            lambda: tools.compare_regulations(
                retriever, theme=theme, language=language, top_k=top_k
            ),
        )

    @mcp.tool()
    def get_corpus_manifest() -> dict[str, Any]:
        """Report the corpus version answers are grounded in.

        Gives the consolidation date and content hash of every ingested
        document, so an answer can state which version of the text it relied on.
        """
        return guard(limiter, lambda: tools.get_corpus_manifest(manifest_path))

    return mcp


def build_default_server() -> MCPServer:
    """Wire the server against the real Qdrant collection and embedder."""
    from qdrant_client import QdrantClient

    from rag.config import get_settings
    from rag.embedder import build_embedder
    from rag.retriever import Retriever

    settings = get_settings()
    embedder = build_embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    retriever = Retriever(client=client, embedder=embedder, settings=settings)
    return build_server(retriever=retriever, manifest_path=settings.manifest_path)
