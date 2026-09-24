"""Langfuse tracing as a decorator around the graph, never a dependency.

Every helper here degrades to a no-op when no client is configured, so the
graph runs identically — same code path, same tests, no network — whether
or not Langfuse credentials are present. Callers never branch on whether
tracing is active; they always get a context manager or a list back.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from rag.config import Settings, get_settings


def init_observability(settings: Settings | None = None) -> Langfuse | None:
    """Build the Langfuse client, or skip it entirely.

    No client is constructed at all without a key pair: an unauthenticated
    client would silently disable itself but still exist, which would make
    `client is None` an unreliable signal downstream.
    """
    settings = settings or get_settings()

    if not settings.tracing_enabled:
        return None

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )


def build_callbacks(client: Langfuse | None) -> list[Any]:
    """LangChain callbacks for `graph.invoke(state, config={"callbacks": [...]})`."""
    if client is None:
        return []
    return [CallbackHandler()]


class _InertSpan:
    """Stands in for a Langfuse span when no client is configured.

    `update` is a no-op so callers can write one code path for both cases.
    """

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002 - inert by design
        return None


@contextmanager
def _span(client: Langfuse | None, **observation: Any):
    """Open an observation, marking it on failure without swallowing anything.

    A traced block must behave exactly like an untraced one, so the exception
    is annotated and re-raised: tracing observes control flow, never alters it.
    """
    if client is None:
        yield _InertSpan()
        return

    with client.start_as_current_observation(**observation) as span:
        try:
            yield span
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise


def retriever_span(client: Langfuse | None, *, name: str, query: str, top_k: int):
    """Trace a retrieval call. The caller reports hit count and filter use afterwards."""
    return _span(
        client,
        name=name,
        as_type="retriever",
        input={"query": query, "top_k": top_k},
    )


def embedding_span(client: Langfuse | None, *, model_id: str, dimension: int, text: str):
    """Trace an embedding call.

    Cost is declared only when the caller supplies usage via `update`: local
    `fastembed` runs have no cost, and declaring one would misrepresent it.
    """
    return _span(
        client,
        name="embedding",
        as_type="embedding",
        input=text,
        model=model_id,
        metadata={"dimension": dimension},
    )


def generation_span(client: Langfuse | None, *, name: str, model: str, messages: list):
    """Trace an LLM generation. Usage/cost are filled in after the response arrives."""
    return _span(
        client,
        name=name,
        as_type="generation",
        input=messages,
        model=model,
    )


def extract_usage(response: Any) -> dict[str, int] | None:
    """Read token counts off a LangChain response, in whichever shape it used.

    Tries `usage_metadata` first (the current LangChain convention), then
    falls back to `response_metadata["token_usage"]` (older/provider-specific
    shape). Never raises: a response that doesn't match either shape, or
    whose fields aren't dicts, just yields `None`.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        try:
            return {
                "input": usage["input_tokens"],
                "output": usage["output_tokens"],
                "total": usage["total_tokens"],
            }
        except (KeyError, TypeError):
            pass

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            try:
                return {
                    "input": token_usage["prompt_tokens"],
                    "output": token_usage["completion_tokens"],
                    "total": token_usage["total_tokens"],
                }
            except (KeyError, TypeError):
                pass

    return None


def flush(client: Langfuse | None) -> None:
    """Force-send buffered traces. No-op without a client."""
    if client is not None:
        client.flush()
