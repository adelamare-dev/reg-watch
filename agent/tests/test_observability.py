"""Observability must be a decorator, never a dependency: no keys, no network,
same code path either way.
"""

from __future__ import annotations

import pytest

from graph.observability import (
    build_callbacks,
    embedding_span,
    extract_usage,
    flush,
    generation_span,
    init_observability,
    retriever_span,
)
from rag.config import Settings


class FakeSpan:
    """Records every `update` call instead of talking to Langfuse."""

    def __init__(self, name: str, as_type: str, **kwargs) -> None:
        self.name = name
        self.as_type = as_type
        self.init_kwargs = kwargs
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuseClient:
    """A fake client recording `start_as_current_observation` calls."""

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.flushed = False

    def start_as_current_observation(self, *, name: str, as_type: str, **kwargs):
        span = FakeSpan(name, as_type, **kwargs)
        self.spans.append(span)
        return _FakeSpanContext(span)

    def flush(self) -> None:
        self.flushed = True


class _FakeSpanContext:
    """Minimal context manager mirroring the real SDK's return value."""

    def __init__(self, span: FakeSpan) -> None:
        self._span = span

    def __enter__(self) -> FakeSpan:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._span.end()
        return False


# --- init_observability --------------------------------------------------


def test_init_observability_returns_none_without_credentials() -> None:
    settings = Settings(_env_file=None)

    assert init_observability(settings=settings) is None


def test_init_observability_returns_a_client_with_credentials() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-1",
    )

    client = init_observability(settings=settings)

    assert client is not None


# --- build_callbacks -------------------------------------------------------


def test_build_callbacks_is_empty_without_a_client() -> None:
    assert build_callbacks(None) == []


def test_build_callbacks_returns_one_handler_with_a_client() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-1",
    )
    client = init_observability(settings=settings)

    callbacks = build_callbacks(client)

    assert len(callbacks) == 1


# --- span envelopes: inert when client is None ----------------------------


def test_retriever_span_is_inert_without_a_client() -> None:
    with retriever_span(None, name="retrieval", query="AI Act scope", top_k=8) as span:
        span.update(hit_count=3, exact_filter_used=True)


def test_embedding_span_is_inert_without_a_client() -> None:
    with embedding_span(None, model_id="fastembed/x", dimension=384, text="hello") as span:
        span.update()


def test_generation_span_is_inert_without_a_client() -> None:
    with generation_span(None, name="analyst", model="mistral-small-2603", messages=[]) as span:
        span.update(usage_details={"input": 1, "output": 1, "total": 2})


# --- span envelopes: call the SDK when a client is provided ----------------


def test_retriever_span_starts_a_retriever_observation() -> None:
    client = FakeLangfuseClient()

    with retriever_span(client, name="retrieval", query="AI Act scope", top_k=8) as span:
        span.update(hit_count=3, exact_filter_used=True)

    assert len(client.spans) == 1
    fake_span = client.spans[0]
    assert fake_span.as_type == "retriever"
    assert fake_span.init_kwargs["input"] == {"query": "AI Act scope", "top_k": 8}
    assert fake_span.updates == [{"hit_count": 3, "exact_filter_used": True}]
    assert fake_span.ended is True


def test_generation_span_starts_a_generation_observation() -> None:
    client = FakeLangfuseClient()
    messages = [("system", "you are a critic"), ("user", "grade this")]

    with generation_span(client, name="critic", model="mistral-small-2603", messages=messages) as span:
        span.update(usage_details={"input": 10, "output": 5, "total": 15})

    fake_span = client.spans[0]
    assert fake_span.as_type == "generation"
    assert fake_span.init_kwargs["model"] == "mistral-small-2603"
    assert fake_span.updates == [{"usage_details": {"input": 10, "output": 5, "total": 15}}]


# --- embedding_span: cost is conditional ------------------------------------


def test_embedding_span_declares_no_cost_when_none_is_given() -> None:
    client = FakeLangfuseClient()

    with embedding_span(client, model_id="fastembed/x", dimension=384, text="hello") as span:
        pass

    fake_span = client.spans[0]
    assert "usage_details" not in fake_span.init_kwargs
    assert "cost_details" not in fake_span.init_kwargs
    assert fake_span.updates == []


def test_embedding_span_declares_cost_when_usage_is_given() -> None:
    client = FakeLangfuseClient()

    with embedding_span(client, model_id="mistral-embed", dimension=1024, text="hello") as span:
        span.update(usage_details={"input": 3, "output": 0, "total": 3})

    fake_span = client.spans[0]
    assert fake_span.updates == [{"usage_details": {"input": 3, "output": 0, "total": 3}}]


# --- extract_usage -----------------------------------------------------------


class _Response:
    def __init__(self, usage_metadata=None, response_metadata=None) -> None:
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        if response_metadata is not None:
            self.response_metadata = response_metadata


def test_extract_usage_reads_usage_metadata() -> None:
    response = _Response(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )

    assert extract_usage(response) == {"input": 10, "output": 5, "total": 15}


def test_extract_usage_falls_back_to_response_metadata() -> None:
    response = _Response(
        response_metadata={
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
        }
    )

    assert extract_usage(response) == {"input": 8, "output": 2, "total": 10}


def test_extract_usage_returns_none_without_usage_information() -> None:
    response = _Response()

    assert extract_usage(response) is None


def test_extract_usage_never_raises_on_malformed_responses() -> None:
    assert extract_usage(None) is None
    assert extract_usage(object()) is None
    assert extract_usage(_Response(usage_metadata="not a dict")) is None
    assert extract_usage(_Response(response_metadata={"token_usage": "nope"})) is None


# --- error propagation -------------------------------------------------------


def test_an_exception_inside_a_span_propagates_and_marks_the_span_in_error() -> None:
    client = FakeLangfuseClient()

    with pytest.raises(ValueError, match="boom"):
        with retriever_span(client, name="retrieval", query="q", top_k=1):
            raise ValueError("boom")

    fake_span = client.spans[0]
    assert fake_span.updates
    assert fake_span.updates[-1]["level"] == "ERROR"
    assert "boom" in fake_span.updates[-1]["status_message"]


def test_an_exception_inside_an_inert_span_still_propagates() -> None:
    with pytest.raises(ValueError, match="boom"):
        with retriever_span(None, name="retrieval", query="q", top_k=1):
            raise ValueError("boom")


# --- flush ---------------------------------------------------------------


def test_flush_is_a_no_op_without_a_client() -> None:
    flush(None)


def test_flush_calls_the_client() -> None:
    client = FakeLangfuseClient()

    flush(client)

    assert client.flushed is True
