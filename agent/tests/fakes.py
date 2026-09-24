"""Test doubles shared across the graph suites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeResponse:
    """The slice of a chat model response the nodes read."""

    content: str


class FakeLLM:
    """Replays canned responses and records the messages it was handed."""

    def __init__(self, *, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages) -> FakeResponse:
        self.calls.append(messages)
        if not self._responses:
            return FakeResponse(content="")
        return FakeResponse(content=self._responses.pop(0))


class FakeStructuredLLM:
    """A model whose `with_structured_output` replays prepared objects."""

    def __init__(self, *, verdicts: list | None = None) -> None:
        self._verdicts = list(verdicts or [])
        self.calls: list[list[tuple[str, str]]] = []

    def with_structured_output(self, schema):  # noqa: ARG002 - shape only
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self._verdicts.pop(0)


class FakeRaisingStructuredLLM:
    """A model whose structured call always raises, for the critic's failure path."""

    def __init__(self, *, error: Exception) -> None:
        self._error = error
        self.calls: list[list[tuple[str, str]]] = []

    def with_structured_output(self, schema):  # noqa: ARG002 - shape only
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        raise self._error


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


class _FakeSpanContext:
    """Minimal context manager mirroring the real SDK's return value."""

    def __init__(self, span: FakeSpan) -> None:
        self._span = span

    def __enter__(self) -> FakeSpan:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._span.end()
        return False


class FakeLangfuseClient:
    """A fake client recording `start_as_current_observation` calls.

    Shared across node test suites so each one can assert a span was opened
    for its node without talking to Langfuse or importing `unittest.mock`.
    """

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.flushed = False

    def start_as_current_observation(self, *, name: str, as_type: str, **kwargs):
        span = FakeSpan(name, as_type, **kwargs)
        self.spans.append(span)
        return _FakeSpanContext(span)

    def flush(self) -> None:
        self.flushed = True
