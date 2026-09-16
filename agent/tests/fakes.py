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
