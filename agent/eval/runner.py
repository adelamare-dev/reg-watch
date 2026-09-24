"""Run the graph over a validated dataset, one entry at a time.

Evaluation deliberately uses `settings.eval_top_k` (3) instead of the
production `settings.top_k` (8): `context_precision` costs one judge LLM call
per retrieved chunk, so `top_k` is the dominant lever on the cost of a run —
running the graph itself at `eval_top_k` keeps that cost from being paid
twice (once in the graph's own retrieval, once again in RAGAS). This is a
deliberate divergence from production behaviour, not an oversight to be
"fixed" back to `top_k` later.
"""

from __future__ import annotations

from typing import Any, Protocol

from eval.dataset import GoldenEntry
from eval.metrics import EvalSample
from graph.__main__ import run_question as _default_run_question


class _RetrieverLike(Protocol):
    def search(self, question: str, **kwargs: Any) -> Any: ...


class _EvalTopKRetriever:
    """Wraps a retriever so every `.search()` call is capped at `eval_top_k`.

    Forcing the cap here — rather than mutating the caller's retriever, or
    relying on every call site to remember to pass `top_k=` — means a
    retriever shared with production code is never touched, and the graph
    (which never passes `top_k` itself; see `graph/nodes/retrieval.py`)
    doesn't need to know evaluation exists.
    """

    def __init__(self, retriever: Any, *, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)

    @property
    def top_k(self) -> int:
        return self._top_k

    def search(self, question: str, **kwargs: Any) -> Any:
        kwargs["top_k"] = self._top_k
        return self._retriever.search(question, **kwargs)


def run_dataset(
    entries: list[GoldenEntry],
    *,
    retriever: Any,
    llm: Any,
    provider: Any,
    eval_top_k: int,
    client: Any = None,
    run_question: Any = _default_run_question,
) -> list[EvalSample]:
    """Execute the graph on every entry, collecting answer, contexts and state.

    `run_question` is injectable (defaults to `graph.__main__.run_question`)
    so this can be tested with a fake graph and no LLM/Qdrant.
    """
    capped_retriever = _EvalTopKRetriever(retriever, top_k=eval_top_k)
    samples: list[EvalSample] = []

    for entry in entries:
        state = run_question(
            entry.question,
            retriever=capped_retriever,
            llm=llm,
            provider=provider,
            client=client,
        )

        hits = state.get("hits") or []
        samples.append(
            EvalSample(
                entry_id=entry.id,
                state=dict(state),
                question=entry.question,
                answer=state.get("answer"),
                retrieved_contexts=[hit["text"] for hit in hits],
                reference=entry.reference,
            )
        )

    return samples
