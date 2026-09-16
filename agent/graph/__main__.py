"""Ask the graph one question from the command line.

`main()` is the only place that builds Qdrant, the embedder and the chat
model; `run_question` takes them as arguments, which is what lets the tests
exercise the whole path with fakes.
"""

from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient

from graph.build import build_graph
from graph.llm import ProviderInfo, build_llm
from graph.state import GraphState, initial_state
from rag.config import get_settings
from rag.embedder import build_embedder
from rag.retriever import Retriever


def run_question(
    question: str,
    *,
    retriever,
    llm,
    provider: ProviderInfo,
    regulation_filter: str | None = None,
) -> GraphState:
    """Run one question through the graph and stamp the provider on it."""
    graph = build_graph(retriever=retriever, llm=llm)
    state = initial_state(question, regulation_filter=regulation_filter)

    final = graph.invoke(state, {"configurable": {"thread_id": "cli"}})
    final["llm_provider"] = provider.model
    return final


def format_result(state: GraphState, provider: ProviderInfo) -> str:
    """Render what the UI will later render: answer, citations, provenance."""
    lines: list[str] = []
    refused = bool(state["refusal_reason"])

    if refused:
        lines.append("=== REFUS ===")
        lines.append(state["refusal_reason"])
    else:
        lines.append("=== RÉPONSE ===")
        lines.append(state["answer"] or "")

    # A refusal for insufficient grounding still carries the last pass's
    # claims and divergences, kept for diagnostics. Printing them here would
    # hand back under "vérifiées" exactly what the refusal exists to withhold.
    if not refused and state["verified_claims"]:
        lines.append("")
        lines.append("=== AFFIRMATIONS VÉRIFIÉES ===")
        for claim in state["verified_claims"]:
            mark = "OK " if claim["is_supported"] else "NON ANCRÉE"
            source = (
                f"{claim['regulation']}, article {claim['article_number']}"
                if claim["regulation"]
                else "aucune source"
            )
            lines.append(f"  [{mark}] {claim['text']} — {source}")

    if not refused and state["divergences"]:
        lines.append("")
        lines.append("=== DIVERGENCES (sans arbitrage) ===")
        for divergence in state["divergences"]:
            lines.append(f"  {divergence['theme']}")
            lines.append(f"    DORA   : {divergence['dora_position']}")
            lines.append(f"    AI Act : {divergence['ai_act_position']}")

    lines.append("")
    lines.append("=== PROVENANCE ===")
    lines.append(f"  Modèle  : {provider.model} ({provider.jurisdiction})")
    lines.append(f"  Corpus  : {state['corpus_version'] or 'non déterminé'}")
    lines.append(f"  Passes  : {state['iteration']}")
    lines.append(f"  Filtre exact : {'oui' if state['used_exact_filter'] else 'non'}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m graph",
        description="Pose une question de conformité au corpus DORA × AI Act.",
    )
    parser.add_argument("question", help="La question, entre guillemets")
    parser.add_argument(
        "--regulation",
        default=None,
        help="Restreindre à un règlement : DORA ou AI_ACT",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = build_embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    retriever = Retriever(client=client, embedder=embedder, settings=settings)
    llm, provider = build_llm(settings)

    final = run_question(
        args.question,
        retriever=retriever,
        llm=llm,
        provider=provider,
        regulation_filter=args.regulation,
    )
    print(format_result(final, provider))
    return 0


if __name__ == "__main__":
    sys.exit(main())
