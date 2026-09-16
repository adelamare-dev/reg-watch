"""The assembled graph: three nodes, two exits, one bounded cycle."""

from __future__ import annotations

from graph.build import build_graph
from graph.schemas import Claim, CriticVerdict
from graph.state import initial_state
from tests.fakes import FakeLLM, FakeStructuredLLM
from tests.test_mcp_tools import FakeRetriever, make_hit

CONFIG = {"configurable": {"thread_id": "test"}}


class ScriptedLLM:
    """Answers as the analyst, then verdicts as the critic, in order."""

    def __init__(self, *, answers: list[str], verdicts: list[CriticVerdict]) -> None:
        self._analyst = FakeLLM(responses=answers)
        self._critic = FakeStructuredLLM(verdicts=verdicts)
        self.analyst_calls = self._analyst.calls
        self.critic_calls = self._critic.calls

    def invoke(self, messages):
        return self._analyst.invoke(messages)

    def with_structured_output(self, schema):
        return self._critic.with_structured_output(schema)


def verdict(*, is_grounded: bool, query: str | None = None) -> CriticVerdict:
    return CriticVerdict(
        is_grounded=is_grounded,
        claims=[
            Claim(
                text="DORA impose un registre.",
                is_supported=is_grounded,
                regulation="DORA" if is_grounded else None,
                article_number="28" if is_grounded else None,
            )
        ],
        divergences=[],
        reformulated_query=query,
    )


def test_a_grounded_question_runs_all_three_nodes():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["DORA, article 28 impose un registre."],
        verdicts=[verdict(is_grounded=True)],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    assert final["answer"] == "DORA, article 28 impose un registre."
    assert final["refusal_reason"] is None
    assert len(final["verified_claims"]) == 1
    assert final["iteration"] == 1


def test_an_out_of_corpus_question_refuses_without_calling_the_llm():
    # The whole point of the early exit: no chunks, no generation spend.
    retriever = FakeRetriever(hits=[])
    llm = ScriptedLLM(answers=[], verdicts=[])
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Quelle est la capitale de la Suisse ?"), CONFIG)

    assert final["answer"] is None
    assert "aucune base réglementaire" in final["refusal_reason"].lower()
    assert llm.analyst_calls == []
    assert llm.critic_calls == []


def test_an_ungrounded_answer_sends_the_run_back_to_retrieval():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["Première tentative.", "Seconde tentative."],
        verdicts=[
            verdict(is_grounded=False, query="registre des prestataires TIC"),
            verdict(is_grounded=True),
        ],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Question vague"), CONFIG)

    assert final["answer"] == "Seconde tentative."
    assert final["iteration"] == 2
    # The second search used the critic's reformulation, not the original.
    assert retriever.search_calls[1]["question"] == "registre des prestataires TIC"


def test_the_cycle_is_bounded_and_degrades_into_a_partial_refusal():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["Tentative 1.", "Tentative 2."],
        verdicts=[
            verdict(is_grounded=False, query="reformulation 1"),
            verdict(is_grounded=False, query="reformulation 2"),
        ],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Question insoluble"), CONFIG)

    assert final["iteration"] == 2
    assert final["answer"] is None
    assert "insuffisant" in final["refusal_reason"].lower()
    # Two passes, not three: the budget held.
    assert len(llm.analyst_calls) == 2


def test_divergences_survive_to_the_final_state():
    from graph.schemas import Divergence

    retriever = FakeRetriever(
        hits=[make_hit(regulation="DORA"), make_hit(regulation="AI_ACT", article_number="9")]
    )
    diverging = CriticVerdict(
        is_grounded=True,
        claims=[],
        divergences=[
            Divergence(
                theme="Gestion des risques",
                dora_position="DORA, article 6.",
                ai_act_position="AI Act, article 9.",
            )
        ],
        reformulated_query=None,
    )
    llm = ScriptedLLM(answers=["Les deux règlements traitent ce point."], verdicts=[diverging])
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Comparaison de la gestion des risques"), CONFIG)

    assert len(final["divergences"]) == 1
    assert final["divergences"][0]["theme"] == "Gestion des risques"


def test_the_final_state_carries_the_corpus_version():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(answers=["Réponse."], verdicts=[verdict(is_grounded=True)])
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    assert final["corpus_version"] == "2025-01-17"
