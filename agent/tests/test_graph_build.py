"""The assembled graph: three nodes, two exits, one bounded cycle."""

from __future__ import annotations

from graph.build import build_graph
from graph.schemas import Claim, CriticVerdict
from graph.state import initial_state
from rag.references import References
from rag.retriever import RetrievalResult
from tests.fakes import FakeLangfuseClient, FakeLLM, FakeRaisingStructuredLLM, FakeStructuredLLM
from tests.test_mcp_tools import FakeRetriever, make_hit

CONFIG = {"configurable": {"thread_id": "test"}}


class SequencedRetriever:
    """Replays one `RetrievalResult` per call, from a fixed sequence of hit lists.

    `FakeRetriever` always returns the same hits on every call, which cannot
    express a retry that finds something on the first pass and nothing on the
    reformulated second pass. This double scripts that per-call sequence.
    """

    def __init__(self, *, hits_sequence: list[list]) -> None:
        self._hits_sequence = list(hits_sequence)
        self.search_calls: list[dict] = []

    def search(self, question: str, *, language=None, regulation=None, top_k=None):
        self.search_calls.append(
            {"question": question, "language": language, "regulation": regulation, "top_k": top_k}
        )
        hits = self._hits_sequence.pop(0)
        return RetrievalResult(
            hits=hits,
            references=References(),
            used_exact_filter=False,
            is_grounded=bool(hits),
        )


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


def test_a_sterile_reformulation_still_refuses_partially():
    # The first pass found articles; the critic's reformulation found none.
    # That is a failure of this run, not a boundary of the corpus, and the
    # refusal has to say so.
    retriever = SequencedRetriever(hits_sequence=[[make_hit()], []])
    llm = ScriptedLLM(
        answers=["Première tentative."],
        verdicts=[verdict(is_grounded=False, query="requête stérile")],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Question difficile"), CONFIG)

    assert final["answer"] is None
    assert "insuffisant" in final["refusal_reason"].lower()


def test_an_english_question_stays_english_across_a_retry():
    # The critic reformulates into keywords, which carry no function words.
    # Detecting the language on that reformulation would silently flip the
    # answer to French.
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["First attempt.", "Second attempt."],
        verdicts=[
            verdict(is_grounded=False, query="third-party ICT service provider register"),
            verdict(is_grounded=True),
        ],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("What does article 28 of DORA say?"), CONFIG)

    assert final["language"] == "en"
    assert final["iteration"] == 2


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


class AnalystThenRaisingCriticLLM:
    """Answers as the analyst, then raises when invoked as the critic."""

    def __init__(self, *, answers: list[str], error: Exception) -> None:
        self._analyst = FakeLLM(responses=answers)
        self._critic = FakeRaisingStructuredLLM(error=error)
        self.analyst_calls = self._analyst.calls
        self.critic_calls = self._critic.calls

    def invoke(self, messages):
        return self._analyst.invoke(messages)

    def with_structured_output(self, schema):
        return self._critic.with_structured_output(schema)


def test_a_raising_critic_ends_the_run_in_a_technical_refusal_not_a_crash():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = AnalystThenRaisingCriticLLM(
        answers=["DORA, article 28 impose un registre."],
        error=RuntimeError("HTTP 429: rate limited"),
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    assert final["answer"] is None
    assert final["critic_error"] is not None
    reason = final["refusal_reason"].lower()
    assert "aucune base réglementaire" not in reason
    assert "insuffisant" not in reason


def test_the_graph_runs_identically_without_a_tracing_client():
    # Non-regression: the default (`client=None`) must behave exactly as
    # before `build_graph` learned about tracing.
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["DORA, article 28 impose un registre."],
        verdicts=[verdict(is_grounded=True)],
    )
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    assert final["answer"] == "DORA, article 28 impose un registre."


def test_a_traced_run_opens_a_span_for_each_of_the_three_nodes():
    client = FakeLangfuseClient()
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(
        answers=["DORA, article 28 impose un registre."],
        verdicts=[verdict(is_grounded=True)],
    )
    graph = build_graph(retriever=retriever, llm=llm, client=client)

    graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    as_types = [span.as_type for span in client.spans]
    assert as_types.count("retriever") == 1
    assert as_types.count("generation") == 2  # analyst, then critic


def test_the_final_state_carries_the_corpus_version():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(answers=["Réponse."], verdicts=[verdict(is_grounded=True)])
    graph = build_graph(retriever=retriever, llm=llm)

    final = graph.invoke(initial_state("Que dit l'article 28 de DORA ?"), CONFIG)

    assert final["corpus_version"] == "2025-01-17"
