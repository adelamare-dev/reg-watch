"""The demo entry point: run a question, print what the UI will later show."""

from __future__ import annotations

from graph.__main__ import format_result, run_question
from graph.llm import ProviderInfo
from graph.schemas import Claim, CriticVerdict
from graph.state import initial_state
from tests.fakes import FakeLLM, FakeStructuredLLM
from tests.test_mcp_tools import FakeRetriever, make_hit

PROVIDER = ProviderInfo(name="Test Provider", model="test-model", jurisdiction="EU")


class ScriptedLLM:
    def __init__(self, *, answers, verdicts):
        self._analyst = FakeLLM(responses=answers)
        self._critic = FakeStructuredLLM(verdicts=verdicts)

    def invoke(self, messages):
        return self._analyst.invoke(messages)

    def with_structured_output(self, schema):
        return self._critic.with_structured_output(schema)


def grounded() -> CriticVerdict:
    return CriticVerdict(
        is_grounded=True,
        claims=[
            Claim(
                text="DORA impose un registre.",
                is_supported=True,
                regulation="DORA",
                article_number="28",
            )
        ],
        divergences=[],
        reformulated_query=None,
    )


def test_run_question_returns_the_final_state():
    retriever = FakeRetriever(hits=[make_hit()])
    llm = ScriptedLLM(answers=["DORA, article 28 impose un registre."], verdicts=[grounded()])

    final = run_question(
        "Que dit l'article 28 de DORA ?",
        retriever=retriever,
        llm=llm,
        provider=PROVIDER,
    )

    assert final["answer"] == "DORA, article 28 impose un registre."
    assert final["llm_provider"] == "test-model"


def test_the_output_names_the_model_and_the_corpus_version():
    state = initial_state("Que dit l'article 28 de DORA ?")
    state["answer"] = "DORA, article 28 impose un registre."
    state["corpus_version"] = "2025-01-17"
    state["verified_claims"] = [
        {
            "text": "DORA impose un registre.",
            "is_supported": True,
            "regulation": "DORA",
            "article_number": "28",
        }
    ]
    state["iteration"] = 1

    output = format_result(state, PROVIDER)

    assert "test-model" in output
    assert "EU" in output
    assert "2025-01-17" in output
    assert "DORA, article 28" in output


def test_a_refusal_reads_differently_from_an_answer():
    state = initial_state("Quelle est la capitale de la Suisse ?")
    state["refusal_reason"] = "Aucune base réglementaire trouvée dans le corpus."

    output = format_result(state, PROVIDER)

    assert "REFUS" in output.upper()
    assert "Aucune base réglementaire" in output


def test_a_refusal_does_not_print_unsupported_claims():
    # The run refused because these claims were not grounded. Printing them
    # under "affirmations vérifiées" would hand back exactly what the refusal
    # exists to withhold.
    state = initial_state("Question difficile")
    state["refusal_reason"] = "Des dispositions ont été trouvées, mais leur ancrage est insuffisant."
    state["verified_claims"] = [
        {
            "text": "DORA impose une amende de 10 millions.",
            "is_supported": False,
            "regulation": None,
            "article_number": None,
        }
    ]

    output = format_result(state, PROVIDER)

    assert "amende de 10 millions" not in output
    assert "AFFIRMATIONS" not in output.upper()
    assert "PROVENANCE" in output.upper()


def test_divergences_appear_side_by_side():
    state = initial_state("Comparaison")
    state["answer"] = "Les deux règlements traitent ce point."
    state["divergences"] = [
        {
            "theme": "Gestion des risques",
            "dora_position": "DORA, article 6.",
            "ai_act_position": "AI Act, article 9.",
        }
    ]

    output = format_result(state, PROVIDER)

    assert "Gestion des risques" in output
    assert "DORA, article 6." in output
    assert "AI Act, article 9." in output
