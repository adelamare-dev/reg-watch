"""The critic decides whether the answer stands, and whether to try again."""

from __future__ import annotations

from graph.nodes.critic import critic_node, route_after_critic
from graph.schemas import Claim, CriticVerdict, Divergence
from graph.state import GraphState, initial_state
from mcp_server.tools import _serialise_hit
from tests.fakes import FakeStructuredLLM
from tests.test_mcp_tools import make_hit


def grounded_verdict(**overrides) -> CriticVerdict:
    defaults = dict(
        is_grounded=True,
        claims=[
            Claim(
                text="DORA impose un registre des prestataires.",
                is_supported=True,
                regulation="DORA",
                article_number="28",
            )
        ],
        divergences=[],
        reformulated_query=None,
    )
    defaults.update(overrides)
    return CriticVerdict(**defaults)


def state_with_answer() -> GraphState:
    state = initial_state("Que dit l'article 28 de DORA ?")
    state["hits"] = [_serialise_hit(make_hit())]
    state["answer"] = "DORA, article 28 impose un registre."
    state["is_grounded"] = True
    return state


def test_the_critic_records_verified_claims():
    llm = FakeStructuredLLM(verdicts=[grounded_verdict()])

    update = critic_node(state_with_answer(), llm=llm)

    assert len(update["verified_claims"]) == 1
    assert update["verified_claims"][0]["is_supported"] is True
    assert update["verified_claims"][0]["article_number"] == "28"


def test_an_unsupported_claim_is_flagged_not_dropped():
    # Dropping it silently would leave a shorter answer with no trace of why.
    verdict = grounded_verdict(
        is_grounded=False,
        claims=[
            Claim(
                text="DORA impose une amende de 10 millions.",
                is_supported=False,
                regulation=None,
                article_number=None,
            )
        ],
        reformulated_query="sanctions DORA",
    )
    llm = FakeStructuredLLM(verdicts=[verdict])

    update = critic_node(state_with_answer(), llm=llm)

    assert update["verified_claims"][0]["is_supported"] is False


def test_the_critic_increments_the_iteration_counter():
    llm = FakeStructuredLLM(verdicts=[grounded_verdict()])

    update = critic_node(state_with_answer(), llm=llm)

    assert update["iteration"] == 1


def test_an_insufficient_verdict_replaces_the_search_query():
    verdict = grounded_verdict(
        is_grounded=False, reformulated_query="registre des prestataires TIC"
    )
    llm = FakeStructuredLLM(verdicts=[verdict])

    update = critic_node(state_with_answer(), llm=llm)

    assert update["search_query"] == "registre des prestataires TIC"


def test_a_sufficient_verdict_leaves_the_question_untouched():
    llm = FakeStructuredLLM(verdicts=[grounded_verdict()])

    update = critic_node(state_with_answer(), llm=llm)

    assert "search_query" not in update


def test_divergences_are_recorded_without_a_verdict():
    verdict = grounded_verdict(
        divergences=[
            Divergence(
                theme="Gestion des risques",
                dora_position="DORA, article 6 : cadre de gestion du risque TIC.",
                ai_act_position="AI Act, article 9 : système de gestion des risques.",
            )
        ]
    )
    llm = FakeStructuredLLM(verdicts=[verdict])

    update = critic_node(state_with_answer(), llm=llm)

    assert len(update["divergences"]) == 1
    recorded = update["divergences"][0]
    assert "dora_position" in recorded
    assert "ai_act_position" in recorded


def test_the_divergence_schema_cannot_express_an_arbitration():
    # The guarantee is structural: juxtaposition is the whole contract, and no
    # field may carry which regulation wins.
    assert set(Divergence.model_fields) == {"theme", "dora_position", "ai_act_position"}


def test_the_critic_sends_the_chunks_it_verifies_against():
    llm = FakeStructuredLLM(verdicts=[grounded_verdict()])

    critic_node(state_with_answer(), llm=llm)
    human = llm.calls[0][1][1]

    assert "<corpus_chunk" in human
    assert "DORA, article 28 impose un registre." in human


def test_a_grounded_answer_ends_the_run():
    state = state_with_answer()
    state["is_grounded"] = True
    state["iteration"] = 1

    assert route_after_critic(state) == "end"


def test_an_ungrounded_answer_goes_back_to_retrieval():
    state = state_with_answer()
    state["is_grounded"] = False
    state["iteration"] = 1

    assert route_after_critic(state) == "retrieval"


def test_the_retry_budget_is_enforced():
    # Two passes is the budget. A third would spend two more LLM calls
    # reformulating a query that has already failed twice.
    state = state_with_answer()
    state["is_grounded"] = False
    state["iteration"] = 2

    assert route_after_critic(state) == "refusal"
