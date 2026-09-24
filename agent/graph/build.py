"""Wire the three nodes into a graph with one bounded cycle.

LangGraph hands each node the state and nothing else, while the nodes take
their collaborators as keyword arguments. `partial` binds them here, at
assembly time, so dependency injection survives contact with the framework
and the tests keep handing in fakes.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes.analyst import analyst_node
from graph.nodes.critic import critic_node, route_after_critic
from graph.nodes.refusal import refusal_node
from graph.nodes.retrieval import retrieval_node, route_after_retrieval
from graph.state import GraphState


def build_graph(*, retriever, llm, checkpointer=None, client=None):
    """Compile the graph. Pass fakes in tests, real clients in `__main__`.

    `client` is the Langfuse client (or `None`), bound into the three nodes
    the same way `retriever` and `llm` are: a keyword with a default, so
    every existing caller keeps working unchanged and tracing stays a
    decorator rather than a dependency of the graph's shape.
    """
    builder = StateGraph(GraphState)

    builder.add_node("retrieval", partial(retrieval_node, retriever=retriever, client=client))
    builder.add_node("analyst", partial(analyst_node, llm=llm, client=client))
    builder.add_node("critic", partial(critic_node, llm=llm, client=client))
    builder.add_node("refusal", refusal_node)

    builder.add_edge(START, "retrieval")
    builder.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"analyst": "analyst", "refusal": "refusal"},
    )
    builder.add_edge("analyst", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"end": END, "retrieval": "retrieval", "refusal": "refusal"},
    )
    builder.add_edge("refusal", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
