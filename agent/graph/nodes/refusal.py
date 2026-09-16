"""Refusals, in the language of the question.

Two of them, and they must not read alike. Nothing found is a corpus
boundary; found but never sufficiently grounded is a failure of this run.
Collapsing them would tell a user their question is out of scope when it is
not.

The two are told apart by `iteration`, not by `hits`: `iteration` is
monotone - only the critic increments it, and nothing ever resets it - so
reaching this node with `iteration > 0` means by construction that an answer
was produced and then judged insufficiently grounded. `hits` offers no such
guarantee, since a retry's retrieval overwrites it with that pass's results.
"""

from __future__ import annotations

from graph.state import GraphState

NO_BASIS_FR = (
    "Aucune base réglementaire trouvée dans le corpus pour répondre à cette "
    "question."
)
NO_BASIS_EN = (
    "No regulatory basis was found in the corpus to answer this question."
)
PARTIAL_FR = (
    "Des dispositions ont été trouvées, mais leur ancrage est insuffisant "
    "pour produire une réponse citable."
)
PARTIAL_EN = (
    "Provisions were found, but their grounding is insufficient to produce a "
    "citable answer."
)


def refusal_node(state: GraphState) -> dict:
    """Produce the refusal, leaving `answer` unset so the UI can style it."""
    english = state["language"] == "en"

    if state["iteration"] > 0:
        reason = PARTIAL_EN if english else PARTIAL_FR
    else:
        reason = NO_BASIS_EN if english else NO_BASIS_FR

    return {"answer": None, "refusal_reason": reason}
