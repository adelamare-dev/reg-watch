"""Refusals, in the language of the question.

Three of them, and they must not read alike. Nothing found is a corpus
boundary; found but never sufficiently grounded is a failure of this run;
unable to verify at all is neither, and saying otherwise would misstate what
went wrong. Collapsing any two of these would misrepresent the run to the
user.

The technical case is checked first, ahead of the other two, because
`critic_error` is set on a dedicated field that nothing ever clears - unlike
`iteration` or `hits`, which a retry can carry over from an unrelated earlier
pass. Once it is set, none of the other conditions get a vote: the analyst's
answer exists in state at this point, but a run the critic never verified
does not have the property this project promises, so it must not be
published under a milder-sounding refusal.

The other two are told apart by `iteration`, not by `hits`: `iteration` is
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
TECHNICAL_FR = (
    "La vérification de la réponse n'a pas pu être effectuée en raison d'un "
    "problème technique."
)
TECHNICAL_EN = (
    "The answer could not be verified due to a technical issue."
)


def refusal_node(state: GraphState) -> dict:
    """Produce the refusal, leaving `answer` unset so the UI can style it."""
    english = state["language"] == "en"

    if state["critic_error"]:
        reason = TECHNICAL_EN if english else TECHNICAL_FR
    elif state["iteration"] > 0:
        reason = PARTIAL_EN if english else PARTIAL_FR
    else:
        reason = NO_BASIS_EN if english else NO_BASIS_FR

    return {"answer": None, "refusal_reason": reason}
