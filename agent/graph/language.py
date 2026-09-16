"""Two-language detection, by function words.

The field it fills serves display and post-filtering — the answer comes back
in the language of the question. It never routes the retrieval: a single
multilingual vector space is the whole point of the corpus design.

Function words are the signal that survives a short question. Legal nouns
("article", "information", "service") are shared between French and English
and are deliberately absent from both lists.
"""

from __future__ import annotations

import re

_FRENCH_MARKERS = frozenset(
    {
        "que", "quelles", "quels", "quelle", "quel", "comment", "pourquoi",
        "est", "sont", "dit", "de", "des", "du", "la", "le", "les", "un",
        "une", "dans", "pour", "sur", "avec", "aux", "au", "par", "en",
        "obligations", "doit", "peut", "faut", "quoi", "lie", "liee",
    }
)

_ENGLISH_MARKERS = frozenset(
    {
        "what", "which", "how", "why", "does", "do", "is", "are", "the",
        "a", "an", "of", "in", "for", "on", "with", "to", "by", "should",
        "must", "can", "say", "says", "apply", "applies", "be",
    }
)

_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")


def detect_language(question: str) -> str:
    """Return "fr" or "en", defaulting to French when the signal is a tie."""
    words = re.findall(r"[a-z]+", question.lower().translate(_ACCENTS))

    french = sum(1 for word in words if word in _FRENCH_MARKERS)
    english = sum(1 for word in words if word in _ENGLISH_MARKERS)

    return "en" if english > french else "fr"
