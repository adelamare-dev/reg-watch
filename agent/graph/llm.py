"""Build the chat model, and name it.

The factory returns the model *and* who served it: each answer has to say
which model produced it and under which jurisdiction, and a bare client
carries neither in a form the UI can read.

One provider today. The shape takes a chain because a second one is coming,
but no fallback is wired here: `.with_fallbacks()` only fires on a raised
exception, and a 429 that comes back as a payload never raises. That needs a
wrapper that inspects responses, which belongs with the tracing that makes a
provider switch visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_mistralai import ChatMistralAI

from rag.config import Settings, get_settings


@dataclass(frozen=True)
class ProviderInfo:
    """What the UI shows next to an answer."""

    name: str
    model: str
    jurisdiction: str


MISTRAL = ProviderInfo(
    name="Mistral La Plateforme",
    model="mistral-small-2603",
    jurisdiction="EU",
)


def build_llm(settings: Settings | None = None) -> tuple[ChatMistralAI, ProviderInfo]:
    """Return the model and its provider card.

    Temperature defaults to 0: the analyst quotes regulation and the critic
    grades grounding. Neither improves with sampling.
    """
    settings = settings or get_settings()

    if not settings.mistral_api_key:
        raise ValueError(
            "MISTRAL_API_KEY is required to run the graph. "
            "Copy .env.example to .env and fill it in."
        )

    model = ChatMistralAI(
        model=settings.llm_model,
        api_key=settings.mistral_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )

    return model, ProviderInfo(
        name=MISTRAL.name,
        model=settings.llm_model,
        jurisdiction=MISTRAL.jurisdiction,
    )
