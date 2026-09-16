"""The critic's output, as a typed object rather than prose.

The judgement is a model's; the routing that follows must not be. A schema is
what makes "go back and search again" a branch taken on a boolean instead of
on a string someone has to parse.

Note what the divergence schema cannot express: there is no field for which
regulation prevails. Juxtaposition is the whole contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One atomic assertion from the answer, and its supporting chunk."""

    text: str = Field(description="The atomic claim, restated on its own")
    is_supported: bool = Field(description="Whether a retrieved chunk supports it")
    regulation: str | None = Field(
        default=None, description="Regulation of the supporting chunk"
    )
    article_number: str | None = Field(
        default=None, description="Article of the supporting chunk"
    )


class Divergence(BaseModel):
    """The same matter, treated differently by the two regulations."""

    theme: str = Field(description="The matter both regulations address")
    dora_position: str = Field(description="What DORA provides, with its article")
    ai_act_position: str = Field(
        description="What the AI Act provides, with its article"
    )


class CriticVerdict(BaseModel):
    """The whole critique, in one structured answer."""

    is_grounded: bool = Field(
        description="Whether every claim is supported by a retrieved chunk"
    )
    claims: list[Claim] = Field(default_factory=list)
    divergences: list[Divergence] = Field(default_factory=list)
    reformulated_query: str | None = Field(
        default=None,
        description="A better search query, when grounding is insufficient",
    )
