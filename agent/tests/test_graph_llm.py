"""Building the model, and saying which one answered."""

from __future__ import annotations

import pytest

from graph.llm import MISTRAL, ProviderInfo, build_llm
from rag.config import Settings


def test_the_provider_declares_its_jurisdiction():
    # The UI states which model answered and under which jurisdiction; that
    # claim has to come from somewhere typed.
    assert MISTRAL.jurisdiction == "EU"
    assert MISTRAL.name == "Mistral La Plateforme"


def make_settings(**overrides) -> Settings:
    """Build settings without reading the developer's own .env.

    `Settings` loads the repository `.env` by default, so a machine that has a
    real key would quietly pass the "no key" test. Pinning `_env_file=None`
    makes these tests answer the same everywhere.
    """
    defaults = {"_env_file": None, "mistral_api_key": "test-key"}
    defaults.update(overrides)
    return Settings(**defaults)


def test_build_llm_returns_the_model_and_its_provider_info():
    settings = make_settings()

    model, provider = build_llm(settings)

    assert isinstance(provider, ProviderInfo)
    assert provider.model == settings.llm_model
    assert hasattr(model, "invoke")


def test_build_llm_refuses_to_run_without_an_api_key():
    settings = make_settings(mistral_api_key=None)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_llm(settings)


def test_the_model_honours_the_configured_name():
    settings = make_settings(llm_model="mistral-large-latest")

    _, provider = build_llm(settings)

    assert provider.model == "mistral-large-latest"
