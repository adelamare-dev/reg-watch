"""Langfuse settings must stay optional: no key, no tracing, no failure."""

from __future__ import annotations

from rag.config import Settings


def test_langfuse_credentials_default_to_none() -> None:
    settings = Settings(_env_file=None)

    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None


def test_langfuse_base_url_defaults_to_cloud() -> None:
    settings = Settings(_env_file=None)

    assert settings.langfuse_base_url == "https://cloud.langfuse.com"


def test_tracing_is_disabled_without_credentials() -> None:
    settings = Settings(_env_file=None)

    assert settings.tracing_enabled is False


def test_tracing_requires_both_halves_of_the_key_pair() -> None:
    public_only = Settings(_env_file=None, langfuse_public_key="pk-lf-1")
    secret_only = Settings(_env_file=None, langfuse_secret_key="sk-lf-1")

    assert public_only.tracing_enabled is False
    assert secret_only.tracing_enabled is False


def test_tracing_is_enabled_when_both_keys_are_present() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-1",
    )

    assert settings.tracing_enabled is True


def test_a_self_hosted_instance_only_needs_the_base_url() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-1",
        langfuse_base_url="http://langfuse.internal:3000",
    )

    assert settings.tracing_enabled is True
    assert settings.langfuse_base_url == "http://langfuse.internal:3000"


def test_eval_top_k_is_lower_than_the_runtime_one() -> None:
    """Deliberate divergence: evaluation trades recall for call volume."""
    settings = Settings(_env_file=None)

    assert settings.eval_top_k == 3
    assert settings.top_k == 8
