"""Embedding back-ends behind a single interface.

Two implementations: a local one (fastembed, offline and reproducible) and a
cloud one (Mistral, EU-hosted). The local back-end is the default, because the
whole corpus must be reproducible on a third-party machine without credentials.

The asymmetric prefix handling lives here. Some retrieval models are trained
with `query: ` / `passage: ` markers and degrade *silently* without them.
fastembed does not add them: `query_embed` and `passage_embed` both delegate to
`embed` with no prefix logic, so this module owns that responsibility.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Models trained with the asymmetric e5 convention. Paraphrase models and
# `mistral-embed` are deliberately absent: for them the prefix is not a marker,
# it is just noise embedded as part of the text.
MODELS_REQUIRING_PREFIXES: frozenset[str] = frozenset(
    {
        "intfloat/multilingual-e5-large",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-small",
    }
)

# Multilingual models verified to cover French and English. The registry also
# flags some models as multilingual that are really German- or code-oriented,
# so an explicit list beats trusting the description string alone.
SUPPORTED_MULTILINGUAL: frozenset[str] = frozenset(
    {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
    }
)

MISTRAL_DIMENSION = 1024


@runtime_checkable
class Embedder(Protocol):
    """Everything the ingestion and retrieval paths need from an embedder."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def needs_prefixes(model_id: str) -> bool:
    """Whether this model expects the `query:` / `passage:` convention."""
    return model_id in MODELS_REQUIRING_PREFIXES


def apply_query_prefix(text: str, model_id: str) -> str:
    """Prefix a query, unless the model does not use the convention."""
    if not needs_prefixes(model_id) or text.startswith(QUERY_PREFIX):
        return text
    return f"{QUERY_PREFIX}{text}"


def apply_passage_prefix(text: str, model_id: str) -> str:
    """Prefix a passage, unless the model does not use the convention."""
    if not needs_prefixes(model_id) or text.startswith(PASSAGE_PREFIX):
        return text
    return f"{PASSAGE_PREFIX}{text}"


class FastEmbedEmbedder:
    """Local ONNX embeddings. Offline after the first model download."""

    def __init__(self, model_id: str) -> None:
        from fastembed import TextEmbedding

        registry = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        if model_id not in registry:
            raise ValueError(
                f"unknown fastembed model {model_id!r}; "
                f"pick one of {sorted(SUPPORTED_MULTILINGUAL)}"
            )
        if model_id not in SUPPORTED_MULTILINGUAL:
            raise ValueError(
                f"{model_id!r} is not a supported multilingual model; the corpus "
                "indexes French and English in a single vector space"
            )

        self._model_id = model_id
        # Read from the registry rather than hard-coded: switching models must
        # not require touching the collection setup.
        self._dimension = int(registry[model_id]["dim"])
        self._model: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load(self):  # noqa: ANN202 - fastembed has no exported type
        """Load the weights on first use.

        Deferred so that constructing an embedder — as configuration checks and
        tests do — never triggers a model download.
        """
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_id)
        return self._model

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        prefixed = [apply_passage_prefix(t, self._model_id) for t in texts]
        return [vector.tolist() for vector in self._load().embed(prefixed)]

    def embed_query(self, text: str) -> list[float]:
        prefixed = apply_query_prefix(text, self._model_id)
        return next(iter(self._load().embed([prefixed]))).tolist()


class MistralEmbedder:
    """Cloud embeddings via the Mistral API, EU-hosted.

    Trades reproducibility for zero local footprint: vectors depend on a
    server-side model that can change without notice, so an ingested corpus is
    no longer reproducible offline.
    """

    def __init__(self, model_id: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required for the mistral provider")
        self._model_id = model_id
        self._api_key = api_key
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return MISTRAL_DIMENSION

    def _load(self):  # noqa: ANN202 - mistralai has no exported client type
        if self._client is None:
            from mistralai import Mistral

            self._client = Mistral(api_key=self._api_key)
        return self._client

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        response = self._load().embeddings.create(
            model=self._model_id, inputs=list(texts)
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._load().embeddings.create(model=self._model_id, inputs=[text])
        return response.data[0].embedding


def build_embedder(
    *, provider: str, model: str, api_key: str | None = None
) -> Embedder:
    """Build the embedder selected by configuration."""
    if provider == "fastembed":
        return FastEmbedEmbedder(model)
    if provider == "mistral":
        return MistralEmbedder(model, api_key or "")
    raise ValueError(
        f"unknown embedding provider {provider!r}; expected 'fastembed' or 'mistral'"
    )
