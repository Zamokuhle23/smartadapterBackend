"""
Embedding providers.

- OpenAIEmbedder: uses the configured OpenAI-compatible endpoint when a key is set.
- HashingEmbedder: deterministic feature-hashing bag-of-words embedder used for
  local/offline development so the whole RAG pipeline runs with zero external
  dependencies. Not semantically deep, but stable and useful for testing.
"""

import hashlib
import math
import re
from urllib import request as http_request

from django.conf import settings

WORD_RE = re.compile(r"[a-zA-Z']+")


class BaseEmbedder:
    dim = settings.EMBEDDING_DIM

    def embed_texts(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str):
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """Feature-hashing bag-of-words vector, L2-normalised."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.EMBEDDING_DIM

    def embed_query(self, text: str):
        vec = [0.0] * self.dim
        for word in WORD_RE.findall(text.lower()):
            digest = hashlib.md5(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder(BaseEmbedder):
    """
    Calls POST {EMBEDDINGS_BASE_URL}/embeddings (OpenAI-compatible).
    Uses EMBEDDINGS_* settings - NOT the chat LLM settings, because OpenRouter
    does not serve an embeddings endpoint.
    """

    model = "text-embedding-3-small"

    def _post(self, payload: dict) -> dict:
        req = http_request.Request(
            f"{settings.EMBEDDINGS_BASE_URL.rstrip('/')}/embeddings",
            data=__import__("json").dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.EMBEDDINGS_API_KEY}",
            },
            method="POST",
        )
        with http_request.urlopen(req, timeout=60) as resp:
            import json

            return json.loads(resp.read().decode("utf-8"))

    def embed_texts(self, texts):
        response = self._post({"model": self.model, "input": texts})
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text: str):
        return self.embed_texts([text])[0]


class LocalEmbedder(BaseEmbedder):
    """
    In-process open-source model via sentence-transformers (CPU-friendly).

    Default model: BAAI/bge-small-en-v1.5 (384-dim, MIT). Swap via
    EMBEDDING_MODEL_NAME; E5 models work too - set the EMBEDDING_*_PREFIX
    settings ("query: " / "passage: ") which they require.
    The model is loaded lazily once per process and cached on the class.
    """

    _model = None

    def __init__(self):
        self.query_prefix = settings.EMBEDDING_QUERY_PREFIX
        self.passage_prefix = settings.EMBEDDING_PASSAGE_PREFIX

    @classmethod
    def _load(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls._model

    def embed_texts(self, texts):
        """Embed passages (used at ingestion time)."""
        model = self._load()
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=32,
            normalize_embeddings=True,   # cosine-ready
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str):
        """Embed a query (chat path - latency sensitive)."""
        model = self._load()
        vector = model.encode(
            [f"{self.query_prefix}{text}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()


def get_embedder() -> BaseEmbedder:
    provider = settings.EMBEDDING_PROVIDER
    if provider == "openai" and settings.EMBEDDINGS_API_KEY:
        return OpenAIEmbedder()
    if provider == "local":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "EMBEDDING_PROVIDER=local but sentence-transformers is not installed; "
                "falling back to HashingEmbedder. Install it with: "
                "pip install sentence-transformers"
            )
            return HashingEmbedder()
        return LocalEmbedder()
    return HashingEmbedder()
