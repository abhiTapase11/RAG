"""Embedding backends.

The default backend is a local sentence-transformers model
(``BAAI/bge-small-en-v1.5``) so retrieval needs no API key. ``hash`` is a
dependency-free fallback (hashing trick over word + character n-grams) that
keeps the demo working on machines without torch; it is weaker but real.

Add a backend by registering a factory in ``_BACKENDS``.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from collections import Counter
from typing import Callable, Protocol, Sequence, runtime_checkable

from .config import Settings, get_settings
from .utils import tokenize

logger = logging.getLogger(__name__)

# bge-* models are trained with an instruction prefix on the query side only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings through sentence-transformers."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.name = f"sentence-transformers:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 6; keep working on older versions.
        dim_getter = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_getter())
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


class HashEmbedder:
    """Deterministic hashing-trick embedder - no model download, no torch.

    Words and character 4-grams are hashed into a fixed number of buckets with
    sub-linear term frequency weighting, then L2-normalised. Good enough to make
    hybrid retrieval work offline; swap in a real model for quality.
    """

    def __init__(self, dim: int = 768):
        self.name = f"hash:{dim}"
        self.dim = dim

    def _features(self, text: str) -> Counter:
        counts: Counter = Counter()
        words = tokenize(text, drop_stopwords=False)
        counts.update(words)
        counts.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
        squashed = re.sub(r"\s+", " ", text.lower())
        counts.update(squashed[i : i + 4] for i in range(0, max(0, len(squashed) - 3), 2))
        return counts

    @staticmethod
    def _stable_hash(feature: str, salt: int) -> int:
        # crc32, not hash(): Python randomises string hashing per process, and
        # these vectors are persisted on disk between runs.
        return zlib.crc32(feature.encode("utf-8"), salt)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, count in self._features(text).items():
            bucket = self._stable_hash(feature, 7919) % self.dim
            sign = 1.0 if self._stable_hash(feature, 104729) % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _make_sentence_transformers(cfg: Settings) -> Embedder:
    return SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_batch_size)


def _make_hash(cfg: Settings) -> Embedder:
    return HashEmbedder()


_BACKENDS: dict[str, Callable[[Settings], Embedder]] = {
    "sentence-transformers": _make_sentence_transformers,
    "st": _make_sentence_transformers,
    "hash": _make_hash,
}

_cache: dict[tuple[str, str], Embedder] = {}


def get_embedder(settings: Settings | None = None, allow_fallback: bool = True) -> Embedder:
    """Build (and cache) the configured embedder."""
    cfg = settings or get_settings()
    backend = cfg.embedding_backend.strip().lower()
    key = (backend, cfg.embedding_model)
    if key in _cache:
        return _cache[key]

    factory = _BACKENDS.get(backend)
    if factory is None:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND {cfg.embedding_backend!r}. "
            f"Available: {', '.join(sorted(set(_BACKENDS)))}"
        )
    try:
        embedder = factory(cfg)
    except Exception as exc:
        if not allow_fallback or backend == "hash":
            raise
        logger.warning(
            "Embedding backend %r unavailable (%s); falling back to the dependency-free "
            "'hash' backend. Install sentence-transformers for better retrieval.",
            cfg.embedding_backend,
            exc,
        )
        embedder = _make_hash(cfg)
    _cache[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    _cache.clear()
