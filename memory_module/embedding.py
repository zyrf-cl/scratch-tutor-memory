"""Embedders for the memory module.

Two implementations behind a common :class:`Embedder` Protocol:

* :class:`HashEmbedder` — zero-dependency hashed bag-of-tokens. Cheap,
  deterministic, but has no real semantic meaning. Used by the demo.
* :class:`STEmbedder` — wraps a ``sentence-transformers`` model.
  Real semantic vectors at the cost of a model download + torch
  dependency (install with ``uv sync --extra st``).

The tokenizer for HashEmbedder emits whitespace-separated words *and*
each CJK character, so the same hashing scheme gives non-trivial
overlap on both English and Chinese text — important because Chinese
is written without inter-word spaces.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterator, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DIM = 64


def _is_cjk(ch: str) -> bool:
    # CJK Unified Ideographs basic block — covers the Chinese characters
    # we care about for the demo.
    return "一" <= ch <= "鿿"


def _tokenize(text: str) -> Iterator[str]:
    """Yield both word-level and CJK-char-level tokens.

    For an English sentence we get the words. For Chinese we get the
    whitespace-bounded chunks (often the whole sentence) plus every
    individual CJK character — the character overlap is what makes
    semantic recall actually score above zero.
    """
    lowered = text.lower()
    for word in lowered.split():
        yield word
    for ch in lowered:
        if _is_cjk(ch):
            yield ch


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Bag-of-hashed-tokens embedder. Cheap, offline, deterministic."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        if not text:
            return vec
        for tok in _tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------- sentence-transformers backed ----------------


_DEFAULT_ST_MODEL = "BAAI/bge-small-zh-v1.5"


class STEmbedder:
    """Sentence-Transformers embedder.

    Loads the model lazily on first :meth:`embed` call so that just
    *constructing* an :class:`STEmbedder` doesn't trigger the (large)
    download — handy when the env-var toggle picks ST in a context
    where the model isn't actually available.

    Vectors are L2-normalised at emission so downstream cosine becomes
    a plain dot product.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_ST_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None  # lazy
        self._dim: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: uv sync --extra st"
            ) from exc
        logger.info(
            "STEmbedder: loading model=%s device=%s (first call may download weights)",
            self._model_name,
            self._device or "auto",
        )
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("STEmbedder: ready, dim=%d", self._dim)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        if not text:
            return np.zeros(self._dim or 0, dtype=np.float32)
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


class FallbackEmbedder:
    """Try a primary embedder and fall back if local model deps are absent."""

    def __init__(self, primary: Embedder, fallback: Embedder) -> None:
        self._primary = primary
        self._fallback = fallback
        self._using_fallback = False
        self._warned = False

    @property
    def dim(self) -> int:
        if self._using_fallback:
            return self._fallback.dim
        try:
            return self._primary.dim
        except Exception as exc:  # noqa: BLE001
            self._switch_to_fallback(exc)
            return self._fallback.dim

    def embed(self, text: str) -> np.ndarray:
        if self._using_fallback:
            return self._fallback.embed(text)
        try:
            return self._primary.embed(text)
        except Exception as exc:  # noqa: BLE001
            self._switch_to_fallback(exc)
            return self._fallback.embed(text)

    def _switch_to_fallback(self, exc: Exception) -> None:
        self._using_fallback = True
        if not self._warned:
            logger.warning(
                "embedding primary unavailable; falling back to %s: %s",
                type(self._fallback).__name__,
                exc,
            )
            self._warned = True


def build_embedder(name: str | None = None) -> Embedder:
    """Build the configured embedder.

    ``MEMORY_MODULE_EMBEDDER`` accepts ``bge``/``st`` or ``hash``.
    BGE is the default target. If local sentence-transformers/model
    dependencies are missing, it falls back to HashEmbedder unless
    ``MEMORY_MODULE_EMBEDDING_STRICT=1`` is set.
    """
    selected = (name or os.environ.get("MEMORY_MODULE_EMBEDDER") or "bge").lower()
    if selected in {"hash", "stub"}:
        return HashEmbedder()

    if selected in {"bge", "st", "sentence-transformers", "sentence_transformers"}:
        model_name = os.environ.get("MEMORY_MODULE_EMBEDDING_MODEL") or _DEFAULT_ST_MODEL
        device = os.environ.get("MEMORY_MODULE_EMBEDDING_DEVICE") or None
        primary = STEmbedder(model_name=model_name, device=device)
        strict = os.environ.get("MEMORY_MODULE_EMBEDDING_STRICT", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        return primary if strict else FallbackEmbedder(primary, HashEmbedder())

    raise ValueError(
        f"Unknown MEMORY_MODULE_EMBEDDER={selected!r}; expected 'bge' or 'hash'"
    )


__all__ = [
    "Embedder",
    "HashEmbedder",
    "STEmbedder",
    "FallbackEmbedder",
    "build_embedder",
    "cosine",
    "DEFAULT_DIM",
]
