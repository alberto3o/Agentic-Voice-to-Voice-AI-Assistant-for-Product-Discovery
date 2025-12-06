"""Embedding utilities for vector indexing.

Supports OpenAI embeddings by default with a "dummy" fallback for offline use.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable, Iterable, List, Literal, Sequence

import numpy as np

try:
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - chromadb should be installed via requirements
    embedding_functions = None  # type: ignore

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["openai", "dummy"]


class DummyEmbeddingFunction:
    """Deterministic embedding based on hashing text.

    This keeps the pipeline runnable without network credentials, but should be
    swapped for a real model (e.g., OpenAI or local sentence-transformers) for
    production quality search.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - simple deterministic behavior
        vectors: List[List[float]] = []
        for text in texts:
            # Create a deterministic hash-based vector
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeats = int(np.ceil(self.dim / len(digest)))
            raw = (digest * repeats)[: self.dim]
            vectors.append([b / 255.0 for b in raw])
        return vectors


def get_embedding_function(backend: EmbeddingBackend = "openai") -> Callable[[Sequence[str]], List[List[float]]]:
    """Return an embedding function compatible with Chroma.

    Parameters
    ----------
    backend: "openai" | "dummy"
        Embedding provider to use. OpenAI requires the ``OPENAI_API_KEY`` and
        ``EMBEDDING_MODEL_NAME`` environment variables.
    """

    if backend == "openai":
        if embedding_functions is None:
            raise ImportError("chromadb is required for OpenAI embedding support")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY or LLM_API_KEY must be set for OpenAI embeddings.")
        logger.info("Using OpenAI embeddings with model %s", model_name)
        return embedding_functions.OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)

    logger.warning("Using dummy embeddings; search quality will be poor. Set EMBEDDING_BACKEND=openai for real embeddings.")
    return DummyEmbeddingFunction()
