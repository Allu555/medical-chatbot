"""
src/rag/retriever.py
=====================
FAISS-based retriever for the medical RAG pipeline.

Features:
  - Query embedding + FAISS search
  - Score threshold filtering
  - Optional cross-encoder reranking
  - Configurable top-k
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/rag.yaml")
FAISS_INDEX_PATH = Path("artifacts/faiss/index.faiss")
METADATA_PATH = Path("artifacts/faiss/metadata.json")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MedicalRetriever:
    """
    Retrieve relevant medical documents for a given query using FAISS.
    """

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = METADATA_PATH,
    ) -> None:
        self.config_path = config_path
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        cfg = self._load_config()
        embed_cfg = cfg.get("embedding", {})
        retrieval_cfg = cfg.get("retrieval", {})

        self.embedding_model_name: str = embed_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        self.normalize: bool = embed_cfg.get("normalize_embeddings", True)
        self.top_k: int = retrieval_cfg.get("top_k", 5)
        self.score_threshold: float = retrieval_cfg.get("score_threshold", 0.35)
        self.use_rerank: bool = retrieval_cfg.get("rerank", True)

        self._embed_model: Any = None
        self._rerank_model: Any = None
        self._index: Any = None
        self._metadata: list[dict] = []
        self._tfidf_vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._is_tfidf_fallback: bool = False

    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as fh:
                return yaml.safe_load(fh) or {}
        return {}

    # ------------------------------------------------------------------
    def _init_tfidf_fallback(self) -> None:
        """Initialize TF-IDF fallback retriever if FAISS/PyTorch is blocked by OS."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not self._metadata and self.meta_path.exists():
            with open(self.meta_path, encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in self._metadata]
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        self._is_tfidf_fallback = True
        logger.info("TF-IDF fallback retriever initialized with %d documents.", len(self._metadata))

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the FAISS index, metadata, and embedding model, falling back to TF-IDF if needed."""
        if not self.index_path.exists():
            logger.warning(
                "FAISS index not found at %s. Attempting TF-IDF fallback.",
                self.index_path
            )
            self._init_tfidf_fallback()
            return

        # Load metadata first
        if self.meta_path.exists():
            with open(self.meta_path, encoding="utf-8") as fh:
                self._metadata = json.load(fh)

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            logger.info("Loading FAISS index from %s …", self.index_path)
            self._index = faiss.read_index(str(self.index_path))

            logger.info(
                "Index loaded: %d vectors | Metadata: %d records",
                self._index.ntotal, len(self._metadata),
            )

            logger.info("Loading embedding model: %s …", self.embedding_model_name)
            self._embed_model = SentenceTransformer(self.embedding_model_name)

            if self.use_rerank:
                try:
                    from sentence_transformers import CrossEncoder
                    self._rerank_model = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        max_length=512,
                    )
                    logger.info("Cross-encoder reranker loaded.")
                except Exception as exc:
                    logger.warning("Could not load cross-encoder: %s. Reranking disabled.", exc)
                    self.use_rerank = False
        except Exception as exc:
            logger.warning("FAISS or neural embedder unavailable (%s). Falling back to TF-IDF retriever.", exc)
            self._init_tfidf_fallback()

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._index is None and not self._is_tfidf_fallback:
            self.load()

    # ------------------------------------------------------------------
    def _normalize_query(self, query: str) -> str:
        """Basic query normalization."""
        import re
        query = re.sub(r"\s+", " ", query).strip()
        # Remove trailing question mark for embedding (slightly helps recall)
        return query.rstrip("?")

    # ------------------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embedding = self._embed_model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k relevant documents for the query.

        Args:
            query:  User's medical question.
            top_k:  Override default top-k if provided.

        Returns:
            List of document dicts with added 'score' key, ordered by relevance.
        """
        self._ensure_loaded()

        k = top_k or self.top_k

        if self._is_tfidf_fallback:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]
            top_indices = scores.argsort()[::-1]
            candidates: list[dict[str, Any]] = []
            for idx in top_indices:
                score = float(scores[idx])
                if score < 0.05 and len(candidates) >= 1:
                    break
                doc = dict(self._metadata[idx])
                doc["score"] = score
                candidates.append(doc)
                if len(candidates) >= k:
                    break
            return candidates

        # Retrieve more candidates if reranking
        search_k = min(k * 3 if self.use_rerank else k, self._index.ntotal)

        # Normalize and embed query
        norm_query = self._normalize_query(query)
        query_emb = self._embed_query(norm_query)

        # FAISS search
        scores, indices = self._index.search(query_emb, search_k)
        scores = scores[0]
        indices = indices[0]

        # Build candidate list
        candidates: list[dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if float(score) < self.score_threshold:
                continue
            doc = dict(self._metadata[idx])
            doc["score"] = float(score)
            candidates.append(doc)

        if not candidates:
            logger.info("No documents above score threshold %.2f for query: %r",
                        self.score_threshold, query[:80])
            return []

        # Rerank if enabled
        if self.use_rerank and self._rerank_model is not None and len(candidates) > 1:
            candidates = self._rerank(query, candidates)

        return candidates[:k]

    # ------------------------------------------------------------------
    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cross-encoder reranking of retrieved candidates."""
        try:
            pairs = [(query, doc["text"][:512]) for doc in candidates]
            rerank_scores = self._rerank_model.predict(pairs)
            for doc, rs in zip(candidates, rerank_scores):
                doc["rerank_score"] = float(rs)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            logger.debug("Reranked %d candidates.", len(candidates))
        except Exception as exc:
            logger.warning("Reranking failed: %s. Using original order.", exc)
        return candidates

    # ------------------------------------------------------------------
    def retrieve_context_string(self, query: str, top_k: int | None = None) -> str:
        """
        Convenience method that returns retrieved context as a single string
        suitable for prompt injection.
        """
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            header = f"[Source {i}: {doc.get('title', 'Unknown')} — {doc.get('source', '')}]"
            parts.append(f"{header}\n{doc['text']}")
        return "\n\n---\n\n".join(parts)
