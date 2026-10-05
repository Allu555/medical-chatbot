"""
src/rag/build_index.py
=======================
Build the FAISS vector index from the medical knowledge base.

Pipeline:
    Documents → Cleaning → Chunking → Embeddings → FAISS

Run:
    python -m src.rag.build_index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console
from rich.progress import track

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
console = Console()

CONFIG_PATH = Path("configs/rag.yaml")
FAISS_INDEX_PATH = Path("artifacts/faiss/index.faiss")
METADATA_PATH = Path("artifacts/faiss/metadata.json")


# ---------------------------------------------------------------------------
def _load_rag_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if path.exists():
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _load_embedding_model(model_name: str, normalize: bool = True):
    """Load a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def _embed_texts(
    model: Any,
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of texts and return as float32 numpy array."""
    logger.info("Embedding %d texts in batches of %d …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "Flat",
    nlist: int = 100,
):
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings:  (n, d) float32 array
        index_type:  "Flat" (exact) or "IVF" (approximate)
        nlist:       IVF number of clusters (unused for Flat)
    """
    import faiss

    dim = embeddings.shape[1]
    logger.info("Building FAISS %s index (dim=%d, n=%d) …", index_type, dim, len(embeddings))

    if index_type == "IVF":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(embeddings) // 4))
        index.train(embeddings)
    else:
        # Inner product (cosine similarity if embeddings are normalized)
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    logger.info("FAISS index built. Total vectors: %d", index.ntotal)
    return index


def save_index(index: Any, metadata: list[dict], index_path: Path, meta_path: Path) -> None:
    """Save the FAISS index and metadata to disk."""
    import faiss

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to %s  (%d records)", meta_path, len(metadata))


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index(
    config_path: Path = CONFIG_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    meta_path: Path = METADATA_PATH,
) -> None:
    """Full index build pipeline."""
    console.rule("[bold blue]RAG Index Builder")

    cfg = _load_rag_config(config_path)
    embed_cfg = cfg.get("embedding", {})
    faiss_cfg = cfg.get("faiss", {})

    embedding_model_name: str = embed_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    batch_size: int = embed_cfg.get("batch_size", 64)
    normalize: bool = embed_cfg.get("normalize_embeddings", True)
    index_type: str = faiss_cfg.get("index_type", "Flat")
    nlist: int = faiss_cfg.get("nlist", 100)

    # ── Load knowledge base ───────────────────────────────────────────────
    from src.rag.knowledge_base import MedicalKnowledgeBase

    kb = MedicalKnowledgeBase(config_path=config_path)

    # Add foundational docs (idempotent)
    kb.add_synthetic_medical_documents()

    # Load all documents
    docs = kb.load_json_documents()
    if not docs:
        logger.error("No documents found. Ensure knowledge_base/documents/ contains JSON files.")
        return

    # Chunk
    chunks = kb.chunk_documents(docs)
    kb.save_chunks(chunks)

    console.print(f"[green]Total chunks to embed: {len(chunks):,}[/green]")

    # ── Embed ─────────────────────────────────────────────────────────────
    embed_model = _load_embedding_model(embedding_model_name, normalize=normalize)
    texts = [c.text for c in chunks]
    embeddings = _embed_texts(embed_model, texts, batch_size=batch_size, normalize=normalize)

    # ── Build FAISS ───────────────────────────────────────────────────────
    faiss_index = build_faiss_index(embeddings, index_type=index_type, nlist=nlist)

    # ── Save ──────────────────────────────────────────────────────────────
    metadata = [c.to_dict() for c in chunks]
    save_index(faiss_index, metadata, index_path, meta_path)

    console.rule("[bold green]Index Build Complete")
    console.print(f"  Chunks indexed: {len(chunks):,}")
    console.print(f"  Embedding dim:  {embeddings.shape[1]}")
    console.print(f"  FAISS index:    {index_path}")
    console.print(f"  Metadata:       {meta_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS RAG index.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--index-path", default=str(FAISS_INDEX_PATH))
    parser.add_argument("--meta-path", default=str(METADATA_PATH))
    args = parser.parse_args()

    build_index(
        config_path=Path(args.config),
        index_path=Path(args.index_path),
        meta_path=Path(args.meta_path),
    )
