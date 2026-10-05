"""
src/evaluation/evaluate_rag.py
================================
RAG-specific evaluation: retrieval quality, faithfulness, coverage.

Run:
    python -m src.evaluation.evaluate_rag
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
console = Console()

BENCHMARK_PATH = Path("src/evaluation/benchmark.json")
FAISS_INDEX = Path("artifacts/faiss/index.faiss")


def evaluate_retrieval(
    retriever: Any,
    benchmark_path: Path = BENCHMARK_PATH,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate retrieval quality on benchmark questions.

    For each question, checks whether retrieved documents contain
    at least one expected keyword (recall proxy).
    """
    if not benchmark_path.exists():
        logger.warning("Benchmark not found at %s", benchmark_path)
        return {}

    with open(benchmark_path) as fh:
        benchmark = json.load(fh)

    hits = 0
    total = 0
    results: list[dict] = []

    for item in benchmark:
        question = item.get("question", "")
        expected_keywords = item.get("expected_keywords", [])
        if not expected_keywords:
            continue

        docs = retriever.retrieve(question, top_k=top_k)
        combined_text = " ".join(d.get("text", "") for d in docs).lower()

        found = [kw for kw in expected_keywords if kw.lower() in combined_text]
        recall = len(found) / len(expected_keywords) if expected_keywords else 0
        hits += recall
        total += 1

        results.append({
            "question": question[:100],
            "expected_keywords": expected_keywords,
            "found_keywords": found,
            "recall": round(recall, 3),
            "n_retrieved": len(docs),
            "top_source": docs[0].get("source", "none") if docs else "none",
        })

    avg_recall = hits / max(total, 1)
    report = {
        "avg_keyword_recall": round(avg_recall, 3),
        "questions_evaluated": total,
        "top_k": top_k,
        "per_question": results,
    }
    console.print(f"\n[bold green]RAG Retrieval Evaluation[/bold green]")
    console.print(f"  Avg keyword recall: {avg_recall:.3f}")
    console.print(f"  Questions evaluated: {total}")
    return report


def main() -> None:
    if not FAISS_INDEX.exists():
        console.print("[red]FAISS index not found. Run: python -m src.rag.build_index[/red]")
        return

    from src.rag.retriever import MedicalRetriever
    retriever = MedicalRetriever()
    retriever.load()

    report = evaluate_retrieval(retriever, BENCHMARK_PATH, top_k=5)

    out_path = Path("artifacts/evaluation/rag_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    console.print(f"\nRAG evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
