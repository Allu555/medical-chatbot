"""
src/data/inspect_dataset.py
============================
Automated dataset inspection for ruslanmv/ai-medical-chatbot.

Run:
    python -m src.data.inspect_dataset
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME: str = "ruslanmv/ai-medical-chatbot"
REPORT_PATH: Path = Path("artifacts/evaluation/dataset_inspection_report.txt")

# Heuristic column aliases — the actual columns are auto-detected.
QUESTION_ALIASES: list[str] = [
    "question", "query", "input", "prompt", "patient", "user",
    "Patient", "Query", "Question",
]
ANSWER_ALIASES: list[str] = [
    "answer", "response", "output", "doctor", "assistant",
    "Doctor", "Response", "Answer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_column(columns: list[str], aliases: list[str]) -> str | None:
    """Return the first column name that matches one of the aliases."""
    lower_col = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower_col:
            return lower_col[alias.lower()]
    return None


def _text_stats(series: pd.Series) -> dict[str, Any]:
    """Compute basic statistics for a text column."""
    lengths = series.dropna().str.len()
    word_counts = series.dropna().str.split().str.len()
    return {
        "count_non_null": int(lengths.count()),
        "count_null": int(series.isna().sum()),
        "count_empty": int((series.str.strip() == "").sum()),
        "char_min": int(lengths.min()) if len(lengths) else 0,
        "char_max": int(lengths.max()) if len(lengths) else 0,
        "char_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "char_median": float(lengths.median()) if len(lengths) else 0.0,
        "word_min": int(word_counts.min()) if len(word_counts) else 0,
        "word_max": int(word_counts.max()) if len(word_counts) else 0,
        "word_mean": float(word_counts.mean()) if len(word_counts) else 0.0,
    }


def _count_duplicates(series: pd.Series) -> int:
    """Count rows with duplicated (non-null) text."""
    non_null = series.dropna()
    return int(non_null.duplicated().sum())


def _sample_extremes(df: pd.DataFrame, col: str, n: int = 3) -> dict[str, list[str]]:
    """Return the shortest and longest n samples from a column."""
    lengths = df[col].dropna().str.len()
    idx_short = lengths.nsmallest(n).index.tolist()
    idx_long = lengths.nlargest(n).index.tolist()
    return {
        "shortest": df.loc[idx_short, col].tolist(),
        "longest": [s[:300] + " …" if len(s) > 300 else s
                    for s in df.loc[idx_long, col].tolist()],
    }


# ---------------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_name: str = DATASET_NAME) -> dict[str, Any]:
    """
    Load and inspect the HuggingFace dataset.

    Returns a report dictionary containing all findings.
    """
    console.rule("[bold blue]AI Medical Chatbot — Dataset Inspection")
    logger.info("Loading dataset: %s", dataset_name)

    ds: DatasetDict | Dataset = load_dataset(dataset_name)

    # ── Basic structure ────────────────────────────────────────────────────
    console.print("\n[bold yellow]Dataset object:[/bold yellow]")
    console.print(repr(ds))

    report: dict[str, Any] = {"dataset_name": dataset_name, "splits": {}}

    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    for split_name, split_ds in ds.items():
        console.rule(f"[bold green]Split: {split_name}")
        logger.info("Processing split: %s  (%d rows)", split_name, len(split_ds))

        df = split_ds.to_pandas()

        # ── Schema ────────────────────────────────────────────────────────
        console.print(f"\n[bold]Columns ({len(df.columns)}):[/bold] {list(df.columns)}")
        console.print(f"[bold]Row count:[/bold] {len(df):,}")

        # ── Auto-detect Q/A columns ───────────────────────────────────────
        q_col = _detect_column(list(df.columns), QUESTION_ALIASES)
        a_col = _detect_column(list(df.columns), ANSWER_ALIASES)
        console.print(f"[bold]Detected question column:[/bold] {q_col!r}")
        console.print(f"[bold]Detected answer column:[/bold]   {a_col!r}")

        split_report: dict[str, Any] = {
            "n_rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
            "question_col": q_col,
            "answer_col": a_col,
        }

        # ── Missing values ─────────────────────────────────────────────────
        null_counts = df.isnull().sum().to_dict()
        console.print("\n[bold]Null counts per column:[/bold]")
        for col, cnt in null_counts.items():
            color = "red" if cnt > 0 else "green"
            console.print(f"  {col}: [{color}]{cnt}[/{color}]")
        split_report["null_counts"] = {k: int(v) for k, v in null_counts.items()}

        # ── Per-column text stats ──────────────────────────────────────────
        text_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
        stats_table = Table(title="Text Column Statistics", show_header=True)
        stats_table.add_column("Column")
        stats_table.add_column("Non-null", justify="right")
        stats_table.add_column("Empty", justify="right")
        stats_table.add_column("Char min", justify="right")
        stats_table.add_column("Char max", justify="right")
        stats_table.add_column("Char mean", justify="right")
        stats_table.add_column("Word mean", justify="right")

        col_stats: dict[str, Any] = {}
        for col in text_cols:
            s = _text_stats(df[col])
            col_stats[col] = s
            stats_table.add_row(
                col,
                str(s["count_non_null"]),
                str(s["count_empty"]),
                str(s["char_min"]),
                str(s["char_max"]),
                f"{s['char_mean']:.0f}",
                f"{s['word_mean']:.1f}",
            )
        console.print(stats_table)
        split_report["text_stats"] = col_stats

        # ── Duplicate detection ────────────────────────────────────────────
        total_dups = int(df.duplicated().sum())
        console.print(f"\n[bold]Exact full-row duplicates:[/bold] {total_dups:,}")
        split_report["full_row_duplicates"] = total_dups

        for col in [q_col, a_col]:
            if col:
                ndups = _count_duplicates(df[col])
                console.print(
                    f"  Duplicate values in [{col!r}]: [yellow]{ndups:,}[/yellow]"
                )
                split_report[f"duplicates_{col}"] = ndups

        # ── Sample records ─────────────────────────────────────────────────
        console.print("\n[bold]First 3 records:[/bold]")
        for i, row in df.head(3).iterrows():
            console.print(f"\n  --- Record {i} ---")
            for col in df.columns:
                val = str(row[col])
                val_display = val[:200] + " …" if len(val) > 200 else val
                console.print(f"  {col}: {val_display}")

        # ── Extreme examples ──────────────────────────────────────────────
        for col in [q_col, a_col]:
            if col:
                extremes = _sample_extremes(df, col, n=2)
                console.print(f"\n[bold]Shortest [{col}]:[/bold]")
                for ex in extremes["shortest"]:
                    console.print(f"  {ex!r}")
                console.print(f"[bold]Longest  [{col}] (truncated):[/bold]")
                for ex in extremes["longest"]:
                    console.print(f"  {ex!r}")
                split_report[f"extremes_{col}"] = extremes

        # ── Unusually long examples (>95th percentile) ─────────────────────
        if q_col:
            threshold_q = df[q_col].str.len().quantile(0.95)
            n_long_q = int((df[q_col].str.len() > threshold_q).sum())
            console.print(
                f"\n[bold]Questions above 95th-percentile length "
                f"({threshold_q:.0f} chars):[/bold] {n_long_q:,}"
            )
            split_report["long_questions_95pct"] = n_long_q

        if a_col:
            threshold_a = df[a_col].str.len().quantile(0.95)
            n_long_a = int((df[a_col].str.len() > threshold_a).sum())
            console.print(
                f"[bold]Answers above 95th-percentile length "
                f"({threshold_a:.0f} chars):[/bold] {n_long_a:,}"
            )
            split_report["long_answers_95pct"] = n_long_a

        # ── Empty Q/A ─────────────────────────────────────────────────────
        for col in [q_col, a_col]:
            if col:
                n_empty = int(
                    df[col].isna().sum() + (df[col].str.strip() == "").sum()
                )
                console.print(
                    f"[bold]Empty/null [{col}]:[/bold] "
                    f"[{'red' if n_empty else 'green'}]{n_empty}[/]"
                )
                split_report[f"empty_{col}"] = n_empty

        report["splits"][split_name] = split_report

    # ── Summary ───────────────────────────────────────────────────────────
    console.rule("[bold]Inspection Summary")
    for split_name, sr in report["splits"].items():
        console.print(
            f"  {split_name}: {sr['n_rows']:,} rows | "
            f"Q col: {sr['question_col']!r} | A col: {sr['answer_col']!r}"
        )

    # ── Save report text ──────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        import json
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Inspection report saved to %s", REPORT_PATH)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = inspect_dataset()
    console.print("\n[bold green][OK] Inspection complete.[/bold green]")
    console.print(f"  Report saved to: {REPORT_PATH}")
