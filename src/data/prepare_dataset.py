"""
src/data/prepare_dataset.py
============================
Preprocessing pipeline for ruslanmv/ai-medical-chatbot.

Steps:
  1. Load raw dataset
  2. Auto-detect question/answer columns
  3. Clean: remove nulls, empties, whitespace, corrupted records
  4. Remove exact duplicates
  5. Remove length outliers
  6. Format into conversation template for MedGemma
  7. Save to data/processed/

Run:
    python -m src.data.prepare_dataset
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console
from rich.progress import track

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
OUTPUT_DIR: Path = Path("data/processed")

QUESTION_ALIASES: list[str] = [
    "question", "query", "input", "prompt", "patient", "user",
    "Patient", "Query", "Question",
]
ANSWER_ALIASES: list[str] = [
    "answer", "response", "output", "doctor", "assistant",
    "Doctor", "Response", "Answer",
]

# Conservative length limits (characters)
MIN_QUESTION_LEN: int = 10
MAX_QUESTION_LEN: int = 3000
MIN_ANSWER_LEN: int = 20
MAX_ANSWER_LEN: int = 8000

# MedGemma system prompt
SYSTEM_PROMPT: str = (
    "You are a medical AI assistant providing educational health information. "
    "You always distinguish general information from personalised medical advice. "
    "You never invent medical facts, medications, dosages, or citations. "
    "You clearly express uncertainty when the evidence is limited. "
    "You always recommend that users seek professional medical care for diagnosis "
    "and treatment. For emergencies, you direct users to emergency services immediately."
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _detect_column(columns: list[str], aliases: list[str]) -> str:
    """Detect the first matching column from alias list; raise if not found."""
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ValueError(
        f"Could not auto-detect column. Tried aliases: {aliases}. "
        f"Available columns: {columns}"
    )


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines and strip leading/trailing whitespace."""
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    return text.strip()


def _is_corrupted(text: str) -> bool:
    """
    Heuristic check for corrupted / nonsense text.
    Returns True if the text looks corrupted.
    """
    if not text or not text.strip():
        return True
    # Excessive special character ratio
    special_ratio = len(re.findall(r"[^\w\s.,!?;:()\-\"']", text)) / max(len(text), 1)
    if special_ratio > 0.35:
        return True
    # Very high digit ratio (likely tables or code, not medical text)
    digit_ratio = len(re.findall(r"\d", text)) / max(len(text), 1)
    if digit_ratio > 0.6:
        return True
    return False


def _text_hash(text: str) -> str:
    """Return a short SHA-256 hash of the normalised text for dedup."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]


def _build_conversation(question: str, answer: str) -> str:
    """
    Build the conversation string in the format expected by MedGemma / Gemma chat template.

    MedGemma uses the standard Gemma instruction format:
      <start_of_turn>user\\n{message}<end_of_turn>
      <start_of_turn>model\\n{message}<end_of_turn>

    We also embed the system context at the top of the first user turn.
    """
    conversation = (
        f"<start_of_turn>user\n"
        f"[System]: {SYSTEM_PROMPT}\n\n"
        f"{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{answer}<end_of_turn>"
    )
    return conversation


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

class MedicalDatasetPreprocessor:
    """
    End-to-end preprocessing pipeline for the AI Medical Chatbot dataset.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        output_dir: Path = OUTPUT_DIR,
        min_question_len: int = MIN_QUESTION_LEN,
        max_question_len: int = MAX_QUESTION_LEN,
        min_answer_len: int = MIN_ANSWER_LEN,
        max_answer_len: int = MAX_ANSWER_LEN,
        max_examples: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.min_question_len = min_question_len
        self.max_question_len = max_question_len
        self.min_answer_len = min_answer_len
        self.max_answer_len = max_answer_len
        self.max_examples = max_examples

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.q_col: str = ""
        self.a_col: str = ""
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load the HuggingFace dataset and return as a DataFrame."""
        logger.info("Loading dataset: %s", self.dataset_name)
        ds = load_dataset(self.dataset_name)

        # Handle both Dataset and DatasetDict
        if isinstance(ds, DatasetDict):
            # Merge all splits — we will re-split later
            frames = [split.to_pandas() for split in ds.values()]
            df = pd.concat(frames, ignore_index=True)
            logger.info("Merged %d split(s) → %d rows", len(ds), len(df))
        else:
            df = ds.to_pandas()  # type: ignore[union-attr]
            logger.info("Single split → %d rows", len(df))

        self.stats["raw_rows"] = len(df)
        return df

    # ------------------------------------------------------------------
    def detect_columns(self, df: pd.DataFrame) -> None:
        """Auto-detect the question and answer columns."""
        columns = list(df.columns)
        self.q_col = _detect_column(columns, QUESTION_ALIASES)
        self.a_col = _detect_column(columns, ANSWER_ALIASES)
        console.print(
            f"[green]Detected columns[/green] — "
            f"Question: [cyan]{self.q_col!r}[/cyan]  "
            f"Answer: [cyan]{self.a_col!r}[/cyan]"
        )

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline."""
        original = len(df)
        logger.info("Starting cleaning. Rows: %d", original)

        # 1. Keep only relevant columns
        df = df[[self.q_col, self.a_col]].copy()
        df.columns = ["question", "answer"]

        # 2. Drop nulls
        df = df.dropna(subset=["question", "answer"])
        logger.info("After dropping nulls: %d rows", len(df))

        # 3. Normalize whitespace
        df["question"] = df["question"].astype(str).apply(_normalize_whitespace)
        df["answer"] = df["answer"].astype(str).apply(_normalize_whitespace)

        # 4. Drop empty after normalization
        df = df[df["question"].str.len() > 0]
        df = df[df["answer"].str.len() > 0]
        logger.info("After dropping empties: %d rows", len(df))

        # 5. Remove corrupted
        mask_ok = ~df["question"].apply(_is_corrupted) & ~df["answer"].apply(_is_corrupted)
        n_corrupt = (~mask_ok).sum()
        df = df[mask_ok].reset_index(drop=True)
        logger.info("Removed %d corrupted rows. Remaining: %d", n_corrupt, len(df))

        # 6. Length filtering
        q_len = df["question"].str.len()
        a_len = df["answer"].str.len()
        mask_len = (
            (q_len >= self.min_question_len)
            & (q_len <= self.max_question_len)
            & (a_len >= self.min_answer_len)
            & (a_len <= self.max_answer_len)
        )
        n_out_of_range = (~mask_len).sum()
        df = df[mask_len].reset_index(drop=True)
        logger.info(
            "Removed %d length-outlier rows. Remaining: %d",
            n_out_of_range, len(df),
        )

        # 7. Exact duplicate removal (question + answer)
        q_hash = df["question"].apply(_text_hash)
        a_hash = df["answer"].apply(_text_hash)
        combined_hash = q_hash + "_" + a_hash
        n_before = len(df)
        df = df[~combined_hash.duplicated()].reset_index(drop=True)
        logger.info(
            "Removed %d exact duplicates. Remaining: %d",
            n_before - len(df), len(df),
        )

        # 8. Remove near-identical questions (same q, different a) — keep first
        q_hash2 = df["question"].apply(_text_hash)
        n_before2 = len(df)
        df = df[~q_hash2.duplicated(keep="first")].reset_index(drop=True)
        logger.info(
            "Removed %d duplicate-question rows. Remaining: %d",
            n_before2 - len(df), len(df),
        )

        # 9. Cap examples if requested
        if self.max_examples and len(df) > self.max_examples:
            df = df.sample(n=self.max_examples, random_state=42).reset_index(drop=True)
            logger.info("Capped to %d examples.", self.max_examples)

        self.stats["cleaned_rows"] = len(df)
        self.stats["removed_nulls"] = original - len(df)
        return df

    # ------------------------------------------------------------------
    def format_conversations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Q/A pairs into the MedGemma conversation format."""
        logger.info("Formatting %d conversations …", len(df))
        rows = []
        for _, row in track(df.iterrows(), total=len(df), description="Formatting"):
            conv = _build_conversation(row["question"], row["answer"])
            rows.append({
                "question": row["question"],
                "answer": row["answer"],
                "text": conv,          # full conversation string for SFT
                "q_len": len(row["question"]),
                "a_len": len(row["answer"]),
            })
        out_df = pd.DataFrame(rows)
        logger.info("Formatted %d conversations.", len(out_df))
        return out_df

    # ------------------------------------------------------------------
    def save(self, df: pd.DataFrame) -> Path:
        """Save the processed dataset as a HuggingFace Dataset (parquet + arrow)."""
        out_path = self.output_dir / "medical_dataset_processed"
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.save_to_disk(str(out_path))

        # Also save as CSV for quick inspection
        csv_path = self.output_dir / "medical_dataset_processed.csv"
        df.to_csv(csv_path, index=False)

        logger.info("Processed dataset saved to %s", out_path)
        self.stats["output_path"] = str(out_path)
        return out_path

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full preprocessing pipeline."""
        console.rule("[bold blue]Medical Dataset Preprocessing")

        df = self.load()
        self.detect_columns(df)
        df_clean = self.clean(df)
        df_formatted = self.format_conversations(df_clean)
        out_path = self.save(df_formatted)

        # ── Report ──────────────────────────────────────────────────────
        console.rule("[bold green]Preprocessing Summary")
        console.print(f"  Raw rows:           {self.stats['raw_rows']:>10,}")
        console.print(f"  Cleaned rows:       {self.stats['cleaned_rows']:>10,}")
        retention = self.stats['cleaned_rows'] / max(self.stats['raw_rows'], 1) * 100
        console.print(f"  Retention rate:     {retention:>9.1f}%")
        console.print(f"  Output path:        {out_path}")
        console.print(
            f"\n  Question column:    {self.q_col!r}\n"
            f"  Answer column:      {self.a_col!r}"
        )

        return df_formatted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess the AI Medical Chatbot dataset."
    )
    parser.add_argument(
        "--dataset", default=DATASET_NAME, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR), help="Output directory"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum number of examples to keep (default: all)"
    )
    parser.add_argument(
        "--max-question-len", type=int, default=MAX_QUESTION_LEN
    )
    parser.add_argument(
        "--max-answer-len", type=int, default=MAX_ANSWER_LEN
    )
    args = parser.parse_args()

    preprocessor = MedicalDatasetPreprocessor(
        dataset_name=args.dataset,
        output_dir=Path(args.output_dir),
        max_examples=args.max_examples,
        max_question_len=args.max_question_len,
        max_answer_len=args.max_answer_len,
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
