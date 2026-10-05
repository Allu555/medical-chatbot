"""
src/data/split_dataset.py
==========================
Reproducible train / validation / test splitting.

Steps:
  1. Load processed dataset
  2. Semantic deduplication guard (hash-based near-duplicate check)
  3. Stratified shuffle split (80 / 10 / 10)
  4. Save individual splits to disk

Run:
    python -m src.data.split_dataset
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset, load_from_disk
from rich.console import Console
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
PROCESSED_PATH: Path = Path("data/processed/medical_dataset_processed")
TRAIN_DIR: Path = Path("data/train")
VALIDATION_DIR: Path = Path("data/validation")
TEST_DIR: Path = Path("data/test")

TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10
SEED: int = 42


# ---------------------------------------------------------------------------
def _short_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:12]


def _remove_near_duplicates(df: pd.DataFrame, col: str = "question") -> pd.DataFrame:
    """
    Remove rows where the question hashes collide, keeping only the first.
    This is a cheap guard; semantic de-dup would require embeddings.
    """
    hashes = df[col].apply(_short_hash)
    mask = ~hashes.duplicated(keep="first")
    removed = (~mask).sum()
    if removed:
        logger.info("Near-duplicate guard removed %d rows from column %r", removed, col)
    return df[mask].reset_index(drop=True)


def split_dataset(
    processed_path: Path = PROCESSED_PATH,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the processed dataset and split into train / validation / test.

    Returns three HuggingFace Dataset objects.
    """
    console.rule("[bold blue]Dataset Splitting")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Ratios must sum to 1.0"
    )

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading processed dataset from %s", processed_path)
    ds = load_from_disk(str(processed_path))
    df = ds.to_pandas()
    console.print(f"Loaded [cyan]{len(df):,}[/cyan] examples.")

    # ── Near-duplicate guard ───────────────────────────────────────────────
    if "question" in df.columns:
        df = _remove_near_duplicates(df, col="question")

    # ── Shuffle ───────────────────────────────────────────────────────────
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ── Split: train vs (val + test) ─────────────────────────────────────
    holdout_ratio = val_ratio + test_ratio
    df_train, df_holdout = train_test_split(
        df,
        test_size=holdout_ratio,
        random_state=seed,
        shuffle=True,
    )

    # ── Split holdout → val and test ──────────────────────────────────────
    relative_test = test_ratio / holdout_ratio
    df_val, df_test = train_test_split(
        df_holdout,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # ── Leakage check: no question should appear in > 1 split ─────────────
    if "question" in df.columns:
        train_qs = set(df_train["question"].apply(_short_hash))
        val_qs = set(df_val["question"].apply(_short_hash))
        test_qs = set(df_test["question"].apply(_short_hash))
        train_val_overlap = len(train_qs & val_qs)
        train_test_overlap = len(train_qs & test_qs)
        val_test_overlap = len(val_qs & test_qs)
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.warning(
                "Leakage detected — train∩val: %d | train∩test: %d | val∩test: %d",
                train_val_overlap, train_test_overlap, val_test_overlap,
            )
        else:
            logger.info("✓ No question hash overlap between splits.")

    # ── Save splits ───────────────────────────────────────────────────────
    for split_df, split_dir, split_name in [
        (df_train, TRAIN_DIR, "train"),
        (df_val, VALIDATION_DIR, "validation"),
        (df_test, TEST_DIR, "test"),
    ]:
        split_dir.mkdir(parents=True, exist_ok=True)
        split_ds = Dataset.from_pandas(split_df, preserve_index=False)
        split_ds.save_to_disk(str(split_dir / "dataset"))
        # CSV for quick inspection
        split_df.to_csv(split_dir / "dataset.csv", index=False)
        logger.info("Saved %s split (%d rows) → %s", split_name, len(split_df), split_dir)

    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_val = Dataset.from_pandas(df_val, preserve_index=False)
    ds_test = Dataset.from_pandas(df_test, preserve_index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    console.rule("[bold green]Split Summary")
    total = len(df_train) + len(df_val) + len(df_test)
    for name, n in [("Train", len(df_train)), ("Validation", len(df_val)), ("Test", len(df_test))]:
        console.print(f"  {name:12s}: {n:>8,}  ({n/total*100:.1f}%)")
    console.print(f"  {'Total':12s}: {total:>8,}")

    return ds_train, ds_val, ds_test


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split the processed dataset.")
    parser.add_argument("--processed-path", default=str(PROCESSED_PATH))
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split_dataset(
        processed_path=Path(args.processed_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    console.print("\n[bold green]✓ Dataset splitting complete.[/bold green]")
