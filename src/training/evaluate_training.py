"""
src/training/evaluate_training.py
===================================
Post-training evaluation: validation loss convergence, perplexity,
and simple generation quality check.

Run:
    python -m src.training.evaluate_training
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

HISTORY_PATH = Path("artifacts/evaluation/training_history.json")
MODEL_DIR = Path("artifacts/model")
VAL_DIR = Path("data/validation/dataset")


def compute_perplexity_from_history(history: list[dict[str, Any]]) -> None:
    """Convert eval_loss to perplexity and print final values."""
    eval_entries = [e for e in history if "eval_loss" in e]
    if not eval_entries:
        logger.warning("No eval_loss entries found in training history.")
        return

    best = min(eval_entries, key=lambda x: x["eval_loss"])
    final = eval_entries[-1]

    logger.info("Best eval loss : %.4f (step %d)", best["eval_loss"], best["step"])
    logger.info("Best perplexity: %.2f", math.exp(best["eval_loss"]))
    logger.info("Final eval loss: %.4f (step %d)", final["eval_loss"], final["step"])
    logger.info("Final perplexity: %.2f", math.exp(final["eval_loss"]))

    # Check for overfitting: if training loss drops but eval doesn't follow
    train_entries = [e for e in history if "loss" in e and "eval_loss" not in e]
    if train_entries and eval_entries:
        final_train = train_entries[-1]["loss"]
        final_eval = final["eval_loss"]
        gap = final_eval - final_train
        if gap > 0.5:
            logger.warning(
                "Possible overfitting: train loss=%.4f, eval_loss=%.4f, gap=%.4f",
                final_train, final_eval, gap,
            )
        else:
            logger.info(
                "Train/eval gap looks healthy: train=%.4f, eval=%.4f",
                final_train, final_eval,
            )


def quick_generation_test(model_dir: Path) -> None:
    """
    Load the saved adapter and run a small set of generation tests
    to verify the model can produce coherent medical text.
    """
    if not model_dir.exists():
        logger.warning("Model directory %s not found. Skipping generation test.", model_dir)
        return

    logger.info("Loading model from %s for generation test …", model_dir)
    try:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        meta_path = model_dir / "training_metadata.json"
        base_name = "google/medgemma-4b-it"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            base_name = meta.get("model_name", base_name)

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        model.eval()

        test_questions = [
            "What are the main symptoms of type 2 diabetes?",
            "What is the difference between a cold and influenza?",
            "When should chest pain be treated as an emergency?",
        ]

        for q in test_questions:
            prompt = (
                "<start_of_turn>user\n"
                f"[System]: You are a medical AI assistant.\n\n{q}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            logger.info("\nQ: %s\nA: %s\n%s", q, response[:400], "-" * 60)

    except Exception as exc:
        logger.error("Generation test failed: %s", exc)


def main() -> None:
    # Load training history
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as fh:
            history = json.load(fh)
        compute_perplexity_from_history(history)
    else:
        logger.warning("Training history not found at %s", HISTORY_PATH)

    # Generation test
    quick_generation_test(MODEL_DIR)

    logger.info("Training evaluation complete.")


if __name__ == "__main__":
    main()
