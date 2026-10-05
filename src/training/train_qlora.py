"""
src/training/train_qlora.py
============================
QLoRA fine-tuning of MedGemma 4B on the AI Medical Chatbot dataset.

Features:
  - Auto-detects GPU, VRAM, bfloat16/float16 support
  - 4-bit NF4 quantization via BitsAndBytes
  - LoRA/QLoRA via PEFT with auto-detected target modules
  - SFTTrainer from TRL
  - Gradient checkpointing + paged optimizer
  - Evaluation on validation set during training
  - Best-model checkpoint selection
  - Training curve saved to artifacts/evaluation/

Run:
    python -m src.training.train_qlora
    python -m src.training.train_qlora --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from datasets import Dataset, load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
load_dotenv()

# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Path = Path("configs/training.yaml")
TRAIN_DIR: Path = Path("data/train/dataset")
VALIDATION_DIR: Path = Path("data/validation/dataset")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

class HardwareInfo:
    """Detect and report available hardware."""

    def __init__(self) -> None:
        self.cuda_available: bool = torch.cuda.is_available()
        self.gpu_name: str = ""
        self.vram_gb: float = 0.0
        self.bf16_supported: bool = False
        self.fp16_supported: bool = False
        self.device: str = "cpu"

        if self.cuda_available:
            self.device = "cuda"
            self.gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            self.vram_gb = total_mem / (1024 ** 3)
            self.bf16_supported = torch.cuda.is_bf16_supported()
            self.fp16_supported = True  # all modern CUDA GPUs support fp16

    def print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("  Device    : %s", self.device.upper())
        logger.info("  GPU       : %s", self.gpu_name or "N/A")
        logger.info("  VRAM      : %.1f GB", self.vram_gb)
        logger.info("  CUDA      : %s", torch.version.cuda or "N/A")
        logger.info("  BF16      : %s", self.bf16_supported)
        logger.info("  FP16      : %s", self.fp16_supported)
        logger.info("=" * 60)

        if not self.cuda_available:
            logger.error(
                "CUDA is not available. Training a 4B model on CPU is extremely slow "
                "and is not recommended. Please use a CUDA-capable GPU."
            )

        if self.cuda_available and self.vram_gb < 10:
            logger.warning(
                "VRAM is %.1f GB, which may be insufficient for 4B QLoRA training. "
                "Recommendations: reduce batch_size to 1, max_length to 512, "
                "gradient_accumulation_steps to 8, lora_rank to 8.",
                self.vram_gb,
            )

    def recommend_config_adjustments(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Return adjusted config keys if hardware is constrained."""
        adjustments: dict[str, Any] = {}
        if self.vram_gb < 10 and self.cuda_available:
            adjustments.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 512,
                "lora_r": 8,
            })
            logger.warning("Applying low-VRAM config adjustments: %s", adjustments)
        return adjustments


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config and return as a flat-ish dictionary."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def build_bnb_config(cfg: dict[str, Any], hw: HardwareInfo) -> BitsAndBytesConfig | None:
    """Build BitsAndBytesConfig with automatic dtype fallback."""
    try:
        import bitsandbytes
    except ImportError:
        logger.warning("bitsandbytes not found, falling back to non-quantized model.")
        return None

    quant_cfg = cfg.get("quantization", {})
    if hw.bf16_supported:
        compute_dtype = torch.bfloat16
        dtype_name = "bfloat16"
    elif hw.fp16_supported:
        compute_dtype = torch.float16
        dtype_name = "float16"
    else:
        compute_dtype = torch.float32
        dtype_name = "float32"
    logger.info("Using compute dtype: %s", dtype_name)

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def _find_target_modules(model: torch.nn.Module) -> list[str]:
    """
    Programmatically identify LoRA target modules from the loaded model.
    Targets linear projection layers in attention blocks.
    """
    target_keywords = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                       "up_proj", "down_proj"}
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short_name = name.split(".")[-1]
            if short_name in target_keywords:
                found.add(short_name)
    if not found:
        logger.warning(
            "Could not detect standard target modules; falling back to broad search."
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "embed" not in name:
                found.add(name.split(".")[-1])
    logger.info("Detected LoRA target modules: %s", sorted(found))
    return sorted(found)


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None,
    max_length: int,
) -> tuple[Any, Any]:
    """Load the base model and tokenizer."""
    logger.info("Loading tokenizer: %s", model_name)
    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model: %s (this may take several minutes) …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None,
        trust_remote_code=True,
        torch_dtype="auto",
        token=token,
    )

    logger.info("Model architecture: %s", type(model).__name__)
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def build_lora_config(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> LoraConfig:
    """Create LoraConfig with auto-detected target modules."""
    lora_cfg = cfg.get("lora", {})
    target_modules = _find_target_modules(model)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return lora_config


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print total, trainable, and percentage of trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total > 0 else 0
    logger.info("Total parameters:     %s", f"{total:,}")
    logger.info("Trainable parameters: %s", f"{trainable:,}")
    logger.info("Trainable %%:          %.4f%%", pct)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TrainingCurveCallback(TrainerCallback):
    """Save training/validation loss history for plotting."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.history: list[dict[str, Any]] = []

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            entry = {"step": state.global_step, "epoch": state.epoch}
            entry.update(logs)
            self.history.append(entry)

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._save()
        self._plot()

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "training_history.json"
        with open(out, "w") as fh:
            json.dump(self.history, fh, indent=2)
        logger.info("Training history saved to %s", out)

    def _plot(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [e["step"] for e in self.history]
            train_loss = [e.get("loss") for e in self.history]
            eval_loss = [e.get("eval_loss") for e in self.history]

            fig, ax = plt.subplots(figsize=(10, 5))
            if any(v is not None for v in train_loss):
                t_steps = [s for s, v in zip(steps, train_loss) if v is not None]
                t_vals = [v for v in train_loss if v is not None]
                ax.plot(t_steps, t_vals, label="Train Loss", color="steelblue")
            if any(v is not None for v in eval_loss):
                e_steps = [s for s, v in zip(steps, eval_loss) if v is not None]
                e_vals = [v for v in eval_loss if v is not None]
                ax.plot(e_steps, e_vals, label="Validation Loss",
                        color="coral", linestyle="--")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("MedGemma QLoRA Training Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)
            out_png = self.output_dir / "training_curves.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Training curves saved to %s", out_png)
        except Exception as exc:
            logger.warning("Could not plot training curves: %s", exc)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: Path = DEFAULT_CONFIG, overrides: dict[str, Any] | None = None) -> None:
    """Main training entry point."""
    start_time = time.time()
    cfg = load_config(config_path)
    if overrides:
        for key, val in overrides.items():
            cfg = _set_nested(cfg, key, val)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lora_cfg_raw = cfg.get("lora", {})

    model_name: str = model_cfg.get("base_model_name", "google/medgemma-4b-it")
    max_length: int = model_cfg.get("max_length", 1024)
    output_dir: Path = Path(train_cfg.get("output_dir", "artifacts/checkpoints"))
    final_model_dir: Path = Path(train_cfg.get("final_model_dir", "artifacts/model"))
    eval_dir: Path = Path("artifacts/evaluation")

    hw = HardwareInfo()
    hw.print_summary()
    hw_adj = hw.recommend_config_adjustments(cfg)
    if hw_adj:
        for k, v in hw_adj.items():
            if k in train_cfg:
                train_cfg[k] = v
            elif k in lora_cfg_raw:
                lora_cfg_raw[k] = v
            else:
                train_cfg[k] = v

    bnb_config = build_bnb_config(cfg, hw)
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config, max_length)

    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )

    lora_config = build_lora_config(model, cfg)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    ds_train = load_from_disk(str(TRAIN_DIR))
    ds_val = load_from_disk(str(VALIDATION_DIR))

    num_train_epochs = train_cfg.get("num_train_epochs", 3)
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    total_steps = (len(ds_train) // int(train_cfg.get("per_device_train_batch_size", 2))) * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "paged_adamw_8bit" if bnb_config else "adamw_torch"),
        bf16=hw.bf16_supported,
        fp16=(not hw.bf16_supported) and hw.fp16_supported,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=int(train_cfg.get("logging_steps", 25)),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=train_cfg.get("report_to", "none"),
        dataset_text_field="text",
        max_length=max_length,              # TRL 1.12: was max_seq_length
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        callbacks=[TrainingCurveCallback(output_dir=eval_dir)],
    )

    trainer.train()
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    elapsed = time.time() - start_time
    logger.info("Training complete in %s", _fmt_duration(elapsed))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> dict:
    """Set a dotted-key in a nested dict, e.g. 'training.learning_rate'."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning of MedGemma 4B")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument(
        "--set", nargs="+", metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.learning_rate=1e-4",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.set:
        for kv in args.set:
            k, _, v = kv.partition("=")
            # Attempt numeric casting
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
            overrides[k] = v

    train(config_path=Path(args.config), overrides=overrides)


if __name__ == "__main__":
    main()
