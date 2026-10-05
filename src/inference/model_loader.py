"""
src/inference/model_loader.py
==============================
Singleton model/tokenizer loader for the inference pipeline.

Handles:
  - 4-bit QLoRA model loading
  - LoRA adapter merging (optional)
  - Hardware auto-detection
  - Tokenizer setup
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_CONFIG = Path("configs/inference.yaml")
DEFAULT_MODEL_DIR = Path("artifacts/model")


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _compute_dtype(cfg: dict[str, Any]) -> torch.dtype:
    raw = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    if raw == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif raw in ("float16", "bfloat16"):
        return torch.float16
    return torch.float32


def _load_inference_config(path: Path = DEFAULT_INFERENCE_CONFIG) -> dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    logger.warning("Inference config not found at %s; using defaults.", path)
    return {}


# ---------------------------------------------------------------------------
# Model loader class (singleton-friendly)
# ---------------------------------------------------------------------------

class MedGemmaLoader:
    """
    Loads the fine-tuned MedGemma 4B QLoRA model and tokenizer.

    Usage:
        loader = MedGemmaLoader()
        model, tokenizer = loader.load()
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        config_path: Path = DEFAULT_INFERENCE_CONFIG,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.cfg = _load_inference_config(config_path)
        self.model_cfg = self.cfg.get("model", {})
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    def _base_model_name(self) -> str:
        """Determine base model name from adapter config or inference config."""
        adapter_cfg = self.model_dir / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as fh:
                ac = json.load(fh)
            base = ac.get("base_model_name_or_path", "")
            if base:
                logger.info("Base model from adapter_config.json: %s", base)
                return base

        meta = self.model_dir / "training_metadata.json"
        if meta.exists():
            with open(meta) as fh:
                m = json.load(fh)
            base = m.get("model_name", "")
            if base:
                logger.info("Base model from training_metadata.json: %s", base)
                return base

        fallback = self.model_cfg.get("base_model_name", "google/medgemma-4b-it")
        logger.info("Using base model from config: %s", fallback)
        return fallback

    # ------------------------------------------------------------------
    def _build_bnb_config(self) -> BitsAndBytesConfig:
        mc = self.model_cfg
        return BitsAndBytesConfig(
            load_in_4bit=mc.get("load_in_4bit", True),
            bnb_4bit_quant_type=mc.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_compute_dtype(mc),
            bnb_4bit_use_double_quant=mc.get("bnb_4bit_use_double_quant", True),
        )

    # ------------------------------------------------------------------
    def load(self, force_reload: bool = False) -> tuple[Any, Any]:
        """
        Load model and tokenizer. Returns cached version on subsequent calls.

        Returns:
            (model, tokenizer)
        """
        if self._model is not None and not force_reload:
            return self._model, self._tokenizer

        bnb_config = self._build_bnb_config()
        base_name = self._base_model_name()

        logger.info("Loading tokenizer from %s …", self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info("Loading base model: %s …", base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        # Check whether model_dir contains a LoRA adapter
        adapter_cfg_path = self.model_dir / "adapter_config.json"
        if adapter_cfg_path.exists():
            logger.info("Loading LoRA adapter from %s …", self.model_dir)
            model = PeftModel.from_pretrained(base_model, str(self.model_dir))
        else:
            logger.warning(
                "No adapter_config.json found in %s. "
                "Loading base model only (no fine-tuning applied).",
                self.model_dir,
            )
            model = base_model

        model.eval()
        logger.info("Model loaded. Device map: %s", getattr(model, "hf_device_map", "N/A"))

        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    def load_base_only(self) -> tuple[Any, Any]:
        """Load the base MedGemma model without any LoRA adapter (for comparison)."""
        base_name = self._base_model_name()
        bnb_config = self._build_bnb_config()

        logger.info("Loading BASE model (no LoRA) from %s …", base_name)
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )
        model.eval()
        return model, tokenizer
