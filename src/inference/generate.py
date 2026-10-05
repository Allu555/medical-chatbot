"""
src/inference/generate.py
==========================
Text generation pipeline for the AI Medical Chatbot.

Features:
  - Configurable generation parameters
  - Conversation history management
  - RAG context injection
  - Safety layer integration
  - Source citation display
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from src.inference.safety import SafetyLayer, Severity, CAUTION_FOOTER
from src.inference.model_loader import MedGemmaLoader

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/inference.yaml")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    do_sample: bool = True
    deterministic: bool = False      # set True for evaluation (greedy)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "GenerationConfig":
        if not path.exists():
            return cls()
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        gen = cfg.get("generation", {})
        return cls(
            max_new_tokens=gen.get("max_new_tokens", 512),
            temperature=gen.get("temperature", 0.3),
            top_p=gen.get("top_p", 0.9),
            top_k=gen.get("top_k", 50),
            repetition_penalty=gen.get("repetition_penalty", 1.15),
            do_sample=gen.get("do_sample", True),
        )


@dataclass
class ConversationTurn:
    role: str           # "user" or "assistant"
    content: str


@dataclass
class ConversationHistory:
    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 5

    def add(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))
        # Trim oldest turns (keep pairs)
        while len(self.turns) > self.max_turns * 2:
            self.turns.pop(0)

    def clear(self) -> None:
        self.turns.clear()

    def to_formatted_string(self, system_prompt: str) -> str:
        """
        Build the full prompt string in MedGemma chat format.
        """
        parts: list[str] = []
        for i, turn in enumerate(self.turns):
            if turn.role == "user":
                # Inject system prompt into the first user turn only
                sys_prefix = f"[System]: {system_prompt}\n\n" if i == 0 else ""
                parts.append(
                    f"<start_of_turn>user\n{sys_prefix}{turn.content}<end_of_turn>"
                )
            else:
                parts.append(
                    f"<start_of_turn>model\n{turn.content}<end_of_turn>"
                )
        # Open the final model turn for generation
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical AI assistant providing educational health information. "
    "You always distinguish general information from personalised medical advice. "
    "You never invent medical facts, medications, dosages, or citations. "
    "You clearly express uncertainty when the evidence is limited. "
    "You always recommend that users seek professional medical care for diagnosis "
    "and treatment. For emergencies, you direct users to emergency services immediately."
)

RAG_SYSTEM_PROMPT_TEMPLATE = """\
You are a medical AI assistant providing educational health information.

Use the retrieved medical context below when it is relevant and reliable.

IMPORTANT RULES:
1. Do not invent medical facts, medications, or dosages.
2. Do not claim certainty when evidence is uncertain.
3. Do not diagnose the user from symptoms alone.
4. Do not prescribe medication or dosages.
5. Clearly express uncertainty when the evidence is limited or absent.
6. If the situation may be urgent or life-threatening, recommend emergency care immediately.
7. If retrieved context does not support the answer, say so explicitly.
8. Distinguish general information from personalised medical advice.
9. Keep responses clear, concise, and understandable.

Retrieved medical context:
---
{context}
---
"""


class MedicalChatGenerator:
    """
    Manages text generation for the medical chatbot.

    Integrates: model loading · safety checks · RAG context · conversation history.
    """

    def __init__(
        self,
        model_dir: Path = Path("artifacts/model"),
        config_path: Path = DEFAULT_CONFIG_PATH,
        rag_retriever: Any | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.config_path = config_path
        self.gen_cfg = GenerationConfig.from_yaml(config_path)
        self.safety = SafetyLayer()
        self.rag = rag_retriever
        self.history = ConversationHistory(max_turns=5)

        # Lazy-loaded — call .load() explicitly or let generate() load on demand
        self._loader: MedGemmaLoader | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the model and tokenizer."""
        self._loader = MedGemmaLoader(
            model_dir=self.model_dir,
            config_path=self.config_path,
        )
        self._model, self._tokenizer = self._loader.load()
        logger.info("MedicalChatGenerator: model loaded.")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ------------------------------------------------------------------
    def _build_system_prompt(self, rag_context: str | None) -> str:
        if rag_context:
            return RAG_SYSTEM_PROMPT_TEMPLATE.format(context=rag_context)
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def _retrieve_context(self, query: str) -> tuple[str | None, list[dict[str, Any]]]:
        """Retrieve RAG context if retriever is available."""
        if self.rag is None:
            return None, []
        try:
            results = self.rag.retrieve(query)
            if not results:
                return None, []
            context_parts = [r["text"] for r in results if r.get("text")]
            context = "\n\n---\n\n".join(context_parts)
            return context, results
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return None, []

    # ------------------------------------------------------------------
    def _format_sources(self, docs: list[dict[str, Any]]) -> str:
        """Format retrieved document metadata into a citation block."""
        if not docs:
            return ""
        lines = ["\n\n---\n**Sources:**"]
        seen: set[str] = set()
        for i, doc in enumerate(docs, start=1):
            title = doc.get("title", "Unknown")
            source = doc.get("source", "")
            url = doc.get("url", "")
            key = f"{title}_{url}"
            if key in seen:
                continue
            seen.add(key)
            if url:
                lines.append(f"{i}. [{title}]({url}) — {source}")
            else:
                lines.append(f"{i}. {title} — {source}")
        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str,
        use_rag: bool = True,
        deterministic: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Generate a medical response to the user message.

        Args:
            user_message:   The user's input text.
            use_rag:        Whether to use RAG retrieval.
            deterministic:  If True, use greedy decoding (for evaluation).

        Returns:
            (response_text: str, sources: list of document metadata dicts)
        """
        self._ensure_loaded()

        # ── 1. Safety check on input ─────────────────────────────────────
        safety_result = self.safety.check_input(user_message)
        if safety_result.is_emergency:
            return safety_result.emergency_response, []

        # ── 2. RAG retrieval ──────────────────────────────────────────────
        rag_context: str | None = None
        source_docs: list[dict[str, Any]] = []
        if use_rag and self.rag is not None:
            rag_context, source_docs = self._retrieve_context(user_message)

        # ── 3. Build prompt ───────────────────────────────────────────────
        system_prompt = self._build_system_prompt(rag_context)
        self.history.add("user", user_message)
        prompt = self.history.to_formatted_string(system_prompt)

        # ── 4. Tokenize ───────────────────────────────────────────────────
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.gen_cfg.max_new_tokens,
        ).to(self._model.device)

        # ── 5. Generate ───────────────────────────────────────────────────
        use_sampling = self.gen_cfg.do_sample and not deterministic
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.gen_cfg.max_new_tokens,
            "repetition_penalty": self.gen_cfg.repetition_penalty,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if use_sampling:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.gen_cfg.temperature,
                "top_p": self.gen_cfg.top_p,
                "top_k": self.gen_cfg.top_k,
            })
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # ── 6. Safety validation on output ────────────────────────────────
        response, _ = self.safety.validate_output(response)

        # ── 7. Add caution footer if needed ──────────────────────────────
        if safety_result.is_caution:
            response += CAUTION_FOOTER

        # ── 8. Append sources ─────────────────────────────────────────────
        sources_block = self._format_sources(source_docs)
        if sources_block:
            response += sources_block

        # ── 9. Update history with assistant turn ─────────────────────────
        self.history.add("assistant", response)

        return response, source_docs

    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history.clear()
        logger.debug("Conversation history cleared.")
