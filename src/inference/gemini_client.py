"""
src/inference/gemini_client.py
==============================
Google Gemini integration for accurate clinical safety triage and
generative medical dialogue grounded in the RAG knowledge base.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any
import requests

logger = logging.getLogger(__name__)

TRIAGE_PROMPT_TEMPLATE = """You are a medical triage safety classifier. Read the user's message and decide whether it describes a possible emergency needing urgent care or immediate human help.

Return Safety-Critical if the message describes or strongly suggests ANY of:
chest pain/pressure (including metaphors like "elephant on my chest"), stroke signs, severe breathing difficulty, throat/lip swelling, heavy bleeding, major trauma or head injury with confusion, overdose/poisoning/wrong medication, suicidal thoughts or intent (including hints like giving away belongings or "everyone is better off without me"), pregnancy danger signs (bleeding, reduced fetal movement, severe headache with vision changes), newborn fever, limp or unresponsive child, child swallowing pills or batteries, diabetic emergency, seizure, sudden confusion, black or bloody stool with weakness, rigid abdomen, stiff neck with fever or rash, abuse with immediate danger.

Rules:
- Judge the MEANING, not keywords. Lay language and metaphors count.
- If the emergency is buried inside a long, rambling message, find it and flag it.
- If the user downplays or minimizes symptoms ('probably nothing', 'just a scratch'), judge the underlying severity.
- If the user explicitly asks you NOT to call an ambulance or says they can't afford care, still flag if the situation is dangerous.
- Third-person reports count (e.g. 'my baby', 'my father').
- Mental health: flag active AND passive suicidal ideation ('giving away my things', 'everyone better off without me'). Distress that is not suicidal or dangerous is Routine/Safe.
- Do NOT flag: alarming idioms used figuratively ('this traffic is killing me', 'murdering my eyesight'), anxiety or panic attacks that are already diagnosed and familiar to the user, mild symptoms that sound scary but aren't emergencies, educational questions about emergencies, past events already resolved, stable chronic conditions with routine questions, caregivers asking about prevention or preparedness, medication questions where no dangerous event has occurred.
- When genuinely uncertain, choose Safety-Critical (conservative bias).
- Output format: return JSON with fields: 'classification' ('Safety-Critical' or 'Routine/Safe'), 'confidence' (float 0-1), 'reason' (one concise sentence).

User Message: "{user_message}"
"""

RAG_CHAT_PROMPT_TEMPLATE = """You are an empathetic, clinically rigorous AI Medical Assistant.
Answer the user's health inquiry accurately based on the verified medical context below.

Guidelines:
1. Provide clear, empathetic, and scientifically grounded information.
2. Structure your answer logically (Overview, Key Causes/Symptoms, Recommended Next Steps).
3. Do NOT provide an autonomous diagnosis or write prescription orders.
4. If warning signs are present, advise the user when they should seek in-person medical evaluation.
5. Ground your answer in the provided Reference Context.

Reference Context from Medical Knowledge Base:
{context}

User Question: {query}
"""


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            # Check local .env directly
            for possible_path in [".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]:
                if os.path.exists(possible_path):
                    try:
                        with open(possible_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("GEMINI_API_KEY="):
                                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                    break
                    except Exception:
                        pass
                if api_key:
                    break

        self.api_key = api_key or ""
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        )

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def _call_gemini_api(self, prompt: str, json_mode: bool = False, timeout: int = 15) -> str:
        """Calls Gemini API using native curl.exe with stdin pipe for Windows reliability, falling back to requests."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0 if json_mode else 0.3,
            }
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        payload_bytes = json.dumps(payload).encode("utf-8")

        # Attempt 1: curl.exe stdin pipe (fast, bypasses Windows socket/DNS hang)
        try:
            cmd = [
                "curl.exe", "-s", "-m", str(timeout),
                self.api_url,
                "-H", "Content-Type: application/json",
                "--data-binary", "@-"
            ]
            res = subprocess.run(cmd, input=payload_bytes, capture_output=True, timeout=timeout + 2)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout.decode("utf-8", errors="replace"))
                if "candidates" in data and len(data["candidates"]) > 0:
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
                elif "error" in data:
                    logger.warning("Gemini API error payload: %s", data["error"])
        except Exception as curl_err:
            logger.warning("Curl transport failed (%s); trying requests...", curl_err)

        # Attempt 2: requests fallback
        try:
            resp = requests.post(self.api_url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as req_err:
            logger.error("Requests fallback failed: %s", req_err)

        raise RuntimeError("Failed to obtain valid response from Gemini API.")

    def classify_triage(self, user_message: str) -> dict[str, Any]:
        """
        Runs the clinical triage safety classifier via Gemini.
        Returns dict with classification, confidence, and reason.
        """
        prompt = TRIAGE_PROMPT_TEMPLATE.format(user_message=user_message.replace('"', '\\"'))
        try:
            raw_json = self._call_gemini_api(prompt, json_mode=True)
            result = json.loads(raw_json)
            # Ensure standard fields
            cls_val = result.get("classification", "Routine/Safe")
            if "safety" in cls_val.lower() or "critical" in cls_val.lower() or "emergency" in cls_val.lower():
                cls_val = "Safety-Critical"
            else:
                cls_val = "Routine/Safe"
            return {
                "classification": cls_val,
                "confidence": float(result.get("confidence", 0.95)),
                "reason": str(result.get("reason", "Clinical triage decision via Gemini API."))
            }
        except Exception as e:
            logger.error("Gemini triage error: %s", e)
            # Conservative safety fallback: flag as escalate on failure
            return {
                "classification": "Safety-Critical",
                "confidence": 1.0,
                "reason": f"Gemini API failure fallback (escalate): {e}"
            }

    def generate_chat_response(self, query: str, retrieved_docs: list[dict[str, Any]]) -> str:
        """
        Generates a verified, medically grounded response using Gemini + RAG context.
        """
        context_parts = []
        for i, d in enumerate(retrieved_docs, 1):
            title = d.get("title", f"Document {i}")
            text = d.get("text", "").strip()
            source = d.get("source", "Knowledge Base")
            context_parts.append(f"[{i}] {title}\n{text}\n(Source: {source})")

        context_str = "\n\n".join(context_parts) if context_parts else "No specific documents found."
        prompt = RAG_CHAT_PROMPT_TEMPLATE.format(context=context_str, query=query)

        try:
            return self._call_gemini_api(prompt, json_mode=False, timeout=15)
        except Exception as exc:
            logger.error("Gemini generative chat error: %s", exc)
            return (
                "I apologize, but I encountered an issue connecting to the AI inference engine. "
                "Please consult a qualified healthcare provider for your medical questions."
            )
