"""
app.py
=======
Sorin-AI Clinical Medical Assistant — Multi-Theme SaaS UI

Themes:
  1. 📜 Warm Editorial (Claude / Anthropic style — Warm paper & Terracotta) [Default]
  2. ☕ Warm Espresso (Claude Dark — Rich charcoal & Amber)
  3. ⚡ Midnight Cyan (Futuristic Biotech / Cyberpunk)
  4. 🔮 Minimal Indigo (Raycast / Linear style)
  5. ❄️ Nordic Clinical (Clean Slate & Frost Blue)

Features:
  - Instant Theme Switcher in Top Navigation
  - 100vh Full Viewport (Zero outer page scrolling)
  - Sleek ChatGPT-style capsule input bar with custom SVG Mic, Speaker & Send icons
  - RAG Retrieval + Gemini Clinical Generative Intelligence
  - Multi-session consultation management (+ New Chat, Delete, Auto-naming)
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
MODEL_DIR = Path(os.getenv("OUTPUT_MODEL_DIR", "artifacts/model"))
FAISS_INDEX = Path("artifacts/faiss/index.faiss")
BASE_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

BOT_AVATAR = str(ASSETS_DIR / "bot_avatar.png") if (ASSETS_DIR / "bot_avatar.png").exists() else None
USER_AVATAR = str(ASSETS_DIR / "user_avatar.png") if (ASSETS_DIR / "user_avatar.png").exists() else None

# ---------------------------------------------------------------------------
# Lazy-loaded backend
# ---------------------------------------------------------------------------
_generator = None
_retriever = None
_backend_initialized = False


def _initialize_backend():
    """Load model and RAG retriever once at startup."""
    global _generator, _retriever, _backend_initialized
    if _backend_initialized:
        return _generator, _retriever

    from src.rag.retriever import MedicalRetriever

    try:
        logger.info("Loading RAG retriever …")
        _retriever = MedicalRetriever()
        _retriever.load()
        logger.info("RAG retriever loaded.")
    except Exception as exc:
        logger.warning("RAG retriever failed to load: %s", exc)
        _retriever = None

    has_weights = any(
        (MODEL_DIR / fn).exists()
        for fn in ["adapter_model.safetensors", "adapter_model.bin", "model.safetensors"]
    )
    if has_weights:
        from src.inference.generate import MedicalChatGenerator
        try:
            logger.info("Loading local model from %s …", MODEL_DIR)
            _generator = MedicalChatGenerator(
                model_dir=MODEL_DIR,
                rag_retriever=_retriever,
            )
            _generator.load()
            logger.info("Local model loaded.")
        except Exception as exc:
            logger.error("Failed to load local model: %s", exc)
            _generator = None
    else:
        logger.info("Using Gemini Generative Clinical Intelligence + RAG retrieval.")
        _generator = None

    _backend_initialized = True
    return _generator, _retriever


# ---------------------------------------------------------------------------
# Conversational / Small Talk Handler
# ---------------------------------------------------------------------------

def _handle_conversational_input(message: str) -> str | None:
    cleaned = re.sub(r"[^\w\s]", "", message.lower()).strip()
    words = cleaned.split()

    if not words:
        return None

    if any(cleaned.startswith(p) for p in ["haha", "hehe", "ehe", "lol", "lmao", "rofl", "xd"]):
        return (
            "😊 I am here to assist with health inquiries. "
            "Feel free to ask about symptoms, health conditions, medications, or preventive care."
        )

    greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup", "yo"}
    if cleaned in greetings or (len(words) <= 2 and words[0] in greetings):
        return (
            "Hello! 👋 I am **Sorin-AI**, your clinical medical assistant.\n\n"
            "How can I assist you today? You can ask about symptoms, medications, health conditions, or clinical guidelines."
        )

    thanks = {"thanks", "thank you", "thx", "thank you so much", "many thanks", "appreciate it", "great thanks"}
    if cleaned in thanks or (len(words) <= 3 and any(w in ["thanks", "thank"] for w in words)):
        return (
            "You are very welcome! If any other health questions arise, feel free to ask. "
            "Wishing you good health! 🩺"
        )

    about_queries = {"who are you", "what are you", "what can you do", "help", "what do you do"}
    if cleaned in about_queries:
        return (
            "I am **Sorin-AI**, an evidence-based clinical AI assistant powered by medical RAG retrieval "
            "grounded in verified resources (CDC, NIH, PubMed, WHO).\n\n"
            "⚠️ *Educational Notice: I provide evidence-based guidance but do not replace "
            "in-person diagnosis or emergency medical care.*"
        )

    return None


# ---------------------------------------------------------------------------
# Chat Handler
# ---------------------------------------------------------------------------

def generate_clinical_answer(message: str) -> str:
    from src.inference.safety import detect_emergency, CAUTION_FOOTER
    from src.inference.gemini_client import GeminiClient

    gemini_client = GeminiClient()
    safety_res = detect_emergency(message)

    if safety_res.is_emergency and safety_res.emergency_response:
        return safety_res.emergency_response

    if gemini_client.is_available:
        try:
            triage = gemini_client.classify_triage(message)
            if triage.get("classification") == "Safety-Critical":
                return (
                    "🚨 **EMERGENCY CLINICAL WARNING** 🚨\n\n"
                    f"{triage.get('reason', 'Your symptoms describe a potentially critical condition requiring urgent care.')}\n\n"
                    "**Immediate Actions Required:**\n"
                    "• Call **911** (or local emergency services: 112 / 999) immediately.\n"
                    "• Seek immediate evaluation at the nearest Emergency Department.\n"
                    "• Do not attempt to drive yourself if feeling faint, short of breath, or disoriented.\n\n"
                    "*(Classified as Safety-Critical by Dual-Layer Clinical Triage)*"
                )
        except Exception as triage_err:
            logger.debug("Triage check note: %s", triage_err)

    conv_response = _handle_conversational_input(message)
    if conv_response:
        return conv_response

    generator, retriever = _initialize_backend()

    if generator is not None:
        try:
            response, _sources = generator.generate(message, use_rag=True)
        except Exception as exc:
            logger.error("Local generation error: %s", exc)
            response = f"I encountered an error generating a response. ({type(exc).__name__})"
    elif retriever is not None:
        try:
            docs = retriever.retrieve(message, top_k=3)
            if not docs and hasattr(retriever, "_metadata") and retriever._metadata:
                stopwords = {"what", "the", "and", "for", "with", "how", "can", "should", "are", "about", "when", "is", "a", "an", "in", "to", "or", "of", "be"}
                query_tokens = [w for w in re.findall(r"[a-zA-Z0-9]+", message.lower()) if len(w) > 2 and w not in stopwords]
                scored = []
                for entry in retriever._metadata:
                    full_text = (entry.get("title", "") + " " + entry.get("text", "")).lower()
                    overlap = sum(1 for tok in query_tokens if tok in full_text)
                    if overlap > 0:
                        scored.append((overlap, entry))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored:
                    docs = [item[1] for item in scored[:3]]

            if gemini_client.is_available:
                response = gemini_client.generate_chat_response(message, docs)
                if docs:
                    sources = list({d.get("source", "Verified Clinical Source") for d in docs})
                    sources_str = ", ".join(sources)
                    response += f"\n\n---\n*📚 Verified Medical Citations: {sources_str} | Clinical Knowledge Grounding*"
            elif docs:
                seen_titles = set()
                parts = ["### 🩺 Verified Clinical Guidance\n"]
                for doc in docs:
                    t = doc.get("title", "Clinical Summary")
                    if t not in seen_titles:
                        parts.append(f"#### **{t}**")
                        seen_titles.add(t)
                    parts.append(f"{doc.get('text', '').strip()}\n")
                    parts.append(f"> 📚 *Source: {doc.get('source', 'Medical Knowledge Base')}*")
                parts.append("\n---\n*💡 Note: This information is for educational purposes. Always consult a licensed doctor for personalized medical evaluation.*")
                response = "\n\n".join(parts)
            else:
                response = (
                    "I am an educational AI medical assistant, but I couldn't locate specific medical guidance "
                    "for your query in the verified clinical index.\n\n"
                    "💡 **Suggestions:**\n"
                    "- Try asking about symptoms, conditions, or medication precautions.\n"
                    "- For emergency symptoms like chest pain or breathing difficulty, call emergency services (911 / 112) immediately."
                )
        except Exception as err:
            logger.error("RAG retrieval error: %s", err)
            response = "⚠️ Could not retrieve medical documentation at this moment. Please try again."
    else:
        response = "⚠️ **Knowledge Base Not Loaded:** FAISS index was not found."

    if safety_res.is_caution and not safety_res.is_emergency:
        response += CAUTION_FOOTER

    return response


def respond_stream(message: str, chat_history: list[dict[str, str]] | None):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        yield chat_history
        return

    # 1. Evaluate emergency triage status immediately
    from src.inference.safety import detect_emergency
    safety_res = detect_emergency(message.strip())
    is_emergency = safety_res.is_emergency

    if is_emergency:
        thinking_html = (
            '<div class="clinical-thinking-card emergency-mode" data-emergency="true">'
            '  <div class="thinking-glow-bar emergency"></div>'
            '  <div class="thinking-body">'
            '    <div class="thinking-siren-wrap">'
            '      <span class="siren-icon">🚨</span>'
            '    </div>'
            '    <div class="thinking-content">'
            '      <div class="thinking-header emergency">'
            '        <span class="emergency-badge">'
            '          <span class="emergency-dot"></span>'
            '          CRITICAL EMERGENCY TRIAGE'
            '        </span>'
            '        <span class="emergency-dots"><span></span><span></span><span></span></span>'
            '      </div>'
            '      <div class="thinking-subtext emergency">'
            '        ⚠️ High-risk clinical flags detected! Formulating emergency triage protocol...'
            '      </div>'
            '    </div>'
            '  </div>'
            '</div>'
        )
    else:
        thinking_html = (
            '<div class="clinical-thinking-card">'
            '  <div class="thinking-glow-bar"></div>'
            '  <div class="thinking-body">'
            '    <div class="thinking-ecg-wrap">'
            '      <svg class="thinking-medical-svg" viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '        <path class="heart-bg" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>'
            '        <path class="ecg-pulse-line" d="M3.5 11h3.2l1.8-3.6 2.8 7.4 2.2-5.4 1.5 1.6h5.5"/>'
            '      </svg>'
            '    </div>'
            '    <div class="thinking-content">'
            '      <div class="thinking-header">'
            '        <span class="thinking-badge">'
            '          <span class="badge-dot"></span>'
            '          CLINICAL KNOWLEDGE RETRIEVAL'
            '        </span>'
            '        <span class="thinking-dots"><span></span><span></span><span></span></span>'
            '      </div>'
            '      <div class="thinking-subtext">'
            '        Synthesizing verified medical literature · PubMed · CDC · WHO'
            '      </div>'
            '    </div>'
            '  </div>'
            '</div>'
        )

    # Immediately append user message & live pulsating clinical thinking indicator
    chat_history.append({"role": "user", "content": message.strip()})
    chat_history.append({
        "role": "assistant",
        "content": thinking_html
    })
    yield chat_history

    answer = generate_clinical_answer(message.strip())

    # 2. Live streaming typewriter token animation
    words = answer.split(" ")
    accumulated = ""
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if accumulated:
            accumulated += " " + chunk
        else:
            accumulated = chunk

        chat_history[-1]["content"] = accumulated + " ▌"
        yield chat_history
        time.sleep(0.012)

    # 3. Final clean update
    chat_history[-1]["content"] = answer
    yield chat_history


def respond(
    message: str,
    chat_history: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], str]:
    if chat_history is None:
        chat_history = []
    if not message or not message.strip():
        return chat_history, ""
    answer = generate_clinical_answer(message.strip())
    chat_history.append({"role": "user", "content": message.strip()})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, ""


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------

def create_new_chat(session_data: dict) -> tuple:
    if session_data is None:
        session_data = {"sessions": {}, "active_id": "Consultation 1", "counter": 1}

    session_data["counter"] += 1
    new_id = f"Consultation {session_data['counter']}"
    session_data["sessions"][new_id] = []
    session_data["active_id"] = new_id

    titles = list(session_data["sessions"].keys())
    return (
        [],
        "",
        gr.Radio(choices=titles, value=new_id),
        session_data,
    )


def delete_active_chat(session_data: dict) -> tuple:
    if session_data is None:
        session_data = {"sessions": {}, "active_id": "Consultation 1", "counter": 1}

    active_id = session_data.get("active_id", "")
    if active_id in session_data["sessions"]:
        del session_data["sessions"][active_id]

    if not session_data["sessions"]:
        session_data["counter"] += 1
        new_id = f"Consultation {session_data['counter']}"
        session_data["sessions"][new_id] = []
        session_data["active_id"] = new_id
    else:
        session_data["active_id"] = list(session_data["sessions"].keys())[0]

    current_id = session_data["active_id"]
    current_messages = session_data["sessions"][current_id]
    titles = list(session_data["sessions"].keys())

    return (
        current_messages,
        "",
        gr.Radio(choices=titles, value=current_id),
        session_data,
    )


def switch_chat(selected_title: str, session_data: dict) -> tuple:
    if session_data is None or not selected_title:
        return [], session_data

    if selected_title in session_data["sessions"]:
        session_data["active_id"] = selected_title
        return session_data["sessions"][selected_title], session_data

    return [], session_data


def handle_user_send(message: str, chat_history: list, session_data: dict):
    if not message or not message.strip():
        titles = list(session_data["sessions"].keys()) if session_data else ["Consultation 1"]
        active = session_data.get("active_id", "Consultation 1") if session_data else "Consultation 1"
        yield chat_history, "", gr.Radio(choices=titles, value=active), session_data
        return

    if session_data is None:
        session_data = {"sessions": {"Consultation 1": []}, "active_id": "Consultation 1", "counter": 1}

    active_id = session_data.get("active_id")
    if not active_id or active_id not in session_data["sessions"]:
        active_id = "Consultation 1"
        session_data["sessions"][active_id] = []
        session_data["active_id"] = active_id

    for partial_history in respond_stream(message, chat_history):
        session_data["sessions"][active_id] = partial_history
        titles = list(session_data["sessions"].keys())
        yield partial_history, "", gr.Radio(choices=titles, value=session_data["active_id"]), session_data

    if active_id.startswith("Consultation"):
        cleaned_title = message.strip().replace("\n", " ")
        new_title = (cleaned_title[:24] + "…") if len(cleaned_title) > 24 else cleaned_title
        if new_title not in session_data["sessions"]:
            session_data["sessions"][new_title] = session_data["sessions"].pop(active_id)
            session_data["active_id"] = new_title

    titles = list(session_data["sessions"].keys())
    yield session_data["sessions"][session_data["active_id"]], "", gr.Radio(choices=titles, value=session_data["active_id"]), session_data


def handle_quick_prompt(prompt_text: str, chat_history: list, session_data: dict):
    yield from handle_user_send(prompt_text, chat_history, session_data)


# ---------------------------------------------------------------------------
# Multi-Theme Design System CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

/* ========================================================================= */
/* THEME 1: Executive Clinical (Apple Health / Clean Slate) — DEFAULT        */
/* ========================================================================= */
/* ========================================================================= */
/* THEME 1: Executive Clinical (Apple Health / Clean Slate) — DEFAULT        */
/* ========================================================================= */
:root, [data-theme="executive"] {
    --bg-page: #f1f5f9;
    --bg-shell: #ffffff;
    --bg-surface: #ffffff;
    --bg-sidebar: #f8fafc;
    --border-subtle: #e2e8f0;
    --border-strong: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --primary-light: #eff6ff;
    --send-btn-bg: #0f172a;
    --send-btn-hover: #1e293b;
    --send-btn-color: #ffffff;
    --user-bubble-bg: #1e293b;
    --user-bubble-text: #ffffff;
    --user-bubble-border: #1e293b;
    --bot-bubble-bg: #ffffff;
    --bot-bubble-border: #e2e8f0;
    --bot-bubble-text: #0f172a;
    --chip-bg: #ffffff;
    --chip-border: #cbd5e1;
    --chip-text: #334155;
    --badge-bg: #eff6ff;
    --badge-text: #1d4ed8;
    --body-text-color: #0f172a !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    color-scheme: light;
}

/* ========================================================================= */
/* THEME 2: Sage & Serenity (Calm Clinical Wellness)                         */
/* ========================================================================= */
[data-theme="sage"] {
    --bg-page: #f7f6f2;
    --bg-shell: #ffffff;
    --bg-surface: #ffffff;
    --bg-sidebar: #f0eee6;
    --border-subtle: #e3ded6;
    --border-strong: #cac3b8;
    --text-primary: #1c1917;
    --text-secondary: #44403c;
    --text-muted: #78716c;
    --primary: #0f766e;
    --primary-hover: #115e59;
    --primary-light: #f0fdfa;
    --send-btn-bg: #0f766e;
    --send-btn-hover: #115e59;
    --send-btn-color: #ffffff;
    --user-bubble-bg: #134e4a;
    --user-bubble-text: #ffffff;
    --user-bubble-border: #134e4a;
    --bot-bubble-bg: #ffffff;
    --bot-bubble-border: #e3ded6;
    --bot-bubble-text: #1c1917;
    --chip-bg: #ffffff;
    --chip-border: #cac3b8;
    --chip-text: #44403c;
    --badge-bg: #ccfbf1;
    --badge-text: #0f766e;
    --body-text-color: #1c1917 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f7f6f2 !important;
    color-scheme: light;
}

/* ========================================================================= */
/* THEME 3: Obsidian Pro Dark (Linear / Perplexity AI Dark)                  */
/* ========================================================================= */
[data-theme="obsidian-dark"] {
    --bg-page: #09090b;
    --bg-shell: #121215;
    --bg-surface: #18181b;
    --bg-sidebar: #0f0f12;
    --border-subtle: #27272a;
    --border-strong: #3f3f46;
    --text-primary: #f4f4f5;
    --text-secondary: #d4d4d8;
    --text-muted: #a1a1aa;
    --primary: #3b82f6;
    --primary-hover: #60a5fa;
    --primary-light: rgba(59, 130, 246, 0.15);
    --send-btn-bg: #ffffff;
    --send-btn-hover: #f4f4f5;
    --send-btn-color: #09090b;
    --user-bubble-bg: #27272a;
    --user-bubble-text: #fafafa;
    --user-bubble-border: #3f3f46;
    --bot-bubble-bg: #18181b;
    --bot-bubble-border: #27272a;
    --bot-bubble-text: #f4f4f5;
    --chip-bg: #18181b;
    --chip-border: #27272a;
    --chip-text: #d4d4d8;
    --badge-bg: rgba(59, 130, 246, 0.15);
    --badge-text: #93c5fd;
    --body-text-color: #f4f4f5 !important;
    --background-fill-primary: #18181b !important;
    --background-fill-secondary: #09090b !important;
    color-scheme: dark !important;
}

/* ========================================================================= */
/* THEME 4: Warm Editorial (Claude / Anthropic style)                        */
/* ========================================================================= */
[data-theme="claude"] {
    --bg-page: #fbf8f3;
    --bg-shell: #ffffff;
    --bg-surface: #ffffff;
    --bg-sidebar: #f5f0e8;
    --border-subtle: #ebdcd0;
    --border-strong: #d6c6b8;
    --text-primary: #292524;
    --text-secondary: #57534e;
    --text-muted: #78716c;
    --primary: #b45309;
    --primary-hover: #92400e;
    --primary-light: #fef3c7;
    --send-btn-bg: #292524;
    --send-btn-hover: #1c1917;
    --send-btn-color: #fafaf9;
    --user-bubble-bg: #292524;
    --user-bubble-text: #fafaf9;
    --user-bubble-border: #292524;
    --bot-bubble-bg: #ffffff;
    --bot-bubble-border: #ebdcd0;
    --bot-bubble-text: #292524;
    --chip-bg: #ffffff;
    --chip-border: #d6c6b8;
    --chip-text: #57534e;
    --badge-bg: #fef3c7;
    --badge-text: #b45309;
    --body-text-color: #292524 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fbf8f3 !important;
    color-scheme: light;
}

/* ========================================================================= */
/* THEME 5: Warm Espresso (Claude Dark — Rich Charcoal & Amber)              */
/* ========================================================================= */
[data-theme="claude-dark"] {
    --bg-page: #141210;
    --bg-shell: #1c1917;
    --bg-surface: #1c1917;
    --bg-sidebar: #171412;
    --border-subtle: #292524;
    --border-strong: #44403c;
    --text-primary: #f5f5f4;
    --text-secondary: #d6d3d1;
    --text-muted: #a8a29e;
    --primary: #f59e0b;
    --primary-hover: #d97706;
    --primary-light: rgba(245, 158, 11, 0.15);
    --send-btn-bg: #f5f5f4;
    --send-btn-hover: #e7e5e4;
    --send-btn-color: #1c1917;
    --user-bubble-bg: #292524;
    --user-bubble-text: #fafaf9;
    --user-bubble-border: #44403c;
    --bot-bubble-bg: #1c1917;
    --bot-bubble-border: #292524;
    --bot-bubble-text: #f5f5f4;
    --chip-bg: #292524;
    --chip-border: #44403c;
    --chip-text: #d6d3d1;
    --badge-bg: rgba(245, 158, 11, 0.15);
    --badge-text: #fcd34d;
    --body-text-color: #f5f5f4 !important;
    --background-fill-primary: #1c1917 !important;
    --background-fill-secondary: #141210 !important;
    color-scheme: dark !important;
}

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

/* 100vh Full Viewport — Strictly Zero Outer Page Scrolling */
html, body {
    height: 100vh !important;
    max-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

body, .gradio-container {
    background-color: var(--bg-page) !important;
    background: var(--bg-page) !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 8px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    transition: background-color 0.2s ease !important;
}

/* App Outer Shell */
#sorin-app-shell {
    max-width: 1440px !important;
    width: 100% !important;
    margin: 0 auto !important;
    height: 100% !important;
    max-height: calc(100vh - 16px) !important;
    background: var(--bg-shell) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.06) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}

/* Top Navbar */
#sorin-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 48px;
    min-height: 48px;
    max-height: 48px;
    padding: 0 16px;
    background: var(--bg-shell);
    border-bottom: 1px solid var(--border-subtle);
    flex-shrink: 0;
}

.sorin-brand-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.sorin-logo-badge {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}
.sorin-brand-title {
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.2px;
}
.sorin-brand-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-weight: 500;
}

.sorin-navbar-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Interactive Theme Selector Dropdown */
.theme-picker-wrap {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 2px 8px;
}
.theme-picker-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
}
.theme-select-dropdown {
    background: transparent;
    border: none;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    cursor: pointer;
    outline: none;
    padding: 3px 2px;
}
.theme-select-dropdown option {
    background: var(--bg-surface);
    color: var(--text-primary);
}

.theme-quick-toggle-btn {
    background: var(--bg-sidebar);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 3px 9px;
    font-size: 0.76rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-primary);
    transition: all 0.15s ease;
}
.theme-quick-toggle-btn:hover {
    border-color: var(--primary);
    background: var(--primary-light);
}

.status-pill {
    background: var(--badge-bg);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 3px 9px;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--badge-text);
    display: flex;
    align-items: center;
    gap: 5px;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary);
}

/* Main 2-Column Layout */
#sorin-layout {
    flex: 1 1 0% !important;
    min-height: 0 !important;
    height: calc(100% - 48px) !important;
    display: flex !important;
    gap: 10px !important;
    padding: 8px 12px 8px 12px !important;
    overflow: hidden !important;
    align-items: stretch !important;
}

/* Sidebar */
#sorin-sidebar {
    width: 240px !important;
    min-width: 240px !important;
    max-width: 240px !important;
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 10px !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* Interactive Robot Avatar Companion Card with Variants & Drag Physics     */
/* ───────────────────────────────────────────────────────────────────────── */
:root,
.robot-companion-card,
#sorin-robot-character,
.robot-character,
[data-variant="classic"] {
    --r-metal-top: #ffffff;
    --r-metal-mid: #e2e8f0;
    --r-metal-bot: #cbd5e1;
    --r-visor-bg: #090d16;
    --r-visor-stroke: #38bdf8;
    --r-eye-glow: #38bdf8;
    --r-pulse-color: #38bdf8;
    --r-accent: #2563eb;
    --r-badge-bg: rgba(37, 99, 235, 0.12);
}

.robot-companion-card[data-variant="cyber"],
#sorin-robot-character[data-variant="cyber"],
[data-variant="cyber"] {
    --r-metal-top: #2e1065;
    --r-metal-mid: #0f172a;
    --r-metal-bot: #020617;
    --r-visor-bg: #090014;
    --r-visor-stroke: #ec4899;
    --r-eye-glow: #f43f5e;
    --r-pulse-color: #a855f7;
    --r-accent: #ec4899;
    --r-badge-bg: rgba(236, 72, 153, 0.15);
}

.robot-companion-card[data-variant="sage"],
#sorin-robot-character[data-variant="sage"],
[data-variant="sage"] {
    --r-metal-top: #f0fdf4;
    --r-metal-mid: #dcfce7;
    --r-metal-bot: #bbf7d0;
    --r-visor-bg: #062419;
    --r-visor-stroke: #10b981;
    --r-eye-glow: #34d399;
    --r-pulse-color: #059669;
    --r-accent: #059669;
    --r-badge-bg: rgba(16, 185, 129, 0.15);
}

.robot-companion-card[data-variant="cryo"],
#sorin-robot-character[data-variant="cryo"],
[data-variant="cryo"] {
    --r-metal-top: #f0f9ff;
    --r-metal-mid: #e0f2fe;
    --r-metal-bot: #bae6fd;
    --r-visor-bg: #041f2d;
    --r-visor-stroke: #06b6d4;
    --r-eye-glow: #22d3ee;
    --r-pulse-color: #38bdf8;
    --r-accent: #0284c7;
    --r-badge-bg: rgba(6, 182, 212, 0.15);
}

.robot-companion-card[data-variant="gold"],
#sorin-robot-character[data-variant="gold"],
[data-variant="gold"] {
    --r-metal-top: #292524;
    --r-metal-mid: #1c1917;
    --r-metal-bot: #0c0a09;
    --r-visor-bg: #1c1304;
    --r-visor-stroke: #f59e0b;
    --r-eye-glow: #fbbf24;
    --r-pulse-color: #d97706;
    --r-accent: #d97706;
    --r-badge-bg: rgba(245, 158, 11, 0.15);
}

.robot-companion-card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 8px 8px 8px 8px !important;
    margin-bottom: 10px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    position: relative !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    user-select: none !important;
    flex-shrink: 0 !important;
}

.robot-companion-card:hover {
    border-color: var(--r-visor-stroke) !important;
    box-shadow: 0 4px 14px var(--r-badge-bg) !important;
}

/* Robot Card Topbar & Settings */
.robot-card-topbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    gap: 6px !important;
    position: relative !important;
    margin-bottom: 4px !important;
}

.robot-settings-btn {
    width: 24px !important;
    height: 24px !important;
    min-width: 24px !important;
    min-height: 24px !important;
    border-radius: 50% !important;
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-muted) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    font-size: 0.72rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
}

.robot-settings-btn:hover {
    color: var(--primary) !important;
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
    transform: rotate(45deg);
}

/* Settings Popover Drawer (Hidden by default, opened by ⚙️) */
.robot-settings-panel {
    display: none;
    position: absolute !important;
    top: 36px !important;
    left: 4px !important;
    right: 4px !important;
    background: var(--bg-surface) !important;
    border: 1.5px solid var(--border-strong) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28) !important;
    z-index: 120 !important;
    animation: settingsPop 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.robot-settings-panel.open {
    display: block !important;
}

@keyframes settingsPop {
    0% { opacity: 0; transform: translateY(-6px) scale(0.96); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

.settings-panel-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 6px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.settings-panel-title {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-primary) !important;
}

.settings-close-btn {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    cursor: pointer !important;
    padding: 0 4px !important;
    line-height: 1 !important;
}

.settings-close-btn:hover {
    color: var(--text-primary) !important;
}

/* Solo Floating Robot Character (DRAGGING ONLY HIM) */
.robot-docking-bay {
    position: relative !important;
    width: 100% !important;
    min-height: 124px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#sorin-robot-character {
    position: relative;
    width: 110px;
    height: 124px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: grab;
    user-select: none;
    transition: transform 0.15s ease-out;
}

#sorin-robot-character:hover {
    filter: drop-shadow(0 4px 12px var(--r-badge-bg));
}

#sorin-robot-character.is-flying {
    position: fixed !important;
    z-index: 999999 !important;
    width: 110px !important;
    height: 124px !important;
    margin: 0 !important;
    cursor: grab !important;
    pointer-events: auto !important;
    filter: drop-shadow(0 14px 28px rgba(0, 0, 0, 0.4)) drop-shadow(0 0 16px var(--r-badge-bg)) !important;
    animation: dropBounce 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}

#sorin-robot-character.is-dragging {
    cursor: grabbing !important;
    transition: none !important;
    transform: scale(1.06) !important;
}

#sorin-robot-character .character-dock-pin {
    position: absolute !important;
    top: -6px !important;
    right: -4px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 50% !important;
    width: 22px !important;
    height: 22px !important;
    font-size: 0.68rem !important;
    display: none !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    z-index: 1000000 !important;
    padding: 0 !important;
}

#sorin-robot-character.is-flying .character-dock-pin {
    display: flex !important;
}

#sorin-robot-character .character-dock-pin:hover {
    transform: scale(1.18);
    background: var(--primary-light) !important;
    border-color: var(--primary) !important;
}

/* Docking Bay Placeholder in sidebar when Dr. Sorin is in flight */
#robot-dock-placeholder {
    display: none;
    width: 100% !important;
    height: 118px !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: 12px !important;
    background: rgba(0, 0, 0, 0.04) !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 5px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    margin: 3px 0 !important;
}

#robot-dock-placeholder.visible {
    display: flex !important;
}

#robot-dock-placeholder:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
}

.placeholder-icon {
    font-size: 1.3rem !important;
    animation: floatGhost 2s infinite ease-in-out !important;
}

@keyframes floatGhost {
    0%, 100% { transform: translateY(0); opacity: 0.6; }
    50% { transform: translateY(-4px); opacity: 1; }
}

.placeholder-text {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}

.recall-btn {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    font-size: 0.64rem !important;
    font-weight: 600 !important;
    color: var(--primary) !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
}

.recall-btn:hover {
    background: var(--primary) !important;
    color: #ffffff !important;
}

@keyframes dropBounce {
    0% { transform: scale(1.08) translateY(-8px); }
    100% { transform: scale(1) translateY(0); }
}

/* Variant Dots Bar */
.robot-variant-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    margin-bottom: 6px !important;
    width: 100% !important;
}

.variant-label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    margin-right: 2px !important;
}

.variant-dot {
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    border: 2px solid transparent !important;
    cursor: pointer !important;
    transition: all 0.18s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    padding: 0 !important;
    margin: 0 !important;
    position: relative !important;
}

.variant-dot:hover {
    transform: scale(1.3) !important;
}

.variant-dot.active {
    transform: scale(1.25) !important;
    border-color: var(--text-primary) !important;
    box-shadow: 0 0 8px currentColor !important;
}

.variant-dot.classic { background: #38bdf8; color: #38bdf8; }
.variant-dot.cyber   { background: #ec4899; color: #ec4899; }
.variant-dot.sage    { background: #10b981; color: #10b981; }
.variant-dot.cryo    { background: #06b6d4; color: #06b6d4; }
.variant-dot.gold    { background: #f59e0b; color: #f59e0b; }

/* Speech Bubble */
.robot-speech-bubble {
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    padding: 5px 8px !important;
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    text-align: center !important;
    width: 100% !important;
    min-height: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    margin-bottom: 6px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    line-height: 1.25 !important;
}

.robot-speech-bubble .speech-bubble-arrow {
    position: absolute !important;
    bottom: -5px !important;
    left: 50% !important;
    transform: translateX(-50%) rotate(45deg) !important;
    width: 8px !important;
    height: 8px !important;
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-strong) !important;
    border-bottom: 1px solid var(--border-strong) !important;
}

/* Stage & Floating SVG */
.robot-stage {
    position: relative !important;
    width: 110px !important;
    height: 120px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: grab !important;
}

.robot-svg {
    overflow: visible !important;
    display: block !important;
    width: 100px !important;
    height: 112px !important;
}

/* Float Bobbing */
.robot-float-group {
    animation: robotBob 3.2s ease-in-out infinite !important;
    transform-origin: 80px 100px !important;
    transition: transform 0.15s ease-out !important;
}

@keyframes robotBob {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

/* Shadow Breathing */
.robot-shadow-ellipse {
    fill: rgba(0, 0, 0, 0.2) !important;
    animation: shadowPulse 3.2s ease-in-out infinite !important;
    transform-origin: 80px 172px !important;
}

@keyframes shadowPulse {
    0%, 100% { transform: scale(1); opacity: 0.25; }
    50% { transform: scale(0.8); opacity: 0.12; }
}

/* Antenna Beacon Pulsing */
.antenna-orb {
    animation: antennaGlow 1.8s ease-in-out infinite !important;
}

@keyframes antennaGlow {
    0%, 100% { filter: drop-shadow(0 0 2px var(--r-eye-glow)); opacity: 0.85; }
    50% { filter: drop-shadow(0 0 8px var(--r-eye-glow)); opacity: 1; transform: scale(1.15); }
}

/* Eyes Blinking */
.robot-eye {
    transform-origin: 50% 50% !important;
    animation: eyeBlink 4.5s ease-in-out infinite !important;
}

@keyframes eyeBlink {
    0%, 92%, 100% { transform: scaleY(1); }
    96% { transform: scaleY(0.08); }
}

/* Thruster Flames */
.thruster-flame {
    opacity: 0;
    transform-origin: 50% 0%;
    transition: opacity 0.2s ease;
}

.robot-flying .thruster-flame,
.is-dragging .thruster-flame,
.robot-boosting .thruster-flame {
    opacity: 1 !important;
    animation: flamePulse 0.12s infinite alternate ease-in-out !important;
}

@keyframes flamePulse {
    0% { transform: scale(1, 0.8) translateY(0); fill: #f97316; filter: drop-shadow(0 0 4px #f97316); }
    100% { transform: scale(1.18, 1.45) translateY(2px); fill: var(--r-eye-glow); filter: drop-shadow(0 0 10px var(--r-eye-glow)); }
}

.robot-boosting .robot-float-group {
    animation: rocketBoost 1.8s ease-in-out !important;
}

@keyframes rocketBoost {
    0% { transform: translateY(0); }
    20% { transform: translateY(6px); }
    48% { transform: translateY(-28px) scale(1.1); }
    75% { transform: translateY(-16px); }
    100% { transform: translateY(0); }
}

/* Interactive States: Waving */
.robot-waving .right-hand {
    animation: handWave 0.75s ease-in-out 3 !important;
    transform-origin: 132px 115px !important;
}

@keyframes handWave {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(-45deg) translateY(-14px) translateX(-4px); }
    50% { transform: rotate(15deg) translateY(-6px); }
    75% { transform: rotate(-35deg) translateY(-12px); }
}

/* Interactive States: Spin Flip */
.robot-spinning .robot-float-group {
    animation: acrobatFlip 0.85s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

@keyframes acrobatFlip {
    0% { transform: rotate(0deg) scale(1); }
    40% { transform: rotate(180deg) scale(1.15) translateY(-18px); }
    100% { transform: rotate(360deg) scale(1); }
}

/* Interactive States: Poked / Jiggle */
.robot-poked .robot-float-group {
    animation: robotJiggle 0.5s ease-in-out !important;
}

@keyframes robotJiggle {
    0%, 100% { transform: scale(1); }
    25% { transform: scale(1.1, 0.9) translateY(4px); }
    50% { transform: scale(0.95, 1.1) translateY(-8px); }
    75% { transform: scale(1.05, 0.98); }
}

/* Interactive States: Dance Mode */
.robot-dancing .robot-float-group {
    animation: robotDance 0.55s infinite alternate ease-in-out !important;
}

@keyframes robotDance {
    0% { transform: translateY(-5px) rotate(-10deg); }
    50% { transform: translateY(0px) rotate(0deg) scale(1.06); }
    100% { transform: translateY(-5px) rotate(10deg); }
}

.robot-dancing .right-hand {
    animation: handDanceRight 0.28s infinite alternate ease-in-out !important;
    transform-origin: 132px 115px !important;
}
.robot-dancing .left-hand {
    animation: handDanceLeft 0.28s infinite alternate ease-in-out !important;
    transform-origin: 28px 115px !important;
}

@keyframes handDanceRight {
    0% { transform: rotate(-35deg) translateY(-10px); }
    100% { transform: rotate(25deg) translateY(4px); }
}
@keyframes handDanceLeft {
    0% { transform: rotate(25deg) translateY(4px); }
    100% { transform: rotate(-35deg) translateY(-10px); }
}

/* Interactive States: Love / Care Mode */
.robot-loving .eye-shape {
    fill: #f43f5e !important;
    animation: heartBeat 0.5s infinite ease-in-out !important;
}

@keyframes heartBeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.22); }
}

.floating-heart {
    position: absolute !important;
    font-size: 1rem !important;
    pointer-events: none !important;
    animation: floatHeartUp 1.3s ease-out forwards !important;
    z-index: 100 !important;
}

@keyframes floatHeartUp {
    0% { opacity: 1; transform: translateY(0) scale(0.6) rotate(0deg); }
    100% { opacity: 0; transform: translateY(-75px) scale(1.3) rotate(25deg); }
}

/* Interactive States: Thinking */
.robot-thinking .antenna-orb {
    animation: antennaFast 0.4s infinite alternate !important;
    fill: #ec4899 !important;
}

@keyframes antennaFast {
    0% { filter: drop-shadow(0 0 2px #ec4899); opacity: 0.5; }
    100% { filter: drop-shadow(0 0 12px #ec4899); opacity: 1; transform: scale(1.3); }
}

.robot-thinking .robot-eyes-group {
    animation: scanEyes 0.9s infinite alternate ease-in-out !important;
}

@keyframes scanEyes {
    0% { transform: translateX(-6px); }
    100% { transform: translateX(6px); }
}

/* Interactive States: Speaking */
.robot-speaking .mouth-line {
    animation: mouthVibrate 0.2s infinite alternate ease-in-out !important;
    stroke: #10b981 !important;
    stroke-width: 3.5 !important;
}

@keyframes mouthVibrate {
    0% { transform: scaleY(0.6); }
    100% { transform: scaleY(3.2); }
}

/* Interactive States: Vitals Pulse */
.robot-vitals-active #robot-chest-pulse {
    stroke: #22c55e !important;
    filter: drop-shadow(0 0 6px #22c55e);
    animation: ecgBeat 0.6s infinite ease-in-out !important;
}

@keyframes ecgBeat {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    30% { transform: scale(1.2); opacity: 1; }
}

/* ───────────────────────────────────────────────────────────────────────── */
/* Emergency & Nervous Alert State for Dr. Sorin                             */
/* ───────────────────────────────────────────────────────────────────────── */
.robot-nervous-alert {
    border-color: rgba(239, 68, 68, 0.6) !important;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25) !important;
}

.robot-nervous-alert #sorin-robot-character,
#sorin-robot-character.robot-nervous-alert,
.robot-nervous-alert #robot-float-group,
#sorin-robot-character.robot-nervous-alert #robot-float-group {
    animation: robotNervousShiver 0.16s infinite ease-in-out !important;
}

@keyframes robotNervousShiver {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    20% { transform: translate(-2.5px, -1.5px) rotate(-2deg); }
    40% { transform: translate(2.5px, 1px) rotate(1.8deg); }
    60% { transform: translate(-2px, 1.5px) rotate(-1.5deg); }
    80% { transform: translate(2px, -1px) rotate(2deg); }
}

/* Flashing Emergency Siren Beacon on Antenna */
.robot-nervous-alert #robot-antenna-orb,
#sorin-robot-character.robot-nervous-alert #robot-antenna-orb {
    animation: emergencySirenStrobe 0.35s infinite alternate ease-in-out !important;
    fill: #ef4444 !important;
}

@keyframes emergencySirenStrobe {
    0% { fill: #ef4444; filter: drop-shadow(0 0 4px #ef4444); transform: scale(1.1); }
    100% { fill: #f59e0b; filter: drop-shadow(0 0 16px #ef4444); transform: scale(1.6); }
}

/* Alarmed, Wide Nervous Eyes */
.robot-nervous-alert .robot-eye .eye-shape,
#sorin-robot-character.robot-nervous-alert .robot-eye .eye-shape {
    fill: #ef4444 !important;
    filter: drop-shadow(0 0 8px #ef4444) !important;
    animation: nervousEyeWiden 0.3s infinite alternate ease-in-out !important;
}

@keyframes nervousEyeWiden {
    0% { transform: scale(1.1, 1.25); }
    100% { transform: scale(1.25, 1.45); }
}

.robot-nervous-alert .robot-eyes-group,
#sorin-robot-character.robot-nervous-alert .robot-eyes-group {
    animation: nervousEyesDart 0.36s infinite alternate ease-in-out !important;
}

@keyframes nervousEyesDart {
    0% { transform: translateX(-5px); }
    50% { transform: translateX(0); }
    100% { transform: translateX(5px); }
}

/* Rapid Cardiac Panic Chest Monitor (Tachycardia 160+ BPM) */
.robot-nervous-alert #robot-chest-pulse,
#sorin-robot-character.robot-nervous-alert #robot-chest-pulse {
    stroke: #ef4444 !important;
    filter: drop-shadow(0 0 8px #ef4444) !important;
    animation: tachycardiaPanic 0.24s infinite ease-in-out !important;
}

@keyframes tachycardiaPanic {
    0%, 100% { transform: scale(1); stroke-width: 2.2; }
    50% { transform: scale(1.35); stroke-width: 3.2; }
}

/* Alarmed Raised Hands */
.robot-nervous-alert .left-hand,
#sorin-robot-character.robot-nervous-alert .left-hand {
    transform: translateY(-12px) rotate(-24deg) !important;
    transition: transform 0.2s ease !important;
}
.robot-nervous-alert .right-hand,
#sorin-robot-character.robot-nervous-alert .right-hand {
    transform: translateY(-12px) rotate(24deg) !important;
    transition: transform 0.2s ease !important;
}

/* Trembling Alert Mouth */
.robot-nervous-alert .mouth-line,
#sorin-robot-character.robot-nervous-alert .mouth-line {
    stroke: #ef4444 !important;
    stroke-width: 3 !important;
    animation: nervousMouthTremble 0.14s infinite alternate ease-in-out !important;
}

@keyframes nervousMouthTremble {
    0% { transform: scaleY(0.6) translateX(-1px); }
    100% { transform: scaleY(2.6) translateX(1px); }
}

/* Speech Bubble Emergency Alert Styling */
.robot-speech-bubble.speech-bubble-emergency {
    background: #450a0a !important;
    border: 1.5px solid #ef4444 !important;
    color: #fef2f2 !important;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.45) !important;
    animation: emergencyBubblePulse 0.9s infinite alternate ease-in-out !important;
}
.robot-speech-bubble.speech-bubble-emergency .speech-bubble-arrow {
    background: #450a0a !important;
    border-color: #ef4444 !important;
}
@keyframes emergencyBubblePulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.02); }
}

/* Info Row */
.robot-info-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    padding: 2px 4px 6px 4px !important;
}

.robot-name {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

.robot-status-pill {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    padding: 2px 7px !important;
    border-radius: 9999px !important;
    text-transform: capitalize !important;
    transition: all 0.2s ease !important;
}

.robot-status-pill.ready {
    background: var(--badge-bg) !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
}

.robot-status-pill.thinking {
    background: rgba(236, 72, 153, 0.15) !important;
    color: #ec4899 !important;
    border: 1px solid rgba(236, 72, 153, 0.3) !important;
}

.robot-status-pill.speaking {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}

.robot-status-pill.listening {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #f59e0b !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}

.robot-status-pill.emergency {
    background: #ef4444 !important;
    color: #ffffff !important;
    border: 1px solid #b91c1c !important;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.7) !important;
    animation: emergencyPillPulse 0.5s infinite alternate ease-in-out !important;
    font-weight: 800 !important;
}
@keyframes emergencyPillPulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.08); }
}

/* Action Buttons Grid */
.robot-actions-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 4px !important;
    width: 100% !important;
}

.robot-act-btn {
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 4px 2px !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    text-align: center !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}

.robot-act-btn:hover {
    background: var(--primary-light) !important;
    border-color: var(--r-visor-stroke) !important;
    color: var(--r-visor-stroke) !important;
    transform: translateY(-1px) !important;
}

.robot-act-btn:active {
    transform: translateY(1px) !important;
}

.sidebar-action-row {
    display: flex !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
    flex-shrink: 0 !important;
}
#new-chat-btn {
    flex: 2 !important;
    background: var(--primary) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    transition: opacity 0.15s ease !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
}
#new-chat-btn:hover {
    opacity: 0.9 !important;
}

#delete-chat-btn {
    flex: 1 !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    color: #ef4444 !important;
    font-weight: 600 !important;
    font-size: 0.76rem !important;
    padding: 6px 6px !important;
    cursor: pointer !important;
}
#delete-chat-btn:hover {
    background: rgba(239, 68, 68, 0.1) !important;
}

.sidebar-title {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 8px 0 4px 2px;
    flex-shrink: 0;
}

/* History List Radio */
#history-radio,
#history-radio > div,
#history-radio fieldset,
#history-radio .wrap {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 3px !important;
}
#history-radio input[type="radio"] {
    display: none !important;
}
#history-radio label {
    display: flex !important;
    align-items: center !important;
    padding: 6px 9px !important;
    margin: 1px 0 !important;
    border-radius: 6px !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-surface) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
#history-radio label:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}
#history-radio label.selected,
#history-radio label:has(input:checked) {
    background: var(--primary-light) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
}

.sidebar-topic-btn {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 5px 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 6px !important;
    margin-bottom: 2px !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.sidebar-topic-btn:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* Main Panel */
#sorin-main-panel {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 8px 12px 6px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0% !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}

/* Prompt Chips */
#prompt-chips-row {
    display: flex !important;
    gap: 6px !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    padding-bottom: 2px !important;
    margin-bottom: 6px !important;
    flex-shrink: 0 !important;
    scrollbar-width: none !important;
}
#prompt-chips-row::-webkit-scrollbar {
    display: none !important;
}
.quick-chip {
    background: var(--chip-bg) !important;
    border: 1px solid var(--chip-border) !important;
    border-radius: 6px !important;
    padding: 3px 9px !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: var(--chip-text) !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
.quick-chip:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--primary-light) !important;
}

/* Chatbot Messages Stream */
#sorin-chat-history {
    flex: 1 1 0% !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: none !important;
    margin-bottom: 6px !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    border: none !important;
    background: transparent !important;
}

#sorin-chat-history > div,
#sorin-chat-history .wrap,
#sorin-chat-history .message-wrap,
#sorin-chat-history .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    padding: 4px 6px !important;
}

/* Scrollbar */
#sorin-chat-history *::-webkit-scrollbar,
#sorin-sidebar::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}
#sorin-chat-history *::-webkit-scrollbar-thumb,
#sorin-sidebar::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 4px;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* User Message Bubble — STRICT SINGLE BUBBLE (No double nested boxes)       */
/* ───────────────────────────────────────────────────────────────────────── */
/* ───────────────────────────────────────────────────────────────────────── */
/* User Message Bubble — STRICT SINGLE BUBBLE (No squished words)            */
/* ───────────────────────────────────────────────────────────────────────── */
#sorin-chat-history .message.user,
#sorin-chat-history .message-row:has([data-testid="user"]) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 4px 0 10px auto !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-end !important;
    text-align: right !important;
}

#sorin-chat-history [data-testid="user"],
#sorin-chat-history .message.user > .message-bubble-border,
#sorin-chat-history .message.user > div:first-child {
    background: var(--user-bubble-bg) !important;
    background-color: var(--user-bubble-bg) !important;
    color: var(--user-bubble-text) !important;
    border: 1px solid var(--user-bubble-border) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 10px 18px !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
    margin: 0 !important;
    width: fit-content !important;
    min-width: 52px !important;
    max-width: 80% !important;
    white-space: pre-wrap !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    flex-shrink: 0 !important;
    text-align: left !important;
    display: inline-block !important;
}

/* Neutralize any inner nested divs/paragraphs from Gradio inside user bubble */
#sorin-chat-history [data-testid="user"] *,
#sorin-chat-history .message.user .message-bubble-border *,
#sorin-chat-history [data-testid="user"] div,
#sorin-chat-history [data-testid="user"] p,
#sorin-chat-history [data-testid="user"] span {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--user-bubble-text) !important;
    word-break: normal !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    margin: 0 !important;
    padding: 0 !important;
    display: inline !important;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* Assistant Message Bubble — STRICT SINGLE BUBBLE                           */
/* ───────────────────────────────────────────────────────────────────────── */
#sorin-chat-history .message.bot,
#sorin-chat-history .message-row:has([data-testid="bot"]) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 4px auto 10px 0 !important;
    max-width: 94% !important;
}

#sorin-chat-history [data-testid="bot"],
#sorin-chat-history .message.bot > .message-bubble-border,
#sorin-chat-history .message.bot > div:first-child {
    background: var(--bot-bubble-bg) !important;
    background-color: var(--bot-bubble-bg) !important;
    border: 1px solid var(--bot-bubble-border) !important;
    border-radius: 16px 16px 16px 4px !important;
    color: var(--bot-bubble-text) !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    line-height: 1.58 !important;
    margin: 0 !important;
}

/* Neutralize any inner nested containers inside bot bubble */
#sorin-chat-history [data-testid="bot"] > div,
#sorin-chat-history .message.bot .message-bubble-border > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#sorin-chat-history [data-testid="bot"] *,
#sorin-chat-history .bot * {
    color: var(--bot-bubble-text) !important;
    font-size: 0.88rem !important;
}

#sorin-chat-history .bot h3,
#sorin-chat-history [data-testid="bot"] h3 {
    color: var(--primary) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin: 4px 0 6px 0 !important;
}
#sorin-chat-history .bot h4,
#sorin-chat-history [data-testid="bot"] h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    margin: 6px 0 2px 0 !important;
}
#sorin-chat-history .bot ul,
#sorin-chat-history [data-testid="bot"] ul {
    padding-left: 16px !important;
    margin: 4px 0 !important;
}

/* Animated live typing dots indicator & Clinical Thinking Capsule */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    vertical-align: middle;
    margin-right: 6px;
}
.typing-indicator span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--primary);
    display: inline-block;
    animation: typing-bounce 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
.typing-indicator span:nth-child(3) { animation-delay: 0s; }

@keyframes typing-bounce {
    0%, 80%, 100% {
        transform: scale(0.6);
        opacity: 0.35;
    }
    40% {
        transform: scale(1.15);
        opacity: 1;
    }
}

/* ───────────────────────────────────────────────────────────────────────── */
/* Premium Clinical Thinking Capsule & Emergency Triage Alert Card           */
/* ───────────────────────────────────────────────────────────────────────── */
.clinical-thinking-card {
    display: flex !important;
    flex-direction: column !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin: 2px 0 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
    position: relative !important;
    overflow: hidden !important;
    max-width: 480px !important;
    transition: all 0.25s ease !important;
}

.thinking-glow-bar {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981, #3b82f6) !important;
    background-size: 200% 100% !important;
    animation: shimmerGlow 2s infinite linear !important;
}

@keyframes shimmerGlow {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

.thinking-body {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    width: 100% !important;
}

.thinking-ecg-wrap {
    width: 38px !important;
    height: 38px !important;
    border-radius: 10px !important;
    background: var(--primary-light) !important;
    border: 1px solid var(--border-strong) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
    color: var(--primary) !important;
    position: relative !important;
    box-shadow: 0 0 12px var(--primary-light) !important;
    animation: pulseWrapGlow 2.2s infinite ease-in-out !important;
}

@keyframes pulseWrapGlow {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 8px var(--primary-light);
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 0 16px var(--primary-light), 0 0 6px var(--primary);
    }
}

.thinking-medical-svg {
    display: block !important;
    width: 22px !important;
    height: 22px !important;
    color: var(--primary) !important;
    filter: drop-shadow(0 0 3px var(--primary));
    overflow: visible !important;
}

.thinking-medical-svg .heart-bg {
    fill: var(--primary) !important;
    fill-opacity: 0.2 !important;
    stroke: var(--primary) !important;
    stroke-opacity: 0.6 !important;
    stroke-width: 1.5 !important;
}

.thinking-medical-svg .ecg-pulse-line {
    stroke: var(--primary) !important;
    stroke-width: 2.2 !important;
    stroke-linecap: round !important;
    stroke-linejoin: round !important;
    filter: drop-shadow(0 0 3px var(--primary));
    animation: ecgLineGlow 1.6s infinite ease-in-out !important;
}

@keyframes ecgLineGlow {
    0%, 100% {
        filter: drop-shadow(0 0 2px var(--primary));
        opacity: 0.85;
    }
    50% {
        filter: drop-shadow(0 0 7px var(--primary)) drop-shadow(0 0 10px #38bdf8);
        opacity: 1;
    }
}

.thinking-content {
    flex: 1 !important;
    min-width: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}

.thinking-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
}

.thinking-badge {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    color: var(--primary) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    text-transform: uppercase !important;
}

.badge-dot {
    width: 6px !important;
    height: 6px !important;
    border-radius: 50% !important;
    background: var(--primary) !important;
    box-shadow: 0 0 6px var(--primary) !important;
    animation: pulseDot 1.2s infinite alternate ease-in-out !important;
}

@keyframes pulseDot {
    0% { transform: scale(0.8); opacity: 0.5; }
    100% { transform: scale(1.3); opacity: 1; }
}

.thinking-dots {
    display: inline-flex !important;
    align-items: center !important;
    gap: 3px !important;
}
.thinking-dots span {
    width: 4px !important;
    height: 4px !important;
    border-radius: 50% !important;
    background: var(--primary) !important;
    animation: typing-bounce 1.4s infinite ease-in-out both !important;
}
.thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
.thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
.thinking-dots span:nth-child(3) { animation-delay: 0s; }

.thinking-subtext {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    line-height: 1.3 !important;
}

/* ── EMERGENCY TRIAGE ALERT CARD ────────────────────────────────────────── */
.clinical-thinking-card.emergency-mode {
    border: 1.5px solid #ef4444 !important;
    background: rgba(239, 68, 68, 0.07) !important;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.3) !important;
    animation: emergencyCardThrob 1.2s infinite alternate ease-in-out !important;
}

@keyframes emergencyCardThrob {
    0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
    100% { box-shadow: 0 0 26px rgba(239, 68, 68, 0.5); }
}

.thinking-glow-bar.emergency {
    background: linear-gradient(90deg, #ef4444, #f59e0b, #dc2626, #ef4444) !important;
    animation: shimmerGlow 0.8s infinite linear !important;
}

.thinking-siren-wrap {
    width: 38px !important;
    height: 38px !important;
    border-radius: 10px !important;
    background: rgba(239, 68, 68, 0.16) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.25rem !important;
    flex-shrink: 0 !important;
    position: relative !important;
    animation: sirenShake 0.4s infinite alternate ease-in-out !important;
}

@keyframes sirenShake {
    0% { transform: rotate(-10deg) scale(0.95); }
    100% { transform: rotate(10deg) scale(1.15); }
}

.emergency-badge {
    font-size: 0.67rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    color: #ef4444 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    text-transform: uppercase !important;
}

.emergency-dot {
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: #ef4444 !important;
    box-shadow: 0 0 8px #ef4444 !important;
    animation: pulseDot 0.5s infinite alternate ease-in-out !important;
}

.emergency-dots {
    display: inline-flex !important;
    align-items: center !important;
    gap: 3px !important;
}
.emergency-dots span {
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
    background: #ef4444 !important;
    animation: typing-bounce 0.8s infinite ease-in-out both !important;
}

.thinking-subtext.emergency {
    color: #ef4444 !important;
    font-weight: 600 !important;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* ───────────────────────────────────────────────────────────────────────── */
/* Input Bar Capsule — Pixel-Perfect Optical Alignment                      */
/* ───────────────────────────────────────────────────────────────────────── */
#sorin-input-bar,
#sorin-input-bar.row,
#sorin-input-bar.gradio-row {
    background: var(--bg-surface) !important;
    border: 1.5px solid var(--border-strong) !important;
    border-radius: 9999px !important;
    padding: 0 8px 0 20px !important;
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.04) !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 0 !important;
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    flex-shrink: 0 !important;
    margin-top: auto !important;
    position: relative !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

#sorin-input-bar:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-light) !important;
}

/* Remove default Gradio border, background, and margin from inner elements */
#sorin-input-bar label,
#sorin-input-bar .show_textbox_border,
#sorin-input-bar .block,
#sorin-input-bar .form,
#sorin-input-bar .wrap,
#sorin-input-bar .prose,
#sorin-input-bar [class*="svelte-"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: normal !important;
}

/* Hide any Gradio progress/loading indicators that break the input capsule */
#sorin-input-bar .progress-level,
#sorin-input-bar .progress-bar,
#sorin-input-bar .loading,
#sorin-input-bar .meta-text,
#sorin-input-bar [data-testid="block-info"],
#sorin-input-bar .progress,
#sorin-input-bar .generating,
#sorin-text-input .progress-level,
#sorin-text-input .progress-bar,
#sorin-text-input .loading,
#sorin-text-input .meta-text,
#sorin-text-input [data-testid="block-info"],
#sorin-text-input .progress {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Child 1: Text input container (takes remaining space) */
#sorin-input-bar > label,
#sorin-input-bar > .gradio-textbox,
#sorin-input-bar > div:first-child,
#sorin-input-bar #sorin-text-input {
    flex: 1 1 0% !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 100% !important;
    min-height: 48px !important;
    max-height: 48px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

#sorin-input-bar label span {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

#sorin-text-input input,
#sorin-text-input textarea,
#sorin-input-bar input {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    padding: 0 8px 0 0 !important;
    margin: auto 0 !important;
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    line-height: 36px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    display: block !important;
    vertical-align: middle !important;
}

#sorin-text-input input::placeholder,
#sorin-text-input textarea::placeholder,
#sorin-input-bar input::placeholder {
    color: var(--text-muted) !important;
    line-height: 36px !important;
}

/* Child 2: Action group wrapper (icons & send button) */
#sorin-input-bar > .gradio-html,
#sorin-input-bar > div:nth-child(2),
#sorin-input-bar > *:has(.input-actions-wrap) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

#sorin-input-bar .prose {
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

.input-actions-wrap {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    flex-shrink: 0 !important;
}

.input-action-btn {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    padding: 0 !important;
    margin: auto 0 !important;
    box-sizing: border-box !important;
    vertical-align: middle !important;
    flex-shrink: 0 !important;
}

.input-action-btn svg {
    display: block !important;
    margin: auto !important;
    width: 16px !important;
    height: 16px !important;
    stroke: currentColor !important;
}

.input-action-btn:hover {
    background: var(--bg-sidebar) !important;
    color: var(--text-primary) !important;
}

.input-action-btn.active {
    background: rgba(239, 68, 68, 0.12) !important;
    color: #ef4444 !important;
}

/* 3. Concentric Circular Send Button */
.input-send-circle-btn {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    border-radius: 50% !important;
    background: var(--send-btn-bg) !important;
    color: var(--send-btn-color) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    cursor: pointer !important;
    padding: 0 !important;
    margin: auto 0 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12) !important;
    box-sizing: border-box !important;
    transition: all 0.15s ease !important;
    flex-shrink: 0 !important;
    vertical-align: middle !important;
}

.input-send-circle-btn svg {
    display: block !important;
    margin: auto !important;
    stroke: var(--send-btn-color) !important;
}

.input-send-circle-btn:hover {
    background: var(--send-btn-hover) !important;
    transform: scale(1.06);
}

.input-send-circle-btn:active {
    transform: scale(0.94);
}

/* Move Gradio's background send_btn completely out of the viewport */
#sorin-send-btn,
#sorin-input-bar > button#sorin-send-btn,
#sorin-input-bar > div:has(#sorin-send-btn),
#sorin-input-bar > div:last-child:has(button) {
    position: absolute !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    overflow: hidden !important;
}

/* Minimalist Footer */
.minimal-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 3px;
    padding: 0 4px;
    font-size: 0.62rem;
    color: var(--text-muted);
    flex-shrink: 0;
}

footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    initial_sessions = {
        "Consultation 1": []
    }
    initial_state = {
        "sessions": initial_sessions,
        "active_id": "Consultation 1",
        "counter": 1,
    }

    with gr.Blocks(
        title="Sorin-AI Clinical Assistant",
    ) as demo:

        session_store = gr.State(initial_state)

        with gr.Column(elem_id="sorin-app-shell"):

            # ── Top Navbar ────────────────────────────────────────────────
            gr.HTML(
                """
                <div id="sorin-navbar">
                    <div class="sorin-brand-wrap">
                        <div class="sorin-logo-badge">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                            </svg>
                        </div>
                        <div>
                            <span class="sorin-brand-title">Sorin-AI</span>
                            <span class="sorin-brand-subtitle" style="margin-left: 6px;">Clinical Knowledge Assistant</span>
                        </div>
                    </div>

                    <div class="sorin-navbar-actions">
                        <div class="theme-picker-wrap">
                            <span class="theme-picker-label">🎨</span>
                            <select id="theme-select" class="theme-select-dropdown" onchange="switchTheme(this.value)">
                                <option value="executive">🌟 Executive Clinical</option>
                                <option value="sage">🌿 Sage & Serenity</option>
                                <option value="obsidian-dark">🖤 Obsidian Pro Dark</option>
                                <option value="claude">📜 Warm Editorial</option>
                                <option value="claude-dark">☕ Warm Espresso</option>
                            </select>
                        </div>
                        <button id="theme-quick-toggle" class="theme-quick-toggle-btn" onclick="toggleQuickLightDark()" title="Quick Switch Light / Dark">
                            <span id="quick-toggle-icon">🌙</span>
                        </button>
                        <div class="status-pill">
                            <span class="status-dot"></span> Clinical RAG
                        </div>
                    </div>
                </div>
                """
            )

            # ── Main 2-Column Layout ──────────────────────────────────────
            with gr.Row(elem_id="sorin-layout"):

                # ── Left Sidebar (History & Topics) ───────────────────────
                with gr.Column(elem_id="sorin-sidebar", scale=1, min_width=230):

                    # Interactive Robot Avatar Companion Card
                    gr.HTML(
                        """
                        <div id="sorin-robot-companion" class="robot-companion-card" data-variant="classic">
                            <!-- Top Bar: Speech bubble & sleek Settings Icon -->
                            <div class="robot-card-topbar">
                                <div id="robot-speech-bubble" class="robot-speech-bubble">
                                    <span id="robot-speech-text">Hi! I'm Dr. Sorin 🩺 Grab me to fly!</span>
                                    <div class="speech-bubble-arrow"></div>
                                </div>
                                <button type="button" id="robot-settings-btn" class="robot-settings-btn" onclick="toggleRobotSettings(event)" title="Settings & Actions (Skins, Dance, ECG)">⚙️</button>
                            </div>

                            <!-- Settings & Actions Popover Panel (Hidden by default, opened via ⚙️) -->
                            <div id="robot-settings-panel" class="robot-settings-panel">
                                <div class="settings-panel-header">
                                    <span class="settings-panel-title">Avatar Settings</span>
                                    <button type="button" class="settings-close-btn" onclick="toggleRobotSettings(event)" title="Close">✕</button>
                                </div>

                                <!-- Skin Variants Selector -->
                                <div class="robot-variant-bar">
                                    <span class="variant-label">Skin:</span>
                                    <button type="button" class="variant-dot classic active" onclick="switchRobotVariant('classic', event)" title="Classic Dr. Sorin"></button>
                                    <button type="button" class="variant-dot cyber" onclick="switchRobotVariant('cyber', event)" title="Cyber-Pulse Neon"></button>
                                    <button type="button" class="variant-dot sage" onclick="switchRobotVariant('sage', event)" title="Sage Botanical Healer"></button>
                                    <button type="button" class="variant-dot cryo" onclick="switchRobotVariant('cryo', event)" title="Cryo-Doc Arctic"></button>
                                    <button type="button" class="variant-dot gold" onclick="switchRobotVariant('gold', event)" title="Chrono-Lux Obsidian Gold"></button>
                                </div>

                                <!-- Action Buttons Grid -->
                                <div class="robot-actions-grid">
                                    <button type="button" class="robot-act-btn" onclick="robotAction('wave', event)" title="Wave hello">👋 Wave</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('dance', event)" title="Dance mode">💃 Dance</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('love', event)" title="Care & Compassion">❤️ Love</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('boost', event)" title="Rocket Thruster Boost">🚀 Boost</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('vitals', event)" title="Check vitals ECG">🩺 Vitals</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('scan', event)" title="Scan clinical database">🔍 Scan</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('spin', event)" title="Perform spin flip">✨ Flip</button>
                                    <button type="button" class="robot-act-btn emergency-btn" onclick="robotAction('emergency', event)" title="Emergency Alert Simulation">🚨 Alert</button>
                                    <button type="button" class="robot-act-btn" onclick="robotAction('tip', event)" title="Evidence-based health tip">💡 Tip</button>
                                </div>
                            </div>

                            <!-- Docking Bay: Holds Dr. Sorin himself OR in-flight recall placeholder -->
                            <div id="robot-docking-bay" class="robot-docking-bay">
                                <!-- In-flight Placeholder in sidebar (shown when Dr. Sorin is flying) -->
                                <div id="robot-dock-placeholder" class="robot-dock-placeholder" onclick="dockRobot(event)" title="Click to recall Dr. Sorin">
                                    <div class="placeholder-icon">🛸</div>
                                    <div class="placeholder-text">Dr. Sorin is in flight</div>
                                    <button type="button" class="recall-btn" onclick="dockRobot(event)">📌 Recall</button>
                                </div>

                                <!-- The Draggable Robot Character (DRAGGING ONLY HIM) -->
                                <div id="sorin-robot-character" class="robot-character" data-variant="classic" title="Grab me to drag anywhere!">
                                    <!-- Dock Pin Button (shown when floating) -->
                                    <button type="button" class="character-dock-pin" onclick="dockRobot(event)" title="Return to sidebar">📌</button>

                                    <!-- Floating hearts container for Love mode -->
                                    <div id="robot-hearts-container" class="floating-hearts-container" style="position: absolute; inset: 0; pointer-events: none; overflow: visible;"></div>

                                    <!-- The Animated SVG Robot -->
                                    <svg id="sorin-robot-svg" viewBox="0 0 160 185" width="105" height="118" class="robot-svg">
                                        <defs>
                                            <linearGradient id="robot-metal" x1="0%" y1="0%" x2="100%" y2="100%">
                                                <stop offset="0%" style="stop-color: var(--r-metal-top, #ffffff);"/>
                                                <stop offset="50%" style="stop-color: var(--r-metal-mid, #e2e8f0);"/>
                                                <stop offset="100%" style="stop-color: var(--r-metal-bot, #cbd5e1);"/>
                                            </linearGradient>
                                            <linearGradient id="robot-visor-bg" x1="0%" y1="0%" x2="0%" y2="100%">
                                                <stop offset="0%" style="stop-color: var(--r-visor-bg, #090d16);"/>
                                                <stop offset="100%" style="stop-color: #050810;"/>
                                            </linearGradient>
                                            <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
                                                <feGaussianBlur stdDeviation="2.5" result="blur"/>
                                                <feComposite in="SourceGraphic" in2="blur" operator="over"/>
                                            </filter>
                                        </defs>

                                        <!-- Shadow -->
                                        <ellipse id="robot-shadow" cx="80" cy="176" rx="34" ry="6" class="robot-shadow-ellipse" />

                                        <!-- Robot Float Group -->
                                        <g id="robot-float-group" class="robot-float-group">
                                            <!-- Left Hand -->
                                            <g id="robot-left-hand" class="robot-hand left-hand">
                                                <ellipse cx="28" cy="115" rx="9" ry="11" fill="url(#robot-metal)" stroke="var(--border-subtle, #94a3b8)" stroke-width="1.2"/>
                                                <circle cx="25" cy="118" r="3" fill="var(--r-eye-glow, #38bdf8)"/>
                                            </g>

                                            <!-- Right Hand -->
                                            <g id="robot-right-hand" class="robot-hand right-hand">
                                                <ellipse cx="132" cy="115" rx="9" ry="11" fill="url(#robot-metal)" stroke="var(--border-subtle, #94a3b8)" stroke-width="1.2"/>
                                                <circle cx="135" cy="118" r="3" fill="var(--r-eye-glow, #38bdf8)"/>
                                            </g>

                                            <!-- Thruster Flame Jets (Active when dragging or rocket boosting) -->
                                            <path id="robot-flame-left" class="thruster-flame" d="M63 150 Q67 172 67 176 Q67 172 71 150 Z" fill="var(--r-eye-glow, #38bdf8)"/>
                                            <path id="robot-flame-right" class="thruster-flame" d="M89 150 Q93 172 93 176 Q93 172 97 150 Z" fill="var(--r-eye-glow, #38bdf8)"/>

                                            <!-- Torso Body -->
                                            <g class="robot-torso">
                                                <rect x="52" y="98" width="56" height="52" rx="16" fill="url(#robot-metal)" stroke="var(--border-subtle, #94a3b8)" stroke-width="1.5"/>
                                                <rect x="62" y="146" width="10" height="5" rx="2" fill="var(--border-strong, #64748b)"/>
                                                <rect x="88" y="146" width="10" height="5" rx="2" fill="var(--border-strong, #64748b)"/>
                                                <!-- Holographic Medical Stethoscope / Core -->
                                                <circle cx="80" cy="122" r="14" fill="#090d16" stroke="var(--r-visor-stroke, #38bdf8)" stroke-width="1.5"/>
                                                <path id="robot-chest-pulse" d="M72 122 h4 l2 -5 l3 10 l3 -8 l2 3 h4" fill="none" stroke="var(--r-pulse-color, #38bdf8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                                            </g>

                                            <!-- Neck Ring -->
                                            <rect x="68" y="88" width="24" height="12" rx="4" fill="var(--border-strong, #64748b)"/>

                                            <!-- Robot Head Group -->
                                            <g id="robot-head-group" class="robot-head-group">
                                                <line x1="80" y1="28" x2="80" y2="40" stroke="var(--border-strong, #64748b)" stroke-width="2.5" stroke-linecap="round"/>
                                                <circle id="robot-antenna-orb" cx="80" cy="24" r="5" fill="var(--r-eye-glow, #38bdf8)" filter="url(#neon-glow)" class="antenna-orb"/>

                                                <rect x="34" y="55" width="8" height="18" rx="3" fill="var(--border-strong, #64748b)"/>
                                                <circle cx="38" cy="64" r="2" fill="var(--r-accent, #2563eb)"/>
                                                <rect x="118" y="55" width="8" height="18" rx="3" fill="var(--border-strong, #64748b)"/>
                                                <circle cx="122" cy="64" r="2" fill="var(--r-accent, #2563eb)"/>

                                                <rect x="40" y="38" width="80" height="54" rx="20" fill="url(#robot-metal)" stroke="var(--border-subtle, #94a3b8)" stroke-width="1.5"/>
                                                <rect x="47" y="45" width="66" height="40" rx="14" fill="url(#robot-visor-bg)" stroke="var(--r-visor-stroke, #38bdf8)" stroke-width="1.2"/>

                                                <!-- Interactive Eyes Group -->
                                                <g id="robot-eyes-group" class="robot-eyes-group">
                                                    <g id="robot-left-eye" class="robot-eye">
                                                        <rect class="eye-shape" x="58" y="56" width="13" height="17" rx="5" fill="var(--r-eye-glow, #38bdf8)" filter="url(#neon-glow)"/>
                                                        <circle class="eye-pupil" cx="64" cy="64" r="2.8" fill="#ffffff"/>
                                                    </g>
                                                    <g id="robot-right-eye" class="robot-eye">
                                                        <rect class="eye-shape" x="89" y="56" width="13" height="17" rx="5" fill="var(--r-eye-glow, #38bdf8)" filter="url(#neon-glow)"/>
                                                        <circle class="eye-pupil" cx="95" cy="64" r="2.8" fill="#ffffff"/>
                                                    </g>
                                                </g>

                                                <!-- Visor Mouth Wave -->
                                                <g id="robot-mouth-wave" class="robot-mouth">
                                                    <line x1="68" y1="77" x2="92" y2="77" stroke="var(--r-visor-stroke, #38bdf8)" stroke-width="2" stroke-linecap="round" class="mouth-line"/>
                                                </g>
                                            </g>
                                        </g>
                                    </svg>
                                </div>
                            </div>

                            <!-- Status & Mood Badge -->
                            <div class="robot-info-row">
                                <span class="robot-name">Dr. Sorin 🤖</span>
                                <span id="robot-status-pill" class="robot-status-pill ready">Ready</span>
                            </div>
                        </div>
                        """
                    )

                    with gr.Row(elem_classes=["sidebar-action-row"]):
                        new_chat_btn = gr.Button("＋ New Chat", elem_id="new-chat-btn")
                        delete_chat_btn = gr.Button("Delete", elem_id="delete-chat-btn")

                    gr.HTML('<div class="sidebar-title">Consultations</div>')
                    history_radio = gr.Radio(
                        choices=["Consultation 1"],
                        value="Consultation 1",
                        label="",
                        show_label=False,
                        elem_id="history-radio",
                    )

                    gr.HTML(
                        """
                        <div class="sidebar-title" style="margin-top: 10px;">Clinical Inquiries</div>
                        <div class="sidebar-topics-wrap">
                            <button type="button" class="sidebar-topic-btn" onclick="sendExamplePrompt('What are common symptoms and causes of iron deficiency anemia?')">Iron deficiency anemia</button>
                            <button type="button" class="sidebar-topic-btn" onclick="sendExamplePrompt('What is the difference between a cold and influenza?')">Cold vs. influenza difference</button>
                            <button type="button" class="sidebar-topic-btn" onclick="sendExamplePrompt('What are common side effects and precautions of ibuprofen?')">Ibuprofen precautions & safety</button>
                            <button type="button" class="sidebar-topic-btn" onclick="sendExamplePrompt('When should chest pain be treated as a medical emergency?')">Chest pain emergency red flags</button>
                            <button type="button" class="sidebar-topic-btn" onclick="sendExamplePrompt('What does high blood pressure mean and what are the stages?')">Blood pressure stages & guide</button>
                        </div>
                        """
                    )

                # ── Main Chat Panel ───────────────────────────────────────
                with gr.Column(elem_id="sorin-main-panel", scale=4):

                    # Clean Minimal Prompt Chips with Typewriter animation into input box
                    gr.HTML(
                        """
                        <div id="prompt-chips-row">
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('What are early warning symptoms of Type 2 Diabetes?')">Type 2 Diabetes</button>
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('What are the uses, common side effects, and safety warnings of ibuprofen?')">Ibuprofen Safety</button>
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('When should chest pain be treated as a medical emergency?')">Chest Pain Flags</button>
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('What are symptoms and causes of iron deficiency anemia?')">Anemia Symptoms</button>
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('What is hypertension and how does high blood pressure affect health?')">Blood Pressure Guide</button>
                            <button type="button" class="quick-chip" onclick="sendExamplePrompt('What are common symptoms of dehydration and how is it treated?')">Dehydration Signs</button>
                        </div>
                        """
                    )

                    # Chatbot messages stream (Expands to fill 560px+)
                    chatbot = gr.Chatbot(
                        label="",
                        elem_id="sorin-chat-history",
                        avatar_images=(USER_AVATAR, BOT_AVATAR),
                        render_markdown=True,
                        show_label=False,
                        placeholder="💬 Your clinical consultation will appear here. Choose an inquiry above or enter a health question below.",
                    )

                    # Modern Minimalist Capsule Input Bar
                    with gr.Row(elem_id="sorin-input-bar"):
                        message_input = gr.Textbox(
                            placeholder="Ask a clinical question about symptoms, medications, or guidelines...",
                            show_label=False,
                            elem_id="sorin-text-input",
                            container=False,
                            lines=1,
                            max_lines=1,
                            autofocus=True,
                            scale=10,
                        )

                        # SVG Icon Actions + Visual Send Button (Unified for Perfect Alignment)
                        gr.HTML(
                            """
                            <div class="input-actions-wrap">
                                <button type="button" id="sorin-mic-btn" class="input-action-btn" onclick="toggleMic()" title="Voice input (Speech to Text)">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                        <line x1="12" y1="19" x2="12" y2="23"></line>
                                        <line x1="8" y1="23" x2="16" y2="23"></line>
                                    </svg>
                                </button>
                                <button type="button" id="sorin-tts-btn" class="input-action-btn" onclick="toggleTTS()" title="Read aloud (Text to Speech)">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                                    </svg>
                                </button>
                                <button type="button" id="sorin-visual-send-btn" class="input-send-circle-btn" onclick="triggerSend()" title="Send clinical inquiry">
                                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                        <line x1="12" y1="19" x2="12" y2="5"></line>
                                        <polyline points="5 12 12 5 19 12"></polyline>
                                    </svg>
                                </button>
                            </div>
                            """
                        )

                        send_btn = gr.Button("↑", elem_id="sorin-send-btn", variant="secondary", scale=1)

                    # Minimal 1-Line Footer
                    gr.HTML(
                        """
                        <div class="minimal-footer">
                            <span>Evidence-Based Clinical Retrieval: PubMed · WHO · CDC Guidelines · FAISS</span>
                            <span>Educational medical AI. Always consult a physician for clinical diagnosis.</span>
                        </div>
                        """
                    )

        # ── Interactive Handlers ──────────────────────────────────────────

        send_btn.click(
            fn=handle_user_send,
            inputs=[message_input, chatbot, session_store],
            outputs=[chatbot, message_input, history_radio, session_store],
            show_progress="hidden",
        )

        message_input.submit(
            fn=handle_user_send,
            inputs=[message_input, chatbot, session_store],
            outputs=[chatbot, message_input, history_radio, session_store],
            show_progress="hidden",
        )

        new_chat_btn.click(
            fn=create_new_chat,
            inputs=[session_store],
            outputs=[chatbot, message_input, history_radio, session_store],
            show_progress="hidden",
        )

        delete_chat_btn.click(
            fn=delete_active_chat,
            inputs=[session_store],
            outputs=[chatbot, message_input, history_radio, session_store],
            show_progress="hidden",
        )

        history_radio.change(
            fn=switch_chat,
            inputs=[history_radio, session_store],
            outputs=[chatbot, session_store],
            show_progress="hidden",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Sorin-AI Clinical Assistant …")
    _initialize_backend()

    demo = build_ui()
    _META_HEAD = r"""
    <meta name="description" content="Sorin-AI Medical Assistant: Intelligent clinical knowledge and health Q&A.">
    <script>
    function getStoredTheme() {
        try {
            const saved = localStorage.getItem('sorin_chosen_theme');
            if (saved && ['executive', 'sage', 'obsidian-dark', 'claude', 'claude-dark'].includes(saved)) {
                return saved;
            }
        } catch (e) {}
        return 'executive';
    }

    function switchTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
        const container = document.querySelector('.gradio-container');
        if (container) {
            container.setAttribute('data-theme', theme);
        }
        const select = document.getElementById('theme-select');
        if (select) {
            select.value = theme;
        }
        const quickIcon = document.getElementById('quick-toggle-icon');
        if (quickIcon) {
            quickIcon.innerText = theme.includes('dark') ? '☀️' : '🌙';
        }
        try {
            localStorage.setItem('sorin_chosen_theme', theme);
        } catch (e) {}
    }

    function toggleQuickLightDark() {
        const current = document.documentElement.getAttribute('data-theme') || 'executive';
        if (current.includes('dark')) {
            switchTheme('executive');
        } else {
            switchTheme('obsidian-dark');
        }
    }

    function initTheme() {
        const theme = getStoredTheme();
        switchTheme(theme);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }
    window.addEventListener('load', initTheme);

    // Emergency detection helper for immediate client-side reaction
    function isEmergencyPrompt(text) {
        if (!text) return false;
        const lower = text.toLowerCase();
        const emergencyTerms = [
            'chest pain', 'heart attack', 'cardiac', 'cannot breathe', "can't breathe",
            'shortness of breath', 'stroke', 'slurred speech', 'drooping', 'unconscious',
            'passed out', 'bleeding', 'seizure', 'anaphylaxis', 'overdose', 'poison',
            'suicide', 'kill myself', 'severe trauma', 'emergency', 'fainted'
        ];
        return emergencyTerms.some(term => lower.includes(term));
    }

    // Live typewriter animation into text input box
    let typingAnimationTimer = null;

    function sendExamplePrompt(promptText) {
        if (!promptText) return;

        if (isEmergencyPrompt(promptText)) {
            triggerEmergencyAlert(true, "🚨 EMERGENCY DETECTED: Analyzing acute clinical flags!");
        } else {
            setRobotState('thinking');
        }

        if (typingAnimationTimer) {
            clearInterval(typingAnimationTimer);
            typingAnimationTimer = null;
        }

        const inputArea = document.querySelector('#sorin-text-input input, #sorin-text-input textarea');
        if (!inputArea) return;

        inputArea.focus();
        inputArea.value = '';
        inputArea.dispatchEvent(new Event('input', { bubbles: true }));

        let idx = 0;
        const speed = 10; // ms per char: swift, sleek animation

        typingAnimationTimer = setInterval(() => {
            if (idx < promptText.length) {
                inputArea.value += promptText.charAt(idx);
                inputArea.dispatchEvent(new Event('input', { bubbles: true }));
                idx++;
            } else {
                clearInterval(typingAnimationTimer);
                typingAnimationTimer = null;
                inputArea.dispatchEvent(new Event('change', { bubbles: true }));

                // Automatically click send after text finishes populating the box
                setTimeout(() => {
                    const sendBtn = document.getElementById('sorin-send-btn');
                    if (sendBtn) {
                        sendBtn.click();
                    }
                }, 80);
            }
        }, speed);
    }

    function triggerSend() {
        const inputArea = document.querySelector('#sorin-text-input input, #sorin-text-input textarea');
        const text = inputArea ? inputArea.value : '';
        if (isEmergencyPrompt(text)) {
            triggerEmergencyAlert(true, "🚨 CRITICAL TRIAGE ALERT: Scanning emergency clinical protocols!");
        } else {
            setRobotState('thinking');
        }

        const sendBtn = document.getElementById('sorin-send-btn');
        if (sendBtn) {
            sendBtn.click();
        }
    }

    // Voice recognition
    let speechRecognitionInstance = null;
    let isRecordingMic = false;

    function toggleMic() {
        const micBtn = document.getElementById('sorin-mic-btn');
        const inputArea = document.querySelector('#sorin-text-input input, #sorin-text-input textarea');

        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {
            alert("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
            return;
        }

        if (isRecordingMic && speechRecognitionInstance) {
            speechRecognitionInstance.stop();
            return;
        }

        try {
            speechRecognitionInstance = new SpeechRec();
            speechRecognitionInstance.continuous = false;
            speechRecognitionInstance.interimResults = true;
            speechRecognitionInstance.lang = 'en-US';

            speechRecognitionInstance.onstart = function() {
                isRecordingMic = true;
                if (micBtn) micBtn.classList.add('active');
                setRobotState('listening');
            };

            speechRecognitionInstance.onresult = function(event) {
                let transcriptText = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    transcriptText += event.results[i][0].transcript;
                }
                if (inputArea && transcriptText) {
                    inputArea.value = transcriptText;
                    inputArea.dispatchEvent(new Event('input', { bubbles: true }));
                    inputArea.dispatchEvent(new Event('change', { bubbles: true }));
                }
            };

            speechRecognitionInstance.onerror = function() { stopMic(); };
            speechRecognitionInstance.onend = function() { stopMic(); };
            speechRecognitionInstance.start();
        } catch (e) {
            console.error('Mic error:', e);
            stopMic();
        }
    }

    function stopMic() {
        isRecordingMic = false;
        const micBtn = document.getElementById('sorin-mic-btn');
        if (micBtn) micBtn.classList.remove('active');
        setRobotState('ready');
    }

    // Text-to-Speech
    let isSpeakingAloud = false;

    function toggleTTS() {
        const ttsBtn = document.getElementById('sorin-tts-btn');
        if (!('speechSynthesis' in window)) {
            alert("Text-to-speech is not supported in this browser.");
            return;
        }

        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
            isSpeakingAloud = false;
            if (ttsBtn) ttsBtn.classList.remove('active');
            return;
        }

        const botMessages = document.querySelectorAll('#sorin-chat-history [data-testid="bot"], #sorin-chat-history .bot');
        if (!botMessages || botMessages.length === 0) {
            alert("No clinical message to read yet.");
            return;
        }

        const lastBot = botMessages[botMessages.length - 1];
        let spokenText = lastBot.innerText || lastBot.textContent || "";

        spokenText = spokenText
            .replace(/###+/g, '')
            .replace(/\*\*/g, '')
            .replace(/---+/g, '')
            .replace(/https?:\/\/\S+/g, '')
            .replace(/⚠️/g, 'Warning:')
            .replace(/🩺|💊|🩸|🫀|🧪|💧|🧬|📚|💡/g, '')
            .trim();

        if (!spokenText) return;

        const utter = new SpeechSynthesisUtterance(spokenText);
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.lang = 'en-US';

        utter.onstart = function() {
            isSpeakingAloud = true;
            if (ttsBtn) ttsBtn.classList.add('active');
        };

        utter.onend = function() {
            isSpeakingAloud = false;
            if (ttsBtn) ttsBtn.classList.remove('active');
        };

        utter.onerror = function() {
            isSpeakingAloud = false;
            if (ttsBtn) ttsBtn.classList.remove('active');
        };

        window.speechSynthesis.speak(utter);
    }

    // ── Interactive Robot Companion Logic (Variants, Draggable Physics, Actions) ──
    const ROBOT_TIPS = [
        "💧 Proper hydration optimizes renal filtration & cognitive focus!",
        "🚶 30 mins of daily moderate walking lowers hypertension risk by 20%!",
        "🥗 A balanced diet rich in leafy greens supports arterial elasticity.",
        "😴 7-8 hours of sleep allows neural glymphatic clearance!",
        "🩺 Early symptom tracking prevents chronic disease progression.",
        "🧘 5 minutes of box breathing (4-4-4-4) lowers acute cortisol levels.",
        "💊 Always finish prescribed antibiotics to prevent bacterial resistance."
    ];

    const POKE_QUIPS = [
        "Beep boop! All clinical telemetry nominal! 🩺",
        "Hello! Dr. Sorin ready to assist your inquiry! 👋",
        "Clinical neural circuits running at peak efficiency! ⚡",
        "Did you know? Regular exercise boosts natural killer cell activity!",
        "*happy robotic chirp* Grab me to fly anywhere! 🚀",
        "ECG rhythm steady! How can I help today? ❤️"
    ];

    const VARIANT_GREETINGS = {
        'classic': "Dr. Sorin: Classic Clinical White initialized! 🩺",
        'cyber': "Dr. Sorin: Cyber-Pulse Neon online! ⚡ All systems optimal.",
        'sage': "Dr. Sorin: Botanical Sage Healer mode active! 🌿 Gentle care.",
        'cryo': "Dr. Sorin: Cryo-Doc Arctic active! ❄️ Sub-zero precision.",
        'gold': "Dr. Sorin: Chrono-Lux 24K Obsidian equipped! ✨ Elite diagnostic."
    };

    function playRobotTone(freq1, freq2, duration, type = 'sine') {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = type;
            osc.connect(gain);
            gain.connect(ctx.destination);
            gain.gain.setValueAtTime(0.04, ctx.currentTime);
            osc.frequency.setValueAtTime(freq1, ctx.currentTime);
            if (freq2) {
                osc.frequency.exponentialRampToValueAtTime(Math.max(10, freq2), ctx.currentTime + duration * 0.7);
            }
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
            osc.start();
            osc.stop(ctx.currentTime + duration);
        } catch(e) {}
    }

    function playRobotArpeggio(notes, delay = 0.08) {
        notes.forEach((f, idx) => {
            setTimeout(() => playRobotTone(f, null, 0.16, 'triangle'), idx * (delay * 1000));
        });
    }

    let speechBubbleTimer = null;
    function setRobotSpeech(text, duration = 4000) {
        const bubble = document.getElementById('robot-speech-bubble');
        const bubbleText = document.getElementById('robot-speech-text');
        if (!bubble || !bubbleText) return;
        bubbleText.innerText = text;
        bubble.style.opacity = '1';
        bubble.style.transform = 'scale(1)';
        if (speechBubbleTimer) clearTimeout(speechBubbleTimer);
        if (duration > 0) {
            speechBubbleTimer = setTimeout(() => {
                bubble.style.opacity = '0.9';
            }, duration);
        }
    }

    function playEmergencyAlertSound() {
        playRobotTone(880, 440, 0.22, 'sawtooth');
        setTimeout(() => playRobotTone(980, 520, 0.25, 'sawtooth'), 240);
    }

    let emergencyResetTimer = null;
    function triggerEmergencyAlert(active = true, customMessage = null) {
        const card = document.getElementById('sorin-robot-companion');
        const character = document.getElementById('sorin-robot-character');
        const pill = document.getElementById('robot-status-pill');
        const bubble = document.getElementById('robot-speech-bubble');

        if (emergencyResetTimer) {
            clearTimeout(emergencyResetTimer);
            emergencyResetTimer = null;
        }

        if (active) {
            if (card) {
                card.classList.remove('robot-waving', 'robot-spinning', 'robot-vitals-active', 'robot-dancing', 'robot-loving', 'robot-boosting', 'robot-thinking', 'robot-speaking');
                card.classList.add('robot-nervous-alert');
            }
            if (character) character.classList.add('robot-nervous-alert');
            if (pill) {
                pill.className = 'robot-status-pill emergency';
                pill.innerText = 'EMERGENCY 🚨';
            }
            if (bubble) bubble.classList.add('speech-bubble-emergency');

            playEmergencyAlertSound();

            const msg = customMessage || "🚨 EMERGENCY ALERT: High-risk clinical flags detected! Please seek immediate emergency medical care!";
            setRobotSpeech(msg, 16000);

            // Auto-relax after 24 seconds if not re-triggered
            emergencyResetTimer = setTimeout(() => {
                triggerEmergencyAlert(false);
            }, 24000);
        } else {
            if (card) card.classList.remove('robot-nervous-alert');
            if (character) character.classList.remove('robot-nervous-alert');
            if (bubble) bubble.classList.remove('speech-bubble-emergency');
            if (pill && pill.classList.contains('emergency')) {
                pill.className = 'robot-status-pill ready';
                pill.innerText = 'Ready';
            }
            setRobotSpeech("Clinical vitals stabilized. Standing by for inquiries 🩺", 4000);
        }
    }

    function setRobotState(state) {
        const card = document.getElementById('sorin-robot-companion');
        const pill = document.getElementById('robot-status-pill');
        if (!card) return;
        if (card.classList.contains('robot-nervous-alert') && state !== 'ready') return;
        card.classList.remove('robot-thinking', 'robot-speaking', 'robot-listening', 'robot-vitals-active');
        if (pill && !card.classList.contains('robot-nervous-alert')) {
            pill.className = 'robot-status-pill ' + state;
            pill.innerText = state;
        }
        if (state === 'thinking') {
            card.classList.add('robot-thinking');
            setRobotSpeech("Analyzing verified clinical records...", 3000);
        } else if (state === 'speaking') {
            card.classList.add('robot-speaking');
            setRobotSpeech("Synthesizing clinical consultation...", 3000);
        } else if (state === 'listening') {
            card.classList.add('robot-listening');
            setRobotSpeech("Listening carefully to your voice...", 4000);
        } else if (state === 'ready') {
            if (pill && !card.classList.contains('robot-nervous-alert')) {
                pill.className = 'robot-status-pill ready';
                pill.innerText = 'Ready';
            }
        }
    }

    function pokeRobot() {
        const card = document.getElementById('sorin-robot-companion');
        if (!card) return;
        if (card.classList.contains('robot-nervous-alert')) {
            triggerEmergencyAlert(false);
            playRobotTone(520, 680, 0.2);
            setRobotSpeech("Whew, thanks for checking in! Standing by safely 🩺", 4000);
            return;
        }
        card.classList.remove('robot-poked');
        void card.offsetWidth; // retrigger reflow
        card.classList.add('robot-poked');
        playRobotTone(520, 780, 0.22);
        const randomQuip = POKE_QUIPS[Math.floor(Math.random() * POKE_QUIPS.length)];
        setRobotSpeech(randomQuip, 4500);
        setTimeout(() => { card.classList.remove('robot-poked'); }, 600);
    }

    function switchRobotVariant(variant, event) {
        if (event) event.stopPropagation();
        const card = document.getElementById('sorin-robot-companion');
        const character = document.getElementById('sorin-robot-character');
        if (card) card.setAttribute('data-variant', variant);
        if (character) character.setAttribute('data-variant', variant);
        document.documentElement.setAttribute('data-variant', variant);

        document.querySelectorAll('.variant-dot').forEach(dot => {
            dot.classList.toggle('active', dot.classList.contains(variant));
        });

        try { localStorage.setItem('sorin_robot_variant', variant); } catch(e) {}

        if (variant === 'cyber') playRobotArpeggio([440, 554, 659, 880]);
        else if (variant === 'sage') playRobotArpeggio([392, 523, 659]);
        else if (variant === 'cryo') playRobotArpeggio([587, 880, 1174]);
        else if (variant === 'gold') playRobotArpeggio([330, 415, 494, 659]);
        else playRobotArpeggio([523, 659, 784]);

        const greeting = VARIANT_GREETINGS[variant] || "Dr. Sorin skin updated!";
        setRobotSpeech(greeting, 4000);
    }

    function spawnRobotHearts() {
        const container = document.getElementById('robot-hearts-container');
        if (!container) return;
        const emojis = ['❤️', '💖', '💕', '🩺', '✨'];
        for (let i = 0; i < 7; i++) {
            setTimeout(() => {
                const el = document.createElement('div');
                el.className = 'floating-heart';
                el.innerText = emojis[Math.floor(Math.random() * emojis.length)];
                el.style.left = (25 + Math.random() * 50) + '%';
                el.style.top = (30 + Math.random() * 25) + '%';
                container.appendChild(el);
                setTimeout(() => el.remove(), 1400);
            }, i * 140);
        }
    }

    let robotActionTimeout = null;
    function robotAction(act, event) {
        if (event) event.stopPropagation();
        const card = document.getElementById('sorin-robot-companion');
        if (!card) return;

        card.classList.remove('robot-waving', 'robot-spinning', 'robot-vitals-active', 'robot-dancing', 'robot-loving', 'robot-boosting');
        if (robotActionTimeout) clearTimeout(robotActionTimeout);

        if (act === 'wave') {
            card.classList.add('robot-waving');
            playRobotTone(580, 880, 0.25);
            setRobotSpeech("Hello! I'm Dr. Sorin 👋 Ready for your inquiry.", 4000);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-waving'), 2400);
        } else if (act === 'dance') {
            card.classList.add('robot-dancing');
            playRobotArpeggio([523, 659, 784, 1046, 784, 1046], 0.08);
            setRobotSpeech("Party dance mode activated! 💃 Stay active and keep moving!", 4000);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-dancing'), 3800);
        } else if (act === 'love') {
            card.classList.add('robot-loving');
            playRobotArpeggio([523, 659, 880], 0.12);
            spawnRobotHearts();
            setRobotSpeech("Sending compassionate clinical care! ❤️ Your wellness matters.", 4000);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-loving'), 3200);
        } else if (act === 'boost') {
            card.classList.add('robot-boosting');
            playRobotTone(220, 960, 0.45, 'sawtooth');
            setRobotSpeech("Rocket thrusters engaged! 🚀 Supersonic clinical response!", 3500);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-boosting'), 2200);
        } else if (act === 'vitals') {
            card.classList.add('robot-vitals-active');
            playRobotTone(440, 220, 0.15);
            setTimeout(() => playRobotTone(440, 220, 0.15), 250);
            setRobotSpeech("🩺 Normal Sinus Rhythm · 72 BPM · Vitals Stable!", 4000);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-vitals-active'), 3000);
        } else if (act === 'scan') {
            setRobotState('thinking');
            playRobotTone(350, 700, 0.35);
            setRobotSpeech("🔍 Scanning PubMed, CDC, WHO & clinical index...", 3500);
            robotActionTimeout = setTimeout(() => { setRobotState('ready'); setRobotSpeech("Scan complete! 39+ topics indexed.", 3000); }, 2500);
        } else if (act === 'spin') {
            card.classList.add('robot-spinning');
            playRobotTone(320, 960, 0.3);
            setRobotSpeech("✨ Calibration flip! Diagnostic core primed 100%!", 3500);
            robotActionTimeout = setTimeout(() => card.classList.remove('robot-spinning'), 900);
        } else if (act === 'emergency') {
            triggerEmergencyAlert(true, "🚨 Emergency alert drill! High-priority triage protocol active!");
        } else if (act === 'tip') {
            playRobotTone(660, 880, 0.2);
            const tip = ROBOT_TIPS[Math.floor(Math.random() * ROBOT_TIPS.length)];
            setRobotSpeech(tip, 6000);
        }
    }

    // ── Draggable Physics (Dragging ONLY Dr. Sorin Himself) & Settings ───
    let isDraggingRobot = false;
    let dragStartX = 0, dragStartY = 0;
    let characterOffsetX = 0, characterOffsetY = 0;
    let hasDragMoved = false;
    let lastClientX = 0;

    function toggleRobotSettings(event) {
        if (event) event.stopPropagation();
        const panel = document.getElementById('robot-settings-panel');
        if (!panel) return;
        panel.classList.toggle('open');
        playRobotTone(panel.classList.contains('open') ? 540 : 380, panel.classList.contains('open') ? 780 : 260, 0.14);
    }

    document.addEventListener('click', (e) => {
        const panel = document.getElementById('robot-settings-panel');
        const btn = document.getElementById('robot-settings-btn');
        if (panel && panel.classList.contains('open')) {
            if (!panel.contains(e.target) && !btn.contains(e.target)) {
                panel.classList.remove('open');
            }
        }
    });

    function dockRobot(event) {
        if (event) event.stopPropagation();
        const character = document.getElementById('sorin-robot-character');
        const bay = document.getElementById('robot-docking-bay');
        const placeholder = document.getElementById('robot-dock-placeholder');
        const pill = document.getElementById('robot-status-pill');

        if (!character || !bay) return;

        character.classList.remove('is-flying', 'is-dragging', 'robot-flying');
        character.style.position = '';
        character.style.left = '';
        character.style.top = '';
        character.style.zIndex = '';
        character.style.margin = '';

        if (character.parentElement !== bay) {
            bay.appendChild(character);
        }

        if (placeholder) placeholder.classList.remove('visible');
        if (pill) {
            pill.className = 'robot-status-pill ready';
            pill.innerText = 'Ready';
        }

        const floatGroup = document.getElementById('robot-float-group');
        if (floatGroup) floatGroup.style.transform = '';

        playRobotTone(520, 780, 0.18);
        setRobotSpeech("Docked back to bay 📌", 3000);
    }

    function initDraggableRobot() {
        const character = document.getElementById('sorin-robot-character');
        const bay = document.getElementById('robot-docking-bay');
        const placeholder = document.getElementById('robot-dock-placeholder');
        const card = document.getElementById('sorin-robot-companion');
        if (!character || character.dataset.draggableReady === 'true') return;
        character.dataset.draggableReady = 'true';

        // Restore saved variant if present
        try {
            const savedVariant = localStorage.getItem('sorin_robot_variant');
            if (savedVariant && ['classic', 'cyber', 'sage', 'cryo', 'gold'].includes(savedVariant)) {
                if (card) card.setAttribute('data-variant', savedVariant);
                document.querySelectorAll('.variant-dot').forEach(dot => {
                    dot.classList.toggle('active', dot.classList.contains(savedVariant));
                });
            }
        } catch(e) {}

        character.addEventListener('pointerdown', (e) => {
            if (e.target.closest('.character-dock-pin')) return;

            isDraggingRobot = true;
            hasDragMoved = false;
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            lastClientX = e.clientX;

            const rect = character.getBoundingClientRect();
            characterOffsetX = e.clientX - rect.left;
            characterOffsetY = e.clientY - rect.top;

            try { character.setPointerCapture(e.pointerId); } catch(err) {}
        });

        character.addEventListener('pointermove', (e) => {
            if (!isDraggingRobot) return;

            const dist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
            if (!hasDragMoved && dist > 5) {
                hasDragMoved = true;
                const rect = character.getBoundingClientRect();

                // Append ONLY Dr. Sorin himself to body for free flight
                if (character.parentElement !== document.body) {
                    document.body.appendChild(character);
                }

                character.classList.add('is-flying', 'is-dragging');
                character.style.left = rect.left + 'px';
                character.style.top = rect.top + 'px';

                if (placeholder) placeholder.classList.add('visible');

                const pill = document.getElementById('robot-status-pill');
                if (pill) {
                    pill.className = 'robot-status-pill listening';
                    pill.innerText = 'In Flight 🛸';
                }

                playRobotTone(280, 560, 0.15, 'triangle');
                setRobotSpeech("Wheee! Flying with you! 🚀", 2200);
            }

            if (hasDragMoved) {
                let left = e.clientX - characterOffsetX;
                let top = e.clientY - characterOffsetY;

                const maxLeft = window.innerWidth - character.offsetWidth - 4;
                const maxTop = window.innerHeight - character.offsetHeight - 4;
                left = Math.max(4, Math.min(maxLeft, left));
                top = Math.max(4, Math.min(maxTop, top));

                character.style.left = left + 'px';
                character.style.top = top + 'px';

                const vx = e.clientX - lastClientX;
                lastClientX = e.clientX;
                const tilt = Math.max(-16, Math.min(16, vx * 1.8));
                const floatGroup = document.getElementById('robot-float-group');
                if (floatGroup) {
                    floatGroup.style.transform = `rotate(${tilt}deg) scale(1.08)`;
                }
            }
        });

        const handlePointerEnd = (e) => {
            if (!isDraggingRobot) return;
            isDraggingRobot = false;
            character.classList.remove('is-dragging');

            const floatGroup = document.getElementById('robot-float-group');
            if (floatGroup) floatGroup.style.transform = '';

            try { character.releasePointerCapture(e.pointerId); } catch(err) {}

            if (hasDragMoved) {
                if (bay) {
                    const bayRect = bay.getBoundingClientRect();
                    const charRect = character.getBoundingClientRect();
                    const distToBay = Math.hypot(
                        (charRect.left + charRect.width / 2) - (bayRect.left + bayRect.width / 2),
                        (charRect.top + charRect.height / 2) - (bayRect.top + bayRect.height / 2)
                    );
                    if (distToBay < 140 || charRect.left < 210) {
                        dockRobot();
                        return;
                    }
                }
                playRobotTone(440, 330, 0.12);
                setRobotSpeech("Stationary here! Click 📌 to recall.", 3500);
            } else {
                pokeRobot();
            }
        };

        character.addEventListener('pointerup', handlePointerEnd);
        character.addEventListener('pointercancel', handlePointerEnd);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDraggableRobot);
    } else {
        initDraggableRobot();
    }
    setTimeout(initDraggableRobot, 500);
    setTimeout(initDraggableRobot, 1500);

    // Mouse tracking for robot eyes and head
    document.addEventListener('mousemove', (e) => {
        if (isDraggingRobot) return;
        const svg = document.getElementById('sorin-robot-svg');
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = e.clientX - cx;
        const dy = e.clientY - cy;
        const eyeX = Math.max(-5, Math.min(5, dx / 35));
        const eyeY = Math.max(-3, Math.min(3, dy / 35));
        const eyes = document.getElementById('robot-eyes-group');
        if (eyes) {
            eyes.style.transform = `translate(${eyeX}px, ${eyeY}px)`;
        }
        const head = document.getElementById('robot-head-group');
        if (head) {
            const tilt = Math.max(-3.5, Math.min(3.5, dx / 50));
            head.style.transformOrigin = '80px 65px';
            head.style.transform = `rotate(${tilt}deg)`;
        }
    });

    // Automatic Chatbot State Observer (Live Sync)
    setInterval(() => {
        const botMessages = document.querySelectorAll('#sorin-chat-history [data-testid="bot"], #sorin-chat-history .bot');
        if (!botMessages || botMessages.length === 0) return;
        const lastMsg = botMessages[botMessages.length - 1];
        const html = lastMsg.innerHTML || "";
        const card = document.getElementById('sorin-robot-companion');
        if (!card) return;

        if (html.includes('emergency-mode') || html.includes('EMERGENCY CLINICAL WARNING') || html.includes('data-emergency')) {
            if (!card.classList.contains('robot-nervous-alert')) {
                triggerEmergencyAlert(true);
            }
        } else if (html.includes('clinical-thinking-card') || html.includes('typing-indicator') || html.includes('Analyzing verified clinical records')) {
            if (!card.classList.contains('robot-thinking') && !card.classList.contains('robot-nervous-alert')) {
                setRobotState('thinking');
            }
        } else if (html.includes('▌')) {
            if (!card.classList.contains('robot-speaking') && !card.classList.contains('robot-nervous-alert')) {
                setRobotState('speaking');
            }
        } else if (card.classList.contains('robot-thinking') || card.classList.contains('robot-speaking')) {
            if (!card.classList.contains('robot-nervous-alert')) {
                setRobotState('ready');
                setRobotSpeech("Clinical guidance compiled! 📋", 3000);
            }
        }
    }, 200);
    </script>
    """

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=GRADIO_PORT,
            share=GRADIO_SHARE,
            inbrowser=True,
            allowed_paths=[str(ASSETS_DIR)],
            show_error=True,
            head=_META_HEAD,
            theme=gr.themes.Base(
                primary_hue=gr.themes.colors.blue,
                neutral_hue=gr.themes.colors.slate,
                font=gr.themes.GoogleFont("Plus Jakarta Sans"),
                font_mono=gr.themes.GoogleFont("JetBrains Mono"),
            ),
            css=CUSTOM_CSS,
        )
    except OSError:
        demo.launch(
            server_name="0.0.0.0",
            server_port=None,
            share=GRADIO_SHARE,
            inbrowser=True,
            allowed_paths=[str(ASSETS_DIR)],
            show_error=True,
            head=_META_HEAD,
            theme=gr.themes.Base(
                primary_hue=gr.themes.colors.blue,
                neutral_hue=gr.themes.colors.slate,
                font=gr.themes.GoogleFont("Plus Jakarta Sans"),
                font_mono=gr.themes.GoogleFont("JetBrains Mono"),
            ),
            css=CUSTOM_CSS,
        )
