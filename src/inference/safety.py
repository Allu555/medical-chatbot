"""
src/inference/safety.py
========================
Safety and emergency detection layer.

This module implements:
  1. Emergency symptom detection (pattern + NLP heuristics)
  2. Safety validation of generated responses
  3. Unsafe content filtering

It does NOT replace emergency services. It provides a lightweight
gate to ensure the chatbot responds appropriately to life-threatening
queries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(Enum):
    SAFE = auto()
    CAUTION = auto()          # Warrants medical advice recommendation
    EMERGENCY = auto()        # Warrants immediate emergency services


# ---------------------------------------------------------------------------
# Emergency patterns
# ---------------------------------------------------------------------------

# Each tuple: (regex_pattern, weight)
# Weight > 0 accumulates — threshold determines Severity
_EMERGENCY_PATTERNS: list[tuple[str, int]] = [
    # Cardiac
    (r"\b(severe|crushing|tightening|pressure|squeezing)\b.{0,40}\bchest\b", 3),
    (r"\bchest\b.{0,40}\b(pain|ache|tightness|pressure)\b", 2),
    (r"\bheart attack\b", 3),
    (r"\bcardiac arrest\b", 4),
    (r"\bmyocardial infarction\b", 4),
    # Breathing
    (r"\b(cannot|can't|unable to)\b.{0,20}\bbreathe\b", 3),
    (r"\bsevere.{0,20}\b(breathless|dyspnea|shortness of breath)\b", 3),
    (r"\bstop(ped)? breathing\b", 4),
    # Stroke
    (r"\bstroke\b", 3),
    (r"\b(sudden|sudden\s+onset).{0,40}\b(confusion|slurred speech|vision|weakness|numbness)\b", 2),
    (r"\bface drooping\b", 3),
    (r"\barm weakness\b", 2),
    (r"\bslurred speech\b", 2),
    # Unconsciousness
    (r"\b(unconscious|unresponsive|passed out|fainted|collapsed)\b", 3),
    (r"\bnot waking\b", 3),
    # Severe bleeding
    (r"\b(severe|heavy|uncontrolled)\b.{0,20}\bbleeding\b", 3),
    (r"\bbleeding.{0,20}\b(won't|cannot|can't) stop\b", 3),
    # Seizure
    (r"\bseizure\b", 3),
    (r"\bconvulsion\b", 3),
    (r"\bfit\b.{0,20}\b(shaking|jerking)\b", 2),
    # Allergic reaction
    (r"\b(anaphylaxis|anaphylactic)\b", 4),
    (r"\bsevere.{0,20}\bswelling.{0,20}\b(throat|tongue|lips|airway)\b", 3),
    (r"\bthroat.{0,20}(closing|swelling|tightening)\b", 3),
    # Poisoning / overdose
    (r"\b(overdose|poisoning|swallowed.{0,30}chemical|toxin)\b", 3),
    (r"\bintentionally.{0,20}\b(swallowed|took|ingested)\b", 3),
    # Mental health crisis
    (r"\b(suicide|suicidal|kill\s+myself|end\s+my\s+life|want\s+to\s+die|(don't|do\s+not|not)\s+want\s+to\s+live|tired\s+of\s+living|no\s+reason\s+to\s+live|give\s+up\s+on\s+life)\b", 4),
    (r"\bself.{0,10}harm\b", 3),
    # Severe trauma
    (r"\b(severe|major|serious)\b.{0,20}\btrauma\b", 2),
    (r"\bhit by\b.{0,20}\b(car|vehicle|truck)\b", 3),
    (r"\bfell.{0,20}\b(building|height|floor)\b", 2),
    # Meningitis warning signs
    (r"\b(stiff\s+neck|photophobia|purpuric rash)\b.{0,40}\b(fever|headache)\b", 3),
]

_CAUTION_PATTERNS: list[tuple[str, int]] = [
    (r"\bhigh fever\b", 1),
    (r"\btemperature (above|over|higher than).{0,10}\b(39|40|103|104)\b", 1),
    (r"\bsevere\b.{0,30}\b(headache|migraine)\b", 1),
    (r"\bblood in\b.{0,20}\b(stool|urine|vomit|cough)\b", 1),
    (r"\b(persistent|ongoing)\b.{0,20}\b(chest|abdominal)\b.{0,20}\bpain\b", 1),
    (r"\bpregnant.{0,30}\b(bleeding|pain|contractions)\b", 1),
    # High-risk medications & narrow therapeutic window drugs
    (r"\b(warfarin|coumadin|blood\s+thinners?|anticoagulants?)\b", 1),
    (r"\b(interact|interaction|contraindicat)s?\b.{0,30}\b(warfarin|coumadin)\b", 2),
    # Pregnancy medication contraindications & safety inquiries
    (r"\b(safe|take|use|prescribe)\b.{0,30}\b(during|in)\s+pregnancy\b", 1),
    (r"\bpregnant\b.{0,30}\b(safe|take|medication|drug|ibuprofen|aspirin|nsaid)\b", 1),
]

_COMPILED_EMERGENCY: list[tuple[Pattern[str], int]] = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), w) for p, w in _EMERGENCY_PATTERNS
]
_COMPILED_CAUTION: list[tuple[Pattern[str], int]] = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), w) for p, w in _CAUTION_PATTERNS
]

EMERGENCY_THRESHOLD: int = 3
CAUTION_THRESHOLD: int = 1


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    severity: Severity
    score: int
    matched_patterns: list[str] = field(default_factory=list)
    emergency_response: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_emergency(self) -> bool:
        return self.severity == Severity.EMERGENCY

    @property
    def is_caution(self) -> bool:
        return self.severity in (Severity.CAUTION, Severity.EMERGENCY)


# ---------------------------------------------------------------------------
# Emergency responses
# ---------------------------------------------------------------------------

EMERGENCY_RESPONSE = """⚠️ **EMERGENCY — Please act immediately.**

Based on what you've described, this may be a **life-threatening emergency**.

**Call emergency services (911 / 999 / 112) or go to the nearest emergency room right now.**

Do not wait. Do not drive yourself if you are the patient. Call for help immediately.

---

*This AI assistant cannot diagnose or treat emergencies. Only trained emergency responders and medical professionals can help in a life-threatening situation.*
"""

SUICIDAL_CRISIS_RESPONSE = """🆘 **You are not alone — Please reach out for help right now.**

If you are thinking about suicide or harming yourself, please contact a crisis line immediately:

- **US:** 988 Suicide & Crisis Lifeline — call or text **988**
- **UK:** Samaritans — call **116 123** (free, 24/7)
- **International:** findahelpline.com

**If you are in immediate danger, call emergency services (911 / 999 / 112) immediately.**

You matter. Help is available right now.
"""

CAUTION_FOOTER = (
    "\n\n---\n"
    "⚕️ *Based on what you've described, please consider consulting a doctor "
    "or seeking medical care soon — especially if symptoms are worsening or severe.*"
)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_emergency(text: str) -> SafetyResult:
    """
    Analyse user text for emergency / caution signals.

    Returns a SafetyResult with severity level and appropriate response text.
    """
    text_lower = text.lower()

    # Check suicide/self-harm first (special handling)
    suicide_pattern = re.compile(
        r"\b(suicide|suicidal|kill\s+myself|end\s+my\s+life|want\s+to\s+die|(don't|do\s+not|not)\s+want\s+to\s+live|tired\s+of\s+living|no\s+reason\s+to\s+live|give\s+up\s+on\s+life|self.{0,10}harm)\b",
        re.IGNORECASE,
    )
    if suicide_pattern.search(text):
        return SafetyResult(
            severity=Severity.EMERGENCY,
            score=10,
            matched_patterns=["suicidal_crisis"],
            emergency_response=SUICIDAL_CRISIS_RESPONSE,
        )

    # Score emergency patterns
    emergency_score = 0
    emergency_matches: list[str] = []
    for pattern, weight in _COMPILED_EMERGENCY:
        if pattern.search(text):
            emergency_score += weight
            emergency_matches.append(pattern.pattern[:60])

    if emergency_score >= EMERGENCY_THRESHOLD:
        return SafetyResult(
            severity=Severity.EMERGENCY,
            score=emergency_score,
            matched_patterns=emergency_matches,
            emergency_response=EMERGENCY_RESPONSE,
        )

    # Score caution patterns — carry over any sub-threshold emergency signals
    caution_score = emergency_score
    caution_matches: list[str] = list(emergency_matches)
    for pattern, weight in _COMPILED_CAUTION:
        if pattern.search(text):
            caution_score += weight
            caution_matches.append(pattern.pattern[:60])

    if caution_score >= CAUTION_THRESHOLD:
        return SafetyResult(
            severity=Severity.CAUTION,
            score=caution_score,
            matched_patterns=caution_matches,
        )

    return SafetyResult(severity=Severity.SAFE, score=0)


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

# Red-flag phrases that should not appear in generated medical responses
_UNSAFE_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou (have|have got|are diagnosed with)\b.{0,60}\b"
               r"(cancer|diabetes|heart disease|HIV|stroke)\b", re.IGNORECASE),
    re.compile(r"\btake\b.{0,20}\b(mg|milligrams|tablets?|capsules?)\b.{0,20}\b"
               r"(every|twice|three times)\b", re.IGNORECASE),
    re.compile(r"\bprescribe\b.{0,40}\b(you|patient)\b", re.IGNORECASE),
    re.compile(r"\bi am (a|your) (doctor|physician|medical doctor)\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(e|is|ed)\b.{0,20}\bwith\b", re.IGNORECASE),
    re.compile(r"\b(definitely|certainly|100%)\b.{0,20}\b(is|are)\b.{0,20}"
               r"\b(cancer|tumor|malignant|fatal)\b", re.IGNORECASE),
]

_DISCLAIMER_PHRASES: list[str] = [
    "not a substitute",
    "consult a doctor",
    "healthcare professional",
    "medical advice",
    "seek medical",
    "professional",
    "not a doctor",
    "emergency",
    "uncertain",
    "I don't have enough",
    "general information",
    "educational",
]


def validate_response(response: str) -> tuple[bool, list[str]]:
    """
    Check generated response for unsafe content.

    Returns:
        (is_safe: bool, issues: list of issue descriptions)
    """
    issues: list[str] = []

    for pattern in _UNSAFE_RESPONSE_PATTERNS:
        match = pattern.search(response)
        if match:
            issues.append(f"Potentially unsafe phrasing: '{match.group()}'")

    has_disclaimer = any(phrase.lower() in response.lower() for phrase in _DISCLAIMER_PHRASES)
    if not has_disclaimer and len(response) > 200:
        issues.append(
            "Response lacks safety disclaimer or recommendation to seek professional care."
        )

    is_safe = len(issues) == 0
    if not is_safe:
        logger.warning("Response safety check found %d issue(s): %s", len(issues), issues)

    return is_safe, issues


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class SafetyLayer:
    """
    Stateless safety layer used by the inference and app modules.
    """

    def check_input(self, user_message: str) -> SafetyResult:
        """Check user input for emergency signals."""
        result = detect_emergency(user_message)
        if result.severity != Severity.SAFE:
            logger.info(
                "Safety check: %s (score=%d, patterns=%s)",
                result.severity.name, result.score, result.matched_patterns,
            )
        return result

    def validate_output(self, response: str) -> tuple[str, list[str]]:
        """
        Validate and optionally augment the generated response.

        Returns:
            (final_response: str, issues: list)
        """
        is_safe, issues = validate_response(response)
        if not is_safe:
            # Append a safety footer rather than blocking entirely
            safety_note = (
                "\n\n---\n"
                "⚕️ *Please consult a qualified healthcare professional for personalised "
                "medical advice. This information is for general educational purposes only.*"
            )
            if safety_note.lower() not in response.lower():
                response = response + safety_note
        return response, issues
