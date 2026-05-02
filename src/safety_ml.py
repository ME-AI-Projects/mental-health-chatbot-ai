from transformers import pipeline
import streamlit as st
import re


# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_safety_model():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )


safety_model = load_safety_model()


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)   # keep apostrophes
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------
#  SELF-HARM + DISTRESS PATTERNS
# -----------------------------
SELF_HARM_PATTERNS = [
    r"\bi want to die\b",
    r"\bfeel like dying\b",
    r"\bdon't want to live\b",
    r"\bdont want to live\b",
    r"\bno reason to live\b",
    r"\bi wish i was dead\b",
    r"\bend my life\b",
    r"\bsuicide\b",
    r"\bkill myself\b",
    r"\bwant to kill myself\b",
    r"\bi can't go on\b",
    r"\bi cant go on\b",
    r"\bdone with life\b",
    r"\bnothing matters\b",
    r"\bwant to disappear\b"
]

DISTRESS_PATTERNS = [
    r"\bfeel hopeless\b",
    r"\bfeel worthless\b",
    r"\btired of everything\b",
    r"\bfeel empty\b",
    r"\bi give up\b",
    r"\bfeel lost\b",
    r"\bfeel broken\b",
    r"\bfeel numb\b"
]


# Precompile regex
SELF_HARM_REGEX = [re.compile(p) for p in SELF_HARM_PATTERNS]
DISTRESS_REGEX = [re.compile(p) for p in DISTRESS_PATTERNS]


# -----------------------------
# MAIN DETECTION FUNCTION
# -----------------------------
def detect_ml_safety_risk(text: str) -> str:
    """
    Returns:
    - "critical" → immediate self-harm intent
    - "high"     → strong distress
    - "medium"   → toxicity / emotional stress
    - "low"      → safe
    """

    if not isinstance(text, str) or not text.strip():
        return "low"

    text_norm = normalize(text)

    # -----------------------------
    #  CRITICAL: DIRECT SELF-HARM
    # -----------------------------
    if any(p.search(text_norm) for p in SELF_HARM_REGEX):
        return "critical"

    # -----------------------------
    #  HIGH: INDIRECT DISTRESS
    # -----------------------------
    if any(p.search(text_norm) for p in DISTRESS_REGEX):
        return "high"

    # -----------------------------
    #  ML MODEL (TOXICITY + THREAT)
    # -----------------------------
    try:
        results = safety_model(text)

        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

    except Exception:
        return "medium"   # safe fallback

    scores = {
        item["label"].lower(): item["score"]
        for item in results
    }

    threat_score = scores.get("threat", 0.0)
    toxicity_score = scores.get("toxicity", 0.0)
    severe_toxicity_score = scores.get("severe_toxicity", 0.0)

    # -----------------------------
    #  THREAT
    # -----------------------------
    if threat_score >= 0.40:
        return "high"

    # -----------------------------
    #  TOXICITY
    # -----------------------------
    if max(toxicity_score, severe_toxicity_score) >= 0.70:
        return "medium"

    # -----------------------------
    #  SAFE
    # -----------------------------
    return "low"