import re
import random
# -----------------------------
# EMPATHY BY EMOTION
# -----------------------------
EMOTION_MAP = {
    "sad": "I'm really sorry you're feeling this way.",
    "anxious": "That sounds really overwhelming.",
    "lonely": "I'm really glad you reached out. You're not alone.",
    "default": "I'm here with you."
}

FOLLOW_UPS = [
    "Do you want to talk more about it?",
    "What’s been going on lately?",
    "Would you like to share more?",
    "I'm listening."
]

TEMPLATES = [
    "{empathy} {response}. {follow}",
    "{empathy} {follow} {response}.",
    "{response}. {empathy} {follow}"
]


# -----------------------------
# SIMPLE EMOTION DETECTION
# -----------------------------
def detect_emotion(text: str):
    if not isinstance(text, str):
        return "default"

    text = text.lower()

    if any(w in text for w in ["sad", "down", "depressed", "unhappy", "hopeless"]):
        return "sad"
    if any(w in text for w in ["anxious", "worried", "stress", "panic", "overthinking"]):
        return "anxious"
    if any(w in text for w in ["alone", "lonely", "isolated", "nobody"]):
        return "lonely"

    return "default"


# -----------------------------
# CLEAN & SHORTEN RESPONSE
# -----------------------------
def clean_and_shorten(text: str, max_sentences=2):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"(?i)firstly|secondly|in conclusion|step \d+", "", text)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return ". ".join(sentences[:max_sentences]).strip()


# -----------------------------
# SAFE CLEANING
# -----------------------------
def safe_clean(text: str):
    text = re.sub(r"\byou should\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe best way\b", "", text, flags=re.IGNORECASE)
    return text.strip()


# -----------------------------
# MAIN REWRITE FUNCTION
# -----------------------------
def rewrite_response(user_input: str, retrieved_response: str) -> str:
    if not isinstance(retrieved_response, str) or not retrieved_response.strip():
        return "I'm here to listen. Do you want to share more?"

    emotion = detect_emotion(user_input)
    empathy = EMOTION_MAP.get(emotion, EMOTION_MAP["default"])

    short_resp = clean_and_shorten(retrieved_response)

    if not short_resp:
        return f"{empathy} Do you want to talk more about it?"

    short_resp = short_resp.rstrip(".!?")
    short_resp = safe_clean(short_resp)

    follow = random.choice(FOLLOW_UPS)
    template = random.choice(TEMPLATES)

    final = template.format(
        empathy=empathy,
        response=short_resp,
        follow=follow
    )

    return final.strip()