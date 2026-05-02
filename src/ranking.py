import re
from collections import Counter

URL_PATTERN = re.compile(r"http|www|visit", re.IGNORECASE)

GENERIC_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"it depends",
        r"everyone is different",
        r"you should consider",
        r"seek professional help",
    ]
]


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def repetition_penalty(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    freq = Counter(words)
    repeated = sum((c - 1) for c in freq.values() if c > 1)
    return min(repeated / len(words), 0.3)


def rank_responses(results, query: str = ""):
    if not results:
        return None

    ranked = []

    for res in results:
        similarity = res.get("score", 0.0)
        response = res.get("response", "")

        if not isinstance(response, str):
            continue

        text = response.lower().strip()
        length = len(text)

        # LENGTH SCORE (balanced)
        if 40 <= length <= 300:
            length_score = 1.0
        elif length < 40:
            length_score = 0.7
        else:
            length_score = 0.6

        # STRUCTURE SCORE
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        structure_score = min(len(sentences) / 3, 1.0)

        # EMOTION SCORE (token-based)
        tokens = tokenize(text)
        emotional_keywords = [
            "feel", "understand", "sorry", "alone",
            "hear", "support", "here"
        ]
        emotion_score = min(
            sum(0.1 for word in emotional_keywords if word in tokens),
            1.0
        )

        # QUESTION SCORE
        question_score = 0.2 if "?" in text else 0.0

        # OVERLAP SCORE (reduced importance)
        overlap_score = jaccard_similarity(query, text)

        # GENERIC PENALTY (reduced)
        generic_penalty = sum(
            0.05 for p in GENERIC_PATTERNS if p.search(text)
        )

        # NOISE PENALTIES
        penalty = 0.0

        if length > 800:
            penalty += 0.2

        if URL_PATTERN.search(text):
            penalty += 0.3

        if length < 30:
            penalty += 0.3

        penalty += repetition_penalty(text)

        # FINAL SCORE (rebalanced)
        final_score = (
            0.55 * similarity +
            0.10 * length_score +
            0.10 * structure_score +
            0.15 * emotion_score +
            0.02 * overlap_score +
            0.08 * question_score
        ) - penalty - generic_penalty

        ranked.append({**res, "final_score": final_score})

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    return ranked[0] if ranked else None