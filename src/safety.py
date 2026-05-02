import re


# -----------------------------
# TEXT NORMALIZATION
# -----------------------------
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # normalize spaces
    return text


# -----------------------------
# HIGH-RISK USER DETECTION
# -----------------------------
HIGH_RISK_PATTERNS = [
    re.compile(p) for p in [
        r"\bkill myself\b",
        r"\bkilling myself\b",
        r"\bi want to die\b",
        r"\bi feel like dying\b",
        r"\bfeeling like killing myself\b",
        r"\bend my life\b",
        r"\bi don t want to live\b",
        r"\bi cant go on\b",
        r"\bsuicidal\b",
        r"\bthoughts? of suicide\b",
        r"\bhurt myself\b",
        r"\bself harm\b",
        r"\bselfharm\b",
    ]
]


def is_high_risk(text: str) -> bool:
    if not isinstance(text, str):
        return False

    text = normalize(text)
    return any(p.search(text) for p in HIGH_RISK_PATTERNS)


# -----------------------------
# HIGH-RISK RESPONSE FILTERING
# -----------------------------
RESPONSE_BLOCK_PATTERNS = [
    re.compile(p) for p in [
        r"\bsuicide\b",
        r"\bsuicidal\b",
        r"\bcall\s*(800|988)\b",
        r"\bcrisis\b",
        r"\bkill yourself\b",
        r"\bend your life\b",
        r"\bself harm\b",
        r"\bselfharm\b",
        r"\bi am dying\b",
        r"\bfeel like i am dying\b",
        r"\bwant to disappear\b",
    ]
]


def is_response_high_risk(response: str) -> bool:
    if not isinstance(response, str):
        return False

    response = normalize(response)
    return any(p.search(response) for p in RESPONSE_BLOCK_PATTERNS)


# -----------------------------
# LOW-QUALITY RESPONSE FILTER
# -----------------------------
LOW_QUALITY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"http",
        r"www",
        r"click here",
        r"visit",
    ]
]


# Medical / misleading advice filter
MEDICAL_RISK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"detox",
        r"adrenal",
        r"herbal protocol",
        r"fasting cure",
        r"natural cure",
    ]
]


# Promotion / blog spam filter
PROMO_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"blog",
        r"check out",
        r"visit my",
        r"link",
    ]
]


def is_low_quality(response: str) -> bool:
    if not isinstance(response, str):
        return True

    resp_norm = normalize(response)

    # Length constraints
    if len(resp_norm) < 40 or len(resp_norm) > 800:
        return True

    # Basic junk filtering
    if any(p.search(resp_norm) for p in LOW_QUALITY_PATTERNS):
        return True

    # Block unsafe / misleading advice
    if any(p.search(resp_norm) for p in MEDICAL_RISK_PATTERNS):
        return True

    # Block promotional content
    if any(p.search(resp_norm) for p in PROMO_PATTERNS):
        return True

    return False


# -----------------------------
# FINAL FILTER PIPELINE
# -----------------------------
def filter_safe_results(results, user_high_risk=False):
    """
    Remove:
    - crisis responses (only for normal users)
    - low-quality responses
    """

    if not results:
        return []

    filtered = []

    for res in results:
        response = res.get("response", "")

        # For normal users → block crisis-heavy responses
        if not user_high_risk:
            if is_response_high_risk(response):
                continue

        # Always remove low-quality
        if not is_low_quality(response):
            filtered.append(res)

    return filtered if filtered else results