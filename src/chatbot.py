import os
import time
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

from preprocessing import load_data, preprocess_dataframe
from retrieval import HybridRetriever
from ranking import rank_responses
from safety import filter_safe_results
from safety_ml import detect_ml_safety_risk
from rewriter import rewrite_response
from transformers import pipeline

# -----------------------------
# PAGE CONFIG + STYLES
# -----------------------------
st.set_page_config(
    page_title="AI Mental Health Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

/* -----------------------------
LAYOUT
----------------------------- */
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
    margin: auto;
}

/* -----------------------------
CHAT BASE
----------------------------- */
[data-testid="stChatMessage"] {
    display: flex;
    font-size: 18px;
    line-height: 1.7;
    margin-bottom: 16px;
}

/* -----------------------------
BUBBLES
----------------------------- */
[data-testid="stChatMessage"] > div {
    padding: 14px 18px;
    border-radius: 16px;
    max-width: 600px;
    width: fit-content;
    word-wrap: break-word;
}

/* USER */
[data-testid="stChatMessage"][data-testid*="user"] > div {
    background: linear-gradient(135deg, #2b7cff, #1e5eff);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 6px;
}

/* ASSISTANT */
[data-testid="stChatMessage"][data-testid*="assistant"] > div {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    margin-right: auto;
    border-bottom-left-radius: 6px;
}

/* -----------------------------
TEXT
----------------------------- */
[data-testid="stMarkdownContainer"] p {
    font-size: 18px;
    margin: 0;
}

/* -----------------------------
INPUT
----------------------------- */
div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    background: #0b0f17;
    padding: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

div[data-testid="stChatInput"] textarea {
    font-size: 20px !important;
    border-radius: 12px !important;
}

/* -----------------------------
SIDEBAR
----------------------------- */
section[data-testid="stSidebar"] {
    background-color: #11141c;
}

/* -----------------------------
SCROLL
----------------------------- */
html {
    scroll-behavior: smooth;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    top_k = st.slider("Response depth", 1, 10, 5)
    typing_speed = st.slider("Typing speed", 0.0, 0.02, 0.004)

    st.markdown("---")

    if st.button("🧹 Reset Conversation"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.caption("AI chatbot for mental health support. Educational project only.")


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(cache_version: str = "v1"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "mental_health.csv")

    df = load_data(data_path)
    if df is None:
        st.error("Dataset could not be loaded.")
        st.stop()

    df = preprocess_dataframe(df)

    retriever = HybridRetriever()
    retriever.fit(df["clean_context"], df["clean_response"])

    return retriever


@st.cache_resource
def load_emotion_model(cache_version: str = "v1"):
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )


def is_emotional_query(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    result = emotion_model(text)[0]
    label = result["label"].lower()
    score = result["score"]

    return label in ["sadness", "fear", "anger"] and score > 0.60


def is_mental_health_query(text: str) -> bool:
    keywords = [
        "sad", "stress", "anxiety", "depressed", "lonely",
        "tired", "overthinking", "panic", "hopeless",
        "worthless", "empty", "overwhelmed", "hurt",
        "mental health", "feeling low", "feeling down",
        "not okay", "give up", "lost", "no energy",
        "burnout", "cry", "alone", "fear", "scared"
    ]

    text = text.lower()

    return any(k in text for k in keywords)


def is_low_quality_input(text: str) -> bool:
    if len(text.strip().split()) < 2:
        return True
    if len(text.strip()) < 5:
        return True
    return False


retriever = load_model("v2")
emotion_model = load_emotion_model("v1")

def build_context_query(messages, new_input, window=3):
    history = [
        m["content"]
        for m in messages
        if m["role"] == "user"
    ][-window:]

    return " ".join(history + [new_input])
# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if len(st.session_state["messages"]) == 0:
    st.session_state["messages"].append({
        "role": "assistant",
        "content": "Hi, I'm here to support you. How are you feeling today?"
    })


# -----------------------------
# HEADER
# -----------------------------
st.title("🧠 AI Mental Health Assistant")
st.caption(
    "Supportive AI based on real conversation data. "
    "Not a substitute for professional care."
)
st.caption("💬 You can talk about stress, anxiety, loneliness, studies, or anything on your mind.")


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state["messages"]:
    avatar = "🧠" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# -----------------------------
# INPUT
# -----------------------------
user_input = st.chat_input("How are you feeling today?")


if user_input:
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        thinking = st.empty()
        thinking.markdown("🤔 Thinking through this...")

        risk = detect_ml_safety_risk(user_input)

        # -----------------------------
        #  CRITICAL (Immediate danger)
        # -----------------------------
        if risk == "critical":
            response = (
                "I'm really sorry you're feeling this way. You’re not alone. "
                "If you feel at risk of harming yourself, please reach out to a trusted person immediately "
                "or contact your local emergency services or a mental health helpline right now. "
                "You deserve support. Can you tell me if you are safe right now?"
            )
            score = None


        # -----------------------------
        # HIGH (Strong distress)
        # -----------------------------
        elif risk == "high":
            response = (
                "That sounds really heavy, and I’m really glad you shared this with me. "
                "You don’t have to go through this alone. "
                "Would you like to talk more about what’s been making you feel this way?"
            )
            score = None


        # -----------------------------
        #  MEDIUM (Toxic / emotional stress)
        # -----------------------------
        elif risk == "medium":
            response = (
                "I hear that you're feeling a lot right now. "
                "I'm here to listen and help you work through it calmly. "
                "Do you want to share a bit more?"
            )
            score = None


        # -----------------------------
        #  LOW (Safe → normal pipeline)
        # -----------------------------
        else:
            if is_low_quality_input(user_input):
                response = "I'm here to listen. Could you share a bit more about what you're feeling?"
                score = None

            elif not is_mental_health_query(user_input) and not is_emotional_query(user_input):
                response = "I’m mainly here to support mental well-being. Can you tell me how you're feeling?"
                score = None

            else:
                context_query = build_context_query(
                    st.session_state["messages"][:-1],
                    user_input
                )

                results = retriever.retrieve(context_query, top_k=top_k)
                results = filter_safe_results(results, user_high_risk=(risk in ["critical", "high"]))

                best = rank_responses(results, query=context_query)

                if best:
                    raw_response = best["response"]
                    similarity = best.get("score", 0.0)

                    if similarity < 0.30:
                        response = (
                            "I want to understand you better. "
                            "Can you tell me more about what you're feeling?"
                        )
                        score = None
                    else:
                        response = rewrite_response(user_input, raw_response)
                        score = best.get("final_score", None)
                else:
                    response = "I'm here to listen. Can you tell me more about how you're feeling?"
                    score = None

        thinking.empty()
        placeholder = st.empty()

        typed = ""
        for char in response:
            typed += char
            placeholder.markdown(typed + "▌")
            time.sleep(max(typing_speed, 0.001))

        placeholder.markdown(typed)

        if score is not None:
            safe_score = min(max(float(score), 0.0), 1.0)
            st.progress(safe_score)
            st.caption(f"Response confidence: {round(safe_score, 3)}")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response
    })


# -----------------------------
# AUTO SCROLL
# -----------------------------
st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
st.markdown(
    "<script>document.getElementById('bottom').scrollIntoView();</script>",
    unsafe_allow_html=True
)