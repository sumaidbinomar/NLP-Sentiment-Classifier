"""Interactive web demo for the sentiment classifier (Streamlit).

Usage:
    streamlit run streamlit_app.py
"""
import json
from html import escape
from pathlib import Path

import streamlit as st

from predict import load_model, predict

ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "results" / "metrics.json"
CONFUSION_PATH = ROOT / "results" / "confusion_matrix.png"

# label -> (emoji, accent colour)
STYLE = {
    "positive": ("😊", "#34d399"),
    "neutral": ("😐", "#fbbf24"),
    "negative": ("😠", "#f87171"),
}

EXAMPLES = [
    "I absolutely love this new phone, the camera is incredible!",
    "Worst customer service ever. I'm never ordering from them again.",
    "The meeting has been moved to 3pm tomorrow.",
    "Can't believe how good the concert was last night, what a show!",
    "Traffic is awful today and I'm going to be late again.",
    "Just landed in London, heading to the hotel now.",
]

CSS = """
<style>
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #312e81 0%, transparent 60%),
                radial-gradient(900px 500px at 100% 0%, #0e7490 0%, transparent 55%), #0b1020;
}
.block-container { max-width: 960px; padding-top: 2rem; }
.hero { text-align: center; padding: 10px 0 18px; }
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 8px; color: #a5f3fc;
           text-shadow: 0 0 28px rgba(103,232,249,.45); }
.hero p { color: #cbd5e1; font-size: 1.05rem; margin: 0 auto 14px; max-width: 640px; }
.chips { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.chip { padding: 4px 12px; border-radius: 999px; font-size: .8rem; color: #e2e8f0;
        background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); }
.card { border-radius: 18px; padding: 22px; color: #e2e8f0;
        background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); }
.verdict { text-align: center; margin-bottom: 18px; }
.verdict .emoji { font-size: 3.4rem; line-height: 1; }
.verdict .label { font-size: 1.8rem; font-weight: 800; letter-spacing: .08em; margin-top: 6px; }
.verdict .conf { color: #94a3b8; font-size: .9rem; }
.bar-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.bar-name { width: 78px; text-transform: capitalize; font-size: .9rem; color: #cbd5e1; }
.bar-track { flex: 1; height: 12px; border-radius: 999px; background: rgba(255,255,255,.09); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-pct { width: 56px; text-align: right; font-variant-numeric: tabular-nums; font-size: .9rem; }
.placeholder { text-align: center; color: #94a3b8; padding: 48px 8px; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
.stat { text-align: center; padding: 14px 6px; border-radius: 14px; background: rgba(255,255,255,.06); }
.stat b { display: block; font-size: 1.6rem; color: #67e8f9; }
.stat span { font-size: .78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }
table.perf { width: 100%; border-collapse: collapse; font-size: .95rem; }
table.perf th, table.perf td { padding: 9px 8px; text-align: right; border: none;
                               border-bottom: 1px solid rgba(255,255,255,.1); }
table.perf th:first-child, table.perf td:first-child { text-align: left; text-transform: capitalize; }
table.perf th { color: #94a3b8; font-weight: 600; font-size: .8rem; text-transform: uppercase; }
.foot { text-align: center; color: #64748b; font-size: .82rem; padding: 26px 0 6px; }
</style>
"""

HERO = """
<div class="hero">
  <h1>💬 Tweet Sentiment Classifier</h1>
  <p>Type any short text and see whether it reads as <b>negative</b>, <b>neutral</b> or <b>positive</b>,
     with the model's confidence for every class.</p>
  <div class="chips">
    <span class="chip">NLTK preprocessing</span><span class="chip">TF-IDF</span>
    <span class="chip">Logistic Regression</span><span class="chip">TweetEval</span>
  </div>
</div>
"""

PLACEHOLDER = '<div class="card placeholder">✨ Your prediction will appear here.</div>'


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))["test"]


def render_result(text: str) -> str:
    """Classify `text` and return the result card as HTML."""
    if not text.strip():
        return PLACEHOLDER
    result = predict(get_model(), text)
    label = result["sentiment"]
    emoji, color = STYLE[label]
    bars = ""
    for name in ("positive", "neutral", "negative"):
        p = result["probabilities"][name]
        bars += (
            f'<div class="bar-row"><div class="bar-name">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p:.1%};background:{STYLE[name][1]}"></div></div>'
            f'<div class="bar-pct">{p:.1%}</div></div>'
        )
    return (
        '<div class="card">'
        f'<div class="verdict"><div class="emoji">{emoji}</div>'
        f'<div class="label" style="color:{color}">{escape(label.upper())}</div>'
        f'<div class="conf">{result["probabilities"][label]:.1%} confidence</div></div>{bars}</div>'
    )


def render_performance(metrics: dict) -> str:
    """Build the test-set performance panel from results/metrics.json."""
    rows = ""
    for name, m in metrics["per_class"].items():
        rows += (
            f'<tr><td>{name}</td><td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td>'
            f'<td>{m["f1-score"]:.3f}</td><td>{int(m["support"]):,}</td></tr>'
        )
    return (
        '<div class="card"><div class="stats">'
        f'<div class="stat"><b>{metrics["accuracy"]:.1%}</b><span>Accuracy</span></div>'
        f'<div class="stat"><b>{metrics["macro_f1"]:.3f}</b><span>Macro F1</span></div>'
        f'<div class="stat"><b>{metrics["weighted_f1"]:.3f}</b><span>Weighted F1</span></div></div>'
        '<table class="perf"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>'
        f"{rows}</table></div>"
    )


def set_text(value: str) -> None:
    st.session_state.text = value


st.set_page_config(page_title="Tweet Sentiment Classifier", page_icon="💬", layout="centered")
st.markdown(CSS + HERO, unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = ""

try_tab, perf_tab, about_tab = st.tabs(["🔮 Try it", "📊 Model performance", "ℹ️ About"])

with try_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.text_area("Your text", key="text", height=150,
                     placeholder="e.g. I can't believe how good this was!")
        st.button("Clear", on_click=set_text, args=("",))
    with right:
        st.markdown(render_result(st.session_state.text), unsafe_allow_html=True)

    st.caption("Click an example")
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"ex{i}", on_click=set_text, args=(example,), use_container_width=True)

with perf_tab:
    st.markdown(render_performance(get_metrics()), unsafe_allow_html=True)
    st.image(str(CONFUSION_PATH), caption="Confusion matrix (test set, 12,284 tweets)")

with about_tab:
    st.markdown(
        """
**Pipeline:** lowercase and clean the text (URLs, @mentions, contractions) →
remove stop-words (negations like *not* are kept) → lemmatize with NLTK →
TF-IDF features → Logistic Regression with balanced class weights.

**Data:** [TweetEval – sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
(45,615 train / 2,000 validation / 12,284 test tweets).

**Be aware:** this is a classic bag-of-words model. It can't understand sarcasm or context,
and neutral text is the hardest class – a macro F1 of about 0.59 is typical for this approach.
"""
    )

st.markdown('<div class="foot">Built with scikit-learn, NLTK and Streamlit</div>', unsafe_allow_html=True)
