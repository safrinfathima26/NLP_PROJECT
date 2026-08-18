"""
app.py
------
Streamlit web interface for the Intelligent Document Question
Answering (NLP + RAG) system — "Evidence Desk", a guided 4-step
experience with a futuristic purple/neon theme:

    1. Welcome
    2. Upload documents
    3. Configure & review
    4. Ask questions (dashboard) — with suggested-question chips

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Evidence Desk — Document QA",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------- #
# Design system — deep purple + neon, with motion
# --------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #0b0614;
        --ink-soft: #170f28;
        --ink-soft2: #1f1638;
        --purple: #8b5cf6;
        --purple-dark: #6d28d9;
        --neon: #00f0ff;
        --neon-pink: #ff3ec9;
        --paper: #f5f1fb;
        --paper-dim: #e7defb;
        --coral: #ff5c72;
        --slate: #a89fc4;
        --ink-text: #1b1030;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(139,92,246,0.16) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(0,240,255,0.10) 0%, transparent 40%),
            radial-gradient(rgba(139,92,246,0.10) 1px, transparent 1px) 0 0/26px 26px,
            var(--ink);
    }
    #MainMenu, footer { visibility: hidden; }

    @keyframes floatY { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
    @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 6px rgba(0,240,255,0.35); } 50% { box-shadow: 0 0 16px rgba(0,240,255,0.75); } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes shimmer { 0% { background-position: -200px 0; } 100% { background-position: 200px 0; } }

    /* ---------- Stepper ---------- */
    .stepper { display: flex; align-items: center; justify-content: center; gap: 0.4rem; margin: 1.5rem 0 2rem 0; flex-wrap: wrap; animation: fadeInUp 0.5s ease; }
    .step-pill {
        display: flex; align-items: center; gap: 0.5rem;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.03em;
        padding: 0.4rem 0.8rem; border-radius: 999px; border: 1px solid #2d2350; color: #776a9c; white-space: nowrap;
        transition: all 0.3s ease;
    }
    .step-pill.active { border-color: var(--neon); color: var(--neon); background: rgba(0,240,255,0.08); animation: pulseGlow 2.2s infinite; }
    .step-pill.done { border-color: var(--purple); color: var(--purple); background: rgba(139,92,246,0.08); }
    .step-line { width: 26px; height: 1px; background: #2d2350; }

    /* ---------- Hero / illustration wrap ---------- */
    .hero-wrap { text-align: center; padding: 1.2rem 1rem 0.3rem 1rem; animation: fadeInUp 0.6s ease; }
    .hero-wrap svg { animation: floatY 5s ease-in-out infinite; }
    .hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--neon); display: block; margin-bottom: 0.6rem; text-shadow: 0 0 10px rgba(0,240,255,0.5); }
    .hero-title {
        font-family: 'Lora', serif; font-weight: 700; font-size: 2.9rem; margin: 0; letter-spacing: -0.01em; line-height: 1.15;
        background: linear-gradient(90deg, var(--purple) 0%, var(--neon) 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: var(--purple);
    }
    .hero-sub { font-family: 'Inter', sans-serif; color: #b3a9d4; font-size: 1.05rem; max-width: 560px; margin: 1rem auto 0 auto; line-height: 1.6; }

    /* ---------- Feature cards ---------- */
    .feature-card {
        background: linear-gradient(160deg, var(--ink-soft) 0%, var(--ink-soft2) 100%);
        border: 1px solid #2d2350; border-radius: 10px; padding: 1.3rem 1.1rem; text-align: left; height: 100%;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .feature-card:hover { transform: translateY(-4px); border-color: var(--neon); }
    .feature-card .feature-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .feature-card h4 { font-family: 'Lora', serif; color: var(--paper); font-size: 1.02rem; margin: 0 0 0.35rem 0; }
    .feature-card p { color: #9d92c2; font-size: 0.85rem; line-height: 1.5; margin: 0; }

    /* ---------- Section headers ---------- */
    .section-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--neon); margin-bottom: 0.3rem; display: block; }
    .section-title { font-family: 'Lora', serif; font-weight: 700; font-size: 1.7rem; color: var(--paper); margin: 0 0 0.4rem 0; }
    .section-sub { color: #9d92c2; font-size: 0.95rem; margin-bottom: 1.3rem; }

    /* ---------- Panels ---------- */
    .panel { background: linear-gradient(160deg, var(--ink-soft) 0%, var(--ink-soft2) 100%); border: 1px solid #2d2350; border-radius: 10px; padding: 1.3rem 1.4rem; animation: fadeInUp 0.5s ease; }
    .review-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid #2d2350; color: #c9c0e6; font-size: 0.9rem; }
    .review-row:last-child { border-bottom: none; }
    .review-value { font-family: 'IBM Plex Mono', monospace; color: var(--neon); font-size: 0.85rem; }

    /* ---------- Stat cards ---------- */
    .stat-card {
        background: linear-gradient(160deg, var(--ink-soft) 0%, var(--ink-soft2) 100%);
        border: 1px solid #2d2350; border-left: 3px solid var(--purple); border-radius: 8px;
        padding: 0.9rem 1.1rem; margin-bottom: 1.3rem; display: flex; align-items: center; gap: 0.7rem;
    }
    .stat-icon { font-size: 1.3rem; filter: drop-shadow(0 0 6px rgba(0,240,255,0.5)); }
    .stat-number { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: var(--paper); line-height: 1.1; }
    .stat-label { font-size: 0.78rem; color: #9d92c2; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: var(--ink-soft); border-right: 1px solid #2d2350; }
    .sidebar-brand { font-family: 'Lora', serif; color: var(--paper); font-size: 1.2rem; font-weight: 700; margin-bottom: 0.1rem; }
    .sidebar-section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--neon); margin: 1.2rem 0 0.5rem 0; border-bottom: 1px solid #2d2350; padding-bottom: 0.35rem; }

    /* ---------- Chat bubbles ---------- */
    .chat-turn { margin-bottom: 1.1rem; animation: fadeInUp 0.4s ease; }
    .bubble-question {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: #fff;
        border-radius: 4px 14px 14px 14px; padding: 0.7rem 1rem; font-size: 0.95rem; max-width: 80%; margin-left: auto;
        box-shadow: 0 2px 10px rgba(139,92,246,0.35);
    }
    .bubble-question-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--purple); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; text-align: right; }
    .bubble-answer {
        background: var(--paper); color: var(--ink-text); border-left: 3px solid var(--neon); border-radius: 4px;
        padding: 0.85rem 1.1rem; font-family: 'Lora', serif; font-size: 1rem; line-height: 1.55; max-width: 85%;
        box-shadow: 0 2px 14px rgba(0,240,255,0.15);
    }
    .bubble-answer-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--neon); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }

    /* ---------- Evidence / source cards ---------- */
    .evidence-card { background: var(--paper); border-left: 3px solid var(--neon-pink); border-radius: 4px; padding: 0.65rem 0.9rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
    .evidence-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
    .evidence-source { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--purple-dark); font-weight: 500; }
    .evidence-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; background: linear-gradient(90deg, var(--neon), var(--neon-pink)); color: #1b1030; padding: 0.12rem 0.5rem; border-radius: 10px; font-weight: 600; }
    .evidence-text { font-family: 'Lora', serif; font-style: italic; font-size: 0.88rem; color: #3a3f4a; line-height: 1.5; }

    /* ---------- Suggested question chips ---------- */
    .chip-row-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--neon); margin: 0.6rem 0 0.6rem 0; }

    /* ---------- Empty state ---------- */
    .empty-state { border: 1px dashed #382a5e; border-radius: 10px; padding: 2.2rem 1.5rem; text-align: center; color: #9d92c2; margin-top: 1rem; animation: fadeInUp 0.5s ease; }
    .empty-state .empty-title { font-family: 'Lora', serif; color: var(--paper); font-size: 1.1rem; margin: 0.6rem 0 0.3rem 0; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--purple) 0%, var(--purple-dark) 100%);
        color: #fff; border: none; border-radius: 6px; font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: 0.01em;
        transition: box-shadow 0.25s ease, transform 0.15s ease;
    }
    .stButton > button:hover { box-shadow: 0 0 18px rgba(0,240,255,0.55); transform: translateY(-1px); color: #fff; }

    /* Document chips */
    .doc-chip {
        display: inline-block; background: var(--ink); border: 1px solid #382a5e; color: #c9c0e6;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; padding: 0.25rem 0.6rem; border-radius: 4px; margin: 0.15rem 0.25rem 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- #
# Original SVG illustrations (hand-drawn, no external images used —
# keeps the project copyright-clean and on the neon/purple palette)
# --------------------------------------------------------------------- #
HERO_SVG = """
<svg viewBox="0 0 420 260" width="100%" height="230" xmlns="http://www.w3.org/2000/svg">
  <rect x="70" y="70" width="130" height="160" rx="8" fill="#1f1638" stroke="#382a5e"/>
  <rect x="95" y="50" width="130" height="160" rx="8" fill="#f5f1fb"/>
  <line x1="112" y1="78" x2="208" y2="78" stroke="#8b5cf6" stroke-width="3"/>
  <line x1="112" y1="94" x2="208" y2="94" stroke="#cfc4ec" stroke-width="3"/>
  <line x1="112" y1="110" x2="190" y2="110" stroke="#cfc4ec" stroke-width="3"/>
  <line x1="112" y1="132" x2="208" y2="132" stroke="#cfc4ec" stroke-width="3"/>
  <line x1="112" y1="148" x2="185" y2="148" stroke="#cfc4ec" stroke-width="3"/>
  <rect x="112" y="164" width="96" height="16" rx="3" fill="#c9f5ff"/>
  <line x1="112" y1="192" x2="208" y2="192" stroke="#cfc4ec" stroke-width="3"/>
  <circle cx="285" cy="150" r="42" fill="none" stroke="#00f0ff" stroke-width="7"/>
  <line x1="315" y1="180" x2="345" y2="210" stroke="#00f0ff" stroke-width="9" stroke-linecap="round"/>
  <path d="M270 150 l10 10 l22 -24" fill="none" stroke="#ff3ec9" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M340 60 l4 10 l10 4 l-10 4 l-4 10 l-4 -10 l-10 -4 l10 -4 z" fill="#00f0ff"/>
  <path d="M60 190 l3 7 l7 3 l-7 3 l-3 7 l-3 -7 l-7 -3 l7 -3 z" fill="#ff3ec9"/>
  <circle cx="200" cy="30" r="3" fill="#8b5cf6"/>
  <circle cx="40" cy="100" r="2.5" fill="#00f0ff"/>
</svg>
"""

UPLOAD_SVG = """
<svg viewBox="0 0 420 190" width="100%" height="170" xmlns="http://www.w3.org/2000/svg">
  <rect x="60" y="75" width="300" height="95" rx="10" fill="#1f1638" stroke="#382a5e"/>
  <circle cx="210" cy="88" r="34" fill="#00f0ff" opacity="0.06"/>
  <path d="M210 58 v55 M192 76 l18 -18 l18 18" fill="none" stroke="#00f0ff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="150" y="145" width="120" height="10" rx="5" fill="#8b5cf6"/>
  <circle cx="330" cy="55" r="5" fill="#8b5cf6"/>
  <circle cx="90" cy="50" r="4" fill="#ff3ec9"/>
</svg>
"""

CONFIG_SVG = """
<svg viewBox="0 0 420 170" width="100%" height="150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="150" cy="85" r="38" fill="none" stroke="#8b5cf6" stroke-width="8"/>
  <circle cx="150" cy="85" r="10" fill="#00f0ff"/>
  <g stroke="#8b5cf6" stroke-width="8" stroke-linecap="round">
    <line x1="150" y1="35" x2="150" y2="47"/>
    <line x1="150" y1="123" x2="150" y2="135"/>
    <line x1="100" y1="85" x2="112" y2="85"/>
    <line x1="188" y1="85" x2="200" y2="85"/>
    <line x1="115" y1="50" x2="123" y2="58"/>
    <line x1="177" y1="112" x2="185" y2="120"/>
    <line x1="185" y1="50" x2="177" y2="58"/>
    <line x1="123" y1="112" x2="115" y2="120"/>
  </g>
  <rect x="250" y="55" width="120" height="14" rx="7" fill="#1f1638" stroke="#382a5e"/>
  <rect x="250" y="55" width="80" height="14" rx="7" fill="#00f0ff"/>
  <rect x="250" y="85" width="120" height="14" rx="7" fill="#1f1638" stroke="#382a5e"/>
  <rect x="250" y="85" width="45" height="14" rx="7" fill="#8b5cf6"/>
  <rect x="250" y="115" width="120" height="14" rx="7" fill="#1f1638" stroke="#382a5e"/>
  <rect x="250" y="115" width="100" height="14" rx="7" fill="#ff3ec9"/>
</svg>
"""

NETWORK_SVG = """
<svg viewBox="0 0 700 120" width="100%" height="110" xmlns="http://www.w3.org/2000/svg">
  <g stroke="#382a5e" stroke-width="1.5">
    <line x1="60" y1="60" x2="200" y2="30"/>
    <line x1="60" y1="60" x2="200" y2="60"/>
    <line x1="60" y1="60" x2="200" y2="90"/>
    <line x1="200" y1="30" x2="340" y2="60"/>
    <line x1="200" y1="60" x2="340" y2="60"/>
    <line x1="200" y1="90" x2="340" y2="60"/>
    <line x1="340" y1="60" x2="480" y2="30"/>
    <line x1="340" y1="60" x2="480" y2="90"/>
    <line x1="480" y1="30" x2="620" y2="60"/>
    <line x1="480" y1="90" x2="620" y2="60"/>
  </g>
  <circle cx="60" cy="60" r="7" fill="#8b5cf6"/>
  <circle cx="200" cy="30" r="6" fill="#00f0ff"/>
  <circle cx="200" cy="60" r="6" fill="#00f0ff"/>
  <circle cx="200" cy="90" r="6" fill="#00f0ff"/>
  <circle cx="340" cy="60" r="8" fill="#ff3ec9"/>
  <circle cx="480" cy="30" r="6" fill="#00f0ff"/>
  <circle cx="480" cy="90" r="6" fill="#00f0ff"/>
  <circle cx="620" cy="60" r="7" fill="#8b5cf6"/>
</svg>
"""

EMPTY_SVG = """
<svg viewBox="0 0 160 120" width="90" height="70" xmlns="http://www.w3.org/2000/svg">
  <rect x="30" y="35" width="100" height="70" rx="8" fill="#1f1638" stroke="#382a5e" stroke-width="2"/>
  <path d="M30 45 h100" stroke="#382a5e" stroke-width="2"/>
  <circle cx="80" cy="75" r="16" fill="none" stroke="#00f0ff" stroke-width="4" stroke-dasharray="4 4"/>
  <line x1="90" y1="86" x2="100" y2="96" stroke="#ff3ec9" stroke-width="4" stroke-linecap="round"/>
</svg>
"""


def draw(svg: str) -> None:
    st.markdown(f'<div class="hero-wrap">{svg}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------- #
DEFAULTS = {
    "step": 1,
    "pipeline": None,
    "pipeline_backend": None,
    "ingested_files": [],
    "chat_history": [],
    "top_k": 4,
    "backend_choice": "local",
    "_pending_uploads": [],
    "_trigger_question": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_pipeline(generator_backend: str) -> RAGPipeline:
    if st.session_state.pipeline is None or st.session_state.pipeline_backend != generator_backend:
        with st.spinner(f"Loading models ({generator_backend} backend)... this can take a minute the first time."):
            st.session_state.pipeline = RAGPipeline(generator_backend=generator_backend)
            st.session_state.pipeline_backend = generator_backend
    return st.session_state.pipeline


def render_sources(sources):
    with st.expander(f"📎 Evidence ({len(sources)} passage{'s' if len(sources) != 1 else ''})"):
        for i, src in enumerate(sources, start=1):
            score_pct = max(0, min(100, round(src["score"] * 100)))
            snippet = src["text"][:420] + ("…" if len(src["text"]) > 420 else "")
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-meta">
                        <span class="evidence-source">📄 {src['source']} · passage {i}</span>
                        <span class="evidence-score">{score_pct}% match</span>
                    </div>
                    <div class="evidence-text">“{snippet}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_stepper(current: int) -> None:
    labels = ["① Welcome", "② Upload", "③ Configure & Review", "④ Ask Questions"]
    pills = []
    for i, label in enumerate(labels, start=1):
        cls = "active" if i == current else ("done" if i < current else "")
        pills.append(f'<div class="step-pill {cls}">{label}</div>')
        if i < len(labels):
            pills.append('<div class="step-line"></div>')
    st.markdown(f'<div class="stepper">{"".join(pills)}</div>', unsafe_allow_html=True)


def go_to(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def process_question(question: str) -> None:
    """Render a question turn, run it through the pipeline, and store the result."""
    st.markdown(
        f"""<div class="chat-turn"><div class="bubble-question-label">You asked</div>
        <div class="bubble-question">{question}</div></div>""",
        unsafe_allow_html=True,
    )
    with st.spinner("Retrieving relevant passages and drafting a grounded answer..."):
        result = st.session_state.pipeline.ask(question, top_k=st.session_state.top_k)

    st.markdown(
        f"""<div class="chat-turn"><div class="bubble-answer-label">Evidence Desk answers</div>
        <div class="bubble-answer">{result['answer']}</div></div>""",
        unsafe_allow_html=True,
    )
    if result["sources"]:
        render_sources(result["sources"])

    st.session_state.chat_history.append(
        {"question": question, "answer": result["answer"], "sources": result["sources"]}
    )
    st.rerun()


# --------------------------------------------------------------------- #
# Sidebar — always visible, minimal
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📚 Evidence Desk</div>', unsafe_allow_html=True)
    st.caption("Document QA powered by NLP + RAG")

    st.markdown('<div class="sidebar-section-title">Progress</div>', unsafe_allow_html=True)
    step_names = {1: "Welcome", 2: "Upload", 3: "Configure & Review", 4: "Ask Questions"}
    st.write(f"Step **{st.session_state.step} of 4** — {step_names[st.session_state.step]}")

    if st.session_state.ingested_files:
        st.markdown('<div class="sidebar-section-title">Indexed documents</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="doc-chip">📄 {name}</span>' for name in st.session_state.ingested_files)
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)
    if st.button("↺ Start over", use_container_width=True):
        for key in DEFAULTS:
            st.session_state.pop(key, None)
        st.rerun()

# ======================================================================= #
# STEP 1 — WELCOME
# ======================================================================= #
if st.session_state.step == 1:
    render_stepper(1)
    draw(HERO_SVG)
    st.markdown(
        """
        <div class="hero-wrap">
            <span class="hero-eyebrow">Intelligent Document QA · NLP + RAG</span>
            <h1 class="hero-title">Evidence Desk</h1>
            <p class="hero-sub">
                Upload any document, ask questions in plain language, and get answers
                grounded in the exact passages that support them — not guesses,
                not memorized facts, just your own documents talking back.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">📤</div>
            <h4>Upload anything</h4><p>PDF, Word, or plain text — parsed and indexed in seconds.</p></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">🔍</div>
            <h4>Ask naturally</h4><p>No keyword search. Ask the way you'd ask a person.</p></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">📎</div>
            <h4>See the evidence</h4><p>Every answer links back to the exact passage it came from.</p></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Get started →", type="primary", use_container_width=True):
            go_to(2)

# ======================================================================= #
# STEP 2 — UPLOAD DOCUMENTS
# ======================================================================= #
elif st.session_state.step == 2:
    render_stepper(2)
    draw(UPLOAD_SVG)
    st.markdown(
        """
        <span class="section-eyebrow" style="display:block; text-align:center;">Step 2 of 4</span>
        <div class="section-title" style="text-align:center;">Upload your documents</div>
        <div class="section-sub" style="text-align:center;">Add one or more PDF, Word, or text files. You can add more later.</div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([0.15, 1.7, 0.15])
    with mid:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.session_state["_pending_uploads"] = uploaded_files

        if st.session_state.ingested_files:
            st.write("")
            st.caption("Already indexed:")
            chips = "".join(f'<span class="doc-chip">📄 {name}</span>' for name in st.session_state.ingested_files)
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("← Back", use_container_width=True):
                go_to(1)
        with b2:
            can_continue = bool(uploaded_files) or bool(st.session_state.ingested_files)
            if st.button("Continue →", type="primary", use_container_width=True, disabled=not can_continue):
                go_to(3)
        if not can_continue:
            st.caption("Add at least one file to continue.")

# ======================================================================= #
# STEP 3 — CONFIGURE & REVIEW
# ======================================================================= #
elif st.session_state.step == 3:
    render_stepper(3)
    draw(CONFIG_SVG)
    st.markdown(
        """
        <span class="section-eyebrow" style="display:block; text-align:center;">Step 3 of 4</span>
        <div class="section-title" style="text-align:center;">Configure and review</div>
        <div class="section-sub" style="text-align:center;">Choose how answers are generated, then index your files.</div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([0.15, 1.7, 0.15])
    with mid:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.session_state.backend_choice = st.selectbox(
            "Answer generation backend",
            options=["local", "openai"],
            index=["local", "openai"].index(st.session_state.backend_choice),
            help="'local' runs fully offline (flan-t5, no API key needed). 'openai' needs OPENAI_API_KEY set in your environment.",
        )
        st.session_state.top_k = st.slider("Passages retrieved per question", 1, 10, st.session_state.top_k)

        st.write("")
        pending = st.session_state.get("_pending_uploads") or []
        if pending:
            st.caption(f"{len(pending)} new file(s) ready to index:")
            chips = "".join(f'<span class="doc-chip">📄 {f.name}</span>' for f in pending)
            st.markdown(chips, unsafe_allow_html=True)
        elif not st.session_state.ingested_files:
            st.info("No files uploaded yet — go back to Step 2 to add some.")

        st.write("")
        if st.button("⚙ Index documents", type="primary", use_container_width=True, disabled=not pending):
            pipeline = get_pipeline(st.session_state.backend_choice)
            with st.spinner("Reading, chunking, and embedding documents..."):
                total_chunks = 0
                for uploaded_file in pending:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    added = pipeline.ingest_file(tmp_path)
                    total_chunks += added
                    if uploaded_file.name not in st.session_state.ingested_files:
                        st.session_state.ingested_files.append(uploaded_file.name)
            st.session_state["_pending_uploads"] = []
            st.success(f"Indexed {len(pending)} file(s) · {total_chunks} passages added.")

        if st.session_state.ingested_files:
            st.write("")
            st.markdown(
                f"""
                <div class="review-row"><span>Backend</span><span class="review-value">{st.session_state.backend_choice}</span></div>
                <div class="review-row"><span>Passages retrieved (top-k)</span><span class="review-value">{st.session_state.top_k}</span></div>
                <div class="review-row"><span>Documents indexed</span><span class="review-value">{len(st.session_state.ingested_files)}</span></div>
                <div class="review-row"><span>Total passages in index</span><span class="review-value">{len(st.session_state.pipeline.store) if st.session_state.pipeline else 0}</span></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("← Back", use_container_width=True):
                go_to(2)
        with b2:
            if st.button(
                "Continue to Q&A →", type="primary", use_container_width=True,
                disabled=not st.session_state.ingested_files,
            ):
                go_to(4)
        if not st.session_state.ingested_files:
            st.caption("Index at least one document to continue.")

# ======================================================================= #
# STEP 4 — ASK QUESTIONS (dashboard)
# ======================================================================= #
elif st.session_state.step == 4:
    render_stepper(4)
    draw(NETWORK_SVG)

    st.markdown(
        """
        <div class="hero-wrap" style="text-align:left; padding-left:0;">
            <span class="section-eyebrow">Step 4 of 4</span>
            <div class="section-title">Ask your documents</div>
            <div class="section-sub">Every answer is grounded in a highlighted passage — check the evidence panel to verify it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    num_docs = len(st.session_state.ingested_files)
    num_chunks = len(st.session_state.pipeline.store) if st.session_state.pipeline else 0
    num_questions = len(st.session_state.chat_history)

    col1, col2, col3 = st.columns(3)
    stat_icons = ["📄", "🧩", "💬"]
    for col, icon, number, label in zip(
        [col1, col2, col3], stat_icons,
        [num_docs, num_chunks, num_questions],
        ["Documents indexed", "Passages in index", "Questions asked"],
    ):
        col.markdown(
            f"""<div class="stat-card"><div class="stat-icon">{icon}</div>
            <div><div class="stat-number">{number}</div><div class="stat-label">{label}</div></div></div>""",
            unsafe_allow_html=True,
        )

    if st.button("← Add more documents / change settings"):
        go_to(2)

    # ---- Suggested question chips (shown before the first question) ---- #
    if not st.session_state.chat_history:
        st.markdown('<div class="chip-row-label">💡 Try asking</div>', unsafe_allow_html=True)
        suggestions = [
            "What is this document about?",
            "Summarize the key points",
            "List the main topics covered",
        ]
        chip_cols = st.columns(len(suggestions))
        for col, suggestion in zip(chip_cols, suggestions):
            with col:
                if st.button(suggestion, key=f"chip_{suggestion}", use_container_width=True):
                    st.session_state["_trigger_question"] = suggestion
                    st.rerun()

    for turn in st.session_state.chat_history:
        st.markdown(
            f"""<div class="chat-turn"><div class="bubble-question-label">You asked</div>
            <div class="bubble-question">{turn['question']}</div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="chat-turn"><div class="bubble-answer-label">Evidence Desk answers</div>
            <div class="bubble-answer">{turn['answer']}</div></div>""",
            unsafe_allow_html=True,
        )
        if turn["sources"]:
            render_sources(turn["sources"])

    typed_question = st.chat_input("Ask a question about your uploaded documents…")
    triggered_question = st.session_state.pop("_trigger_question", None)
    final_question = typed_question or triggered_question

    if final_question:
        process_question(final_question)