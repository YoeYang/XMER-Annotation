import streamlit as st
import json
import os

st.set_page_config(page_title="XMER Annotation", layout="wide")

# ---------------------------------------------------------------------------
# Sample data — replace with your real dataset
# ---------------------------------------------------------------------------
SAMPLES = [
    {
        "id": "sample_001",
        "video_url": "https://www.w3schools.com/html/mov_bbb.mp4",
        "description": "Sample video 1",
    },
    {
        "id": "sample_002",
        "video_url": "https://www.w3schools.com/html/movie.mp4",
        "description": "Sample video 2",
    },
]

QUESTIONS = [
    {
        "key": "emotion",
        "label": "Q1: What is the primary emotion expressed in this video?",
        "options": ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Neutral"],
    },
    {
        "key": "intensity",
        "label": "Q2: How intense is the emotion?",
        "options": ["Mild", "Moderate", "Strong"],
    },
    {
        "key": "confidence",
        "label": "Q3: How confident are you in your answers?",
        "options": ["Not confident", "Somewhat confident", "Very confident"],
    },
]

ANNOTATIONS_FILE = "annotations.json"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def load_annotations() -> dict:
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(data: dict) -> None:
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "annotations" not in st.session_state:
    st.session_state.annotations = load_annotations()

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("XMER Annotation Tool")

total = len(SAMPLES)
annotated = sum(1 for s in SAMPLES if s["id"] in st.session_state.annotations)
st.progress(annotated / total, text=f"Progress: {annotated} / {total} samples annotated")

st.divider()

# Navigation
col_prev, col_info, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("← Previous", disabled=st.session_state.current_idx == 0):
        st.session_state.current_idx -= 1
        st.rerun()
with col_info:
    st.markdown(
        f"<div style='text-align:center; padding-top:6px;'>Sample "
        f"<b>{st.session_state.current_idx + 1}</b> of <b>{total}</b></div>",
        unsafe_allow_html=True,
    )
with col_next:
    if st.button("Next →", disabled=st.session_state.current_idx == total - 1):
        st.session_state.current_idx += 1
        st.rerun()

st.divider()

sample = SAMPLES[st.session_state.current_idx]
existing = st.session_state.annotations.get(sample["id"], {})

# Video
st.subheader(f"Sample ID: `{sample['id']}`")
if sample.get("description"):
    st.caption(sample["description"])
st.video(sample["video_url"])

st.divider()

# Questions
st.subheader("Annotation Questions")
answers = {}
for q in QUESTIONS:
    default_idx = 0
    if q["key"] in existing:
        try:
            default_idx = q["options"].index(existing[q["key"]])
        except ValueError:
            default_idx = 0
    answers[q["key"]] = st.radio(
        q["label"],
        options=q["options"],
        index=default_idx,
        horizontal=True,
        key=f"{sample['id']}_{q['key']}",
    )

st.divider()

# Submit
if st.button("Save Annotation", type="primary"):
    st.session_state.annotations[sample["id"]] = answers
    save_annotations(st.session_state.annotations)
    st.success("Annotation saved!")
    # Auto-advance to next sample
    if st.session_state.current_idx < total - 1:
        st.session_state.current_idx += 1
        st.rerun()

# Show already-saved state
if sample["id"] in st.session_state.annotations:
    st.info("This sample has been annotated. You can still update and re-save.")
