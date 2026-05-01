# ============================================================
# app2.py — XMER Seamless Interaction Data Cleaning Platform
# ============================================================
import streamlit as st
import streamlit.components.v1 as components
import json
import os
from datetime import datetime

import traceback

import requests

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

try:
    from huggingface_hub import list_repo_files, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

st.set_page_config(
    page_title="XMER Data Cleaning",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

HF_REPO_ID  = "YoeYang/XMER-Videos"
HF_SPLIT    = "seamless"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_SPLIT}"
BATCH_SIZE  = 10

VISUAL_EMOTIONS = [
    "Neutral", "Happy", "Sad", "Angry",
    "Fearful", "Disgusted", "Surprised", "Contempt", "Unclear",
]

AUDIO_EMOTIONS = [
    "Angry-like (high pitch, loud, fast)",
    "Sad-like (low pitch, quiet, slow)",
    "Happy-like (varied pitch, energetic)",
    "Neutral / Calm",
    "Anxious / Tense",
    "Unclear",
]

TEXT_SENTIMENTS = ["Positive", "Neutral", "Negative"]

KEPT_HEADERS = [
    "sample_id", "timestamp", "dataset_split", "participant_id",
    "visual_emotion_human", "visual_valence_human",
    "audio_emotion_human",  "audio_valence_human",
    "text_sentiment_human", "text_valence_human", "text_notes_human",
    "gemini_visual_emotion",   "gemini_visual_reasoning",
    "gemini_audio_emotion",    "gemini_audio_reasoning",
    "gemini_text_sentiment",   "gemini_text_reasoning",
    "gemini_conflict_detected", "gemini_conflict_description",
    "selfReport_1P_IS", "selfReport_1P_R",
    "thirdParty_3P_IS", "thirdParty_3P_R", "thirdParty_3P_V",
    "transcript_text", "has_conflict_human",
]

DISCARDED_HEADERS = [
    "sample_id", "timestamp", "discard_step",
    "partial_visual", "partial_audio", "partial_text", "discard_reason",
]

# ============================================================
# SECTION 2: HELPERS
# ============================================================

def _get_hf_token() -> str:
    try:
        return st.secrets.get("huggingface", {}).get("token", "") or os.environ.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


def _get_gemini_api_key() -> str:
    try:
        return (
            st.secrets.get("gemini", {}).get("api_key", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
    except Exception:
        return os.environ.get("GOOGLE_API_KEY", "")


def _file_url(sample_id: str, filename: str) -> str:
    token = _get_hf_token()
    url = f"{HF_BASE_URL}/{sample_id}/{filename}"
    if token:
        url += f"?token={token}"
    return url


def _parse_participant_id(sample_id: str) -> str:
    for part in sample_id.split("_"):
        if part.startswith("P") and part[1:].isdigit():
            return part
    return sample_id


def _emotion_valence(label: str) -> int:
    positive = {
        "Happy", "Positive", "Surprised",
        "Happy-like (varied pitch, energetic)",
    }
    negative = {
        "Sad", "Angry", "Fearful", "Disgusted", "Contempt", "Negative",
        "Angry-like (high pitch, loud, fast)",
        "Sad-like (low pitch, quiet, slow)",
        "Anxious / Tense",
    }
    if label in positive:
        return 1
    if label in negative:
        return -1
    return 0


def _derive_has_conflict(annotations: dict) -> bool:
    v = _emotion_valence(annotations.get("visual_emotion_human", ""))
    a = _emotion_valence(annotations.get("audio_emotion_human", ""))
    t = _emotion_valence(annotations.get("text_sentiment_human", ""))
    return any(x * y < 0 for x, y in [(v, a), (v, t), (a, t)])


def _parse_annotation_types(ann_list: list) -> dict:
    """Flexibly extract typed annotation content from a JSONL-parsed list."""
    result = {}
    if not ann_list:
        return result
    known_types = {"1P-IS", "1P-R", "3P-IS", "3P-R", "3P-V"}
    for item in ann_list:
        if not isinstance(item, dict):
            continue
        # Try direct type field
        ann_type = item.get("type", item.get("annotation_type", ""))
        content = (
            item.get("content")
            or item.get("text")
            or item.get("value")
            or item.get("annotation")
        )
        if ann_type in known_types and content is not None:
            result[ann_type] = str(content)
        # Also scan for bare keys matching known types
        for kt in known_types:
            if kt in item and kt not in result:
                result[kt] = str(item[kt])
    return result


# ============================================================
# SECTION 3: DATA LOADING
# ============================================================

@st.cache_data(ttl=3600, show_spinner="Discovering samples from HuggingFace…")
def discover_samples() -> list:
    if not HF_AVAILABLE:
        return []
    token = _get_hf_token() or None
    try:
        all_files = list(list_repo_files(HF_REPO_ID, repo_type="dataset", token=token))
        sample_ids: set[str] = set()
        for f in all_files:
            parts = f.replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[0] == HF_SPLIT and parts[1]:
                sample_ids.add(parts[1])
        return sorted(sample_ids)
    except Exception as e:
        st.warning(f"Could not list HuggingFace samples: {e}")
        return []


@st.cache_data(show_spinner=False)
def fetch_transcript(sample_id: str) -> str:
    url = _file_url(sample_id, f"{sample_id}.txt")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def fetch_annotations_jsonl(sample_id: str) -> list:
    url = _file_url(sample_id, "annotations.jsonl")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        lines = [ln.strip() for ln in resp.text.strip().split("\n") if ln.strip()]
        return [json.loads(ln) for ln in lines]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def download_for_gemini(sample_id: str, filename: str) -> str | None:
    """Download a file from the HF dataset to the local HF cache and return its path."""
    if not HF_AVAILABLE:
        return None
    token = _get_hf_token() or None
    hf_path = f"{HF_SPLIT}/{sample_id}/{filename}"
    try:
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            token=token,
        )
    except Exception as e:
        st.warning(f"Could not download {filename}: {e}")
        return None


# ============================================================
# SECTION 4: GOOGLE SHEETS
# ============================================================

@st.cache_resource
def _gspread_client():
    if not GSHEETS_AVAILABLE:
        return None
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        return gspread.authorize(creds)
    except Exception:
        return None


def _get_or_create_worksheet(sheet_name: str, headers: list):
    client = _gspread_client()
    if client is None:
        return None
    try:
        sheet_id = st.secrets["google_sheets"]["sheet_id"]
        spreadsheet = client.open_by_key(sheet_id)
        try:
            ws = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(
                title=sheet_name, rows=2000, cols=len(headers)
            )
        first_row = ws.row_values(1)
        if not first_row or first_row[0] != headers[0]:
            ws.insert_row(headers, 1)
        return ws
    except Exception:
        return None


def _save_kept(record: dict) -> bool:
    ws = _get_or_create_worksheet("kept_samples", KEPT_HEADERS)
    if ws is None:
        return False
    try:
        row = [str(record.get(h, "")) for h in KEPT_HEADERS]
        ws.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception:
        return False


def _save_discarded(record: dict) -> bool:
    ws = _get_or_create_worksheet("discarded_samples", DISCARDED_HEADERS)
    if ws is None:
        return False
    try:
        row = [str(record.get(h, "")) for h in DISCARDED_HEADERS]
        ws.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception:
        return False


# ============================================================
# SECTION 5: SESSION STATE
# ============================================================

def _init_state():
    defaults: dict = {
        "current_sample":  None,   # {"sample_id": str}
        "current_step":    1,      # 1-5
        "annotations":     {},     # human annotation values
        "gemini_result":   None,   # dict returned by run_gemini_analysis
        "sample_queue":    [],     # ordered list of sample_ids for this session
        "processed_ids":   set(),  # all handled IDs (saved OR discarded)
        "discard_confirm": False,
        "gemini_error":    None,
        "selected_batch":  1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ============================================================
# SECTION 6: SAMPLE LIFECYCLE
# ============================================================

def _load_next_sample(batch_samples: list):
    unprocessed = [s for s in batch_samples if s not in st.session_state.processed_ids]
    if not unprocessed:
        st.session_state.current_sample = None
        st.rerun()
    sample_id = unprocessed[0]
    st.session_state.current_sample   = {"sample_id": sample_id}
    st.session_state.current_step     = 1
    st.session_state.annotations      = {}
    st.session_state.gemini_result    = None
    st.session_state.discard_confirm  = False
    st.session_state.gemini_error     = None
    st.session_state.sample_queue     = unprocessed
    st.rerun()


def _do_save(sample_id: str):
    anns = st.session_state.annotations
    gr   = st.session_state.gemini_result or {}
    transcript = fetch_transcript(sample_id)
    ann_list   = fetch_annotations_jsonl(sample_id)
    typed_anns = _parse_annotation_types(ann_list)

    record = {
        "sample_id":       sample_id,
        "timestamp":       datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset_split":   HF_SPLIT,
        "participant_id":  _parse_participant_id(sample_id),
        # Human annotations
        "visual_emotion_human":  anns.get("visual_emotion_human", ""),
        "visual_valence_human":  anns.get("visual_valence_human", ""),
        "audio_emotion_human":   anns.get("audio_emotion_human", ""),
        "audio_valence_human":   anns.get("audio_valence_human", ""),
        "text_sentiment_human":  anns.get("text_sentiment_human", ""),
        "text_valence_human":    anns.get("text_valence_human", ""),
        "text_notes_human":      anns.get("text_notes_human", ""),
        # Gemini
        "gemini_visual_emotion":      gr.get("video", {}).get("emotion", ""),
        "gemini_visual_reasoning":    gr.get("video", {}).get("description", ""),
        "gemini_audio_emotion":       gr.get("audio", {}).get("emotion", ""),
        "gemini_audio_reasoning":     gr.get("audio", {}).get("description", ""),
        "gemini_text_sentiment":      gr.get("text",  {}).get("emotion", ""),
        "gemini_text_reasoning":      gr.get("text",  {}).get("description", ""),
        "gemini_conflict_detected":   str(gr.get("conflict", {}).get("detected", "")),
        "gemini_conflict_description": gr.get("conflict", {}).get("description", ""),
        # Dataset annotations
        "selfReport_1P_IS": typed_anns.get("1P-IS", "N/A"),
        "selfReport_1P_R":  typed_anns.get("1P-R",  "N/A"),
        "thirdParty_3P_IS": typed_anns.get("3P-IS", "N/A"),
        "thirdParty_3P_R":  typed_anns.get("3P-R",  "N/A"),
        "thirdParty_3P_V":  typed_anns.get("3P-V",  "N/A"),
        "transcript_text":  transcript,
        "has_conflict_human": str(_derive_has_conflict(anns)),
    }
    ok = _save_kept(record)
    st.session_state.processed_ids.add(sample_id)
    st.session_state.current_sample = None
    if ok:
        st.success("✅ Sample saved to Google Sheets!")
    else:
        st.warning("Google Sheets not connected — progress tracked in session only.")
    st.rerun()


def _do_discard(sample_id: str, step: int, reason: str):
    anns = st.session_state.annotations
    record = {
        "sample_id":     sample_id,
        "timestamp":     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "discard_step":  step,
        "partial_visual": anns.get("visual_emotion_human", ""),
        "partial_audio":  anns.get("audio_emotion_human",  ""),
        "partial_text":   anns.get("text_sentiment_human", ""),
        "discard_reason": reason,
    }
    _save_discarded(record)
    st.session_state.processed_ids.add(sample_id)
    st.session_state.current_sample  = None
    st.session_state.discard_confirm = False
    st.rerun()


# ============================================================
# SECTION 7: GEMINI RUNNER
# ============================================================

def _run_gemini(sample_id: str, transcript_text: str) -> dict | None:
    from gemini_analysis import (
        build_gemini_model_simple,
        gen_video_signals_from_path,
        gen_audio_signals_from_path,
        gen_text_signals_from_text,
        gen_conflict_assessment,
        parse_output,
        parse_conflict_output,
    )

    api_key = _get_gemini_api_key()
    if not api_key:
        st.error("Gemini API key not configured. Add it to .streamlit/secrets.toml or GOOGLE_API_KEY env var.")
        return None

    try:
        model = build_gemini_model_simple(api_key)
    except Exception as e:
        st.error(f"Could not initialise Gemini model: {e}")
        return None

    result: dict = {}
    no_audio_filename = f"{sample_id}_no_audio.mp4"
    audio_filename    = f"{sample_id}.wav"

    with st.status("Running Gemini multimodal analysis…", expanded=True) as status:

        # ── Visual ───────────────────────────────────────────────
        st.write("📹 Downloading silent video for visual analysis…")
        no_audio_path = download_for_gemini(sample_id, no_audio_filename)
        if no_audio_path:
            try:
                st.write("📹 Analysing visual cues…")
                raw = gen_video_signals_from_path(model, no_audio_path)
                em, desc = parse_output(raw)
                result["video"] = {"emotion": em, "description": desc, "raw": raw}
                st.markdown(f"**Visual:** `{em}` — {desc}")
            except Exception as e:
                tb = traceback.format_exc()
                result["video"] = {"emotion": "", "description": "", "raw": "", "error": str(e)}
                st.warning(f"Visual analysis failed: {e}")
                with st.expander("🐛 Visual error details"):
                    st.code(tb)
        else:
            result["video"] = {"emotion": "", "description": "", "raw": "", "error": "download failed"}
            st.warning("Could not download the silent video.")

        # ── Audio ────────────────────────────────────────────────
        st.write("🔊 Downloading audio file…")
        audio_path = download_for_gemini(sample_id, audio_filename)
        if audio_path:
            try:
                st.write("🔊 Analysing prosodic cues…")
                raw = gen_audio_signals_from_path(model, audio_path)
                em, desc = parse_output(raw)
                result["audio"] = {"emotion": em, "description": desc, "raw": raw}
                st.markdown(f"**Audio:** `{em}` — {desc}")
            except Exception as e:
                tb = traceback.format_exc()
                result["audio"] = {"emotion": "", "description": "", "raw": "", "error": str(e)}
                st.warning(f"Audio analysis failed: {e}")
                with st.expander("🐛 Audio error details"):
                    st.code(tb)
        else:
            result["audio"] = {"emotion": "", "description": "", "raw": "", "error": "download failed"}
            st.warning("Could not download the audio file.")

        # ── Text ─────────────────────────────────────────────────
        try:
            st.write("📝 Analysing text sentiment…")
            raw = gen_text_signals_from_text(model, transcript_text or "(no transcript)")
            em, desc = parse_output(raw)
            result["text"] = {"emotion": em, "description": desc, "raw": raw}
            st.markdown(f"**Text:** `{em}` — {desc}")
        except Exception as e:
            tb = traceback.format_exc()
            result["text"] = {"emotion": "", "description": "", "raw": "", "error": str(e)}
            st.warning(f"Text analysis failed: {e}")
            with st.expander("🐛 Text error details"):
                st.code(tb)

        # ── Conflict assessment ───────────────────────────────────
        try:
            st.write("🔄 Assessing cross-modal conflict…")
            cr = gen_conflict_assessment(
                model,
                result.get("video", {}).get("emotion", ""),
                result.get("video", {}).get("description", ""),
                result.get("audio", {}).get("emotion", ""),
                result.get("audio", {}).get("description", ""),
                result.get("text",  {}).get("emotion", ""),
                result.get("text",  {}).get("description", ""),
            )
            detected, desc = parse_conflict_output(cr)
            result["conflict"] = {"detected": detected, "description": desc, "raw": cr}
            flag = "⚠️ **Conflict detected**" if detected else "✅ **No conflict**"
            st.markdown(f"{flag}: {desc}")
        except Exception as e:
            tb = traceback.format_exc()
            result["conflict"] = {"detected": False, "description": "", "raw": "", "error": str(e)}
            st.warning(f"Conflict assessment failed: {e}")
            with st.expander("🐛 Conflict error details"):
                st.code(tb)

        status.update(label="✅ Gemini analysis complete!", state="complete")

    return result


# ============================================================
# SECTION 8: STEP RENDERERS
# ============================================================

def _step_indicator(current: int):
    labels = ["1 Visual", "2 Audio", "3 Text", "4 Review", "5 Gemini"]
    cols = st.columns(5)
    for i, label in enumerate(labels):
        with cols[i]:
            n = i + 1
            if n == current:
                st.markdown(
                    f"<div style='text-align:center;background:#1f77b4;color:white;"
                    f"padding:6px;border-radius:6px;font-weight:bold;'>▶ {label}</div>",
                    unsafe_allow_html=True,
                )
            elif n < current:
                st.markdown(
                    f"<div style='text-align:center;background:#d4edda;color:#155724;"
                    f"padding:6px;border-radius:6px;'>✓ {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;background:#e9ecef;color:#6c757d;"
                    f"padding:6px;border-radius:6px;'>{label}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("")


def _render_step1(sample_id: str):
    st.markdown("### Step 1 — Visual Modality (no audio)")
    st.caption("Watch the silent video and annotate facial / body expression.")

    no_audio_url = _file_url(sample_id, f"{sample_id}_no_audio.mp4")
    components.html(
        f"""
        <div style="display:flex;justify-content:center;">
          <video controls muted
            style="max-width:280px;width:100%;border-radius:8px;background:#000;">
            <source src="{no_audio_url}" type="video/mp4">
            <p>Your browser does not support HTML5 video.
               <a href="{no_audio_url}">Download video</a>.</p>
          </video>
        </div>
        """,
        height=520,
    )

    st.markdown("**Facial / body emotion**")
    visual_emotion = st.radio(
        "Facial / body emotion",
        VISUAL_EMOTIONS,
        index=None,
        horizontal=True,
        key=f"vis_em_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("**Emotion valence**")
    visual_valence = st.slider(
        "Visual valence",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
        key=f"vis_val_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("")
    if st.button("Confirm Visual Annotation →", type="primary", key="step1_confirm"):
        if not visual_emotion:
            st.error("Please select a facial / body emotion before continuing.")
        else:
            st.session_state.annotations["visual_emotion_human"] = visual_emotion
            st.session_state.annotations["visual_valence_human"] = visual_valence
            st.session_state.current_step = 2
            st.rerun()


def _render_step2(sample_id: str):
    st.markdown("### Step 2 — Audio Modality (no video, no text)")
    st.caption("Listen to the audio and annotate the prosodic / vocal emotion.")

    audio_url = _file_url(sample_id, f"{sample_id}.wav")
    st.audio(audio_url, format="audio/wav")

    st.markdown("**Prosodic / vocal emotion**")
    audio_emotion = st.radio(
        "Prosodic emotion",
        AUDIO_EMOTIONS,
        index=None,
        horizontal=False,
        key=f"aud_em_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("**Emotion valence**")
    audio_valence = st.slider(
        "Audio valence",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
        key=f"aud_val_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="step2_back"):
            st.session_state.current_step = 1
            st.rerun()
    with col_next:
        if st.button("Confirm Audio Annotation →", type="primary", key="step2_confirm"):
            if not audio_emotion:
                st.error("Please select a prosodic emotion before continuing.")
            else:
                st.session_state.annotations["audio_emotion_human"] = audio_emotion
                st.session_state.annotations["audio_valence_human"] = audio_valence
                st.session_state.current_step = 3
                st.rerun()


def _render_step3(sample_id: str):
    st.markdown("### Step 3 — Text Modality (transcript only)")
    st.caption("Read the transcript and annotate the verbal / semantic sentiment.")

    transcript = fetch_transcript(sample_id)
    if transcript:
        st.markdown(
            f"<div style='background:#f0f2f6;padding:14px 18px;border-radius:8px;"
            f"font-size:0.95rem;line-height:1.6;white-space:pre-wrap;'>{transcript}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Transcript not available (N/A).")

    st.markdown("")
    st.markdown("**Text sentiment**")
    text_sentiment = st.radio(
        "Text sentiment",
        TEXT_SENTIMENTS,
        index=None,
        horizontal=True,
        key=f"txt_sent_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("**Emotion valence**")
    text_valence = st.slider(
        "Text valence",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.1,
        key=f"txt_val_{sample_id}",
        label_visibility="collapsed",
    )

    st.markdown("**Notable linguistic cues** (optional)")
    text_notes = st.text_area(
        "Linguistic cues",
        placeholder="e.g. uses sarcasm, repeated negations, abrupt topic shift…",
        key=f"txt_notes_{sample_id}",
        label_visibility="collapsed",
        height=80,
    )

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="step3_back"):
            st.session_state.current_step = 2
            st.rerun()
    with col_next:
        if st.button("Confirm Text Annotation →", type="primary", key="step3_confirm"):
            if not text_sentiment:
                st.error("Please select a text sentiment before continuing.")
            else:
                st.session_state.annotations["text_sentiment_human"] = text_sentiment
                st.session_state.annotations["text_valence_human"]   = text_valence
                st.session_state.annotations["text_notes_human"]     = text_notes
                st.session_state.current_step = 4
                st.rerun()


def _render_step4(sample_id: str):
    st.markdown("### Step 4 — Review: Full Video + Dataset Annotations")
    st.caption("Watch the video with audio, then expand the dataset annotations below.")

    video_url = _file_url(sample_id, f"{sample_id}.mp4")
    components.html(
        f"""
        <div style="display:flex;justify-content:center;">
          <video controls
            style="max-width:280px;width:100%;border-radius:8px;background:#000;">
            <source src="{video_url}" type="video/mp4">
            <p>Your browser does not support HTML5 video.
               <a href="{video_url}">Download video</a>.</p>
          </video>
        </div>
        """,
        height=520,
    )

    transcript = fetch_transcript(sample_id)
    if transcript:
        with st.expander("📄 Transcript", expanded=False):
            st.text(transcript)

    with st.expander("📂 Dataset Annotations", expanded=False):
        ann_list = fetch_annotations_jsonl(sample_id)
        if ann_list:
            typed = _parse_annotation_types(ann_list)
            labels = {
                "1P-IS": "1P Self-Report — Internal State",
                "1P-R":  "1P Self-Report — Rationale",
                "3P-IS": "3P Third-Party — Internal State",
                "3P-R":  "3P Third-Party — Rationale",
                "3P-V":  "3P Third-Party — Valence",
            }
            for key, label in labels.items():
                val = typed.get(key, "N/A")
                st.markdown(f"**{label}**")
                st.markdown(
                    f"<div style='background:#f8f9fa;padding:8px 12px;"
                    f"border-radius:6px;margin-bottom:8px;'>{val}</div>",
                    unsafe_allow_html=True,
                )
            if not typed:
                st.caption("Raw annotation data:")
                for item in ann_list:
                    st.json(item)
        else:
            st.info("No annotation file found for this sample (N/A).")

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="step4_back"):
            st.session_state.current_step = 3
            st.rerun()
    with col_next:
        if st.button("Continue to Gemini Analysis →", type="primary", key="step4_next"):
            st.session_state.current_step = 5
            st.rerun()


def _render_step5(sample_id: str):
    st.markdown("### Step 5 — LLM Analysis (Gemini)")

    anns = st.session_state.annotations
    gr   = st.session_state.gemini_result

    # Show prior human annotations as a summary
    with st.expander("Your annotations so far", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Visual:** {anns.get('visual_emotion_human', '—')}")
            st.markdown(f"Valence: {anns.get('visual_valence_human', '—')}")
        with col2:
            st.markdown(f"**Audio:** {anns.get('audio_emotion_human', '—')}")
            st.markdown(f"Valence: {anns.get('audio_valence_human', '—')}")
        with col3:
            st.markdown(f"**Text:** {anns.get('text_sentiment_human', '—')}")
            st.markdown(f"Valence: {anns.get('text_valence_human', '—')}")

    st.markdown("")

    if gr is None:
        # ── Gemini API key test ───────────────────────────────────
        if st.button("🔑 Test Gemini API Key", key="test_gemini_key"):
            api_key = _get_gemini_api_key()
            if not api_key:
                st.error("No API key found. Add GOOGLE_API_KEY to env or .streamlit/secrets.toml under [gemini].")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    m = genai.GenerativeModel("gemini-1.5-flash")
                    resp = m.generate_content("Reply with the single word: OK")
                    st.success(f"Gemini reachable. Response: `{resp.text.strip()}`")
                except Exception as e:
                    st.error(f"Gemini test failed: {e}")
                    with st.expander("🐛 Full traceback"):
                        st.code(traceback.format_exc())

        st.markdown("")
        # Show run / retry button
        err = st.session_state.gemini_error
        if err:
            st.error(f"Previous attempt failed: {err}")
        transcript = fetch_transcript(sample_id)
        btn_label = "🔄 Retry Gemini Analysis" if err else "▶ Run Gemini Analysis"
        if st.button(btn_label, type="primary", key="run_gemini"):
            with st.spinner("Calling Gemini API…"):
                result = _run_gemini(sample_id, transcript)
            if result:
                st.session_state.gemini_result = result
                st.session_state.gemini_error  = None
            else:
                st.session_state.gemini_error = "Analysis returned no result."
            st.rerun()
    else:
        # Display cached results
        st.success("Gemini analysis complete.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Visual (Gemini)**")
            st.markdown(f"`{gr.get('video', {}).get('emotion', '—')}`")
            st.caption(gr.get("video", {}).get("description", ""))
        with col2:
            st.markdown("**Audio (Gemini)**")
            st.markdown(f"`{gr.get('audio', {}).get('emotion', '—')}`")
            st.caption(gr.get("audio", {}).get("description", ""))
        with col3:
            st.markdown("**Text (Gemini)**")
            st.markdown(f"`{gr.get('text', {}).get('emotion', '—')}`")
            st.caption(gr.get("text", {}).get("description", ""))

        conflict = gr.get("conflict", {})
        if conflict.get("detected"):
            st.warning(f"⚠️ **Conflict detected:** {conflict.get('description', '')}")
        else:
            st.info(f"✅ **No conflict:** {conflict.get('description', '')}")

        if st.button("🔄 Re-run Gemini Analysis", key="rerun_gemini"):
            st.session_state.gemini_result = None
            st.session_state.gemini_error  = None
            st.rerun()

    st.markdown("")
    if st.button("← Back", key="step5_back"):
        st.session_state.current_step = 4
        st.rerun()


# ============================================================
# SECTION 9: MAIN LAYOUT
# ============================================================

def main():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.title("🔍 XMER Data Cleaning")
        st.divider()

        all_samples = discover_samples()
        filtered = all_samples

        unprocessed = [s for s in filtered if s not in st.session_state.processed_ids]
        total_batches = max(1, -(-len(unprocessed) // BATCH_SIZE))  # ceiling division

        sel_batch = st.number_input(
            "Batch",
            min_value=1,
            max_value=total_batches,
            value=min(st.session_state.selected_batch, total_batches),
            step=1,
        )
        st.session_state.selected_batch = sel_batch

        start = (sel_batch - 1) * BATCH_SIZE
        batch_samples = unprocessed[start : start + BATCH_SIZE]

        st.caption(
            f"{len(unprocessed)} unprocessed · {len(batch_samples)} in this batch"
        )

        if not HF_AVAILABLE:
            st.warning("`huggingface_hub` not installed.\nRun: `pip install huggingface_hub`")

        if st.button("▶ Load Next Sample", type="primary", disabled=not batch_samples):
            _load_next_sample(batch_samples)

        st.divider()
        st.caption(f"Processed this session: **{len(st.session_state.processed_ids)}**")

    # ── No sample loaded ─────────────────────────────────────
    sample = st.session_state.current_sample
    if sample is None:
        st.title("XMER Multimodal Data Cleaning")
        st.info(
            "Select a dataset split and batch in the sidebar, "
            "then click **▶ Load Next Sample** to begin."
        )
        return

    sample_id = sample["sample_id"]
    step      = st.session_state.current_step
    anns      = st.session_state.annotations
    gr        = st.session_state.gemini_result

    # Save is enabled only when all 3 modality annotations exist + Gemini result available
    save_enabled = (
        bool(anns.get("visual_emotion_human"))
        and bool(anns.get("audio_emotion_human"))
        and bool(anns.get("text_sentiment_human"))
        and gr is not None
    )

    # ── Top bar ───────────────────────────────────────────────
    tc1, tc2, tc3, tc4 = st.columns([4, 1, 1, 1])
    with tc1:
        processed_n = len(st.session_state.processed_ids)
        queue_n     = len(st.session_state.sample_queue)
        st.markdown(
            f"**`{sample_id}`** &nbsp;|&nbsp; Step {step}/5 &nbsp;|&nbsp; "
            f"Processed: {processed_n} &nbsp;|&nbsp; Queue: {queue_n}",
            unsafe_allow_html=True,
        )
    with tc2:
        pass  # spacer
    with tc3:
        if st.button("🗑️ Discard", use_container_width=True):
            st.session_state.discard_confirm = True
    with tc4:
        if st.button(
            "💾 Save",
            type="primary",
            use_container_width=True,
            disabled=not save_enabled,
        ):
            _do_save(sample_id)

    # ── Discard confirmation ──────────────────────────────────
    if st.session_state.discard_confirm:
        st.warning("⚠️ Are you sure you want to discard this sample?")
        reason = st.text_input("Reason (optional)", key="discard_reason_field")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("✓ Confirm Discard", type="primary"):
                _do_discard(sample_id, step, reason)
        with dc2:
            if st.button("✗ Cancel"):
                st.session_state.discard_confirm = False
                st.rerun()
        # Render nothing else while the dialog is open
        return

    st.divider()
    _step_indicator(step)

    if step == 1:
        _render_step1(sample_id)
    elif step == 2:
        _render_step2(sample_id)
    elif step == 3:
        _render_step3(sample_id)
    elif step == 4:
        _render_step4(sample_id)
    elif step == 5:
        _render_step5(sample_id)


main()
