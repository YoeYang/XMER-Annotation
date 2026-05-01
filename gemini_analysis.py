import os
import time
import random
import argparse
from pathlib import Path
from typing import Optional, Callable, TypeVar, Dict, List, Tuple

import pandas as pd
import google.generativeai as genai

T = TypeVar("T")

# =========================
# Rate limit + retry knobs
# =========================
UPLOAD_MIN_INTERVAL_S = 3.0
SEND_MIN_INTERVAL_S   = 4.0
FILE_POLL_INTERVAL_S  = 10.0

MAX_RETRIES  = 6
BASE_SLEEP_S = 2.0
MAX_SLEEP_S  = 60.0
COOLDOWN_ON_429_S = 8 * 60

# =========================
# Prompts
# Each prompt asks Gemini to output:
#   <emotion justification (1-2 words)> | <signal description (1-2 sentences)>
# =========================
VIDEO_SIGNALS_PROMPT = """
Task: Analyze the silent video clip and return exactly two fields separated by " | ":
1. Emotion justification — 1 to 2 words naming the primary emotion visible in the clip.
2. Signal description — 1 to 2 sentences explaining which visual cues (facial expression, gaze, head pose, posture, body motion) support that emotion label.

Use visual cues ONLY. Ignore audio, speech, and text completely.

OUTPUT FORMAT (strict — one line, nothing else):
<emotion label> | <signal description>

Example:
Anxious | The speaker's rapid blinking and tightly clasped hands suggest heightened anxiety, while an averted gaze reinforces visible discomfort.
"""

AUDIO_SIGNALS_PROMPT = """
Task: Analyze the audio clip using vocal prosody only and return exactly two fields separated by " | ":
1. Emotion justification — 1 to 2 words naming the primary emotion conveyed by prosodic cues.
2. Signal description — 1 to 2 sentences explaining which prosodic signals (pitch/F0, intensity, rhythm, pauses, voice quality) support that emotion label.

Do NOT transcribe or mention any spoken words or content.

OUTPUT FORMAT (strict — one line, nothing else):
<emotion label> | <signal description>

Example:
Sad | The speaker's low, flat pitch punctuated by long pauses signals emotional withdrawal, and the softened intensity throughout further reinforces a subdued, sorrowful tone.
"""

TEXT_SIGNALS_PROMPT_TEMPLATE = """
Task: Analyze the transcript text below and return exactly two fields separated by " | ":
1. Emotion justification — 1 to 2 words naming the primary emotion expressed in the text.
2. Signal description — 1 to 2 sentences referencing specific phrases or expressions in the transcript that support that emotion label.

OUTPUT FORMAT (strict — one line, nothing else):
<emotion label> | <signal description>

Example:
Frustrated | The repeated use of "I can't believe this" and "nothing ever works" reveals frustration, while the rhetorical question "What's even the point?" signals underlying helplessness.

Transcript:
{transcript}
"""

# =========================
# Helpers
# =========================
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def ensure_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def is_missing_cell(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() == "nan" or s.upper() == "NA"

def is_likely_quota_error(msg: str) -> bool:
    m = msg.lower()
    return ("429" in m) or ("resource has been exhausted" in m) or ("quota" in m) or ("rate limit" in m)

def retry_call(
    fn: Callable[[], T],
    *,
    what: str = "call",
    max_retries: int = MAX_RETRIES,
    base_sleep: float = BASE_SLEEP_S,
    max_sleep: float = MAX_SLEEP_S,
) -> T:
    retriable = [
        "429", "resource has been exhausted", "rate limit", "quota",
        "500", "503", "unavailable", "internal",
        "timeout", "timed out", "temporary failure", "connection reset",
    ]
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)
            ok = any(s in msg.lower() for s in retriable)
            if (not ok) or (attempt == max_retries):
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** attempt)) + random.uniform(0, 1.0)
            print(f"[RETRY] {what} failed: {msg[:180]}... sleep {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_s)

    raise last_err if last_err else RuntimeError(f"{what} failed (unknown)")

class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._next_time = 0.0

    def wait(self):
        now = time.time()
        if now < self._next_time:
            time.sleep(self._next_time - now)
        self._next_time = time.time() + self.min_interval

UPLOAD_LIMITER = RateLimiter(UPLOAD_MIN_INTERVAL_S)
SEND_LIMITER   = RateLimiter(SEND_MIN_INTERVAL_S)

def wait_for_file_active(file_obj, timeout: int = 900, poll_interval: float = FILE_POLL_INTERVAL_S, status_cb=None):
    start = time.time()
    name = getattr(file_obj, "name", None) or getattr(getattr(file_obj, "file", file_obj), "name", None)

    while True:
        state = getattr(file_obj, "state", None)
        if state is not None and "ACTIVE" in str(state):
            return file_obj
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            raise TimeoutError(f"File {name} not ACTIVE after {timeout}s, last state={state}")
        if status_cb:
            status_cb(f"⏳ Waiting for Google to process file… {elapsed}s elapsed (state: {state})")
        time.sleep(poll_interval)
        file_obj = retry_call(lambda: genai.get_file(name), what=f"get_file({name})")

def build_gemini_model(sys_prompt: str, model_name: str, temperature: float, top_p: float, max_output_tokens: int):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_mime_type="text/plain",
    )

    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=sys_prompt,
        generation_config=generation_config,
    )

# =========================
# Output parsing & validation
# =========================
def parse_output(raw: str) -> Tuple[str, str]:
    """
    Parse Gemini output: '<emotion label> | <signal description>'
    Returns (emotion_justification, signal_description).
    """
    raw = raw.strip()
    if "|" in raw:
        parts = raw.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return "", raw

def normalize_raw(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def is_valid_output(raw: str) -> bool:
    raw = normalize_raw(raw)
    if not raw:
        return False
    low = raw.lower()
    if low in {"na", "nan", "none", "null"}:
        return False
    if "output format" in low or "output rules" in low:
        return False
    if "|" not in raw:
        return False

    emotion, description = parse_output(raw)
    if not emotion or not description:
        return False
    # emotion should be short (1-3 words; allow slight leeway)
    if len(emotion.split()) > 4:
        return False
    return True

# =========================
# Core generators
# =========================
def gen_video_signals(model: genai.GenerativeModel, clip_dir: Path) -> str:
    video_path = clip_dir / "video.mp4"
    ensure_file(video_path, "video.mp4")

    chat = model.start_chat(history=[])
    UPLOAD_LIMITER.wait()
    video_file = retry_call(lambda: genai.upload_file(path=str(video_path)), what=f"upload video {clip_dir}")
    video_file = wait_for_file_active(video_file)

    SEND_LIMITER.wait()
    resp = retry_call(lambda: chat.send_message([video_file, VIDEO_SIGNALS_PROMPT]), what=f"send video {clip_dir}")
    return normalize_raw((resp.text or "").strip())

def gen_audio_signals(model: genai.GenerativeModel, clip_dir: Path) -> str:
    audio_path = clip_dir / "audio.wav"
    ensure_file(audio_path, "audio.wav")

    chat = model.start_chat(history=[])
    UPLOAD_LIMITER.wait()
    audio_file = retry_call(lambda: genai.upload_file(path=str(audio_path)), what=f"upload audio {clip_dir}")
    audio_file = wait_for_file_active(audio_file)

    SEND_LIMITER.wait()
    resp = retry_call(lambda: chat.send_message([audio_file, AUDIO_SIGNALS_PROMPT]), what=f"send audio {clip_dir}")
    return normalize_raw((resp.text or "").strip())

def gen_text_signals(model: genai.GenerativeModel, clip_dir: Path) -> str:
    text_path = clip_dir / "text.txt"
    ensure_file(text_path, "text.txt")

    transcript = load_text(text_path)
    prompt = TEXT_SIGNALS_PROMPT_TEMPLATE.format(transcript=transcript)

    chat = model.start_chat(history=[])
    SEND_LIMITER.wait()
    resp = retry_call(lambda: chat.send_message(prompt), what=f"send text {clip_dir}")
    return normalize_raw((resp.text or "").strip())

# =========================
# Scan data_root -> rows
# =========================
def scan_data_root(data_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Expect: data_root/video_id/clip_id/(video.mp4, audio.wav, text.txt)
    Return list of (video_id, clip_id, clip_dir)
    """
    rows: List[Tuple[str, str, Path]] = []
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    for video_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        video_id = video_dir.name
        for clip_dir in sorted([p for p in video_dir.iterdir() if p.is_dir()]):
            clip_id = clip_dir.name
            rows.append((video_id, clip_id, clip_dir))
    return rows

# CSV columns: for each modality, store raw output + parsed emotion + parsed description
MODALITY_COLS = {
    "video": ("video_raw", "video_emotion", "video_signals"),
    "audio": ("audio_raw", "audio_emotion", "audio_signals"),
    "text":  ("text_raw",  "text_emotion",  "text_signals"),
}
ALL_EXTRA_COLS = ["status", "error"]

def load_or_init_df(out_csv: Path, scanned_rows: List[Tuple[str, str, Path]]) -> pd.DataFrame:
    base = pd.DataFrame([{"video_id": vid, "clip_id": cid} for vid, cid, _ in scanned_rows])
    if out_csv.exists():
        old = pd.read_csv(out_csv, dtype={"video_id": str, "clip_id": str})
        df = base.merge(old, on=["video_id", "clip_id"], how="left", suffixes=("", "_old"))
    else:
        df = base

    all_cols = (
        [c for tup in MODALITY_COLS.values() for c in tup]
        + ALL_EXTRA_COLS
    )
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA

    for c in list(df.columns):
        if c.endswith("_old"):
            df.drop(columns=[c], inplace=True)

    return df

def store_modality(df: pd.DataFrame, i: int, modality: str, raw: str) -> None:
    raw_col, emotion_col, signals_col = MODALITY_COLS[modality]
    emotion, description = parse_output(raw)
    df.at[i, raw_col]     = raw
    df.at[i, emotion_col] = emotion
    df.at[i, signals_col] = description

def needs_regen(df: pd.DataFrame, i: int, modality: str, retry_invalid: bool) -> bool:
    raw_col = MODALITY_COLS[modality][0]
    cur = df.at[i, raw_col] if raw_col in df.columns else pd.NA
    if is_missing_cell(cur):
        return True
    return retry_invalid and (not is_valid_output(str(cur)))

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Root: video_id/clip_id folders containing video.mp4, audio.wav, text.txt")
    ap.add_argument("--out_csv", type=Path, default=None,
                    help="Output CSV (default: data_root/gemini_signals.csv)")
    ap.add_argument("--sys_prompt", type=Path, required=True, help="System prompt file path")
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")

    ap.add_argument("--temperature",       type=float, default=0.4)
    ap.add_argument("--top_p",             type=float, default=0.9)
    ap.add_argument("--max_output_tokens", type=int,   default=512)

    ap.add_argument("--do_video", action="store_true", help="Generate video emotion + signals")
    ap.add_argument("--do_audio", action="store_true", help="Generate audio emotion + signals")
    ap.add_argument("--do_text",  action="store_true", help="Generate text emotion + signals")

    ap.add_argument("--save_every",        type=int,  default=20,
                    help="Flush CSV every N successful updates")
    ap.add_argument("--retry_invalid",     action="store_true",
                    help="Retry invalid existing outputs (otherwise only fill missing)")
    ap.add_argument("--max_regen_per_cell", type=int, default=3,
                    help="Max regen attempts per cell if invalid/empty")
    args = ap.parse_args()

    out_csv = args.out_csv or (args.data_root / "gemini_signals.csv")

    ensure_file(args.sys_prompt, "system prompt")
    sys_prompt = load_text(args.sys_prompt)

    model = build_gemini_model(
        sys_prompt=sys_prompt,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )

    scanned = scan_data_root(args.data_root)
    df = load_or_init_df(out_csv, scanned)

    run_modalities = []
    if args.do_video: run_modalities.append("video")
    if args.do_audio: run_modalities.append("audio")
    if args.do_text:  run_modalities.append("text")
    if not run_modalities:
        run_modalities = ["video", "audio", "text"]

    generators = {
        "video": gen_video_signals,
        "audio": gen_audio_signals,
        "text":  gen_text_signals,
    }

    clip_map: Dict[Tuple[str, str], Path] = {(vid, cid): cdir for vid, cid, cdir in scanned}

    success_updates = 0
    total_rows = len(df)

    for i, row in df.iterrows():
        video_id = str(row["video_id"])
        clip_id  = str(row["clip_id"])
        clip_dir = clip_map.get((video_id, clip_id))

        if clip_dir is None or (not clip_dir.exists()):
            df.at[i, "status"] = "missing_clip_dir"
            df.at[i, "error"]  = f"Missing clip dir: {args.data_root}/{video_id}/{clip_id}"
            continue

        try:
            for modality in run_modalities:
                if not needs_regen(df, i, modality, args.retry_invalid):
                    continue

                print(f"[RUN ] {video_id}/{clip_id} gen {modality}")
                raw = ""
                for k in range(args.max_regen_per_cell):
                    raw = generators[modality](model, clip_dir)
                    if is_valid_output(raw):
                        break
                    print(f"[WARN] invalid {modality} output (try {k+1}/{args.max_regen_per_cell}): {raw[:120]}")
                    time.sleep(1)

                store_modality(df, i, modality, raw)
                if is_valid_output(raw):
                    success_updates += 1

            # update status
            all_ok = all(
                is_valid_output(str(df.at[i, MODALITY_COLS[m][0]]))
                for m in run_modalities
                if not is_missing_cell(df.at[i, MODALITY_COLS[m][0]])
            )
            any_done = any(
                not is_missing_cell(df.at[i, MODALITY_COLS[m][0]])
                for m in run_modalities
            )
            if all_ok and any_done:
                df.at[i, "status"] = "ok"
                df.at[i, "error"]  = pd.NA
            else:
                df.at[i, "status"] = "partial_or_invalid"
                df.at[i, "error"]  = "Some modalities missing/invalid (use --retry_invalid to repair)"

        except Exception as e:
            msg = str(e)
            df.at[i, "status"] = "error"
            df.at[i, "error"]  = msg
            print(f"[ERROR] {video_id}/{clip_id}: {msg}")

            if is_likely_quota_error(msg):
                print(f"[COOLDOWN] quota/rate limit; sleeping {COOLDOWN_ON_429_S}s")
                time.sleep(COOLDOWN_ON_429_S)
            else:
                time.sleep(3)

        if success_updates > 0 and (success_updates % args.save_every == 0):
            df.to_csv(out_csv, index=False, na_rep="NA")
            print(f"[SAVE] wrote {out_csv} (success_updates={success_updates}/{total_rows})")

    df.to_csv(out_csv, index=False, na_rep="NA")
    print(f"[DONE] wrote {out_csv} | success_updates={success_updates} | rows={len(df)}")

if __name__ == "__main__":
    main()


# =========================
# Streamlit integration API
# (added to support app2.py; existing CLI main() is unchanged)
# =========================

CONFLICT_ASSESSMENT_PROMPT_TEMPLATE = """
Task: Given independent emotion analyses of three modalities from the same video clip, determine whether emotional conflict exists across modalities.

Visual emotion (silent video): {video_emotion}
Visual signals: {video_signals}

Audio emotion (prosody only): {audio_emotion}
Audio signals: {audio_signals}

Text emotion (transcript): {text_emotion}
Text signals: {text_signals}

Conflict means two or more modalities convey meaningfully different or contradictory emotions \
(e.g., smiling face but tense, sad-sounding voice; positive words but angry prosody).

Return exactly two fields separated by " | ":
1. Conflict verdict — "YES" if conflict exists across any two modalities, "NO" if all modalities align.
2. Assessment — 1-2 sentences naming which modalities conflict and the nature of the discrepancy, \
or confirming cross-modal consistency if no conflict.

OUTPUT FORMAT (strict — one line, nothing else):
<YES or NO> | <assessment>

Examples:
YES | The visual cues suggest calm contentment while the audio prosody signals suppressed anger, indicating the speaker may be masking their true emotional state.
NO | All three modalities consistently indicate a subdued, sorrowful state with no cross-modal discrepancy.
"""


def build_gemini_model_simple(
    api_key: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_output_tokens: int = 512,
) -> genai.GenerativeModel:
    """Build a Gemini model with an explicit API key (no env-var required)."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            response_mime_type="text/plain",
        ),
    )


def gen_video_signals_from_path(model: genai.GenerativeModel, video_path: str, status_cb=None) -> str:
    """Analyze visual emotion from an arbitrary video file path (silent video)."""
    vp = Path(video_path)
    ensure_file(vp, "video file")

    chat = model.start_chat(history=[])
    UPLOAD_LIMITER.wait()
    if status_cb:
        status_cb(f"📤 Uploading {vp.name} to Google Files API…")
    video_file = retry_call(
        lambda: genai.upload_file(path=str(vp)),
        what=f"upload video {vp.name}",
    )
    video_file = wait_for_file_active(video_file, status_cb=status_cb)

    SEND_LIMITER.wait()
    if status_cb:
        status_cb("📹 File ready — sending to Gemini for visual analysis…")
    resp = retry_call(
        lambda: chat.send_message([video_file, VIDEO_SIGNALS_PROMPT]),
        what=f"send video {vp.name}",
    )
    return normalize_raw((resp.text or "").strip())


def gen_audio_signals_from_path(model: genai.GenerativeModel, audio_path: str, status_cb=None) -> str:
    """Analyze prosodic emotion from an arbitrary audio file path."""
    ap = Path(audio_path)
    ensure_file(ap, "audio file")

    chat = model.start_chat(history=[])
    UPLOAD_LIMITER.wait()
    if status_cb:
        status_cb(f"📤 Uploading {ap.name} to Google Files API…")
    audio_file = retry_call(
        lambda: genai.upload_file(path=str(ap)),
        what=f"upload audio {ap.name}",
    )
    audio_file = wait_for_file_active(audio_file, status_cb=status_cb)

    SEND_LIMITER.wait()
    if status_cb:
        status_cb("🔊 File ready — sending to Gemini for audio analysis…")
    resp = retry_call(
        lambda: chat.send_message([audio_file, AUDIO_SIGNALS_PROMPT]),
        what=f"send audio {ap.name}",
    )
    return normalize_raw((resp.text or "").strip())


def gen_text_signals_from_text(model: genai.GenerativeModel, transcript_text: str) -> str:
    """Analyze text sentiment directly from a transcript string (no file needed)."""
    prompt = TEXT_SIGNALS_PROMPT_TEMPLATE.format(transcript=transcript_text)
    chat = model.start_chat(history=[])
    SEND_LIMITER.wait()
    resp = retry_call(
        lambda: chat.send_message(prompt),
        what="send text",
    )
    return normalize_raw((resp.text or "").strip())


def gen_conflict_assessment(
    model: genai.GenerativeModel,
    video_emotion: str,
    video_signals: str,
    audio_emotion: str,
    audio_signals: str,
    text_emotion: str,
    text_signals: str,
) -> str:
    """Synthesize per-modality results into a cross-modal conflict verdict."""
    prompt = CONFLICT_ASSESSMENT_PROMPT_TEMPLATE.format(
        video_emotion=video_emotion or "N/A",
        video_signals=video_signals or "N/A",
        audio_emotion=audio_emotion or "N/A",
        audio_signals=audio_signals or "N/A",
        text_emotion=text_emotion or "N/A",
        text_signals=text_signals or "N/A",
    )
    chat = model.start_chat(history=[])
    SEND_LIMITER.wait()
    resp = retry_call(
        lambda: chat.send_message(prompt),
        what="conflict_assessment",
    )
    return normalize_raw((resp.text or "").strip())


def parse_conflict_output(raw: str) -> Tuple[bool, str]:
    """Parse 'YES | description' or 'NO | description' into (detected: bool, description: str)."""
    raw = raw.strip()
    if "|" in raw:
        verdict, desc = raw.split("|", 1)
        detected = verdict.strip().upper().startswith("YES")
        return detected, desc.strip()
    upper = raw.upper()
    if upper.startswith("YES"):
        return True, raw[3:].strip(" :")
    if upper.startswith("NO"):
        return False, raw[2:].strip(" :")
    return False, raw
