import streamlit as st
import json
import os
from datetime import datetime

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

st.set_page_config(page_title="XMER CPM Annotation", page_icon="🎬", layout="wide")

# ============================================================
# SECTION 1: 配置
# ============================================================

# 1a. Hugging Face 视频托管
#   上传结构：CPM/{DATASET}/{DATASET}_{video_id}_{clip_id}.mp4
HF_DATASET_URL = "https://huggingface.co/datasets/YoeYang/XMER-Videos/resolve/main"

# 数据集名规范化（与 prepare_data_CPM.py / stage_videos_CPM.py 保持一致）
DATASET_DISPLAY = {
    "ch-sims": "CH-SIMS",
    "meld":    "MELD",
    "iemocap": "IEMOCAP",
    "mosi":    "MOSI",
}

# 1b. CPM 四维打分协议（Scherer CPM，范围 -2 ~ +2）
CPM_DIMS = [
    {
        "key": "R",
        "name": "R — Relevance 相关性",
        "desc": "即时的感官与生理registration（在任何推理之前）。"
                "正：感官愉悦、身体舒适、被刺激吸引；负：身体不适、疼痛、厌恶、本能排斥。",
    },
    {
        "key": "I",
        "name": "I — Implication 蕴含",
        "desc": "这件事对当事人目标与关切意味着什么。"
                "正：利于目标、机会、解脱、想要的结果；负：阻碍目标、损失、威胁、不想要的后果。",
    },
    {
        "key": "C",
        "name": "C — Coping Potential 应对潜能",
        "desc": "当事人表达出的应对能力感。"
                "正：自信、掌控、有能力、积极投入；负：不堪重负、无力、退缩、被动。",
    },
    {
        "key": "N",
        "name": "N — Normative Significance 规范意义",
        "desc": "社会/自我规范在此刻驱动的表达方向（不是判断内在情绪是否被压抑）。"
                "正：规范驱动的开放、温暖、得体、亲和；负：规范违背带来的抑制、含糊、羞愧、回避。",
    },
]
CPM_MIN, CPM_MAX = -2, 2

# 1c. 质量评价选项
QUALITY_OPTIONS = ["好", "坏"]

# 1d. 整体情绪选项
SENTIMENT_OPTIONS = ["positive", "neutral", "negative"]


# ============================================================
# SECTION 2: 数据加载
# ============================================================

@st.cache_data
def load_samples():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "samples_CPM.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["samples"]


def get_video_url(sample: dict) -> str:
    ds_disp = DATASET_DISPLAY.get(sample["dataset"], sample["dataset"])
    return f"{HF_DATASET_URL}/CPM/{ds_disp}/{sample['video_file']}"


def signals_to_lines(raw: str) -> str:
    """把 ' | ' 分隔的信号转成逐行文本，便于编辑。"""
    if not raw:
        return ""
    return "\n".join(s.strip() for s in str(raw).split("|") if s.strip())


def render_bullets(raw: str):
    """只读展示：把 ' | ' 分隔的信号渲染为项目符号列表。"""
    if not raw:
        st.caption("（无）")
        return
    for s in str(raw).split("|"):
        s = s.strip()
        if s:
            st.markdown(f"- {s}")


# ============================================================
# SECTION 3: Google Sheets
# ============================================================

SHEET_HEADERS = [
    "timestamp", "annotator_id", "dataset", "sample_id", "video_id", "clip_id",
    "video_desc", "audio_desc", "text_desc",
    "R", "I", "C", "N",
    "overall_sentiment", "quality", "notes",
]


@st.cache_resource
def get_worksheet():
    if not GSHEETS_AVAILABLE:
        return None
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"],
        )
        client = gspread.authorize(creds)
        ws = client.open_by_key(st.secrets["google_sheets"]["sheet_id"]).sheet1
        if ws.row_count == 0 or ws.cell(1, 1).value != "timestamp":
            ws.insert_row(SHEET_HEADERS, 1)
        return ws
    except Exception:
        return None


def save_to_gsheet(sample: dict, answers: dict, annotator_id: str) -> bool:
    ws = get_worksheet()
    if ws is None:
        return False
    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        annotator_id,
        sample.get("dataset", ""),
        sample.get("sample_id", ""),
        sample.get("video_id", ""),
        sample.get("clip_id", ""),
        answers.get("video_desc", ""),
        answers.get("audio_desc", ""),
        answers.get("text_desc", ""),
        answers.get("R", ""),
        answers.get("I", ""),
        answers.get("C", ""),
        answers.get("N", ""),
        answers.get("overall_sentiment", ""),
        answers.get("quality", ""),
        answers.get("notes", ""),
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")
    return True


def load_progress_from_gsheet(annotator_id: str) -> dict:
    """读取该标注者历史记录，同一 sample_id 取最后一条。返回 {sample_id: answers}。"""
    ws = get_worksheet()
    if ws is None:
        return {}
    try:
        records = ws.get_all_records()
        result = {}
        for r in records:
            if r.get("annotator_id") != annotator_id or not r.get("sample_id"):
                continue
            sid = r["sample_id"]
            answers = {
                "video_desc":        r.get("video_desc", ""),
                "audio_desc":        r.get("audio_desc", ""),
                "text_desc":         r.get("text_desc", ""),
                "overall_sentiment": r.get("overall_sentiment") or None,
                "quality":           r.get("quality") or None,
                "notes":             r.get("notes", ""),
            }
            for dim in ("R", "I", "C", "N"):
                v = r.get(dim, "")
                if v != "":
                    try:
                        answers[dim] = int(v)
                    except (ValueError, TypeError):
                        pass
            result[sid] = answers
        return result
    except Exception:
        return {}


# ============================================================
# SECTION 4: Session state
# ============================================================

def init_state():
    defaults = {
        "page":              "instructions",
        "annotator_id":      "",
        "current_idx":       0,
        "local_annotations": {},
        "resume_needed":     False,
        "gsheet_loaded":     False,
        "displayed_sid":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# SECTION 5: 须知页面
# ============================================================

def show_instructions():
    st.title("XMER · CPM 标注系统")
    st.markdown("---")
    st.markdown("""
### 标注流程（每个样本三步）

**① 模态描述提取**　文本框已预填 Gemini 生成的 video / audio / text 信号，
请你**删改、精炼**成最终正确的模态描述（每个模态一个框）。

**② CPM 四维打分**　依据**情境卡片 + 上下文**，从当事人视角对四个维度打分
（范围 −2 ~ +2）。滑块已用 LLM 预打分初始化，你可以调整。

**③ 整体情绪判断**　综合视频、音频、文本，判断该片段整体情绪倾向：positive / neutral / negative，三选一。

**④ 质量评价**　对该样本整体质量给出「好 / 坏」判断。

> 顶部展示视频、情境卡片、上下文；冲突参考信息（is_conflict / reasoning 等）
> 折叠在「参考信息」里，仅供参考。
""")
    st.subheader("输入你的标注者 ID 后开始")
    name = st.text_input("姓名或 ID", value=st.session_state.annotator_id,
                         placeholder="例如：Yoe")
    if st.button("开始标注 →", type="primary", disabled=not name.strip()):
        st.session_state.annotator_id = name.strip()
        st.session_state.page = "annotation"
        st.session_state.current_idx = 0
        st.session_state.resume_needed = True
        st.session_state.gsheet_loaded = False   # 触发 annotation 页自动同步
        st.session_state.displayed_sid = None
        st.rerun()


# ============================================================
# SECTION 6: 标注主页面
# ============================================================

def seed_widgets(sid, sample, existing, force=False):
    """把已保存答案 / 预填默认值写入 widget session_state。
    force=True 时强制覆盖已有值（用于 gsheets 回溯同步）。"""
    def put(wk, val):
        if force or wk not in st.session_state:
            st.session_state[wk] = val

    put(f"video_desc_{sid}", existing.get("video_desc") or signals_to_lines(sample["video_signals"]))
    put(f"audio_desc_{sid}", existing.get("audio_desc") or signals_to_lines(sample["audio_signals"]))
    put(f"text_desc_{sid}",  existing.get("text_desc")  or signals_to_lines(sample["text_signals"]))
    for dim in CPM_DIMS:
        k = dim["key"]
        put(f"{k}_{sid}", int(existing.get(k, sample[f"{k}_valence"])))
    if existing.get("overall_sentiment"):
        put(f"overall_sentiment_{sid}", existing["overall_sentiment"])
    elif force:
        st.session_state.pop(f"overall_sentiment_{sid}", None)
    if existing.get("quality"):
        put(f"quality_{sid}", existing["quality"])
    elif force:
        st.session_state.pop(f"quality_{sid}", None)
    put(f"notes_{sid}", existing.get("notes", ""))


def collect_answers(sid, sample):
    a = {
        "video_desc":        st.session_state.get(f"video_desc_{sid}", ""),
        "audio_desc":        st.session_state.get(f"audio_desc_{sid}", ""),
        "text_desc":         st.session_state.get(f"text_desc_{sid}", ""),
        "overall_sentiment": st.session_state.get(f"overall_sentiment_{sid}"),
        "quality":           st.session_state.get(f"quality_{sid}"),
        "notes":             st.session_state.get(f"notes_{sid}", ""),
    }
    for dim in CPM_DIMS:
        k = dim["key"]
        a[k] = st.session_state.get(f"{k}_{sid}", sample[f"{k}_valence"])
    return a


def validate(answers):
    missing = []
    if not answers.get("video_desc", "").strip():
        missing.append("① 视频模态描述")
    if not answers.get("audio_desc", "").strip():
        missing.append("① 音频模态描述")
    if not answers.get("text_desc", "").strip():
        missing.append("① 文本模态描述")
    if not answers.get("overall_sentiment"):
        missing.append("③ 整体情绪判断")
    if not answers.get("quality"):
        missing.append("④ 质量评价")
    return missing


def show_annotation(samples):
    total = len(samples)

    # ── 自动从 gsheets 同步历史（每次会话首次进入标注页触发）──
    if not st.session_state.gsheet_loaded and st.session_state.annotator_id:
        with st.spinner("正在从 Google Sheets 同步历史标注..."):
            history = load_progress_from_gsheet(st.session_state.annotator_id)
        st.session_state.gsheet_loaded = True
        if history:
            st.session_state.local_annotations.update(history)
        st.session_state.displayed_sid = None   # 同步后强制重新回填当前样本

    # 恢复进度 → 跳到第一条未标注
    if st.session_state.get("resume_needed"):
        st.session_state.resume_needed = False
        done = st.session_state.local_annotations
        for i, s in enumerate(samples):
            if s["sample_id"] not in done:
                st.session_state.current_idx = i
                break
        else:
            st.session_state.current_idx = total - 1

    annotated = sum(1 for s in samples if s["sample_id"] in st.session_state.local_annotations)

    sample = samples[st.session_state.current_idx]
    sid = sample["sample_id"]
    existing = st.session_state.local_annotations.get(sid, {})
    # 仅在切换到新样本时回填一次；同一样本后续 rerun 不再覆盖，保护用户的编辑
    if st.session_state.get("displayed_sid") != sid:
        seed_widgets(sid, sample, existing, force=bool(existing))
        st.session_state.displayed_sid = sid

    # ── 侧栏 ──
    with st.sidebar:
        st.markdown(f"👤 标注者：**{st.session_state.annotator_id}**")
        st.caption(f"进度：{annotated} / {total}")
        st.divider()
        # 快速跳转
        jump = st.number_input("跳转到第几条", min_value=1, max_value=total,
                               value=st.session_state.current_idx + 1, step=1)
        if st.button("跳转"):
            st.session_state.local_annotations[sid] = collect_answers(sid, sample)
            st.session_state.current_idx = int(jump) - 1
            st.rerun()
        st.divider()
        if st.button("← 返回须知页"):
            st.session_state.local_annotations[sid] = collect_answers(sid, sample)
            st.session_state.page = "instructions"
            st.rerun()

    # ── 顶部进度 + 导航 ──
    st.title("XMER · CPM 标注")
    st.progress(annotated / total if total else 0,
                text=f"进度：{annotated} / {total} 已完成")

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← 上一条", disabled=st.session_state.current_idx == 0):
            st.session_state.local_annotations[sid] = collect_answers(sid, sample)
            st.session_state.current_idx -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>"
            f"[{sample['dataset']}] <code>{sid}</code> · 第 "
            f"<b>{st.session_state.current_idx + 1}</b> / {total} 条</div>",
            unsafe_allow_html=True)
    with col_next:
        if st.button("下一条 →", disabled=st.session_state.current_idx == total - 1):
            st.session_state.local_annotations[sid] = collect_answers(sid, sample)
            st.session_state.current_idx += 1
            st.rerun()

    # ── 顶部展示：视频 + 情境卡片/上下文 ──
    col_video, col_card = st.columns([1, 1], gap="large")
    with col_video:
        st.video(get_video_url(sample))
    with col_card:
        st.subheader("情境卡片")
        for label, field in [("Subject 主体", "Subject"),
                             ("Stance 处境", "Stance"),
                             ("Power / Interest 权力·利益", "Power_Interest")]:
            val = sample.get(field, "")
            st.markdown(f"**{label}**")
            st.markdown(
                f"<div style='background:#f0f2f6;padding:8px 12px;border-radius:6px;"
                f"margin-bottom:8px;'>{val if val else '（无）'}</div>",
                unsafe_allow_html=True)
        st.markdown("**Context 上下文**")
        ctx = sample.get("context", "")
        if ctx:
            ctx_html = "<br>".join(s.strip() for s in ctx.split("|") if s.strip())
            st.markdown(
                f"<div style='background:#fff8e6;padding:8px 12px;border-radius:6px;"
                f"max-height:220px;overflow-y:auto;font-size:0.9rem;'>{ctx_html}</div>",
                unsafe_allow_html=True)
        else:
            st.caption("（该数据集无上下文 / CH-SIMS）")

    # ── 参考信息（折叠）──
    with st.expander("📎 参考信息：冲突判断 & 原始模态信号 & CPM 预打分理由", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**is_conflict**：{sample.get('is_conflict','')}　"
                        f"**confidence**：{sample.get('confidence','')}")
            st.markdown(f"**reasoning**：{sample.get('reasoning','')}")
            st.markdown(f"**mechanism**：{sample.get('mechanism','')}")
            st.markdown(f"**Predicted_polarity**：{sample.get('Predicted_polarity','')}")
            st.markdown(f"**Justification**：{sample.get('Justification','')}")
        with c2:
            st.markdown("**原始 video_signals**")
            render_bullets(sample.get("video_signals", ""))
            st.markdown("**原始 audio_signals**")
            render_bullets(sample.get("audio_signals", ""))
            st.markdown("**原始 text_signals**")
            render_bullets(sample.get("text_signals", ""))

    st.divider()

    # ── 阶段1：模态描述提取 ──
    st.markdown("### ① 模态描述提取")
    st.caption("文本框已预填生成的信号（每行一条），请删改 / 精炼为最终正确的模态描述。")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**🎞 视频模态描述**")
        st.text_area("video_desc", key=f"video_desc_{sid}", height=180,
                     label_visibility="collapsed")
    with d2:
        st.markdown("**🔊 音频模态描述**")
        st.text_area("audio_desc", key=f"audio_desc_{sid}", height=180,
                     label_visibility="collapsed")
    with d3:
        st.markdown("**💬 文本模态描述**")
        st.text_area("text_desc", key=f"text_desc_{sid}", height=180,
                     label_visibility="collapsed")

    st.divider()

    # ── 阶段2：CPM 四维打分 ──
    st.markdown("### ② CPM 四维打分（−2 ~ +2）")
    st.caption("依据情境卡片 + 上下文，从当事人视角评估每个维度表达的方向。滑块已用 LLM 预打分初始化。")
    for dim in CPM_DIMS:
        k = dim["key"]
        st.markdown(f"**{dim['name']}**")
        st.markdown(
            f"<p style='color:#888;font-size:0.85rem;margin:-4px 0 2px 0;'>{dim['desc']}　"
            f"<span style='color:#1f77b4;'>LLM 预打分：{sample[f'{k}_valence']:+d}</span></p>",
            unsafe_allow_html=True)
        st.slider(k, min_value=CPM_MIN, max_value=CPM_MAX, step=1,
                  key=f"{k}_{sid}", label_visibility="collapsed")

    st.divider()

    # ── 阶段3：整体情绪判断 ──
    st.markdown("### ③ 整体情绪判断")
    st.caption("综合视频、音频、文本，判断该片段的整体情绪倾向。")
    st.radio("整体情绪", SENTIMENT_OPTIONS, index=None, horizontal=True,
             key=f"overall_sentiment_{sid}", label_visibility="collapsed")

    st.divider()

    # ── 阶段4：质量评价 ──
    st.markdown("### ④ 样本质量评价")
    st.radio("该样本整体质量", QUALITY_OPTIONS, index=None, horizontal=True,
             key=f"quality_{sid}", label_visibility="collapsed")

    st.text_area("备注（可选）", key=f"notes_{sid}", height=70,
                 placeholder="如有特殊说明请填写")

    # ── 保存 ──
    st.markdown("")
    answers = collect_answers(sid, sample)
    missing = validate(answers)
    if st.button("保存并继续 ✓", type="primary", key=f"save_{sid}"):
        if missing:
            st.error("请完成以下必填项后再保存：\n- " + "\n- ".join(missing))
        else:
            st.session_state.local_annotations[sid] = answers
            ok = save_to_gsheet(sample, answers, st.session_state.annotator_id)
            if ok:
                st.success("已保存到 Google Sheets ✓")
            else:
                st.warning("Google Sheets 未连接，仅保存在本地（关闭页面后丢失）")
            if st.session_state.current_idx < total - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.success("🎉 全部样本已标注完成！感谢你的参与。")

    if sid in st.session_state.local_annotations:
        st.caption("该样本已标注，可修改后重新保存。")


# ============================================================
# SECTION 7: 路由
# ============================================================

def main():
    samples = load_samples()
    if st.session_state.page == "instructions":
        show_instructions()
    else:
        show_annotation(samples)


main()
