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

st.set_page_config(page_title="XMER Annotation", page_icon="🎬", layout="wide")

# ============================================================
# SECTION 1: CONFIGURATION  ← 只需修改这里
# ============================================================

# 1a. Hugging Face 视频托管
# 在 HuggingFace 创建 dataset repo 后，把视频上传到对应子文件夹
# 视频 URL = HF_DATASET_URL + "/" + VIDEO_FOLDERS[dataset] + "/" + video_file
HF_DATASET_URL = "https://huggingface.co/datasets/YoeYang/XMER-Videos/resolve/main"

VIDEO_FOLDERS = {
    "CH-SIMS": "CH-SIMS/video",
    "MELD":    "MELD/clean",
}

# 1b. 情景卡片字段显示控制（True=默认显示，False=默认隐藏；侧栏可实时切换）
SITUATION_CARD_FIELDS = {
    "人物身份":       True,
    "触发事件":       True,
    "当前目标_利益":  True,
    "权力关系":       False,
    "标注提示":       True,
}

# 1c. 标注问题（按需修改选项和题目）
QUESTIONS = [
    {
        "key":     "audio_emotion",
        "label":   "Q1  仅凭说话人的 **声音/语调**，你感知到的情绪是？",
        "options": ["积极/高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"],
    },
    {
        "key":     "visual_emotion",
        "label":   "Q2  仅凭说话人的 **面部表情/肢体动作**，你感知到的情绪是？",
        "options": ["积极/高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"],
    },
    {
        "key":     "overall_emotion",
        "label":   "Q3  综合所有信息（声音+画面+语言内容），整体情绪判断是？",
        "options": ["积极/高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中性"],
    },
    {
        "key":     "modality_conflict",
        "label":   "Q4  声音与表情之间是否存在 **情绪冲突**？",
        "options": ["无冲突（一致）", "轻微冲突", "明显冲突"],
    },
    {
        "key":     "confidence",
        "label":   "Q5  你对以上判断的 **把握程度**？",
        "options": ["不确定", "较有把握", "非常确定"],
    },
]

# 1d. Batch 分组系统
# 格式: {"batch_name": [("dataset", "sample_id"), ...]}
# 空 dict → 不分组，所有样本在同一链接
# 访问方式: ?batch=batch_1  ?batch=batch_2
# 示例（取消注释并填入 sample_id 即可启用）:
# BATCHES = {
#     "batch_1": [
#         ("CH-SIMS", "aqgy3_0001_00016"),
#         ("CH-SIMS", "aqgy5_0008_00031"),
#         ("MELD",    "dia29_utt6"),
#     ],
#     "batch_2": [
#         ("CH-SIMS", "aqgy5_0015_00015"),
#         ("MELD",    "dia34_utt4"),
#     ],
# }
BATCHES = {}


# ============================================================
# SECTION 2: 数据加载
# ============================================================

@st.cache_data
def load_all_samples():
    base = os.path.dirname(os.path.abspath(__file__))
    samples = []

    ch_path = os.path.join(base, "CH-SIMS-emo-conflict", "situation_cards_CH-SIMS.json")
    if os.path.exists(ch_path):
        with open(ch_path, "r", encoding="utf-8") as f:
            for s in json.load(f)["samples"]:
                s["dataset"] = "CH-SIMS"
                samples.append(s)

    meld_path = os.path.join(base, "MELD_emo_conflict", "situation_cards_MELD.json")
    if os.path.exists(meld_path):
        with open(meld_path, "r", encoding="utf-8") as f:
            for s in json.load(f)["samples"]:
                s["dataset"] = "MELD"
                samples.append(s)

    return samples


def get_video_url(dataset: str, video_file: str) -> str:
    folder = VIDEO_FOLDERS.get(dataset, dataset)
    return f"{HF_DATASET_URL}/{folder}/{video_file}"


def filter_by_batch(all_samples, batch_name):
    if not BATCHES or not batch_name:
        return all_samples
    batch_ids = BATCHES.get(batch_name)
    if not batch_ids:
        st.error(f"Batch '{batch_name}' 不存在，可用: {list(BATCHES.keys())}")
        return all_samples
    id_set = {(ds, sid) for ds, sid in batch_ids}
    return [s for s in all_samples if (s["dataset"], s["sample_id"]) in id_set]


# ============================================================
# SECTION 3: Google Sheets
# ============================================================
# 配置说明（在 Streamlit Cloud → App settings → Secrets 中填入）:
#
# [gcp_service_account]
# type = "service_account"
# project_id = "your-project-id"
# private_key_id = "..."
# private_key = "-----BEGIN PRIVATE KEY-----\n..."
# client_email = "your-sa@project.iam.gserviceaccount.com"
# client_id = "..."
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
#
# [google_sheets]
# sheet_id = "your-google-sheet-id"
# ============================================================

SHEET_HEADERS = (
    ["timestamp", "annotator_id", "dataset", "sample_id",
     "video_file", "speaker", "utterance"]
    + [q["key"] for q in QUESTIONS]
    + ["notes"]
)


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
        sheet = client.open_by_key(st.secrets["google_sheets"]["sheet_id"])
        ws = sheet.sheet1
        # 如果表头行不存在则自动写入
        if ws.row_count == 0 or ws.cell(1, 1).value != "timestamp":
            ws.insert_row(SHEET_HEADERS, 1)
        return ws
    except Exception:
        return None


def save_to_gsheet(sample: dict, answers: dict, annotator_id: str) -> bool:
    ws = get_worksheet()
    if ws is None:
        return False
    row = (
        [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
         annotator_id,
         sample.get("dataset", ""),
         sample.get("sample_id", ""),
         sample.get("video_file", ""),
         sample.get("speaker", ""),
         sample.get("utterance", "")]
        + [answers.get(q["key"], "") for q in QUESTIONS]
        + [answers.get("notes", "")]
    )
    ws.append_row(row, value_input_option="USER_ENTERED")
    return True


def load_progress_from_gsheet(annotator_id: str) -> set:
    """从 Google Sheets 读取该标注者已完成的 sample_id 集合。"""
    ws = get_worksheet()
    if ws is None:
        return set()
    try:
        records = ws.get_all_records()
        return {
            r["sample_id"]
            for r in records
            if r.get("annotator_id") == annotator_id and r.get("sample_id")
        }
    except Exception:
        return set()


# ============================================================
# SECTION 4: Session state 初始化
# ============================================================

def init_state():
    defaults = {
        "page":               "instructions",
        "annotator_id":       "",
        "current_idx":        0,
        "local_annotations":  {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# SECTION 5: 用户须知页面
# ============================================================

def show_instructions():
    st.title("XMER 跨模态情感标注系统")
    st.markdown("---")

    st.markdown("""
## 欢迎参加本次标注实验

本实验旨在研究**跨模态情感冲突**现象——即说话人的声音情绪与面部表情情绪不一致的情形。
你的标注数据将用于学术研究，请认真作答。

---

## 标注流程

1. **观看视频**：每个样本包含一段 5–15 秒的视频片段。
2. **阅读情景卡片**（可选）：了解说话人的背景、目标与情境，有助于更准确地判断情绪。
3. **回答问题**：分别从声音、表情两个维度判断情绪，再给出综合判断，并评估是否存在冲突。
4. **点击「保存并继续」**：结果将自动上传，进入下一条。

---

## 注意事项

- **每个视频可反复播放**，建议至少看两遍（一遍关注声音，一遍关注画面）。
- 情景卡片不会提示答案，仅帮助你理解背景；**以你实际感知为准**。
- 如果确实判断不了，在备注中说明即可。
- 标注过程中**请勿关闭页面**，未保存的进度会丢失。
- 每个 batch 约含 **10–15 个样本**，预计完成时间 **20–30 分钟**。

---

## 情绪选项说明

| 选项 | 含义 |
|------|------|
| 积极/高兴 | 喜悦、满足、轻松、兴奋等正面情绪 |
| 悲伤 | 失落、沮丧、难过 |
| 愤怒 | 生气、不满、激动 |
| 恐惧 | 害怕、担忧、紧张 |
| 惊讶 | 意外、震惊（中性方向） |
| 厌恶 | 反感、鄙视 |
| 中性 | 情绪平淡，无明显倾向 |

---
""")

    st.subheader("请输入你的标注者 ID 后开始")
    name = st.text_input(
        "姓名或 ID（用于区分不同标注者）",
        value=st.session_state.annotator_id,
        placeholder="例如：ZhangSan",
    )
    if st.button("开始标注 →", type="primary", disabled=not name.strip()):
        annotator_id = name.strip()
        st.session_state.annotator_id = annotator_id

        # 从 Google Sheets 恢复已标注进度
        with st.spinner("正在读取你的历史进度..."):
            done_ids = load_progress_from_gsheet(annotator_id)

        if done_ids:
            st.session_state.local_annotations = {sid: {} for sid in done_ids}
            st.info(f"检测到你已完成 {len(done_ids)} 条标注，将自动跳转到第一条未完成的样本。")

        st.session_state.page = "annotation"
        st.session_state.current_idx = 0   # 路由后在 show_annotation 里修正
        st.session_state.resume_needed = True
        st.rerun()


# ============================================================
# SECTION 6: 标注主页面
# ============================================================

def show_annotation(samples):
    total = len(samples)

    # 恢复进度后自动跳到第一条未标注的样本
    if st.session_state.get("resume_needed"):
        st.session_state.resume_needed = False
        done = st.session_state.local_annotations
        for i, s in enumerate(samples):
            if s["sample_id"] not in done:
                st.session_state.current_idx = i
                break
        else:
            st.session_state.current_idx = total - 1  # 全部完成时停在最后一条

    annotated = sum(
        1 for s in samples if s["sample_id"] in st.session_state.local_annotations
    )

    # --- 侧栏 ---
    with st.sidebar:
        st.markdown(f"👤 标注者：**{st.session_state.annotator_id}**")
        st.caption(f"进度：{annotated} / {total}")
        st.divider()
        if st.button("← 返回须知页"):
            st.session_state.page = "instructions"
            st.rerun()

    # --- 顶部进度 + 导航 ---
    st.title("XMER 跨模态情感标注")
    st.progress(
        annotated / total if total else 0,
        text=f"当前进度：{annotated} / {total} 已完成",
    )

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← 上一条", disabled=st.session_state.current_idx == 0):
            st.session_state.current_idx -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>"
            f"第 <b>{st.session_state.current_idx + 1}</b> 条，共 <b>{total}</b> 条</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("下一条 →", disabled=st.session_state.current_idx == total - 1):
            st.session_state.current_idx += 1
            st.rerun()

    st.divider()

    sample = samples[st.session_state.current_idx]
    existing = st.session_state.local_annotations.get(sample["sample_id"], {})

    # --- 视频 + 情景卡片（左右两栏）---
    col_video, col_card = st.columns([1, 1], gap="large")

    with col_video:
        st.subheader(f"[{sample['dataset']}]  `{sample['sample_id']}`")
        speaker = sample.get("speaker", "未知")
        utterance = sample.get("utterance", "")
        st.caption(f"说话人：**{speaker}**")
        if utterance:
            st.info(f'"{utterance}"')
        video_url = get_video_url(sample["dataset"], sample["video_file"])
        st.video(video_url)

    with col_card:
        st.subheader("情景卡片")
        card = sample.get("situation_card", {})
        visible = [f for f, v in SITUATION_CARD_FIELDS.items() if v]
        if not visible:
            st.info("情景卡片当前已全部隐藏（由管理员配置）。")
        for field in visible:
            if field in card:
                st.markdown(f"**{field}**")
                st.markdown(
                    f"<div style='background:#f0f2f6;padding:8px 12px;"
                    f"border-radius:6px;margin-bottom:8px;'>{card[field]}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # --- 标注问题 ---
    st.subheader("标注问题")
    answers: dict = {}
    for q in QUESTIONS:
        default = 0
        if q["key"] in existing:
            try:
                default = q["options"].index(existing[q["key"]])
            except ValueError:
                default = 0
        answers[q["key"]] = st.radio(
            q["label"],
            options=q["options"],
            index=default,
            horizontal=True,
            key=f"{sample['sample_id']}_{q['key']}",
        )

    answers["notes"] = st.text_area(
        "备注（可选）",
        value=existing.get("notes", ""),
        height=80,
        placeholder="如有疑问或特殊说明，请在此填写",
        key=f"{sample['sample_id']}_notes",
    )

    st.divider()

    # --- 保存按钮 ---
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        if st.button("保存并继续 ✓", type="primary"):
            st.session_state.local_annotations[sample["sample_id"]] = answers
            ok = save_to_gsheet(sample, answers, st.session_state.annotator_id)
            if ok:
                st.success("已保存到 Google Sheets")
            else:
                st.warning("Google Sheets 未连接，仅保存在本地（页面关闭后丢失）")
            if st.session_state.current_idx < total - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.success("🎉 本 batch 所有样本已标注完成！感谢你的参与。")

    with col_msg:
        if sample["sample_id"] in st.session_state.local_annotations:
            st.info("该样本在本次会话中已标注，可修改后重新保存。")


# ============================================================
# SECTION 7: 路由入口
# ============================================================

def main():
    batch_name = st.query_params.get("batch", None)
    all_samples = load_all_samples()
    samples = filter_by_batch(all_samples, batch_name)

    if st.session_state.page == "instructions":
        show_instructions()
    else:
        show_annotation(samples)


main()
