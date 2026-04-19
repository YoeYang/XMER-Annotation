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
HF_DATASET_URL = "https://huggingface.co/datasets/YoeYang/XMER-Videos/resolve/main"
VIDEO_FOLDERS = {
    "CH-SIMS": "CH-SIMS/video",
    "MELD":    "MELD/clean",
}

# 1b. 情景卡片字段（True=显示，False=隐藏，标注者不可更改）
SITUATION_CARD_FIELDS = {
    "人物身份":       True,
    "触发事件":       True,
    "当前目标_利益":  True,
    "权力关系":       True,
    "标注提示":       False,
}

# 1c. 标注协议 —— 三步

# ── Part 1：模态观察 ──────────────────────────────
PART1_FACE = {
    "嘴角":    ["上扬", "下降", "中性", "不确定"],
    "眉头":    ["皱起", "舒展", "中性", "不确定"],
    "眼神":    ["直视", "回避", "游移", "不确定"],
    "整体面部": ["紧绷", "放松", "中性", "不确定"],
}
PART1_VOICE = {
    "语调": ["偏高", "偏低", "平稳", "不确定"],
    "语速": ["较快", "较慢", "正常", "不确定"],
    "音量": ["较大", "较小", "正常", "不确定"],
    "语气": ["有力/坚定", "迟疑/颤抖", "平淡", "不确定"],
}
PART1_LANGUAGE = {
    "内容倾向": ["积极", "消极", "中性", "不确定"],
    "表达方式": ["直接", "回避/委婉", "反问/质疑", "不确定"],
}
PART1_CONFLICT_OPTIONS = [
    "face vs voice", "face vs language", "voice vs language",
    "多组同时冲突", "没有明显冲突", "不确定",
]

# ── Part 2：S-E 整体静态评估 ─────────────────────
PART2_SLIDERS = [
    {
        "key": "pleasantness",
        "label": "Q1  这件事对 TA 来说是愉快的吗？（Pleasantness）",
        "min_val": -3, "max_val": 3,
        "min_label": "-3 非常不愉快", "max_label": "+3 非常愉快",
    },
    {
        "key": "certainty",
        "label": "Q2  TA 对正在发生的事情清楚吗，知道结果会怎样吗？（Certainty）",
        "min_val": -3, "max_val": 3,
        "min_label": "-3 完全不确定", "max_label": "+3 非常确定",
    },
    {
        "key": "attentional_activity",
        "label": "Q3  TA 想继续关注、深入了解这件事吗？（Attentional Activity）",
        "min_val": -3, "max_val": 3,
        "min_label": "-3 想回避", "max_label": "+3 高度投入",
    },
    {
        "key": "anticipated_effort",
        "label": "Q4  TA 觉得对于发生的事，需要付出很大努力吗？（Anticipated Effort）",
        "min_val": 0, "max_val": 3,
        "min_label": "0 不需要努力", "max_label": "+3 需要极大努力",
    },
    {
        "key": "situational_control",
        "label": "Q6  TA 觉得自己能控制、改变这个局面吗？（Situational Control）",
        "min_val": -3, "max_val": 3,
        "min_label": "-3 完全无能为力", "max_label": "+3 完全掌控",
    },
]
PART2_AGENCY_OPTIONS = ["主要是 TA 自己", "主要是他人", "外部环境或命运"]

# ── Part 3：CPM 动态评估 ──────────────────────────
PART3_QUESTIONS = [
    {
        "key": "A",
        "label": "□ A  这件事有没有引起 TA 的注意，TA 觉得和自己有关吗？",
        "options": [
            "有，TA 明显在意 / 觉得与自己相关",
            "没有，TA 显得漠然 / 觉得与自己无关",
            "无法判断",
        ],
    },
    {
        "key": "B",
        "label": "□ B  TA 觉得这件事是好事还是坏事，本能上喜欢还是排斥？",
        "options": ["本能喜欢 / 好事", "本能排斥 / 坏事", "混合 / 无法判断"],
    },
    {
        "key": "C",
        "label": "□ C  TA 觉得这件事对自己的目标是有利还是有害？",
        "options": ["有利 / 朝着目标前进", "有害 / 阻碍目标实现", "混合 / 无法判断"],
    },
    {
        "key": "D",
        "label": "□ D  TA 觉得自己有能力应对这件事吗？",
        "options": ["有能力，感到掌控", "没有能力，感到无力", "混合 / 无法判断"],
    },
    {
        "key": "E",
        "label": "□ E  TA 觉得这件事在道德上、社会规范上是否合适？",
        "options": ["合适 / 符合规范", "不合适 / 违背规范", "这个维度在此情境中不适用", "无法判断"],
    },
]
PART3_CONSISTENCY_OPTIONS = [
    "高度一致", "基本一致（有的不太能判断）", "有明显的模态冲突", "不太清楚",
]

# 1d. Batch 分组系统
# 格式: {"batch_name": [("dataset", "sample_id"), ...]}
# 空 dict → 不分组；访问方式: ?batch=batch_1
# BATCHES = {
#     "batch_1": [("CH-SIMS", "aqgy3_0001_00016"), ("MELD", "dia29_utt6")],
#     "batch_2": [("CH-SIMS", "aqgy5_0015_00015"), ("MELD", "dia34_utt4")],
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
    return f"{HF_DATASET_URL}/{VIDEO_FOLDERS.get(dataset, dataset)}/{video_file}"


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

# Google Sheets 列定义（与 save_to_gsheet 的 row 构建顺序严格对应）
SHEET_HEADERS = (
    ["timestamp", "annotator_id", "dataset", "sample_id", "video_file", "speaker", "utterance"]
    # Part 1 - Face
    + [f"face_{k}" for k in PART1_FACE]
    # Part 1 - Voice
    + [f"voice_{k}" for k in PART1_VOICE]
    # Part 1 - Language
    + [f"lang_{k}" for k in PART1_LANGUAGE]
    + ["conflict"]
    # Part 2
    + [s["key"] for s in PART2_SLIDERS]
    + ["agency"]
    # Part 3
    + [f"p3_{q['key']}" for q in PART3_QUESTIONS]
    + [f"p3_{q['key']}1" for q in PART3_QUESTIONS]
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
    conflict_val = ", ".join(answers.get("conflict", []))
    row = (
        [datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
         annotator_id,
         sample.get("dataset", ""),
         sample.get("sample_id", ""),
         sample.get("video_file", ""),
         sample.get("speaker", ""),
         sample.get("utterance", "")]
        + [answers.get(f"face_{k}", "") for k in PART1_FACE]
        + [answers.get(f"voice_{k}", "") for k in PART1_VOICE]
        + [answers.get(f"lang_{k}", "") for k in PART1_LANGUAGE]
        + [conflict_val]
        + [answers.get(s["key"], "") for s in PART2_SLIDERS]
        + [answers.get("agency", "")]
        + [answers.get(f"p3_{q['key']}", "") for q in PART3_QUESTIONS]
        + [answers.get(f"p3_{q['key']}1", "") for q in PART3_QUESTIONS]
        + [answers.get("notes", "")]
    )
    ws.append_row(row, value_input_option="USER_ENTERED")
    return True


def load_progress_from_gsheet(annotator_id: str) -> dict:
    """从 Google Sheets 加载该标注者的所有历史答案。
    同一 sample_id 有多条记录时取最后一条（时间最晚）。
    返回 {sample_id: {data_key: value, ...}}。
    """
    ws = get_worksheet()
    if ws is None:
        return {}
    try:
        records = ws.get_all_records()
        # 只保留当前标注者的记录，遍历顺序即时间顺序，后面的覆盖前面的
        result = {}
        for r in records:
            if r.get("annotator_id") != annotator_id or not r.get("sample_id"):
                continue
            sid = r["sample_id"]
            answers = {}
            # Part 1 - Face / Voice / Language
            for k in PART1_FACE:
                if r.get(f"face_{k}"):
                    answers[f"face_{k}"] = r[f"face_{k}"]
            for k in PART1_VOICE:
                if r.get(f"voice_{k}"):
                    answers[f"voice_{k}"] = r[f"voice_{k}"]
            for k in PART1_LANGUAGE:
                if r.get(f"lang_{k}"):
                    answers[f"lang_{k}"] = r[f"lang_{k}"]
            # conflict 存为逗号分隔字符串，还原为 list
            if r.get("conflict"):
                answers["conflict"] = [c.strip() for c in r["conflict"].split(",") if c.strip()]
            # Part 2
            for s in PART2_SLIDERS:
                if r.get(s["key"]) != "":
                    try:
                        answers[s["key"]] = int(r[s["key"]])
                    except (ValueError, TypeError):
                        pass
            if r.get("agency"):
                answers["agency"] = r["agency"]
            # Part 3
            for q in PART3_QUESTIONS:
                k = q["key"]
                if r.get(f"p3_{k}"):
                    answers[f"p3_{k}"] = r[f"p3_{k}"]
                if r.get(f"p3_{k}1"):
                    answers[f"p3_{k}1"] = r[f"p3_{k}1"]
            if r.get("notes"):
                answers["notes"] = r["notes"]
            result[sid] = answers
        return result
    except Exception:
        return {}


# ============================================================
# SECTION 4: Session state 初始化
# ============================================================

def init_state():
    defaults = {
        "page":              "instructions",
        "annotator_id":      "",
        "current_idx":       0,
        "local_annotations": {},
        "annotation_step":   1,       # 当前标注步骤 1/2/3
        "_last_sample_idx":  -1,      # 用于检测样本切换，自动重置步骤
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# SECTION 5: 用户须知页面
# ============================================================

@st.cache_data
def load_guide() -> str:
    guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ANNOTATOR_GUIDE.md")
    if os.path.exists(guide_path):
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    return "*(Annotator guide not found. Please contact the administrator.)*"


def show_instructions():
    st.title("XMER Annotation System")
    st.markdown("---")
    st.markdown(load_guide(), unsafe_allow_html=True)
    st.subheader("请输入你的标注者 ID 后开始")
    name = st.text_input(
        "姓名或 ID（用于区分不同标注者）",
        value=st.session_state.annotator_id,
        placeholder="例如：ZhangSan",
    )
    if st.button("开始标注 →", type="primary", disabled=not name.strip()):
        annotator_id = name.strip()
        st.session_state.annotator_id = annotator_id
        with st.spinner("正在读取你的历史进度..."):
            history = load_progress_from_gsheet(annotator_id)
        if history:
            st.session_state.local_annotations = history
        st.session_state.page = "annotation"
        st.session_state.current_idx = 0
        st.session_state.resume_needed = True
        st.rerun()


# ============================================================
# SECTION 6: 标注主页面
# ============================================================

def _radio(label, options, key, existing, hide_label=False):
    """带回填的 radio。session_state 已由 restore_widget_state() 预填，直接渲染即可。"""
    lv = "collapsed" if hide_label else "visible"
    return st.radio(label, options, index=None, horizontal=True,
                    key=key, label_visibility=lv)


def restore_widget_state(sid, existing):
    """把已保存答案写入 session_state widget key，仅在该 key 尚未被用户交互设置时写入。
    data key（如 face_嘴角）→ widget key（如 face_嘴角_sampleA）。
    """
    for data_key, value in existing.items():
        if value is None:
            continue
        widget_key = f"{data_key}_{sid}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = value


def collect_answers(sid, existing):
    """从 session_state 和已保存答案中收集所有字段值。"""
    a = {}
    for k in PART1_FACE:
        a[f"face_{k}"] = st.session_state.get(f"face_{k}_{sid}") or existing.get(f"face_{k}")
    for k in PART1_VOICE:
        a[f"voice_{k}"] = st.session_state.get(f"voice_{k}_{sid}") or existing.get(f"voice_{k}")
    for k in PART1_LANGUAGE:
        a[f"lang_{k}"] = st.session_state.get(f"lang_{k}_{sid}") or existing.get(f"lang_{k}")
    a["conflict"] = st.session_state.get(f"conflict_{sid}", existing.get("conflict", []))
    for s in PART2_SLIDERS:
        wk = f"{s['key']}_{sid}"
        a[s["key"]] = st.session_state.get(wk) if st.session_state.get(wk) is not None \
                      else existing.get(s["key"], (s["min_val"] + s["max_val"]) // 2)
    a["agency"] = st.session_state.get(f"agency_{sid}") or existing.get("agency")
    for q in PART3_QUESTIONS:
        k = q["key"]
        a[f"p3_{k}"]  = st.session_state.get(f"p3_{k}_{sid}")  or existing.get(f"p3_{k}")
        a[f"p3_{k}1"] = st.session_state.get(f"p3_{k}1_{sid}") or existing.get(f"p3_{k}1")
    a["notes"] = st.session_state.get(f"notes_{sid}", existing.get("notes", ""))
    return a


def validate_step(answers, step):
    """返回当前步骤未填写的必填项列表。"""
    missing = []
    if step == 1:
        for k in PART1_FACE:
            if not answers.get(f"face_{k}"):
                missing.append(f"Face · {k}")
        for k in PART1_VOICE:
            if not answers.get(f"voice_{k}"):
                missing.append(f"Voice · {k}")
        for k in PART1_LANGUAGE:
            if not answers.get(f"lang_{k}"):
                missing.append(f"Language · {k}")
        if not answers.get("conflict"):
            missing.append("冲突判断（至少选一项）")
    elif step == 2:
        if not answers.get("agency"):
            missing.append("Q5 Agency")
    elif step == 3:
        for q in PART3_QUESTIONS:
            k = q["key"]
            if not answers.get(f"p3_{k}"):
                missing.append(f"□ {k} 主判断")
            if not answers.get(f"p3_{k}1"):
                missing.append(f"□ {k}1 模态一致性")
    return missing


def show_annotation(samples):
    total = len(samples)

    # 恢复进度 → 自动跳到第一条未标注
    if st.session_state.get("resume_needed"):
        st.session_state.resume_needed = False
        done = st.session_state.local_annotations
        for i, s in enumerate(samples):
            if s["sample_id"] not in done:
                st.session_state.current_idx = i
                break
        else:
            st.session_state.current_idx = total - 1

    # 检测样本切换 → 重置到第一步
    if st.session_state._last_sample_idx != st.session_state.current_idx:
        st.session_state.annotation_step = 1
        st.session_state._last_sample_idx = st.session_state.current_idx

    annotated = sum(1 for s in samples
                    if s["sample_id"] in st.session_state.local_annotations)
    step = st.session_state.annotation_step

    # 侧栏
    with st.sidebar:
        st.markdown(f"👤 标注者：**{st.session_state.annotator_id}**")
        st.caption(f"样本进度：{annotated} / {total}")
        st.divider()
        if st.button("← 返回须知页"):
            st.session_state.page = "instructions"
            st.rerun()

    # 当前样本（必须在导航按钮之前定义，按钮里需要用到 sid/existing）
    sample = samples[st.session_state.current_idx]
    sid = sample["sample_id"]
    existing = st.session_state.local_annotations.get(sid, {})

    # 把已保存答案预填入 widget session_state，保证回看时答案留存
    restore_widget_state(sid, existing)

    # 顶部进度 + 样本导航
    st.title("XMER 跨模态情感标注")
    st.progress(annotated / total if total else 0,
                text=f"样本进度：{annotated} / {total} 已完成")

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← 上一条", disabled=st.session_state.current_idx == 0):
            st.session_state.local_annotations[sid] = collect_answers(sid, existing)
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
            st.session_state.local_annotations[sid] = collect_answers(sid, existing)
            st.session_state.current_idx += 1
            st.rerun()

    # ── 上方固定区域：视频 + 情景卡片 ──────────────────
    col_video, col_card = st.columns([1, 1], gap="large")

    with col_video:
        st.subheader(f"[{sample['dataset']}]  `{sid}`")
        st.caption(f"说话人：**{sample.get('speaker', '未知')}**")
        utterance = sample.get("utterance", "")
        if utterance:
            st.info(f'"{utterance}"')
        st.video(get_video_url(sample["dataset"], sample["video_file"]))

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

    # ── 步骤指示器 ──────────────────────────────────────
    STEP_NAMES = ["第一步：模态观察", "第二步：S-E 整体评估", "第三步：CPM 动态评估"]
    step_cols = st.columns(3)
    for i, name in enumerate(STEP_NAMES):
        with step_cols[i]:
            if i + 1 == step:
                st.markdown(
                    f"<div style='text-align:center;background:#1f77b4;color:white;"
                    f"padding:6px;border-radius:6px;font-weight:bold;'>▶ {name}</div>",
                    unsafe_allow_html=True)
            elif i + 1 < step:
                st.markdown(
                    f"<div style='text-align:center;background:#d4edda;color:#155724;"
                    f"padding:6px;border-radius:6px;'>✓ {name}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='text-align:center;background:#e9ecef;color:#6c757d;"
                    f"padding:6px;border-radius:6px;'>{name}</div>",
                    unsafe_allow_html=True)
    st.markdown("")

    # ── Step 1：模态观察 ────────────────────────────────
    if step == 1:
        st.markdown("### 👁 Face —— 面部观察")
        cols = st.columns(2)
        for i, (feature, opts) in enumerate(PART1_FACE.items()):
            with cols[i % 2]:
                _radio(f"**{feature}**", opts, f"face_{feature}_{sid}", existing)

        st.markdown("### 🔊 Voice —— 声音观察")
        cols = st.columns(2)
        for i, (feature, opts) in enumerate(PART1_VOICE.items()):
            with cols[i % 2]:
                _radio(f"**{feature}**", opts, f"voice_{feature}_{sid}", existing)

        st.markdown("### 💬 Language —— 语言观察")
        cols = st.columns(2)
        for i, (feature, opts) in enumerate(PART1_LANGUAGE.items()):
            with cols[i % 2]:
                _radio(f"**{feature}**", opts, f"lang_{feature}_{sid}", existing)

        st.markdown("### ⚡ 冲突判断（可多选）")
        st.caption("你认为以上三个模态之间存在不一致吗？")
        st.multiselect("选择存在冲突的模态对", options=PART1_CONFLICT_OPTIONS,
                       default=existing.get("conflict", []), key=f"conflict_{sid}")

    # ── Step 2：S-E 整体评估 ────────────────────────────
    elif step == 2:
        st.markdown("以**视频中当事人的视角**，结合情景卡片信息，回答以下问题：")
        st.markdown("")
        for slider in PART2_SLIDERS:
            k = slider["key"]
            saved_val = existing.get(k, (slider["min_val"] + slider["max_val"]) // 2)
            st.markdown(f"**{slider['label']}**")
            col_l, col_s, col_r = st.columns([2, 5, 2])
            with col_l:
                st.caption(slider["min_label"])
            with col_s:
                st.slider(" ", min_value=slider["min_val"], max_value=slider["max_val"],
                          value=int(saved_val), step=1,
                          key=f"{k}_{sid}", label_visibility="collapsed")
            with col_r:
                st.caption(slider["max_label"])
            st.markdown("")

        st.markdown("**Q5  TA 认为这件事主要是谁造成的？（Agency）**")
        _radio("Agency", PART2_AGENCY_OPTIONS, f"agency_{sid}", existing, hide_label=True)

    # ── Step 3：CPM 动态评估 ────────────────────────────
    elif step == 3:
        st.markdown("根据你的直觉回答以下五个问题，每题完成后评估三个模态的一致性：")
        st.markdown("")
        for q in PART3_QUESTIONS:
            k = q["key"]
            st.markdown(f"**{q['label']}**")
            _radio(q["label"], q["options"], f"p3_{k}_{sid}", existing, hide_label=True)
            st.markdown(f"↳ **{k}1**  你觉得面部、语调、说话内容三个模态都一致地反映了你上面的判断吗？")
            _radio(f"{k}1一致性", PART3_CONSISTENCY_OPTIONS,
                   f"p3_{k}1_{sid}", existing, hide_label=True)
            st.markdown("---")

        st.text_area("备注（可选）", value=existing.get("notes", ""), height=80,
                     placeholder="如有疑问或特殊说明，请在此填写", key=f"notes_{sid}")

    # ── 底部导航按钮 ────────────────────────────────────
    st.markdown("")
    answers = collect_answers(sid, existing)
    missing = validate_step(answers, step)

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if step > 1:
            if st.button("← 上一步"):
                st.session_state.local_annotations[sid] = collect_answers(sid, existing)
                st.session_state.annotation_step -= 1
                st.rerun()
    with nav_right:
        if step < 3:
            if st.button("下一步 →", type="primary"):
                if missing:
                    st.error("请完成所有必填项后再继续：\n- " + "\n- ".join(missing))
                else:
                    # 关键：把当前步骤答案合并存入 local_annotations
                    # 确保切步骤后 session_state 清空时 existing 仍有完整记录
                    st.session_state.local_annotations[sid] = collect_answers(sid, existing)
                    st.session_state.annotation_step += 1
                    st.rerun()
        else:
            if st.button("保存并继续 ✓", type="primary", key=f"save_{sid}"):
                if missing:
                    st.error("请完成所有必填项后再提交：\n- " + "\n- ".join(missing))
                else:
                    st.session_state.local_annotations[sid] = answers
                    ok = save_to_gsheet(sample, answers, st.session_state.annotator_id)
                    if ok:
                        st.success("已保存到 Google Sheets ✓")
                    else:
                        st.warning("Google Sheets 未连接，仅保存在本地（页面关闭后丢失）")
                    if st.session_state.current_idx < total - 1:
                        st.session_state.current_idx += 1
                        st.rerun()
                    else:
                        st.balloons()
                        st.success("🎉 本 batch 所有样本已标注完成！感谢你的参与。")

    if sid in st.session_state.local_annotations:
        st.caption("该样本已标注，可修改后重新保存。")


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
