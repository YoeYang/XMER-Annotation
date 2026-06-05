#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1 数据预处理：把分散的多张表合并成 app-CPM.py 直接使用的单一数据文件。

输入：
  - merged_reasoning_CPM_clean.csv  (信号 + reasoning + Interface)
  - CPM_classify_signature_results.csv  (情境卡片 + CPM 预打分)
  - gemini_signals_{MELD,iemocap,mosi}.csv  (context；CH-SIMS 无)

输出：
  - XMER-Annotation/data/samples_CPM.json  (368 条)

key 统一为 (dataset, video_id, clip_id)。
"""
import os
import json
import pandas as pd

# ── 路径 ────────────────────────────────────────────
CONFLICT_ROOT = "/scratch/project_2017416/yyy/conflict-sampling"
MERGED_CSV   = os.path.join(CONFLICT_ROOT, "step-1-genconflict", "merged_reasoning_CPM_clean.csv")
CLASSIFY_CSV = os.path.join(CONFLICT_ROOT, "step-2-classify", "CPM_classify_signature_results.csv")
GEMINI_DIR   = os.path.join(CONFLICT_ROOT, "step-1-genconflict", "gemini_signals")

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_JSON = os.path.join(OUT_DIR, "samples_CPM.json")

# ── 数据集名规范化（用于视频文件名前缀 & HF 路径）──────
DATASET_DISPLAY = {
    "ch-sims": "CH-SIMS",
    "meld":    "MELD",
    "iemocap": "IEMOCAP",
    "mosi":    "MOSI",
}


def video_filename(dataset: str, video_id, clip_id) -> str:
    """统一视频文件名：{DATASET}_{video_id}_{clip_id}.mp4"""
    ds = DATASET_DISPLAY.get(dataset, dataset)
    return f"{ds}_{video_id}_{clip_id}.mp4"


def load_context_lookup():
    """返回函数 (dataset, video_id, clip_id) -> context 字符串（无则 '')。"""
    # MELD: 用 (video_id, clip_id)
    meld = pd.read_csv(os.path.join(GEMINI_DIR, "gemini_signals_MELD.csv"))
    meld_map = {(str(r.video_id), str(r.clip_id)): r.context for _, r in meld.iterrows()}
    # iemocap / mosi: 用 id = video_id_clipid
    iem = pd.read_csv(os.path.join(GEMINI_DIR, "gemini_signals_iemocap.csv"))
    iem_map = {str(r.id): r.context for _, r in iem.iterrows()}
    mos = pd.read_csv(os.path.join(GEMINI_DIR, "gemini_signals_mosi.csv"))
    mos_map = {str(r.id): r.context for _, r in mos.iterrows()}

    def lookup(dataset, video_id, clip_id):
        vid, cid = str(video_id), str(clip_id)
        if dataset == "meld":
            ctx = meld_map.get((vid, cid))
        elif dataset == "iemocap":
            ctx = iem_map.get(f"{vid}_{cid}")
        elif dataset == "mosi":
            ctx = mos_map.get(f"{vid}_{cid}")
        else:  # ch-sims 无 context
            ctx = ""
        if ctx is None or (isinstance(ctx, float)):
            return ""
        return str(ctx)

    return lookup


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    merged = pd.read_csv(MERGED_CSV)
    classify = pd.read_csv(CLASSIFY_CSV)

    # classify 建索引：(dataset, video_id, clip_id) -> row
    cls_map = {
        (str(r.dataset), str(r.video_id), str(r.clip_id)): r
        for _, r in classify.iterrows()
    }
    ctx_lookup = load_context_lookup()

    samples = []
    missing_cls = 0
    for _, r in merged.iterrows():
        ds, vid, cid = str(r.dataset), str(r.video_id), str(r.clip_id)
        c = cls_map.get((ds, vid, cid))
        if c is None:
            missing_cls += 1
            continue

        sample = {
            # ── 标识 ──
            "sample_id":   f"{ds}_{vid}_{cid}",
            "dataset":     ds,
            "video_id":    vid,
            "clip_id":     cid,
            "video_file":  video_filename(ds, vid, cid),
            # ── 预生成模态信号（阶段1 预填）──
            "video_signals": r.video_signals if pd.notna(r.video_signals) else "",
            "audio_signals": r.audio_signals if pd.notna(r.audio_signals) else "",
            "text_signals":  r.text_signals  if pd.notna(r.text_signals)  else "",
            # ── 冲突参考信息（只读）──
            "is_conflict": r.is_conflict if pd.notna(r.is_conflict) else "",
            "confidence":  r.confidence  if pd.notna(r.confidence)  else "",
            "reasoning":   r.reasoning   if pd.notna(r.reasoning)   else "",
            "mechanism":   r.mechanism   if pd.notna(r.mechanism)   else "",
            # ── 情境卡片（只读）──
            "Subject":        c.Subject        if pd.notna(c.Subject)        else "",
            "Stance":         c.Stance         if pd.notna(c.Stance)         else "",
            "Power_Interest": c.Power_Interest if pd.notna(c.Power_Interest) else "",
            "context":        ctx_lookup(ds, vid, cid),
            # ── CPM 预打分（阶段2 预填，人工可调）──
            "R_valence": int(c.R_valence),
            "I_valence": int(c.I_valence),
            "C_valence": int(c.C_valence),
            "N_valence": int(c.N_valence),
            "Predicted_polarity": c.Predicted_polarity if pd.notna(c.Predicted_polarity) else "",
            "Justification":      c.Justification      if pd.notna(c.Justification)      else "",
        }
        samples.append(sample)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    from collections import Counter
    by_ds = Counter(s["dataset"] for s in samples)
    n_ctx = sum(1 for s in samples if s["context"])
    print(f"✓ 写出 {len(samples)} 条 → {OUT_JSON}")
    print(f"  数据集分布: {dict(by_ds)}")
    print(f"  含 context: {n_ctx} 条（ch-sims 无 context）")
    if missing_cls:
        print(f"  ⚠ {missing_cls} 条在 classify 表中无匹配，已跳过")


if __name__ == "__main__":
    main()
