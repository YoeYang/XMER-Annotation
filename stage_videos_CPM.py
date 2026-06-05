#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2 视频归集：把 368 个视频从各自本地目录复制并统一重命名，
方便上传到 Hugging Face。

输入：merged_reasoning_CPM_clean.csv 的 Interface 列（本地绝对路径）
输出：XMER-Annotation/videos_upload/{DATASET}/{DATASET}_{video_id}_{clip_id}.mp4

上传到 HF 后，app 端访问路径为：
  {HF_DATASET_URL}/CPM/{DATASET}/{DATASET}_{video_id}_{clip_id}.mp4
"""
import os
import shutil
import pandas as pd

CONFLICT_ROOT = "/scratch/project_2017416/yyy/conflict-sampling"
MERGED_CSV = os.path.join(CONFLICT_ROOT, "step-1-genconflict", "merged_reasoning_CPM_clean.csv")

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos_upload")

DATASET_DISPLAY = {
    "ch-sims": "CH-SIMS",
    "meld":    "MELD",
    "iemocap": "IEMOCAP",
    "mosi":    "MOSI",
}


def video_filename(dataset, video_id, clip_id):
    ds = DATASET_DISPLAY.get(dataset, dataset)
    return f"{ds}_{video_id}_{clip_id}.mp4"


def main():
    merged = pd.read_csv(MERGED_CSV)

    copied, skipped, missing = 0, 0, []
    total_bytes = 0
    for _, r in merged.iterrows():
        ds = str(r.dataset)
        src = r.Interface
        if not isinstance(src, str) or not os.path.exists(src):
            missing.append((ds, r.video_id, r.clip_id, src))
            continue

        ds_disp = DATASET_DISPLAY.get(ds, ds)
        out_dir = os.path.join(OUT_ROOT, ds_disp)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, video_filename(ds, r.video_id, r.clip_id))

        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            skipped += 1
            total_bytes += os.path.getsize(dst)
            continue

        shutil.copy2(src, dst)
        copied += 1
        total_bytes += os.path.getsize(dst)

    print(f"✓ 归集完成 → {OUT_ROOT}")
    print(f"  新复制: {copied}  已存在跳过: {skipped}  缺失: {len(missing)}")
    print(f"  总大小: {total_bytes / 1024 / 1024:.1f} MB")
    # 各数据集统计
    for ds_disp in DATASET_DISPLAY.values():
        d = os.path.join(OUT_ROOT, ds_disp)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith(".mp4")])
            sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".mp4"))
            print(f"    {ds_disp}: {n} 个, {sz / 1024 / 1024:.1f} MB")
    if missing:
        print(f"  ⚠ 缺失样例: {missing[:5]}")


if __name__ == "__main__":
    main()
