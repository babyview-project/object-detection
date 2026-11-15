#!/usr/bin/env python3
"""Per child x month of age tables for the within-child development checks (development_within_child.py).

One pass over the BabyView Qwen box store (renamed labels, QOV_LABELS=renamed) on the core categories
(data/included_categories_qwen_core.txt, 183): for each (child, month of age, category), the sum of L2-normalized
DINOv3 vectors (all formats, and real objects only) and the box count. From the box index alone: per box area (share of
the frame) for the object-size check, and per frame the core categories present (for category-distribution divergences).

Outputs (out/private/dev_within_child[_renamed]/, private: per-child tables):
  sums_dinov3.npz   keys (child|month), categories, sum_all / sum_real (key x category x 768, float32), n_all, n_real
  boxes.parquet     core-category boxes: unit, month, age_months, video_id, frame_number, category, format, area, edge
  frames.parquet    every frame: unit, month, age_months, video_id, frame_number, home

  QOV_LABELS=renamed python build_child_month_means.py [--workers 16]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
import variant as V
from build_bv_vedb_data import STORES, FRAME_WH, HOME, age_table, unit_of

STORE = STORES["bv"]
OUT = V.out_dir(HERE / "out" / "private" / "dev_within_child")
CORE = sorted(l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_core.txt") if l.strip())


def accumulate(args):
    stem, cat_idx, month = args                       # cat_idx: per shard row, index into CORE or -1
    keep = cat_idx >= 0
    if not keep.any():
        return None
    with np.load(STORE / "shards" / f"{stem}.npz", allow_pickle=True) as z:
        assert len(z["image_id"]) == len(cat_idx), stem
        X = z["dino_emb"][keep].astype(np.float32)
        fmt = z["format"].astype(str)[keep]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    ci = cat_idx[keep]; real = fmt == "real"
    S = np.zeros((len(CORE), X.shape[1]), np.float64); np.add.at(S, ci, X)
    R = np.zeros_like(S); np.add.at(R, ci[real], X[real])
    return f"{unit_of(stem)}|{month}", S, R, np.bincount(ci, minlength=len(CORE)), np.bincount(ci[real], minlength=len(CORE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from build_open_means import basic_of
    OUT.mkdir(parents=True, exist_ok=True)
    ix = V.read_index(STORE, usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str)
    ix["cat"] = ix.label.map(basic_of(ix.label.unique())).fillna("")
    ix["unit"] = ix.video_id.map(unit_of)
    ages = age_table(ix.video_id.unique())
    ix["age_months"] = ix.video_id.map(ages)
    ix = ix[ix.age_months.notna()].copy() if ix.age_months.isna().any() else ix
    ix["month"] = np.floor(ix.age_months).astype(int)
    W, H = FRAME_WH["bv"]
    ix["area"] = (ix.xmax - ix.xmin) * (ix.ymax - ix.ymin) / (W * H)
    ix["edge"] = (ix.xmin <= .01 * W) | (ix.ymin <= .01 * H) | (ix.xmax >= .99 * W) | (ix.ymax >= .99 * H)
    ci = {c: i for i, c in enumerate(CORE)}
    ix["ci"] = ix.cat.map(ci).fillna(-1).astype(int)

    fr = ix.groupby(["video_id", "frame_number"]).agg(unit=("unit", "first"), month=("month", "first"),
                                                      age_months=("age_months", "first"),
                                                      home=("cat", lambda s: bool(set(s) & HOME))).reset_index()
    fr.to_parquet(OUT / "frames.parquet")
    b = ix[ix.ci >= 0][["unit", "month", "age_months", "video_id", "frame_number", "cat", "format", "area", "edge"]]
    b.rename(columns={"cat": "category"}).to_parquet(OUT / "boxes.parquet")
    print(f"{fr.unit.nunique()} children, {fr.groupby(['unit', 'month']).ngroups} child-months, {len(fr):,} frames, "
          f"{len(b):,} core boxes", flush=True)

    # shard rows follow the store index order within each video (V.read_index keeps row order)
    jobs = [(v, g.ci.to_numpy(), int(g.month.iloc[0])) for v, g in ix.groupby("video_id", sort=False)]
    keys = sorted({f"{u}|{m}" for u, m in fr[["unit", "month"]].drop_duplicates().itertuples(index=False)})
    ki = {k: i for i, k in enumerate(keys)}
    S = np.zeros((len(keys), len(CORE), 768), np.float32); R = np.zeros_like(S)
    NA = np.zeros((len(keys), len(CORE)), np.int64); NR = np.zeros_like(NA)
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(accumulate, jobs, chunksize=4)):
            if r is not None:
                k = ki[r[0]]; S[k] += r[1]; R[k] += r[2]; NA[k] += r[3]; NR[k] += r[4]
            if i % 1000 == 0:
                print(f"  {i:,}/{len(jobs):,} videos", flush=True)
    np.savez(OUT / "sums_dinov3.npz", keys=np.array(keys), categories=np.array(CORE), sum_all=S, sum_real=R, n_all=NA, n_real=NR)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
