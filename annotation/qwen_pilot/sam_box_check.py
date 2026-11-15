#!/usr/bin/env python3
"""Can a segmentation model flag Qwen boxes that are not around a real object? Check on the RA gold set (gold set v2).

Every Qwen box of the RA-reviewed frames (annotation/gold_set/score_ra_qwen_t1.py -> ra_scores/boxes.csv) is given to
SAM 2.1 (hiera-large, transformers) as a box prompt on its frame. Per box: SAM's predicted mask IoU (its own confidence),
mask fill (mask area / box area), the tight box around the mask, and that tight box's IoU with the original box. Then:
how well each score separates boxes the RAs deleted from boxes they kept (AUC, overall and for generic labels), and
whether the tight box agrees better with the RA's final box than Qwen's box does.

Output (private: frame ids): annotation/gold_set/out/private/ra_scores/sam_box_check.csv and printed summary
  CUDA_VISIBLE_DEVICES=0 python sam_box_check.py [--model facebook/sam2.1-hiera-large]          # qwen env
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "gold_set" / "out" / "private"
OUT = GOLD / "ra_scores" / "sam_box_check.csv"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def auc(pos, neg):
    """P(random kept box > random deleted box), ties count half (Mann-Whitney U / n1 n2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    x = np.concatenate([pos, neg]); order = x.argsort(kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                                   # average ranks for ties
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--summary", action="store_true", help="only summarize the saved sam_box_check.csv")
    args = ap.parse_args()
    if args.summary:
        summarize(pd.read_csv(OUT)); return
    os.environ.setdefault("HF_HOME", "/data/j7yang/models/hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    torch.set_num_threads(2)
    b = pd.read_csv(GOLD / "ra_scores" / "boxes.csv")
    q = b[b.outcome != "added"].copy()                 # every Qwen box the RAs saw: kept (matched) or deleted
    fr = pd.read_csv(GOLD / "frames_qwen_t1.csv").drop_duplicates("frame_uid").set_index("frame_uid").frame_path
    dev = "cuda:0"
    proc = Sam2Processor.from_pretrained(args.model, cache_dir="/data/j7yang/models/hf/hub")
    model = Sam2Model.from_pretrained(args.model, cache_dir="/data/j7yang/models/hf/hub").to(dev).eval()
    rows = []
    for uid, g in q.groupby("frame_uid"):
        im = Image.open(fr[uid]).convert("RGB")               # CVAT boxes are on this 512 x 910 frame
        boxes = g[["q_x0", "q_y0", "q_x1", "q_y1"]].values.astype(float).tolist()
        inp = proc(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(**inp, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][:, 0].numpy()
        scores = out.iou_scores.float().cpu().numpy().reshape(len(boxes), -1)[:, 0]
        for (idx, r), m, s, bx in zip(g.iterrows(), masks, scores, boxes):
            area_box = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            ys, xs = np.nonzero(m)
            tight = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [np.nan] * 4
            x0, y0, x1, y1 = (int(v) for v in bx)
            inside = m[max(0, y0):y1, max(0, x0):x1].sum()
            rows.append(dict(row=idx, frame_uid=uid, batch=r.batch, outcome=r.outcome, q_name=r.q_name, ra_name=r.ra_name,
                             sam_score=float(s), fill=inside / area_box, leak=(m.sum() - inside) / area_box,
                             tight_x0=tight[0], tight_y0=tight[1], tight_x1=tight[2], tight_y1=tight[3],
                             tight_vs_qwen_iou=iou(tight, bx) if len(xs) else 0.0,
                             qwen_vs_ra_iou=r.iou if r.outcome == "matched" else np.nan,
                             tight_vs_ra_iou=iou(tight, [r.ra_x0, r.ra_y0, r.ra_x1, r.ra_y1]) if (r.outcome == "matched" and len(xs)) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(OUT, index=False)
    summarize(S)


def summarize(S: pd.DataFrame):
    import sys
    sys.path.insert(0, str(HERE))
    from qwen_generic_crops import GENERIC
    try:
        import qwen_data as qd
        S["generic"] = S.q_name.fillna("").map(lambda l: qd.norm_label(l) in GENERIC if l else False)
    except Exception:
        S["generic"] = S.q_name.fillna("").str.lower().isin(GENERIC)
    print(f"{len(S):,} Qwen boxes on {S.frame_uid.nunique()} frames; deleted by RAs: {(S.outcome == 'deleted').sum()}")
    for name, t in (("all boxes", S), ("generic labels", S[S.generic]), ("other labels", S[~S.generic])):
        d, k = t[t.outcome == "deleted"], t[t.outcome == "matched"]
        print(f"\n{name}: {len(t):,} boxes, {len(d)} deleted")
        for col, sign in (("sam_score", 1), ("fill", 1), ("tight_vs_qwen_iou", 1), ("leak", -1)):
            a = auc(sign * k[col], sign * d[col])
            print(f"  {col:18s} kept median {k[col].median():.3f} | deleted median {d[col].median():.3f} | AUC (kept > deleted) {a:.3f}")
    k = S[S.outcome == "matched"]
    print(f"\nkept boxes: IoU with the RA's final box: Qwen {k.qwen_vs_ra_iou.median():.3f} median, {(k.qwen_vs_ra_iou >= .5).mean():.1%} >= 0.5"
          f" | SAM tight box {k.tight_vs_ra_iou.median():.3f} median, {(k.tight_vs_ra_iou >= .5).mean():.1%} >= 0.5")
    moved = k[k.qwen_vs_ra_iou < .9]
    print(f"boxes the RA moved (IoU < 0.9, n={len(moved)}): Qwen vs RA {moved.qwen_vs_ra_iou.median():.3f}, SAM tight vs RA {moved.tight_vs_ra_iou.median():.3f}")
    for thr in (0.5, 0.6, 0.7, 0.8):
        flag = S.sam_score < thr
        print(f"flag sam_score < {thr}: {flag.mean():.1%} of boxes flagged; catches {flag[S.outcome == 'deleted'].mean():.1%} of deleted, "
              f"{flag[S.outcome == 'matched'].mean():.1%} of kept")


if __name__ == "__main__":
    main()
