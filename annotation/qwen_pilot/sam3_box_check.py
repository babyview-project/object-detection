#!/usr/bin/env python3
"""Label-aware box check with SAM 3 on the RA gold set (compare sam_box_check.py, SAM 2.1, label-blind).

SAM 3 segments every instance of a text concept in an image. Two modes:
  crop (default): for each Qwen box, the padded square crop Qwen saw in the renaming pass (box + 25% on each side, the
    box's own context; SAM 3 upsamples it, which helps small objects), prompted with the box's label. The target is the
    highest-scoring instance that contains the box centre or overlaps the box (IoU >= 0.3); its box (mapped back to the
    frame) is the corrected box and may extend beyond Qwen's box anywhere within the padding. Score 0 if none.
  frame: the whole frame prompted once per distinct label; target = best instance with IoU >= 0.5 with the Qwen box.
Also stored: the corrected box, its IoU with Qwen's box and, for kept boxes, with the RA's final box. Summary: how well the score separates boxes the RAs deleted from boxes they kept (AUC), alone and next to the
SAM 2.1 score (sam_box_check.csv), overall and for generic labels.

Runs locally (weights in /data/j7yang/models/hf after a one-time download; gated, Meta SAM license); no frames leave the machine.
Output (private: frame ids): annotation/gold_set/out/private/ra_scores/sam3_box_check.csv
  python sam3_box_check.py --download                  # once, after huggingface-cli login (no GPU)
  CUDA_VISIBLE_DEVICES=0 python sam3_box_check.py      # qwen env
  python sam3_box_check.py --summary
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "gold_set" / "out" / "private"
OUT = GOLD / "ra_scores" / "sam3_box_check.csv"
def out_path(mode, prompt="label", formats=None):
    tag = ("" if mode == "crop" else "_frame") + ("" if prompt == "label" else f"_{prompt}prompt") + (f"_{'-'.join(formats)}" if formats else "")
    return OUT.with_name(f"sam3_box_check{tag}.csv")


def prompt_for(label, fmt, style):
    if style == "label" or fmt in ("real", "", None) or not isinstance(fmt, str):
        return label
    return {"toy": f"toy {label}", "photo": f"photo of a {label}", "drawing": f"drawing of a {label}",
            "picture": f"picture of a {label}", "screen": f"{label} on a screen"}.get(fmt, label)


def padded(bx, W, H):
    """Square crop around a box, PAD on each side (as qwen_generic_crops.crop_url), clipped to the frame."""
    x0, y0, x1, y1 = bx
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L, T = max(0, cx - side / 2), max(0, cy - side / 2)
    return int(L), int(T), int(min(W, cx + side / 2) + .999), int(min(H, cy + side / 2) + .999)
MODEL, CACHE = "facebook/sam3", "/data/j7yang/models/hf/hub"
MATCH_IOU, MIN_SCORE, PAD, CROP_IOU = 0.5, 0.05, 0.25, 0.3


def iou_m(A, B):
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    i = ix * iy; ar = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return i / np.maximum(ar(A)[:, None] + ar(B)[None, :] - i, 1e-9)


def summarize(S: pd.DataFrame):
    from sam_box_check import auc
    s2p = GOLD / "ra_scores" / "sam_box_check.csv"
    if s2p.exists():
        S = S.merge(pd.read_csv(s2p)[["row", "sam_score"]].rename(columns={"sam_score": "sam2_score"}), on="row", how="left")
    GEN = {"toy", "animal", "character", "figure", "vehicle", "food", "object", "item", "thing"}
    S["generic"] = S.q_name.fillna("").str.lower().isin(GEN)
    print(f"{len(S):,} Qwen boxes on {S.frame_uid.nunique()} frames; deleted by RAs: {(S.outcome == 'deleted').sum()}")
    for name, t in (("all boxes", S), ("generic labels", S[S.generic]), ("other labels", S[~S.generic])):
        k, d = t[t.outcome == "matched"], t[t.outcome == "deleted"]
        print(f"\n{name}: {len(t):,} boxes, {len(d)} deleted")
        for col in ("sam3_score", "sam3_best_iou", "sam2_score"):
            if col in t:
                print(f"  {col:14s} kept median {k[col].median():.3f} | deleted median {d[col].median():.3f} | AUC {auc(k[col], d[col]):.3f}")
        if "sam2_score" in t:
            comb = t.sam3_score.rank(pct=True) + t.sam2_score.rank(pct=True)
            print(f"  combined (rank sum)   AUC {auc(comb[t.outcome == 'matched'], comb[t.outcome == 'deleted']):.3f}")
    if "new_vs_ra_iou" in S:
        k = S[S.outcome == "matched"]; mv = k[k.qwen_vs_ra_iou < .9]
        print(f"\ncorrected box vs RA's final box (kept boxes): SAM 3 {k.new_vs_ra_iou.median():.3f} median "
              f"({(k.new_vs_ra_iou >= .5).mean():.1%} >= 0.5) vs Qwen {k.qwen_vs_ra_iou.median():.3f}")
        print(f"boxes the RA moved (IoU < 0.9, n={len(mv)}): SAM 3 {mv.new_vs_ra_iou.median():.3f} vs Qwen {mv.qwen_vs_ra_iou.median():.3f}; "
              f"SAM 3 box closer to the RA's in {(mv.new_vs_ra_iou > mv.qwen_vs_ra_iou).mean():.0%}; extends beyond Qwen's box in "
              f"{(S.beyond_box > 2).mean():.0%} of all boxes")
    renamed = S[(S.outcome == "matched") & (S.q_name != S.ra_name)]
    print(f"\nkept but renamed by the RA (n={len(renamed)}): SAM 3 score under Qwen's name, median {renamed.sam3_score.median():.3f} "
          f"(kept and not renamed: {S[(S.outcome == 'matched') & (S.q_name == S.ra_name)].sam3_score.median():.3f})")
    for thr in (0.1, 0.2, 0.3, 0.5):
        f = S.sam3_score < thr
        print(f"flag sam3_score < {thr}: {f.mean():.1%} flagged; catches {f[S.outcome == 'deleted'].mean():.1%} of deleted, {f[S.outcome == 'matched'].mean():.1%} of kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N frames (timing / smoke test)")
    ap.add_argument("--mode", choices=["crop", "frame"], default="crop")
    ap.add_argument("--prompt", choices=["label", "format"], default="label",
                    help="format: depictions prompted as depictions ('toy X', 'drawing of a X', 'X on a screen'); real objects as 'X'")
    ap.add_argument("--formats", nargs="*", help="only Qwen boxes with these formats (e.g. toy photo drawing screen)")
    args = ap.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        print(snapshot_download(MODEL, cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"])); return
    if args.summary:
        summarize(pd.read_csv(out_path(args.mode, args.prompt, args.formats))); return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import time
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    torch.set_num_threads(2)
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(MODEL, cache_dir=CACHE)
    model = Sam3Model.from_pretrained(MODEL, cache_dir=CACHE, dtype=torch.bfloat16).to(dev).eval()
    b = pd.read_csv(GOLD / "ra_scores" / "boxes.csv")
    q = b[b.outcome != "added"].copy()
    if args.formats:
        q = q[q.q_format.isin(args.formats)]
    fr = pd.read_csv(GOLD / "frames_qwen_t1.csv").drop_duplicates("frame_uid").set_index("frame_uid").frame_path
    frames = list(q.groupby("frame_uid"))[: args.limit or None]
    rows, t0 = [], time.time()
    for n, (uid, g) in enumerate(frames):
        im = Image.open(fr[uid]).convert("RGB")
        if args.mode == "crop":
            rows += crop_mode(uid, g, im, proc, model, dev, torch, args.prompt)
            if n % 25 == 0:
                print(f"  {n + 1}/{len(frames)} frames, {(time.time() - t0) / (n + 1):.2f} s/frame", flush=True)
            continue
        labels = sorted(set(g.q_name.fillna("object").astype(str)))
        inp = proc(images=[im] * len(labels), text=labels, return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(out, threshold=MIN_SCORE, mask_threshold=0.5,
                                                      target_sizes=[im.size[::-1]] * len(labels))
        inst = {lab: (r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()) for lab, r in zip(labels, res)}
        for idx, r in g.iterrows():
            boxes, scores = inst[str(r.q_name) if isinstance(r.q_name, str) else "object"]
            qb = [r.q_x0, r.q_y0, r.q_x1, r.q_y1]
            ious = iou_m([qb], boxes)[0] if len(boxes) else np.array([])
            ok = ious >= MATCH_IOU
            rows.append(dict(row=idx, frame_uid=uid, batch=r.batch, outcome=r.outcome, q_name=r.q_name, ra_name=r.ra_name,
                             sam3_score=float(scores[ok].max()) if ok.any() else 0.0,
                             sam3_best_iou=float(ious.max()) if len(ious) else 0.0, sam3_instances=len(boxes)))
        if n % 25 == 0:
            print(f"  {n + 1}/{len(frames)} frames, {(time.time() - t0) / (n + 1):.2f} s/frame", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out_path(args.mode, args.prompt, args.formats), index=False)
    summarize(S)


def crop_mode(uid, g, im, proc, model, dev, torch, style="label"):
    W, H = im.size
    items, crops, labels = [], [], []
    for idx, r in g.iterrows():
        qb = [r.q_x0, r.q_y0, r.q_x1, r.q_y1]
        L, T, R, B = padded(qb, W, H)
        if R - L < 2 or B - T < 2:
            continue
        items.append((idx, r, qb, L, T)); crops.append(im.crop((L, T, R, B)))
        labels.append(prompt_for(str(r.q_name) if isinstance(r.q_name, str) and r.q_name else "object", r.q_format, style))
    rows = []
    for s in range(0, len(crops), 16):                       # crops of one frame, 16 at a time
        inp = proc(images=crops[s:s + 16], text=labels[s:s + 16], return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(out, threshold=MIN_SCORE, mask_threshold=0.5,
                                                      target_sizes=[c.size[::-1] for c in crops[s:s + 16]])
        for (idx, r, qb, L, T), rr in zip(items[s:s + 16], res):
            boxes = rr["boxes"].float().cpu().numpy() + np.array([L, T, L, T]); scores = rr["scores"].float().cpu().numpy()
            cx, cy = (qb[0] + qb[2]) / 2, (qb[1] + qb[3]) / 2
            ious = iou_m([qb], boxes)[0] if len(boxes) else np.array([])
            hit = ((boxes[:, 0] <= cx) & (cx <= boxes[:, 2]) & (boxes[:, 1] <= cy) & (cy <= boxes[:, 3])) | (ious >= CROP_IOU) if len(boxes) else np.array([], bool)
            j = int(np.argmax(np.where(hit, scores, -1))) if hit.any() else -1
            nb = boxes[j] if j >= 0 else [np.nan] * 4
            ra = [r.ra_x0, r.ra_y0, r.ra_x1, r.ra_y1]
            rows.append(dict(row=idx, frame_uid=uid, batch=r.batch, outcome=r.outcome, q_name=r.q_name, ra_name=r.ra_name,
                             sam3_score=float(scores[j]) if j >= 0 else 0.0, sam3_instances=len(boxes),
                             sam3_best_iou=float(iou_m([qb], [nb])[0, 0]) if j >= 0 else 0.0,
                             new_x0=nb[0], new_y0=nb[1], new_x1=nb[2], new_y1=nb[3],
                             beyond_box=float(max(0, qb[0] - nb[0]) + max(0, nb[2] - qb[2]) + max(0, qb[1] - nb[1]) + max(0, nb[3] - qb[3])) if j >= 0 else 0.0,
                             qwen_vs_ra_iou=r.iou if r.outcome == "matched" else np.nan,
                             new_vs_ra_iou=float(iou_m([nb], [ra])[0, 0]) if (j >= 0 and r.outcome == "matched") else np.nan))
    return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
