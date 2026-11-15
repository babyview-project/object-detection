#!/usr/bin/env python3
"""Data for the context / geometry / crop-quality controls (comparison-datasets notebooks 12, 13, 15, 16) on the
Qwen open Tier 1 categories, BabyView and VEDB. Read by build_controls_nb.py -> controls_analysis.ipynb.

One pass over each Qwen box store (embed_full_run.py) sums L2-normalized CLIP and DINOv3 box vectors per
  geo    (unit, category, real, geometry cell)  cell = size sextile x vertical third x truncated (36 cells), and
                                                 short-side bin (<32, 32-63, >= 64 px) as a separate key
  room   (unit, category, real, room)           BabyView only; room / activity of the 10-s clip
  act    (unit, category, real, activity)       (2025.2 merged_contexts.csv; frame f belongs to clip f // 10)
Unit = BabyView child, VEDB session. Size sextiles: quantiles of log10(box area / frame area) over Tier 1 boxes, both
datasets with equal weight (comparison-datasets notebook 15). Truncated: box within 1% of a frame edge.
Frame co-occurrence (notebook 12): C[i, j] = frames containing both categories, per dataset, from the box index.

Outputs (out/private/controls[_renamed]/, private):
  geo_{bv,vedb}.npz, room_bv.npz, act_bv.npz   keys (unit, cat, real, code), n, sum_clip, sum_dinov3 (float32)
  cooc_{bv,vedb}.npz                           C (cat x cat frames), n_frames, categories
  meta.json                                    size edges, label lists

  QOV_LABELS=renamed nice -n 19 python build_controls_data.py [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
import variant as V
from build_open_means import basic_of, unit_of

STORES = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
          "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults")}
FRAME_WH = {"bv": (512, 910), "vedb": (2048, 1536)}
BV_CONTEXTS = Path("/ccn2a/dataset/babyview/2025.2/outputs/merged_contexts.csv")
OUT = V.out_dir(HERE / "out" / "private" / "controls")
EDGE_FRAC = 0.01
N_SIZE, N_VPOS = 6, 3
SIDE_EDGES = (32, 64)                    # short-side bins: < 32, 32-63, >= 64 px
ROOMS = ["living room", "kitchen", "dining room", "bedroom"]
ACTIVITIES = ["playing", "eating", "reading", "watching tv"]
CATS = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).index.tolist()
CIX = {c: i for i, c in enumerate(CATS)}


def geometry(bbox: np.ndarray, ds: str, size_edges: np.ndarray):
    W, H = FRAME_WH[ds]
    x0, y0, x1, y1 = bbox.T
    la = np.log10(np.maximum((x1 - x0) * (y1 - y0), 1) / (W * H))
    size = np.searchsorted(size_edges, la, side="right")                    # 0..5
    vpos = np.minimum(((y0 + y1) / 2 / H * N_VPOS).astype(int), N_VPOS - 1)
    trunc = ((x0 <= EDGE_FRAC * W) | (y0 <= EDGE_FRAC * H) | (x1 >= (1 - EDGE_FRAC) * W) | (y1 >= (1 - EDGE_FRAC) * H)).astype(int)
    side = np.searchsorted(SIDE_EDGES, np.minimum(x1 - x0, y1 - y0), side="right")   # 0..2
    return (size * N_VPOS + vpos) * 2 + trunc, side


def clip_labels():
    mc = pd.read_csv(BV_CONTEXTS, usecols=["video_id", "Location", "Activity"], low_memory=False)
    mc["vid"] = mc.video_id.astype(str).str.replace(r"_(\d+)\.mp4$", "", regex=True)
    mc["seg"] = mc.video_id.astype(str).str.extract(r"_(\d+)\.mp4$")[0].astype(int)
    out = {}
    for vid, g in mc.groupby("vid"):
        L = np.full(g.seg.max() + 1, len(ROOMS), np.int16); A = np.full_like(L, len(ACTIVITIES))   # last = other / unlabeled
        loc = g.Location.astype(str).str.strip().str.lower(); act = g.Activity.astype(str).str.strip().str.lower()
        L[g.seg.to_numpy()] = [ROOMS.index(x) if x in ROOMS else len(ROOMS) for x in loc]
        A[g.seg.to_numpy()] = [ACTIVITIES.index(x) if x in ACTIVITIES else len(ACTIVITIES) for x in act]
        out[vid] = (L, A)
    return out


def summed(keys: np.ndarray, X: dict) -> tuple:
    """Unique key rows and per-key sums of each model's vectors."""
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    S = {m: np.zeros((len(u), x.shape[1]), np.float64) for m, x in X.items()}
    for m, x in X.items():
        np.add.at(S[m], inv, x)
    return u, np.bincount(inv, minlength=len(u)), S


def per_video(args):
    stem, ds, cmap, labels, ctx, size_edges = args
    with np.load(STORES[ds] / "shards" / f"{stem}.npz", allow_pickle=True) as z:
        if len(z["image_id"]) == 0:
            return None
        lab = z["label"] if labels is None else labels
        ci = np.array([CIX.get(cmap.get(str(l), ""), -1) for l in lab])
        keep = ci >= 0
        if not keep.any():
            return None
        ci = ci[keep]
        real = (z["format"].astype(str)[keep] == "real").astype(int)
        X = {}
        for m, k in (("clip", "clip_emb"), ("dinov3", "dino_emb")):
            x = z[k][keep].astype(np.float32)
            X[m] = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        cell, side = geometry(z["bbox"][keep].astype(np.float64), ds, size_edges)
        frame = z["frame_number"][keep]
    out = {"unit": unit_of(stem), "geo": summed(np.c_[ci, real, cell], X), "side": summed(np.c_[ci, real, side], X)}
    if ds == "bv":
        L, A = ctx if ctx is not None else (np.array([], np.int16), np.array([], np.int16))
        seg = frame // 10
        loc = np.where(seg < len(L), L[np.minimum(seg, max(len(L) - 1, 0))] if len(L) else len(ROOMS), len(ROOMS))
        act = np.where(seg < len(A), A[np.minimum(seg, max(len(A) - 1, 0))] if len(A) else len(ACTIVITIES), len(ACTIVITIES))
        out["room"] = summed(np.c_[ci, real, loc], X)
        out["act"] = summed(np.c_[ci, real, act], X)
    return out


class Acc:
    """Running sums keyed by (unit, cat, real, code)."""
    def __init__(self):
        self.ix, self.n, self.S = {}, [], {"clip": [], "dinov3": []}

    def add(self, unit, res):
        u, n, S = res
        for r in range(len(u)):
            key = (unit, int(u[r, 0]), int(u[r, 1]), int(u[r, 2]))
            j = self.ix.get(key)
            if j is None:
                self.ix[key] = len(self.n); self.n.append(int(n[r]))
                for m in S:
                    self.S[m].append(S[m][r].astype(np.float64))
            else:
                self.n[j] += int(n[r])
                for m in S:
                    self.S[m][j] += S[m][r]

    def save(self, path):
        keys = list(self.ix)
        np.savez(path, unit=np.array([k[0] for k in keys]), cat=np.array([k[1] for k in keys], np.int32),
                 real=np.array([k[2] for k in keys], np.int8), code=np.array([k[3] for k in keys], np.int16),
                 n=np.array(self.n, np.int64), categories=np.array(CATS),
                 **{f"sum_{m}": np.array(self.S[m], np.float32) for m in self.S})


def read_index(ds):
    ix = V.read_index(STORES[ds], usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str)
    cmap = basic_of(ix.label.unique())
    ix["ci"] = ix.label.map(cmap).map(CIX).fillna(-1).astype(int)
    return ix, cmap


def cooccurrence(ix):
    t = ix[ix.ci >= 0]
    f = pd.factorize(t.video_id.astype(str) + "/" + t.frame_number.astype(str))[0]
    M = sparse.csr_matrix((np.ones(len(t), np.float32), (f, t.ci.to_numpy())), shape=(f.max() + 1, len(CATS)))
    M.data[:] = 1; M.sum_duplicates(); M.data[:] = 1
    n_frames = ix.video_id.astype(str).str.cat(ix.frame_number.astype(str), sep="/").nunique()
    return (M.T @ M).toarray().astype(np.int64), n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("labels:", V.LABELS, "->", OUT, flush=True)
    IX = {ds: read_index(ds) for ds in STORES}
    for ds, (ix, _) in IX.items():
        C, nf = cooccurrence(ix)
        np.savez(OUT / f"cooc_{ds}.npz", C=C, n_frames=nf, categories=np.array(CATS))
        print(f"{ds}: {len(ix):,} boxes, {(ix.ci >= 0).mean():.1%} in Tier 1; {nf:,} frames", flush=True)
    # size sextiles over Tier 1 boxes, both datasets weighted equally
    la, w = [], []
    for ds, (ix, _) in IX.items():
        t = ix[ix.ci >= 0]; W, H = FRAME_WH[ds]
        a = np.log10(np.maximum((t.xmax - t.xmin) * (t.ymax - t.ymin), 1) / (W * H)).to_numpy()
        la.append(a); w.append(np.full(len(a), 1 / len(a)))
    la, w = np.concatenate(la), np.concatenate(w)
    o = np.argsort(la); cw = np.cumsum(w[o]) / w.sum()
    size_edges = np.array([la[o][np.searchsorted(cw, q)] for q in np.arange(1, N_SIZE) / N_SIZE])
    json.dump({"size_edges_log10_area": size_edges.tolist(), "side_edges_px": list(SIDE_EDGES), "rooms": ROOMS + ["(other)"],
               "activities": ACTIVITIES + ["(other)"], "edge_frac": EDGE_FRAC, "n_size": N_SIZE, "n_vpos": N_VPOS,
               "frame_wh": FRAME_WH}, open(OUT / "meta.json", "w"), indent=1)
    print("size edges (log10 area):", np.round(size_edges, 3), flush=True)

    ctx = clip_labels()
    for ds, (ix, cmap) in IX.items():
        lab_by_video = {v: g.label.values for v, g in ix.groupby("video_id", sort=False)} if V.LABELS == "renamed" else {}
        del ix
        stems = sorted(p.stem for p in (STORES[ds] / "shards").glob("*.npz") if not p.name.endswith(".tmp.npz"))
        acc = {k: Acc() for k in (("geo", "side", "room", "act") if ds == "bv" else ("geo", "side"))}
        jobs = [(s, ds, cmap, lab_by_video.get(s), ctx.get(s) if ds == "bv" else None, size_edges) for s in stems]
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(per_video, jobs, chunksize=4)):
                if r is not None:
                    for k in acc:
                        acc[k].add(r["unit"], r[k])
                if i % 500 == 0:
                    print(f"  {ds} {i:,}/{len(stems):,} videos", flush=True)
        for k, a in acc.items():
            a.save(OUT / f"{k}_{ds}.npz")
            print(f"  {ds} {k}: {len(a.n):,} keys, {sum(a.n):,} boxes", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
