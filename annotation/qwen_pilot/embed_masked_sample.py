#!/usr/bin/env python3
"""CLIP + DINOv3 embeddings of the BV / VEDB sample boxes after SAM 3 (sam3_crop_sample.py), two new versions per box:
  box     the SAM 3 corrected box (can extend beyond Qwen's box into the padding), background kept
  masked  the same crop with every pixel outside SAM 3's object mask set to mid-grey (background removed)
Boxes where SAM 3 found no instance get no new embedding (NaN). The original-box embeddings are the sample's own
(build_bv_vedb_data.py). Same models and readout as embed_full_run.py.

Output: /data2/.../sam3_crop_sample_bv_vedb/embeddings_{clip,dinov3}.npz  idx, box (n, d) f16, masked (n, d) f16
I/O: frames read once each from /ccn2a under ionice -c3 (2 loader threads per worker); output written once at the end.
  CUDA_VISIBLE_DEVICES=0 python embed_masked_sample.py --worker 0/8 ; then --merge 8      # qwen env
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parents[1] / "analysis" / "qwen-open-vocab" / "out" / "private" / "bv_vedb_renamed" / "sample_clip.npz"
D = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb")
CLIP_MODEL, DINO_MODEL = "openai/clip-vit-base-patch32", "facebook/dinov3-vitb16-pretrain-lvd1689m"
GREY, BATCH, THREADS, MIN_SIDE = 128, 128, 2, 4


def sam_results():
    parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(D / "w*_*.npz")))]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return {"idx": cat("idx"), "score": cat("score"), "box": cat("box"), "crop": cat("crop"),
            "mask": np.concatenate([p["mask"] for p in parts]), "mask_shape": cat("mask_shape")}


def make_pair(S, vid, fno, r):
    """(box crop, background-removed crop) for SAM result row r, or None if the corrected box is tiny."""
    from PIL import Image
    i = S["idx"][r]; v = str(vid[i])
    p = (f"/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps/{v}/{int(fno[i]):08d}.jpg" if v.startswith("session_")
         else f"/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/{v}/{int(fno[i]):05d}.jpg")
    im = np.asarray(Image.open(p).convert("RGB")); H, W = im.shape[:2]
    L, T, R, B = (int(v) for v in S["crop"][r]); h, w = (int(v) for v in S["mask_shape"][r])
    m = np.unpackbits(S["mask"][r])[: h * w].reshape(h, w).astype(bool)
    m = np.asarray(Image.fromarray(m).resize((max(1, R - L), max(1, B - T)), Image.NEAREST))
    full = np.zeros((H, W), bool); full[T:T + m.shape[0], L:L + m.shape[1]] = m[: H - T, : W - L]
    x0, y0, x1, y1 = S["box"][r]
    X0, Y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0))); X1, Y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    if X1 - X0 < MIN_SIDE or Y1 - Y0 < MIN_SIDE:
        return None
    c = im[Y0:Y1, X0:X1]; mc = full[Y0:Y1, X0:X1]
    return c, np.where(mc[..., None], c, GREY).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--merge", type=int, default=0)
    args = ap.parse_args()
    if args.merge:
        for m in ("clip", "dinov3"):
            ps = [np.load(D / f"emb_{m}_w{k}.npz") for k in range(args.merge)]
            np.savez(D / f"embeddings_{m}.npz", **{f: np.concatenate([p[f] for p in ps]) for f in ("idx", "box", "masked")})
            for k in range(args.merge):
                os.remove(D / f"emb_{m}_w{k}.npz")
        print("merged"); return
    k, n = map(int, args.worker.split("/"))
    os.environ.setdefault("HF_HOME", "/ccn2/u/khaiaw/.cache/huggingface")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(2)
    z = np.load(SAMPLE, allow_pickle=True); vid, fno = z["video_id"], z["frame_number"]
    S = sam_results()
    rows = np.flatnonzero(S["score"] > 0)
    rows = rows[np.argsort([f"{vid[S['idx'][r]]}/{fno[S['idx'][r]]:08d}" for r in rows], kind="stable")][k::n]
    print(f"worker {k}/{n}: {len(rows):,} boxes", flush=True)
    dev = "cuda:0"
    cp = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    dp = AutoImageProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModel.from_pretrained(DINO_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    make = lambda r: make_pair(S, vid, fno, r)

    def embed(crops):
        with torch.inference_mode():
            ci = {a: b.to(dev) for a, b in cp(images=crops, return_tensors="pt").items()}
            c = clip.get_image_features(**ci); c = (c.pooler_output if hasattr(c, "pooler_output") else c).float()
            d = dino(**{a: b.to(dev) for a, b in dp(images=crops, return_tensors="pt").items()}).pooler_output.float()
        return c.cpu().numpy(), d.cpu().numpy()

    out = {m: {"box": np.full((len(rows), dm), np.nan, np.float16), "masked": np.full((len(rows), dm), np.nan, np.float16)}
           for m, dm in (("clip", 512), ("dinov3", 768))}
    pool, pre = ThreadPoolExecutor(THREADS), ThreadPoolExecutor(1); t0 = time.time()
    batches = [list(range(s, min(s + BATCH, len(rows)))) for s in range(0, len(rows), BATCH)]
    nxt = pre.submit(lambda b: list(pool.map(lambda j: make(rows[j]), b)), batches[0]) if batches else None
    for bi, b in enumerate(batches):
        made = nxt.result()
        if bi + 1 < len(batches):
            nxt = pre.submit(lambda bb: list(pool.map(lambda j: make(rows[j]), bb)), batches[bi + 1])
        ok = [(j, x) for j, x in zip(b, made) if x is not None]
        if not ok:
            continue
        for ver, slot in (("box", 0), ("masked", 1)):
            c, d = embed([x[slot] for _, x in ok])
            out["clip"][ver][[j for j, _ in ok]] = c; out["dinov3"][ver][[j for j, _ in ok]] = d
        if bi % 50 == 0:
            el = time.time() - t0
            print(f"  {(bi + 1) * BATCH:,}/{len(rows):,} boxes, {(bi + 1) * BATCH / el:.0f} boxes/s", flush=True)
    for m in out:
        np.savez(D / f"emb_{m}_w{k}.npz", idx=S["idx"][rows], box=out[m]["box"], masked=out[m]["masked"])
    print(f"worker {k}/{n} done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
