#!/usr/bin/env python3
"""SAM 3 on padded crops, prompted with the box's category: corrected boxes and object masks for the BabyView / VEDB
embedding sample (analysis/qwen-open-vocab/build_bv_vedb_data.py, renamed labels; up to 150 boxes per category and age
bin, 513k boxes).

Per box: the square crop Qwen saw in the renaming pass (box + 25% on each side), prompted with the category name. The
target is the highest-scoring SAM 3 instance that contains the box centre or overlaps the box (IoU >= 0.3); its box can
extend beyond Qwen's box anywhere within the padding (objects Qwen's box cut off). On the RA gold set this finds the
object for 91-96% of kept toy / picture / screen boxes and moves boxes the RAs had moved closer to the RAs' boxes
(sam3_box_check.py --mode crop). Plain category prompts: depiction prompts ("toy X") did not help.

Output (private; local NVMe /data2, not the shared /ccn2a or the near-full home disk):
  /data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb/w<k>_<chunk>.npz, append-only chunks
  idx (row in the sample), score (0 if no instance), n (instances), box (corrected box, frame px; NaN if none),
  crop (the padded crop, frame px), mask (object mask inside `crop`, resized to <= 256 px on the long side,
  np.packbits of the flattened bool array), mask_shape (h, w)
Resumable: rows already in a chunk file are skipped. I/O kept light: frames are only read from /ccn2a (2 loader
threads per worker), chunks are written once and never rewritten; run under ionice -c3 (run_sam3_crop_sample.sh).
  CUDA_VISIBLE_DEVICES=0 python sam3_crop_sample.py --worker 0/8      # qwen env
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parents[1] / "analysis" / "qwen-open-vocab" / "out" / "private" / "bv_vedb_renamed"
OUT = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb")
MODEL, CACHE = "facebook/sam3", "/data/j7yang/models/hf/hub"
BV_FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
VEDB_FRAMES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps")
PAD, HIT_IOU, MIN_SCORE, BATCH, MASK_SIDE, SAVE_EVERY, THREADS = 0.25, 0.3, 0.05, 16, 256, 4096, 2
FIELDS = ("idx", "score", "n", "box", "crop", "mask", "mask_shape")


def frame_path(video_id, frame):
    return (VEDB_FRAMES / video_id / f"{frame:08d}.jpg") if video_id.startswith("session_") else (BV_FRAMES / video_id / f"{frame:05d}.jpg")


def padded(bx, W, H):
    x0, y0, x1, y1 = bx
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L, T = max(0, cx - side / 2), max(0, cy - side / 2)
    return int(L), int(T), int(min(W, cx + side / 2) + .999), int(min(H, cy + side / 2) + .999)


def iou_m(A, B):
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    i = ix * iy; ar = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return i / np.maximum(ar(A)[:, None] + ar(B)[None, :] - i, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    k, n = map(int, args.worker.split("/"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(SAMPLE / "sample_clip.npz", allow_pickle=True)
    vid, fno, bbox, cat = z["video_id"], z["frame_number"], z["bbox"], z["category"]
    chunks = sorted(OUT.glob(f"w{k}_*.npz"))
    done = {int(i) for c in chunks for i in np.load(c, allow_pickle=True)["idx"]}
    keep = {f: [] for f in FIELDS}
    n_chunk = len(chunks)
    rows = [i for i in range(k, len(vid), n) if i not in done]
    rows.sort(key=lambda i: (vid[i], fno[i]))                 # frames in order: each frame read once
    if args.limit:
        rows = rows[:args.limit]
    print(f"worker {k}/{n}: {len(rows):,} boxes to do ({len(done):,} done)", flush=True)
    dev = "cuda:0"
    proc = Sam3Processor.from_pretrained(MODEL, cache_dir=CACHE)
    model = Sam3Model.from_pretrained(MODEL, cache_dir=CACHE, dtype=torch.bfloat16).to(dev).eval()

    def load(i):
        im = Image.open(frame_path(str(vid[i]), int(fno[i]))).convert("RGB")
        L, T, R, B = padded(bbox[i], *im.size)
        return i, im.crop((L, T, max(R, L + 2), max(B, T + 2))), (L, T, max(R, L + 2), max(B, T + 2))

    def save():
        nonlocal n_chunk
        if not keep["idx"]:
            return
        part = OUT / f"w{k}_{n_chunk:04d}.npz"; n_chunk += 1
        tmp = part.with_name(part.stem + ".tmp.npz")
        np.savez(tmp, idx=np.array(keep["idx"]), score=np.array(keep["score"], np.float32), n=np.array(keep["n"], np.int16),
                 box=np.array(keep["box"], np.float32).reshape(-1, 4), crop=np.array(keep["crop"], np.int32).reshape(-1, 4),
                 mask=np.array(keep["mask"], dtype=object), mask_shape=np.array(keep["mask_shape"], np.int16).reshape(-1, 2))
        os.replace(tmp, part)
        for f in FIELDS:
            keep[f].clear()

    pool, pre = ThreadPoolExecutor(THREADS), ThreadPoolExecutor(1); t0 = time.time(); since = 0
    batches = [rows[s:s + BATCH] for s in range(0, len(rows), BATCH)]
    nxt = pre.submit(lambda b: list(pool.map(load, b)), batches[0]) if batches else None
    for bi, b in enumerate(batches):
        items = nxt.result()
        if bi + 1 < len(batches):
            nxt = pre.submit(lambda bb: list(pool.map(load, bb)), batches[bi + 1])
        crops = [c for _, c, _ in items]
        inp = proc(images=crops, text=[str(cat[i]) for i, _, _ in items], return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = model(**inp)
        res = proc.post_process_instance_segmentation(out, threshold=MIN_SCORE, mask_threshold=0.5, target_sizes=[c.size[::-1] for c in crops])
        for (i, c, (L, T, R, B)), r in zip(items, res):
            boxes = r["boxes"].float().cpu().numpy() + np.array([L, T, L, T]); scores = r["scores"].float().cpu().numpy()
            qb = bbox[i]; cx, cy = (qb[0] + qb[2]) / 2, (qb[1] + qb[3]) / 2
            if len(boxes):
                ious = iou_m([qb], boxes)[0]
                hit = ((boxes[:, 0] <= cx) & (cx <= boxes[:, 2]) & (boxes[:, 1] <= cy) & (cy <= boxes[:, 3])) | (ious >= HIT_IOU)
            else:
                hit = np.array([], bool)
            if hit.any():
                j = int(np.argmax(np.where(hit, scores, -1)))
                m = r["masks"][j].bool().cpu().numpy()
                h, w = m.shape; s = min(1.0, MASK_SIDE / max(h, w))
                if s < 1:
                    m = np.asarray(Image.fromarray(m).resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST))
                keep["score"].append(float(scores[j])); keep["box"].append(boxes[j]); keep["mask"].append(np.packbits(m.ravel()))
                keep["mask_shape"].append(m.shape)
            else:
                keep["score"].append(0.0); keep["box"].append([np.nan] * 4); keep["mask"].append(np.zeros(0, np.uint8)); keep["mask_shape"].append((0, 0))
            keep["idx"].append(i); keep["n"].append(len(boxes)); keep["crop"].append((L, T, R, B))
        since += len(items)
        if since >= SAVE_EVERY or bi == len(batches) - 1:
            save(); since = 0
            el = time.time() - t0; nd = (bi + 1) * BATCH
            print(f"  {min(nd, len(rows)):,}/{len(rows):,} boxes, {nd / el:.1f} boxes/s, {el / 3600:.2f} h", flush=True)
    print(f"worker {k}/{n} done", flush=True)


if __name__ == "__main__":
    main()
