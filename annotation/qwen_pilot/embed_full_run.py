#!/usr/bin/env python3
"""CLIP + DINOv3 embeddings of every Qwen box in the 2025.2 full run, packed per video (no crop images saved).

Boxes: every object in the full run's records (setup B, temperature 1.0: run_fullrun.sh merged file + subject runs),
person included; repeat boxes inside a frame (same label, IoU > 0.7 with a box already kept: the near-duplicates
loops produce, as qwen_data.objects_table flags `dup`) are skipped. Labels are stored as Qwen wrote them (lowercased);
the open-category mapping (qwen_data.norm_label, domains) is applied at analysis time.

Models and readout as the 2025.2 YOLOE + CLIP store (image-embedding/run_cdi_crop_pipeline.py): CLIP ViT-B/32
get_image_features (512-d) and DINOv3 ViT-B/16 pooler_output (768-d), float16, not L2-normalized; clip_sim = cosine of
the L2-normalized image embedding and the text embedding of "a photo of a {label}" (Qwen's own label; a quality score,
not a filter). Crops are cut from the 512 x 910 frame at the box (pixel coordinates); re-cut them on demand with
qwen_store.crop().

Output (OUT_ROOT, default /data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1):
  shards/<video_id>.npz      image_id, label, format, frame_number, obj_idx, bbox (xmin, ymin, xmax, ymax px), clip_sim,
                             clip_emb (n, 512) f16, dino_emb (n, 768) f16
  boxes_index.csv.gz         every embedded box without embeddings (written by --index)
  run_config.json, README.txt

  CUDA_VISIBLE_DEVICES=4 python embed_full_run.py --worker 4/8        # one per GPU; resumable (skips finished videos)
  python embed_full_run.py --index                                   # after all workers: the index (CPU)

--dataset vedb: the same for the VEDB 1612 run (run_vedb.sh; every 5th 1 fps frame of 717 sessions, Qwen saw frames
resized to the BabyView pixel count). Boxes are on 0-1000, so crops are cut from the native 2048 x 1536 frames.
Store: /data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv (run_embed_vedb.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE.parent / "qwen_pilot" / "out" / "runs"
DATASETS = {
    "bv": dict(manifest=HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv",
               store="/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1",
               records=["qwen3.8-27b-fp8__detect_open_format__p1_fullrun2025.2_t1_pp0.jsonl",
                        "qwen3.8-27b-fp8__detect_open_format__p1_subj_*_t1_pp0.jsonl"],
               detector="Qwen3.8-27B-FP8, setup B (open label + format + box, <= 30 objects), temperature 1.0, 512x910 frames",
               frames="every 5th 1 fps frame of 2025.2 BV-main (S00270001 excluded)",
               readme="2025.2 BV-main, every 5th 1 fps frame"),
    "vedb": dict(manifest=HERE / "out" / "private" / "vedb_manifest.csv",
                 store="/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv",
                 records=["qwen3.8-27b-fp8__detect_open_format__p1_vedb1612_t1_pp0_pixbv.jsonl"],
                 detector="Qwen3.8-27B-FP8, setup B (open label + format + box, <= 30 objects), temperature 1.0, frames "
                          "resized to the BabyView pixel count (788x591, aspect kept); crops cut from the native 2048x1536 frames",
                 frames="every 5th 1 fps frame of the 717 VEDB 1612 world-camera sessions",
                 readme="VEDB 1612 world camera, 717 sessions, every 5th 1 fps frame"),
}
DS = DATASETS["bv"]
MANIFEST = DS["manifest"]
OUT_ROOT = Path(os.environ.get("QWEN_STORE", DS["store"]))
CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_HOME_2025_2 = "/ccn2/u/khaiaw/.cache/huggingface"
TEMPLATE = "a photo of a {label}"
DUP_IOU, MIN_SIDE = 0.7, 4


def record_files() -> list[Path]:
    return [p for pat in DS["records"] for p in sorted(PRIV.glob(pat))]


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def boxes_of(raw: str) -> list[tuple]:
    """(obj_idx, label, format, box_2d) kept after dropping repeat boxes; box_2d = [ymin, xmin, ymax, xmax] on 0-1000."""
    out, kept = [], []
    for i, o in enumerate(json.loads(raw).get("objects", [])):
        b = o.get("box_2d", [])
        if len(b) != 4:
            continue
        y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
        if x1 <= x0 or y1 <= y0:
            continue
        lab = str(o.get("label", "")).strip().lower()
        box = [y0, x0, y1, x1]
        if any(l == lab and iou(box, k) > DUP_IOU for l, k in kept):
            continue
        kept.append((lab, box))
        out.append((i, lab, str(o.get("format", "")), box))
    return out


def load_units(k: int, n: int) -> pd.DataFrame:
    """This worker's frames: videos assigned by position in the sorted video list (fixed split, k of n)."""
    m = pd.read_csv(MANIFEST, usecols=["unit_id", "video_id", "frame", "frame_path"])
    vids = sorted(m.video_id.unique())
    mine = set(vids[k::n])
    m = m[m.video_id.isin(mine)]
    want = set(m.unit_id)
    raw = {}
    for p in record_files():
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["unit_id"] in want and r.get("error") is None:
                    raw[r["unit_id"]] = r["raw"]
    m = m[m.unit_id.isin(raw)].copy()
    m["raw"] = m.unit_id.map(raw)
    return m


def worker(k: int, n: int, batch: int, threads: int, limit: int = 0):
    os.environ.setdefault("HF_HOME", HF_HOME_2025_2)
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, CLIPTokenizer
    torch.set_num_threads(2)
    dev = "cuda:0"
    (OUT_ROOT / "shards").mkdir(parents=True, exist_ok=True)
    units = load_units(k, n)
    vids = [v for v in sorted(units.video_id.unique()) if not (OUT_ROOT / "shards" / f"{v}.npz").exists()]
    if limit:
        vids = vids[:limit]
    print(f"worker {k}/{n}: {units.video_id.nunique()} videos, {len(vids)} to do, {len(units):,} frames", flush=True)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModel.from_pretrained(DINO_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    text_bank: dict[str, np.ndarray] = {}

    def unwrap(x):
        return x.pooler_output if hasattr(x, "pooler_output") else x

    def text_emb(labels):
        new = sorted({l for l in labels if l not in text_bank})
        for i in range(0, len(new), 512):
            t = tok([TEMPLATE.format(label=l) for l in new[i:i + 512]], padding=True, truncation=True, return_tensors="pt")
            with torch.inference_mode():
                e = unwrap(clip.get_text_features(**{a: b.to(dev) for a, b in t.items()}))
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
            text_bank.update(zip(new[i:i + 512], e))
        return np.stack([text_bank[l] for l in labels])

    def embed(crops):
        with torch.inference_mode():
            ci = {a: b.to(dev) for a, b in clip_proc(images=crops, return_tensors="pt").items()}
            c = unwrap(clip.get_image_features(**ci)).float()
            di = {a: b.to(dev) for a, b in dino_proc(images=crops, return_tensors="pt").items()}
            d = dino(**di).pooler_output.float()
        return c.cpu().numpy(), d.cpu().numpy()

    def load_frame(path):
        return np.asarray(Image.open(path).convert("RGB"))

    pool, prefetch = ThreadPoolExecutor(threads), ThreadPoolExecutor(1)   # decoding threads; one prefetcher
    t0, n_box = time.time(), 0
    by_vid = dict(tuple(units.groupby("video_id")))
    nxt = prefetch.submit(lambda v: list(pool.map(load_frame, by_vid[v].frame_path)), vids[0]) if vids else None
    for vi, v in enumerate(vids):
        frames = nxt.result()
        if vi + 1 < len(vids):
            nxt = prefetch.submit(lambda vv: list(pool.map(load_frame, by_vid[vv].frame_path)), vids[vi + 1])
        g = by_vid[v]
        meta, crops = [], []
        for im, r in zip(frames, g.itertuples()):
            h, w = im.shape[:2]
            for i, lab, fmt, (y0, x0, y1, x1) in boxes_of(r.raw):
                X0, X1 = int(x0 / 1000 * w), int(np.ceil(x1 / 1000 * w))
                Y0, Y1 = int(y0 / 1000 * h), int(np.ceil(y1 / 1000 * h))
                if X1 - X0 < MIN_SIDE or Y1 - Y0 < MIN_SIDE:
                    continue
                crops.append(im[Y0:Y1, X0:X1])
                meta.append((f"{v}_{r.frame:05d}_{i:02d}", lab, fmt, r.frame, i, X0, Y0, X1, Y1))
        if not meta:
            np.savez(OUT_ROOT / "shards" / f"{v}.npz", image_id=np.array([], dtype=object))
            continue
        C, D = [], []
        for s in range(0, len(crops), batch):
            c, d = embed(crops[s:s + batch])
            C.append(c); D.append(d)
        C, D = np.concatenate(C), np.concatenate(D)
        labels = [m_[1] for m_ in meta]
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        sim = (Cn * text_emb(labels)).sum(axis=1)
        tmp = OUT_ROOT / "shards" / f"{v}.tmp.npz"
        np.savez(tmp, image_id=np.array([m_[0] for m_ in meta], dtype=object), label=np.array(labels, dtype=object),
                 format=np.array([m_[2] for m_ in meta], dtype=object),
                 frame_number=np.array([m_[3] for m_ in meta], np.int32), obj_idx=np.array([m_[4] for m_ in meta], np.int16),
                 bbox=np.array([m_[5:9] for m_ in meta], np.float32), clip_sim=sim.astype(np.float32),
                 clip_emb=C.astype(np.float16), dino_emb=D.astype(np.float16))
        os.replace(tmp, OUT_ROOT / "shards" / f"{v}.npz")
        n_box += len(meta)
        if vi % 20 == 0 or vi == len(vids) - 1:
            el = time.time() - t0
            print(f"  {vi + 1}/{len(vids)} videos, {n_box:,} boxes, {n_box / el:.0f} boxes/s, {el / 60:.0f} min", flush=True)
    print(f"worker {k}/{n} done: {n_box:,} boxes in {(time.time() - t0) / 60:.0f} min", flush=True)


def build_index():
    rows = []
    for p in sorted((OUT_ROOT / "shards").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=True)
        if len(z["image_id"]) == 0:
            continue
        b = z["bbox"]
        rows.append(pd.DataFrame({"image_id": z["image_id"], "video_id": p.stem, "frame_number": z["frame_number"],
                                  "obj_idx": z["obj_idx"], "label": z["label"], "format": z["format"],
                                  "xmin": b[:, 0], "ymin": b[:, 1], "xmax": b[:, 2], "ymax": b[:, 3], "clip_sim": z["clip_sim"]}))
    ix = pd.concat(rows, ignore_index=True)
    ix.to_csv(OUT_ROOT / "boxes_index.csv.gz", index=False)
    print(f"index: {len(ix):,} boxes in {ix.video_id.nunique():,} videos; person {(ix.label == 'person').sum():,}")


def write_config():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "run_config.json").write_text(json.dumps({
        "detections": [str(p) for p in record_files()], "manifest": str(MANIFEST),
        "detector": DS["detector"], "frames": DS["frames"],
        "clip_model": CLIP_MODEL, "dino_model": DINO_MODEL, "clip_text_template": TEMPLATE,
        "dup_rule": f"same label and IoU > {DUP_IOU} with a box already kept in the frame: skipped",
        "min_side_px": MIN_SIDE, "person": "included", "crops_saved": False,
        "readout": {"clip_emb": "get_image_features, 512-d, float16, not L2-normalized",
                    "dino_emb": "pooler_output CLS, 768-d, float16, every box",
                    "clip_sim": "cosine of L2-normalized image and text ('a photo of a {label}') embeddings"}}, indent=2) + "\n")
    (OUT_ROOT / "README.txt").write_text(
        f"Qwen3.8-27B open-vocabulary detections ({DS['readme']}) with CLIP + DINOv3 embeddings of\n"
        "every box (person included, repeat boxes skipped). No crop images: re-cut a crop from the frame with\n"
        "annotation/qwen_pilot/qwen_store.py crop(). Built by annotation/qwen_pilot/embed_full_run.py (object-detection repo);\n"
        "settings in run_config.json; boxes_index.csv.gz lists every box without embeddings.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="bv")
    ap.add_argument("--worker", default="", help="K/N: embed every N-th video from K (one process per GPU)")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4, help="frame-decoding threads per worker")
    ap.add_argument("--limit-videos", type=int, default=0, help="check run: first N videos only")
    args = ap.parse_args()
    global DS, MANIFEST, OUT_ROOT
    DS = DATASETS[args.dataset]
    MANIFEST = DS["manifest"]
    OUT_ROOT = Path(os.environ.get("QWEN_STORE", DS["store"]))
    if args.index:
        build_index(); return
    write_config()
    k, n = map(int, args.worker.split("/"))
    worker(k, n, args.batch, args.threads, args.limit_videos)


if __name__ == "__main__":
    main()
