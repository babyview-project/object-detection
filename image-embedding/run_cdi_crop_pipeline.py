#!/usr/bin/env python3
"""2025.2 CDI crop pipeline: CLIP embed → CLIP-text filter → annotated frames → DINOv3.

Designed for ~20M detections on NFS, where the 868-hour layout (one crop JPG + one
``.npy`` per detection) would blow inode / metadata budgets.

What this does *not* write
--------------------------
- Per-crop JPEG files
- Per-crop ``.npy`` files
- Annotated frames with *all* YOLO boxes (those already live under object_detections/cdi)

What it writes
--------------
``{emb_root}/shards/{video_id}.npz``
    CLIP vector for every scored detection; DINOv3 CLS only for CLIP-passers.
    See ``crop_embedding_store.py``.
``{annotated_root}/{video_id}.tar``  (default)
    JPEGs of source 1fps frames with *only* boxes that passed CLIP, packed one
    tar per video (~3k inodes instead of ~1M JPEGs). ``--annotated-format jpeg``
    writes a tree parallel to ``extracted_frames_1fps`` instead.
``{emb_root}/crops_index.csv.gz``, filter-list txt, ``run_config.json``

Single GPU pass per video: prefetch 1fps JPEGs, crop in memory, CLIP-score,
draw passers on a side thread, DINOv3-embed passers. Resume is per-video
(``.done`` markers).

Manuscript checkpoints
----------------------
CLIP    ``openai/clip-vit-base-patch32``     512-d ``get_image_features``
DINOv3  ``facebook/dinov3-vitb16-pretrain-lvd1689m``  768-d CLS ``pooler_output``
Filter  cosine(L2(image), L2(text)) >= 0.27 with template ``a photo of a {label}``
Drop    ``person``, ``picture`` (same as the 868-hour ``exclude-people`` lists)

CPU scan (no GPU)::

    python run_cdi_crop_pipeline.py --stage scan

GPU embed, one process per GPU (see ``dispatch_cdi_crop_pipeline.sh``)::

    CUDA_VISIBLE_DEVICES=0 python -u run_cdi_crop_pipeline.py --stage embed --shard 0/4

Then::

    python run_cdi_crop_pipeline.py --stage index
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import sys
import tarfile
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from crop_embedding_store import (
    CLIP_DIM,
    DINO_DIM,
    save_video_shard,
    shard_path,
    write_run_config,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FRAMES = "/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps"
DEFAULT_CDI = "/ccn2a/dataset/babyview/2025.2/outputs/object_detections/cdi"
DEFAULT_EMB = "/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full"
DEFAULT_ANN = "/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps_annotated"
CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_EXCLUDE_CATS = ("person", "picture")
KHAIAW_HF_HOME = "/ccn2/u/khaiaw/.cache/huggingface"

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID129_PATH = REPO_ROOT / "data" / "included_categories_valid129.txt"


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["scan", "embed", "index", "all"], default="embed")
    p.add_argument("--frames-root", default=DEFAULT_FRAMES)
    p.add_argument("--cdi-root", default=DEFAULT_CDI)
    p.add_argument("--emb-root", default=DEFAULT_EMB)
    p.add_argument("--annotated-root", default=DEFAULT_ANN)
    p.add_argument("--clip-model", default=CLIP_MODEL)
    p.add_argument("--dino-model", default=DINO_MODEL)
    p.add_argument("--clip-threshold", type=float, default=0.27)
    p.add_argument(
        "--clip-text-template",
        default="a photo of a {label}",
        help="CLIP zero-shot template. `{label}` is replaced by the class name. "
             "Cosine is L2(image_embeds) · L2(text_embeds).",
    )
    p.add_argument("--min-yolo-conf", type=float, default=0.0)
    p.add_argument("--exclude-categories", nargs="*", default=list(DEFAULT_EXCLUDE_CATS))
    p.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["00270001"],
        help="Drop videos whose id contains these subject tokens (868-hour filter list "
             "excluded 00270001). Pass none to keep everyone: `--exclude-subjects`.",
    )
    p.add_argument(
        "--categories-file",
        default=None,
        help="If set, only embed classes listed in this file (e.g. included_categories_valid129.txt). "
             "Default: every CDI class except --exclude-categories.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--prefetch-frames",
        type=int,
        default=8,
        help="Outstanding NFS JPEG decodes (thread pool). 1 disables prefetch.",
    )
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument(
        "--annotated-format",
        choices=["tar", "jpeg", "none"],
        default="tar",
        help="tar (default, one archive per video) | jpeg (loose tree) | none (skip drawing).",
    )
    p.add_argument("--shard", default=None, help="i/N, e.g. 0/4. Embed only videos where index %% N == i.")
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=16, help="CPU workers for --stage scan.")
    p.add_argument("--hf-home", default=None, help="HuggingFace cache. Default: khai's if local dinov3 missing.")
    p.add_argument("--overwrite", action="store_true", help="Redo videos that already have a .done marker.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Paths / video lists
# ---------------------------------------------------------------------------
def list_video_dirs(cdi_root: Path) -> list[Path]:
    return sorted(p for p in cdi_root.iterdir() if p.is_dir())


def parse_shard(spec: str | None, n: int) -> tuple[int, int, list[int]]:
    if not spec:
        return 0, 1, list(range(n))
    i_str, n_str = spec.split("/", 1)
    i, n_shards = int(i_str), int(n_str)
    if not (0 <= i < n_shards):
        raise SystemExit(f"--shard {spec} out of range")
    idx = [k for k in range(n) if k % n_shards == i]
    return i, n_shards, idx


def subject_token(video_id: str) -> str:
    # S00xxxxxx_YYYY-MM-DD_...  or  00xxxxxx_YYYY-...
    head = video_id.split("_", 1)[0]
    return head[1:] if head.startswith("S") and len(head) > 1 else head


def should_drop_video(video_id: str, exclude_subjects: list[str]) -> bool:
    if not exclude_subjects:
        return False
    tok = subject_token(video_id)
    return any(tok == s or s in video_id for s in exclude_subjects)


def frame_jpg(frames_root: Path, video_id: str, frame_number: int) -> Path | None:
    d = frames_root / video_id
    for name in (f"{frame_number:05d}.jpg", f"{frame_number:08d}.jpg", f"{frame_number}.jpg"):
        p = d / name
        if p.is_file():
            return p
    return None


def status_paths(emb_root: Path, video_id: str) -> tuple[Path, Path]:
    status_dir = emb_root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir / f"{video_id}.done", status_dir / f"{video_id}.failed"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Scan (CPU)
# ---------------------------------------------------------------------------
def _scan_one_video(args_tuple):
    video_dir, min_yolo_conf, exclude_cats, exclude_subjects, allowed = args_tuple
    video_dir = Path(video_dir)
    video_id = video_dir.name
    csv_path = video_dir / "bounding_box_predictions.csv"
    out = {
        "video_id": video_id,
        "has_csv": csv_path.is_file(),
        "dropped_subject": should_drop_video(video_id, exclude_subjects),
        "n_csv_rows": 0,
        "n_boxes": 0,
        "n_scored": 0,
        "n_frames": 0,
        "n_excluded_cat": 0,
        "classes": Counter(),
    }
    if not csv_path.is_file() or out["dropped_subject"]:
        return out
    try:
        df = pd.read_csv(csv_path, usecols=["frame_number", "xmin", "class_name", "confidence"])
    except Exception as e:
        out["error"] = str(e)
        return out
    out["n_csv_rows"] = int(len(df))
    df = df[df["xmin"].notna()]
    if min_yolo_conf > 0:
        df = df[df["confidence"] >= min_yolo_conf]
    out["n_boxes"] = int(len(df))
    if df.empty:
        return out
    names = df["class_name"].astype(str).str.strip().str.lower()
    excl = names.isin(exclude_cats)
    out["n_excluded_cat"] = int(excl.sum())
    keep = ~excl
    if allowed is not None:
        keep &= names.isin(allowed)
    scored = df[keep]
    out["n_scored"] = int(len(scored))
    out["n_frames"] = int(scored["frame_number"].nunique()) if len(scored) else 0
    out["classes"] = Counter(names[keep].tolist())
    return out


def stage_scan(args) -> None:
    cdi_root = Path(args.cdi_root)
    emb_root = Path(args.emb_root)
    emb_root.mkdir(parents=True, exist_ok=True)
    exclude_cats = {c.strip().lower() for c in args.exclude_categories}
    allowed = None
    if args.categories_file:
        allowed = {ln.strip().lower() for ln in Path(args.categories_file).read_text().splitlines() if ln.strip()}
        print(f"Restricting to {len(allowed)} classes from {args.categories_file}")

    video_dirs = list_video_dirs(cdi_root)
    if args.limit_videos:
        video_dirs = video_dirs[: args.limit_videos]
    print(f"Scanning {len(video_dirs)} video dirs under {cdi_root}")

    tasks = [
        (str(d), args.min_yolo_conf, exclude_cats, list(args.exclude_subjects), allowed)
        for d in video_dirs
    ]
    totals = Counter()
    class_counts = Counter()
    n_dropped_subj = 0
    n_err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(_scan_one_video, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r.get("error"):
                n_err += 1
            if r["dropped_subject"]:
                n_dropped_subj += 1
            for k in ("n_csv_rows", "n_boxes", "n_scored", "n_frames"):
                totals[k] += r[k]
            class_counts.update(r["classes"])
            if done % 200 == 0 or done == len(tasks):
                print(f"  scanned {done}/{len(tasks)}  scored_boxes={totals['n_scored']:,}  "
                      f"frames_with_boxes={totals['n_frames']:,}", flush=True)

    stats = {
        "n_videos": len(video_dirs),
        "n_dropped_subject": n_dropped_subj,
        "n_csv_errors": n_err,
        "n_csv_rows": totals["n_csv_rows"],
        "n_boxes": totals["n_boxes"],
        "n_scored": totals["n_scored"],
        "n_frames_with_scored_boxes": totals["n_frames"],
        "exclude_categories": sorted(exclude_cats),
        "exclude_subjects": list(args.exclude_subjects),
        "min_yolo_conf": args.min_yolo_conf,
        "clip_threshold": args.clip_threshold,
        "clip_text_template": args.clip_text_template,
        "elapsed_s": round(time.time() - t0, 1),
        "top_classes": class_counts.most_common(30),
        "n_classes": len(class_counts),
        "est_clip_gb": round(totals["n_scored"] * CLIP_DIM * 2 / 1e9, 2),
        "est_dino_gb_if_30pct_pass": round(0.30 * totals["n_scored"] * DINO_DIM * 2 / 1e9, 2),
    }
    out = emb_root / "scan_stats.json"
    write_json(out, stats)
    print(json.dumps(stats, indent=2))
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def resolve_hf_home(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    user_dino = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m"
    if user_dino.is_dir():
        return str(Path.home() / ".cache/huggingface")
    if Path(KHAIAW_HF_HOME).is_dir():
        return KHAIAW_HF_HOME
    return str(Path.home() / ".cache/huggingface")


def _unwrap_embeds(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


class CropModels:
    def __init__(self, clip_name: str, dino_name: str, text_template: str, class_names: list[str],
                 device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.text_template = text_template

        print(f"Loading CLIP {clip_name} on {device} ...", flush=True)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(
            clip_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()

        print(f"Loading DINOv3 {dino_name} on {device} ...", flush=True)
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        self.dino = AutoModel.from_pretrained(
            dino_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()

        self.text_bank = self._encode_texts(class_names)
        print(f"CLIP text bank: {len(self.text_bank)} classes, template={text_template!r}", flush=True)

    def _encode_texts(self, class_names: list[str]) -> dict[str, np.ndarray]:
        torch = self.torch
        names = sorted({c.strip().lower() for c in class_names if c.strip()})
        if not names:
            return {}
        prompts = [self.text_template.format(label=n) for n in names]
        # CLIPProcessor in recent transformers wants `text=` only (no dummy images).
        try:
            inputs = self.clip_proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
        except TypeError:
            inputs = self.clip_proc(
                text=prompts, images=None, return_tensors="pt", padding=True, truncation=True
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            out = _unwrap_embeds(self.clip.get_text_features(**inputs))
            vecs = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
        return {n: vecs[i] for i, n in enumerate(names)}

    def ensure_text(self, class_name: str) -> np.ndarray:
        c = class_name.strip().lower()
        if c not in self.text_bank:
            self.text_bank.update(self._encode_texts([c]))
        return self.text_bank[c]

    def _batch_to_device(self, processor, rgb_crops: list[np.ndarray]) -> dict:
        torch = self.torch
        inputs = processor(images=rgb_crops, return_tensors="pt")
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                out[k] = v.pin_memory().to(self.device, non_blocking=True)
        return out

    def encode_clip(self, rgb_crops: list[np.ndarray]) -> np.ndarray:
        """rgb_crops: list of HxWx3 uint8. Returns float32 (B, 512) raw (not L2-normed)."""
        torch = self.torch
        inputs = self._batch_to_device(self.clip_proc, rgb_crops)
        with torch.inference_mode():
            out = _unwrap_embeds(self.clip.get_image_features(**inputs))
            return out.float().cpu().numpy()

    def encode_dino(self, rgb_crops: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        inputs = self._batch_to_device(self.dino_proc, rgb_crops)
        with torch.inference_mode():
            out = self.dino(**inputs)
            return out.pooler_output.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Crops / drawing
# ---------------------------------------------------------------------------
def clamp_box(xmin, ymin, xmax, ymax, w, h) -> tuple[int, int, int, int] | None:
    x0, x1 = sorted((max(0, min(w, float(xmin))), max(0, min(w, float(xmax)))))
    y0, y1 = sorted((max(0, min(h, float(ymin))), max(0, min(h, float(ymax)))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _imread_rgb(path_str: str) -> np.ndarray | None:
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_rgb(path: Path) -> np.ndarray | None:
    return _imread_rgb(str(path))


def iter_prefetched_frames(jobs, pool: ThreadPoolExecutor | None, prefetch: int):
    """Yield (frame_number, group, rgb_or_none). `jobs` is (frame, group, path_or_none)."""
    it = iter(jobs)
    if pool is None or prefetch <= 1:
        for frame_number, g, path in it:
            rgb = _imread_rgb(str(path)) if path is not None else None
            yield frame_number, g, rgb
        return
    pending: deque = deque()

    def submit(item):
        frame_number, g, path = item
        if path is None:
            fut = pool.submit(lambda: None)
        else:
            fut = pool.submit(_imread_rgb, str(path))
        pending.append((frame_number, g, fut))

    try:
        for _ in range(prefetch):
            try:
                submit(next(it))
            except StopIteration:
                break
        while pending:
            frame_number, g, fut = pending.popleft()
            try:
                submit(next(it))
            except StopIteration:
                pass
            yield frame_number, g, fut.result()
    finally:
        while pending:
            pending.popleft()[2].result()


def class_color(name: str) -> tuple[int, int, int]:
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def draw_passers(rgb: np.ndarray, rows: list[dict], jpeg_quality: int) -> bytes:
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for r in rows:
        x0, y0, x1, y1 = r["box"]
        col = class_color(r["class_name"])
        draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f"{r['class_name']} {r['clip_sim']:.2f}"
        ty = max(0, y0 - 14)
        draw.rectangle([x0, ty, x0 + 8 * len(label), ty + 14], fill=col)
        draw.text((x0 + 2, ty), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    return buf.getvalue()


class AnnotateWriter:
    """JPEG-encode + tar-add off the GPU thread. tarfile is used only here."""

    def __init__(self, tar, jpeg_dir: Path, annotated_format: str, jpeg_quality: int, stats: dict):
        self.tar = tar
        self.jpeg_dir = jpeg_dir
        self.fmt = annotated_format
        self.jpeg_quality = jpeg_quality
        self.stats = stats
        self.err: BaseException | None = None
        self.queue: queue.Queue = queue.Queue(maxsize=16)
        self.thread: threading.Thread | None = None
        self._closed = False
        if annotated_format != "none":
            self.thread = threading.Thread(target=self._run, name="annotate", daemon=True)
            self.thread.start()

    def submit(self, rgb: np.ndarray, passer_rows: list[dict], fname: str) -> None:
        if self.thread is None:
            return
        self.queue.put(("job", rgb, passer_rows, fname))

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item[0] == "stop":
                    return
                _, rgb, passer_rows, fname = item
                jpg = draw_passers(rgb, passer_rows, self.jpeg_quality)
                if self.tar is not None:
                    info = tarfile.TarInfo(name=fname)
                    info.size = len(jpg)
                    self.tar.addfile(info, io.BytesIO(jpg))
                elif self.fmt == "jpeg":
                    self.jpeg_dir.mkdir(parents=True, exist_ok=True)
                    (self.jpeg_dir / fname).write_bytes(jpg)
                self.stats["n_frames_annotated"] += 1
        except BaseException as e:
            self.err = e

    def close(self) -> None:
        if self.thread is None or self._closed:
            if self.err is not None:
                raise self.err
            return
        self._closed = True
        self.queue.put(("stop",))
        self.thread.join()
        if self.err is not None:
            raise self.err


def crop_rgb(rgb: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Per-video embed
# ---------------------------------------------------------------------------
def detections_for_video(
    video_dir: Path,
    min_yolo_conf: float,
    exclude_cats: set[str],
    allowed: set[str] | None,
) -> pd.DataFrame:
    csv_path = video_dir / "bounding_box_predictions.csv"
    if not csv_path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "xmin" not in df.columns:
        return pd.DataFrame()
    df = df[df["xmin"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    if min_yolo_conf > 0:
        df = df[df["confidence"] >= min_yolo_conf]
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df = df[~df["class_name"].isin(exclude_cats)]
    if allowed is not None:
        df = df[df["class_name"].isin(allowed)]
    df["frame_number"] = df["frame_number"].astype(int)
    return df.reset_index(drop=True)


def process_video(
    video_dir: Path,
    args,
    models: CropModels,
    decode_pool: ThreadPoolExecutor | None = None,
) -> dict:
    video_id = video_dir.name
    frames_root = Path(args.frames_root)
    emb_root = Path(args.emb_root)
    ann_root = Path(args.annotated_root)
    shards_dir = emb_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)

    done_path, fail_path = status_paths(emb_root, video_id)
    out_npz = shard_path(shards_dir, video_id)
    tar_path = ann_root / f"{video_id}.tar"
    jpeg_dir = ann_root / video_id

    exclude_cats = {c.strip().lower() for c in args.exclude_categories}
    allowed = None
    if args.categories_file:
        allowed = {ln.strip().lower() for ln in Path(args.categories_file).read_text().splitlines() if ln.strip()}

    df = detections_for_video(video_dir, args.min_yolo_conf, exclude_cats, allowed)
    stats = {
        "video_id": video_id,
        "n_detections": int(len(df)),
        "n_scored": 0,
        "n_passed": 0,
        "n_frames_loaded": 0,
        "n_frames_missing": 0,
        "n_frames_annotated": 0,
        "n_degenerate_box": 0,
    }
    if df.empty:
        save_video_shard(
            out_npz,
            image_id=[], class_name=[], frame_number=[], dup_idx=[], bbox=[],
            yolo_conf=[], clip_sim=[], clip_emb=[], passed=[], dino_idx=[], dino_emb=[],
        )
        write_json(done_path, stats)
        return stats

    # Accumulators
    image_ids, class_names, frames, dup_idxs = [], [], [], []
    bboxes, yolo_confs, clip_sims, passed_flags, dino_idxs = [], [], [], [], []
    clip_chunks, dino_chunks = [], []
    dino_row = 0

    tar_tmp = tar_path.with_suffix(f".tar.tmp{os.getpid()}")
    tar = None
    if args.annotated_format == "tar":
        tar = tarfile.open(tar_tmp, "w")
    writer = AnnotateWriter(tar, jpeg_dir, args.annotated_format, args.jpeg_quality, stats)

    grouped = df.groupby("frame_number", sort=True)
    pending = []  # list of dicts with rgb, dets, crops
    prefetch = max(1, int(getattr(args, "prefetch_frames", 1)))
    max_pending_frames = max(32, prefetch * 4)

    def flush_pending():
        nonlocal dino_row
        if not pending:
            return
        crops = [c for item in pending for c in item["crops"]]
        if not crops:
            pending.clear()
            return
        clip_vecs = []
        bs = args.batch_size
        for i in range(0, len(crops), bs):
            clip_vecs.append(models.encode_clip(crops[i:i + bs]))
        clip_all = np.concatenate(clip_vecs, axis=0)
        clip_norm = clip_all / (np.linalg.norm(clip_all, axis=1, keepdims=True) + 1e-8)

        offset = 0
        dino_crops = []
        dino_slots = []
        for item in pending:
            n = len(item["crops"])
            vecs = clip_all[offset:offset + n]
            norms = clip_norm[offset:offset + n]
            passer_rows = []
            clip_chunks.append(np.ascontiguousarray(vecs))
            for j, det in enumerate(item["dets"]):
                cls = det["class_name"]
                text = models.ensure_text(cls)
                sim = float(np.dot(norms[j], text))
                passed = sim >= args.clip_threshold
                iid = f"{video_id}_{item['frame']:05d}_{cls}_{det['dup_idx']}"
                image_ids.append(iid)
                class_names.append(cls)
                frames.append(item["frame"])
                dup_idxs.append(det["dup_idx"])
                bboxes.append(det["box"])
                yolo_confs.append(det["yolo_conf"])
                clip_sims.append(sim)
                passed_flags.append(passed)
                if passed:
                    dino_crops.append(item["crops"][j])
                    dino_slots.append(len(image_ids) - 1)
                    passer_rows.append({
                        "class_name": cls,
                        "clip_sim": sim,
                        "box": det["box"],
                    })
                    dino_idxs.append(-2)
                else:
                    dino_idxs.append(-1)
            offset += n
            if passer_rows and args.annotated_format != "none":
                writer.submit(item["rgb"], passer_rows, f"{item['frame']:05d}.jpg")

        if dino_crops:
            dino_vecs = []
            for i in range(0, len(dino_crops), bs):
                dino_vecs.append(models.encode_dino(dino_crops[i:i + bs]))
            dino_all = np.concatenate(dino_vecs, axis=0)
            dino_chunks.append(np.ascontiguousarray(dino_all))
            for k, slot in enumerate(dino_slots):
                dino_idxs[slot] = dino_row
                dino_row += 1

        stats["n_scored"] += len(crops)
        pending.clear()

    try:
        jobs = [
            (int(frame_number), g, frame_jpg(frames_root, video_id, int(frame_number)))
            for frame_number, g in grouped
        ]
        for frame_number, g, rgb in iter_prefetched_frames(jobs, decode_pool, prefetch):
            if rgb is None:
                stats["n_frames_missing"] += 1
                continue
            h, w = rgb.shape[:2]
            stats["n_frames_loaded"] += 1

            dup_counter: dict[str, int] = {}
            dets, crops = [], []
            for rec in g.itertuples(index=False):
                box = clamp_box(rec.xmin, rec.ymin, rec.xmax, rec.ymax, w, h)
                if box is None:
                    stats["n_degenerate_box"] += 1
                    continue
                cls = rec.class_name
                idx = dup_counter.get(cls, 0)
                dup_counter[cls] = idx + 1
                crop = crop_rgb(rgb, box)
                if crop.size == 0:
                    stats["n_degenerate_box"] += 1
                    continue
                dets.append({
                    "class_name": cls,
                    "dup_idx": idx,
                    "box": box,
                    "yolo_conf": float(rec.confidence) if pd.notna(rec.confidence) else 0.0,
                })
                crops.append(crop)
            if not crops:
                continue
            pending.append({"frame": int(frame_number), "rgb": rgb, "dets": dets, "crops": crops})
            n_pending_crops = sum(len(x["crops"]) for x in pending)
            if n_pending_crops >= args.batch_size or len(pending) >= max_pending_frames:
                flush_pending()

        flush_pending()
        writer.close()
        if tar is not None:
            tar.close()
            tar = None
            if stats["n_frames_annotated"] > 0:
                os.replace(tar_tmp, tar_path)
            elif tar_tmp.exists():
                tar_tmp.unlink()
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        if tar is not None:
            try:
                tar.close()
            except Exception:
                pass
            if tar_tmp.exists():
                tar_tmp.unlink()
        raise

    if clip_chunks:
        clip_emb = np.concatenate(clip_chunks, axis=0)
    else:
        clip_emb = np.zeros((0, CLIP_DIM), dtype=np.float16)
    if dino_chunks:
        dino_emb = np.concatenate(dino_chunks, axis=0)
    else:
        dino_emb = np.zeros((0, DINO_DIM), dtype=np.float16)

    # Re-map placeholder: passed rows with dino_idx still -2 shouldn't happen
    save_video_shard(
        out_npz,
        image_id=image_ids,
        class_name=class_names,
        frame_number=frames,
        dup_idx=dup_idxs,
        bbox=bboxes,
        yolo_conf=yolo_confs,
        clip_sim=clip_sims,
        clip_emb=clip_emb,
        passed=passed_flags,
        dino_idx=dino_idxs,
        dino_emb=dino_emb,
    )
    stats["n_passed"] = int(sum(1 for p in passed_flags if p))
    # Sidecar metadata (no vectors) so --stage index is a cheap concat, not an 18M-row Python loop.
    meta_path = out_npz.with_suffix(".meta.csv.gz")
    if image_ids:
        bb = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        meta = pd.DataFrame({
            "image_id": image_ids,
            "video_id": video_id,
            "frame_number": frames,
            "class_name": class_names,
            "dup_idx": dup_idxs,
            "xmin": bb[:, 0],
            "ymin": bb[:, 1],
            "xmax": bb[:, 2],
            "ymax": bb[:, 3],
            "yolo_conf": yolo_confs,
            "clip_sim": clip_sims,
            "passed": passed_flags,
            "has_dino": [int(i) >= 0 for i in dino_idxs],
        })
        meta.to_csv(meta_path, index=False, compression="gzip")
    elif meta_path.exists():
        meta_path.unlink()
    write_json(done_path, stats)
    if fail_path.exists():
        fail_path.unlink()
    return stats


def load_cdi_class_list(categories_file: str | None, exclude_cats: set[str]) -> list[str]:
    path = Path(categories_file) if categories_file else (
        REPO_ROOT / "data" / "vocabulary_lists" / "cdi_concrete_nouns_295.txt"
    )
    if not path.is_file():
        path = VALID129_PATH
    names = [ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()]
    return [n for n in names if n not in exclude_cats]


def configure_runtime_threads() -> None:
    """Keep OpenCV/BLAS from spawning a thread storm that fights NFS prefetch."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def stage_embed(args) -> None:
    configure_runtime_threads()
    os.environ.setdefault("HF_HOME", resolve_hf_home(args.hf_home))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this process. Launch via dispatch_cdi_crop_pipeline.sh "
            "on a GPU node (do not use the sandbox). nvidia-smi must show free GPUs."
        )
    device = "cuda"

    cdi_root = Path(args.cdi_root)
    emb_root = Path(args.emb_root)
    emb_root.mkdir(parents=True, exist_ok=True)
    (emb_root / "shards").mkdir(exist_ok=True)
    (emb_root / "_status").mkdir(exist_ok=True)
    Path(args.annotated_root).mkdir(parents=True, exist_ok=True)

    exclude_cats = {c.strip().lower() for c in args.exclude_categories}
    class_names = load_cdi_class_list(args.categories_file, exclude_cats)

    write_run_config(emb_root / "run_config.json", {
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "clip_threshold": args.clip_threshold,
        "clip_text_template": args.clip_text_template,
        "min_yolo_conf": args.min_yolo_conf,
        "exclude_categories": sorted(exclude_cats),
        "exclude_subjects": list(args.exclude_subjects),
        "categories_file": args.categories_file,
        "frames_root": args.frames_root,
        "cdi_root": args.cdi_root,
        "annotated_format": args.annotated_format,
        "batch_size": args.batch_size,
        "prefetch_frames": args.prefetch_frames,
        "hf_home": os.environ.get("HF_HOME"),
        "readout": {
            "clip": "get_image_features (512-d, float16, not L2-normalized on disk)",
            "dinov3": "pooler_output CLS (768-d, float16); passers only",
            "clip_sim": "cosine of L2-normalized image_embeds vs L2-normalized text_embeds",
        },
    })

    video_dirs = list_video_dirs(cdi_root)
    video_dirs = [d for d in video_dirs if not should_drop_video(d.name, args.exclude_subjects)]
    if args.limit_videos:
        video_dirs = video_dirs[: args.limit_videos]
    shard_i, n_shards, idx = parse_shard(args.shard, len(video_dirs))
    mine = [video_dirs[k] for k in idx]
    print(f"Shard {shard_i}/{n_shards}: {len(mine)} / {len(video_dirs)} videos "
          f"(dropped subjects={args.exclude_subjects})", flush=True)

    models = CropModels(args.clip_model, args.dino_model, args.clip_text_template, class_names, device=device)
    prefetch = max(1, int(args.prefetch_frames))
    decode_pool = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 1 else None
    print(
        f"Runtime: batch_size={args.batch_size} prefetch_frames={prefetch} "
        f"cv2_threads={cv2.getNumThreads()}",
        flush=True,
    )

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    try:
        for vi, vdir in enumerate(mine, 1):
            done_path, fail_path = status_paths(emb_root, vdir.name)
            npz = shard_path(emb_root / "shards", vdir.name)
            if not args.overwrite and done_path.is_file() and npz.is_file():
                n_skip += 1
                continue
            try:
                stats = process_video(vdir, args, models, decode_pool=decode_pool)
                n_done += 1
                elapsed = time.time() - t0
                print(
                    f"[{vi}/{len(mine)}] {vdir.name}  scored={stats['n_scored']}  "
                    f"passed={stats['n_passed']}  frames={stats['n_frames_loaded']}  "
                    f"missing={stats['n_frames_missing']}  annotated={stats['n_frames_annotated']}  "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
            except Exception as e:
                import traceback
                n_fail += 1
                tb = traceback.format_exc()
                write_json(fail_path, {"video_id": vdir.name, "error": repr(e), "traceback": tb})
                print(f"[{vi}/{len(mine)}] FAILED {vdir.name}: {e!r}\n{tb}", flush=True)
    finally:
        if decode_pool is not None:
            decode_pool.shutdown(wait=True)
    print(f"Shard {shard_i}/{n_shards} finished: newly_done={n_done} skipped={n_skip} failed={n_fail}",
          flush=True)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def stage_index(args) -> None:
    emb_root = Path(args.emb_root)
    shards_dir = emb_root / "shards"
    metas = sorted(shards_dir.glob("*.meta.csv.gz"))
    if not metas:
        raise SystemExit(f"No *.meta.csv.gz under {shards_dir}; did embed finish?")
    print(f"Concatenating {len(metas)} per-video metadata tables ...", flush=True)
    parts = [pd.read_csv(p) for p in metas]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n_total = int(len(df))
    n_pass = int(df["passed"].sum()) if n_total else 0
    index_path = emb_root / "crops_index.csv.gz"
    df.to_csv(index_path, index=False, compression="gzip")
    print(f"Wrote {index_path}  rows={n_total:,}  passed={n_pass:,} / {n_total:,}")

    # Compatibility filter list: one virtual path per passer (parent=class, stem=image_id)
    # so existing parsers that do Path(line).parent.name / .stem keep working.
    # These paths are NOT real files — load vectors via crop_embedding_store.CropStore.
    thr = args.clip_threshold
    excl = "-".join(["exclude-people"] + [f"exclude-subject-{s}" for s in args.exclude_subjects if s])
    list_name = f"clip_image_embeddings_filtered-by-clip-{thr:.2f}_{excl}.txt"
    list_path = emb_root / list_name
    if len(df) and "passed" in df.columns:
        passed = df[df["passed"].astype(bool)]
    else:
        passed = df.iloc[0:0]
    with open(list_path, "w") as fh:
        if len(passed):
            fh.writelines(
                f"{emb_root}/shards/{c}/{i}.npy\n"
                for c, i in zip(passed["class_name"].astype(str), passed["image_id"].astype(str))
            )
    print(f"Wrote {list_path}  ({len(passed):,} lines; virtual paths, not real npy files)")

    # Also a scores csv (all scored, any threshold) — analogue of the 868-hour all-frames csv.
    scores_path = emb_root / f"clip_image_embeddings_all_scored_{excl}.csv.gz"
    keep_cols = ["image_id", "video_id", "frame_number", "class_name", "clip_sim", "yolo_conf", "passed"]
    if len(df):
        df[keep_cols].to_csv(scores_path, index=False, compression="gzip")
        print(f"Wrote {scores_path}")

    n_status = len(list((emb_root / "_status").glob("*.done")))
    n_fail = len(list((emb_root / "_status").glob("*.failed")))
    n_shards = len(list(shards_dir.glob("*.npz")))
    summary = {
        "n_done_videos": n_status,
        "n_failed_videos": n_fail,
        "n_shards": n_shards,
        "n_scored": int(n_total),
        "n_passed": int(n_pass),
        "pass_rate": (n_pass / n_total) if n_total else None,
        "clip_threshold": thr,
        "index": str(index_path),
        "filter_list": str(list_path),
    }
    write_json(emb_root / "index_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    args = get_args()
    # argparse with nargs='*' and default=['00270001'] treats `--exclude-subjects`
    # with zero tokens as [] (keep everyone). Good.
    if args.stage in ("scan", "all"):
        stage_scan(args)
    if args.stage in ("embed", "all"):
        stage_embed(args)
    if args.stage == "index":
        stage_index(args)
    if args.stage == "all":
        # index only makes sense after *all* shards finish; skip here.
        print("Skip --stage index from `all` (run it once after every GPU shard is done).")


if __name__ == "__main__":
    main()
