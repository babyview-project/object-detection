"""Packed per-video crop-embedding store for the 2025.2 CDI pipeline.

One ``{video_id}.npz`` holds every CLIP-scored detection in that video (and DINOv3
vectors only for CLIP-passers). This is ~3k files instead of tens of millions of
per-crop ``.npy`` files, which NFS inode / metadata ops cannot survive.

NPZ layout
----------
image_id     (N,)  unicode     ``{video_id}_{frame:05d}_{class}_{dup_idx}``
class_name   (N,)  unicode
frame_number (N,)  int32
dup_idx      (N,)  int16       running index of this class in this frame
bbox         (N, 4) float32    xmin, ymin, xmax, ymax in frame pixel space
yolo_conf    (N,)  float32
clip_sim     (N,)  float32     cosine(image, CLIP text of class name)
clip_emb     (N, 512) float16  raw CLIP ``get_image_features`` (not L2-normalized)
passed       (N,)  bool        clip_sim >= threshold used at write time
dino_idx     (N,)  int32       row in ``dino_emb``, or -1 if not computed
dino_emb     (M, 768) float16  DINOv3 CLS ``pooler_output`` for passers only
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CLIP_DIM = 512
DINO_DIM = 768


def shard_path(shards_dir: Path, video_id: str) -> Path:
    return Path(shards_dir) / f"{video_id}.npz"


def atomic_npz_save(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" unless the name already ends with it — a ".npz.tmp<pid>"
    # stem would silently write "<name>.npz.tmp<pid>.npz" and os.replace would miss it.
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def save_video_shard(
    path: Path,
    *,
    image_id,
    class_name,
    frame_number,
    dup_idx,
    bbox,
    yolo_conf,
    clip_sim,
    clip_emb,
    passed,
    dino_idx,
    dino_emb,
) -> None:
    n = len(image_id)
    if n == 0:
        # Keep an empty shard so resume can treat the video as done.
        atomic_npz_save(
            path,
            image_id=np.array([], dtype=object),
            class_name=np.array([], dtype=object),
            frame_number=np.zeros((0,), dtype=np.int32),
            dup_idx=np.zeros((0,), dtype=np.int16),
            bbox=np.zeros((0, 4), dtype=np.float32),
            yolo_conf=np.zeros((0,), dtype=np.float32),
            clip_sim=np.zeros((0,), dtype=np.float32),
            clip_emb=np.zeros((0, CLIP_DIM), dtype=np.float16),
            passed=np.zeros((0,), dtype=bool),
            dino_idx=np.zeros((0,), dtype=np.int32),
            dino_emb=np.zeros((0, DINO_DIM), dtype=np.float16),
        )
        return
    clip_emb = np.asarray(clip_emb, dtype=np.float16)
    if clip_emb.ndim != 2 or clip_emb.shape[1] != CLIP_DIM:
        raise ValueError(f"clip_emb expected (N, {CLIP_DIM}), got {clip_emb.shape}")
    dino_emb = np.asarray(dino_emb, dtype=np.float16)
    if dino_emb.size == 0:
        dino_emb = np.zeros((0, DINO_DIM), dtype=np.float16)
    elif dino_emb.ndim != 2 or dino_emb.shape[1] != DINO_DIM:
        raise ValueError(f"dino_emb expected (M, {DINO_DIM}), got {dino_emb.shape}")
    atomic_npz_save(
        path,
        image_id=np.asarray(image_id, dtype=object),
        class_name=np.asarray(class_name, dtype=object),
        frame_number=np.asarray(frame_number, dtype=np.int32),
        dup_idx=np.asarray(dup_idx, dtype=np.int16),
        bbox=np.asarray(bbox, dtype=np.float32).reshape(n, 4),
        yolo_conf=np.asarray(yolo_conf, dtype=np.float32),
        clip_sim=np.asarray(clip_sim, dtype=np.float32),
        clip_emb=clip_emb,
        passed=np.asarray(passed, dtype=bool),
        dino_idx=np.asarray(dino_idx, dtype=np.int32),
        dino_emb=dino_emb,
    )


def load_video_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def iter_shards(shards_dir: Path):
    shards_dir = Path(shards_dir)
    for p in sorted(shards_dir.glob("*.npz")):
        if ".tmp" in p.name:
            continue
        yield p.stem, load_video_shard(p)


class CropStore:
    """Read-side helper for packed shards (+ optional crops_index.csv.gz)."""

    def __init__(self, emb_root: Path):
        self.root = Path(emb_root)
        self.shards_dir = self.root / "shards"
        self._id_to_video: dict[str, str] | None = None

    def _ensure_id_index(self) -> None:
        if self._id_to_video is not None:
            return
        index_path = self.root / "crops_index.csv.gz"
        mapping: dict[str, str] = {}
        if index_path.is_file():
            import pandas as pd
            df = pd.read_csv(index_path, usecols=["image_id", "video_id"])
            mapping = dict(zip(df["image_id"].astype(str), df["video_id"].astype(str)))
        else:
            for video_id, shard in iter_shards(self.shards_dir):
                for iid in shard["image_id"]:
                    mapping[str(iid)] = video_id
        self._id_to_video = mapping

    def load_clip(self, image_id: str) -> np.ndarray:
        self._ensure_id_index()
        video_id = self._id_to_video.get(image_id)
        if video_id is None:
            raise KeyError(image_id)
        shard = load_video_shard(shard_path(self.shards_dir, video_id))
        ids = shard["image_id"].astype(str)
        hit = np.flatnonzero(ids == image_id)
        if hit.size == 0:
            raise KeyError(image_id)
        return shard["clip_emb"][int(hit[0])]

    def load_dinov3(self, image_id: str) -> np.ndarray:
        self._ensure_id_index()
        video_id = self._id_to_video.get(image_id)
        if video_id is None:
            raise KeyError(image_id)
        shard = load_video_shard(shard_path(self.shards_dir, video_id))
        ids = shard["image_id"].astype(str)
        hit = np.flatnonzero(ids == image_id)
        if hit.size == 0:
            raise KeyError(image_id)
        i = int(hit[0])
        di = int(shard["dino_idx"][i])
        if di < 0:
            raise KeyError(f"{image_id} has no DINOv3 vector (did not pass CLIP filter)")
        return shard["dino_emb"][di]

    def load_category(
        self,
        class_name: str,
        *,
        model: str = "clip",
        passed_only: bool = True,
        min_sim: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, matrix) for one class, scanning shards. Prefer crops_index.csv.gz
        + per-video loads when doing many categories (see pipeline --stage index)."""
        class_name = class_name.strip().lower()
        ids, vecs = [], []
        for _, shard in iter_shards(self.shards_dir):
            names = np.array([str(x).lower() for x in shard["class_name"]])
            mask = names == class_name
            if passed_only:
                mask &= shard["passed"]
            if min_sim is not None:
                mask &= shard["clip_sim"] >= float(min_sim)
            if model == "dinov3":
                mask &= shard["dino_idx"] >= 0
            if not mask.any():
                continue
            sel = np.flatnonzero(mask)
            ids.append(shard["image_id"][sel].astype(str))
            if model == "clip":
                vecs.append(shard["clip_emb"][sel])
            elif model == "dinov3":
                vecs.append(shard["dino_emb"][shard["dino_idx"][sel]])
            else:
                raise ValueError(model)
        dim = CLIP_DIM if model == "clip" else DINO_DIM
        if not vecs:
            return np.array([], dtype=object), np.zeros((0, dim), dtype=np.float16)
        return np.concatenate(ids), np.concatenate(vecs, axis=0)


def write_run_config(path: Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.json")
    tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
