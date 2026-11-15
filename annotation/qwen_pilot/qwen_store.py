"""Read the packed Qwen box store (embed_full_run.py) and re-cut crops on demand (no crop images are saved).

    import qwen_store as qs
    z = qs.shard("S00220001_2024-02-05_1_recfcw2yqs02gdskT")      # dict of arrays for one video
    img = qs.crop(z["image_id"][0], z["bbox"][0])                  # PIL image of that box, from the 512 x 910 frame
    ix = qs.index()                                                # every box without embeddings (boxes_index.csv.gz)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("QWEN_STORE", "/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"))
FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")


def shard(video_id: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / "shards" / f"{video_id}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def index() -> pd.DataFrame:
    return pd.read_csv(ROOT / "boxes_index.csv.gz")


def frame_path(image_id: str) -> Path:
    """image_id = <video_id>_<frame:05d>_<obj:02d>; the video folder may end in _rotated / _processed."""
    video, frame, _ = image_id.rsplit("_", 2)
    return FRAMES / video / f"{frame}.jpg"


def crop(image_id: str, bbox, pad: float = 0.0) -> Image.Image:
    """The box (xmin, ymin, xmax, ymax in frame pixels), optionally padded by a share of its size on each side."""
    im = Image.open(frame_path(image_id)).convert("RGB")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return im.crop((max(0, x0 - dx), max(0, y0 - dy), min(im.width, x1 + dx), min(im.height, y1 + dy)))
