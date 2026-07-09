"""Load 7,018-cohort embeddings from the git-shared zip (no cluster paths).

Usage::

    from load_valid7018_embeddings import load_valid7018_from_zip

    clip_by_cat, dino_by_cat = load_valid7018_from_zip()
"""
from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

CCN_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = (
    CCN_DIR.parent.parent
    / "data"
    / "shared_data_ccn_2026"
    / "embeddings"
    / "valid7018_bv_embeddings.zip"
)


def _load_npy_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        return np.asarray(np.load(BytesIO(f.read()), allow_pickle=False), dtype=np.float64).ravel()


def load_valid7018_from_zip(
    zip_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return per-category embedding matrices for CLIP and DINOv3.

    Keys are lower-case category names; values are ``(n_exemplars, dim)`` float64 arrays.
    """
    zp = Path(zip_path or DEFAULT_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Embedding zip not found: {zp}")

    clip: dict[str, list[np.ndarray]] = defaultdict(list)
    dino: dict[str, list[np.ndarray]] = defaultdict(list)

    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            rows = list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
        for row in tqdm(rows, desc="Load valid7018 zip", unit="ex"):
            cat = row["category"].strip().lower()
            clip[cat].append(_load_npy_from_zip(zf, row["clip_npy"]))
            dino[cat].append(_load_npy_from_zip(zf, row["dinov3_npy"]))

    clip_out = {c: np.stack(v, axis=0) for c, v in sorted(clip.items())}
    dino_out = {c: np.stack(v, axis=0) for c, v in sorted(dino.items())}
    return clip_out, dino_out
