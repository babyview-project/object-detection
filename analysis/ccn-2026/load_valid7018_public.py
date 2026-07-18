"""Load privacy-filtered public valid7018 crops / embeddings (no cluster paths).

Usage::

    from load_valid7018_public import load_public_embeddings, load_public_crops

    clip_by_cat, dino_by_cat = load_public_embeddings()
    crops_by_cat = load_public_crops()
"""
from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

CCN_DIR = Path(__file__).resolve().parent
PUBLIC_ROOT = CCN_DIR.parent.parent / "data" / "shared_data_ccn_2026" / "public"
DEFAULT_EMB_ZIP = PUBLIC_ROOT / "embeddings" / "valid7018_public_embeddings.zip"
DEFAULT_CROPS_ZIP = PUBLIC_ROOT / "crops" / "valid7018_public_crops.zip"


def _load_npy(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        return np.asarray(np.load(BytesIO(f.read()), allow_pickle=False), dtype=np.float64).ravel()


def load_public_embeddings(
    zip_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    zp = Path(zip_path or DEFAULT_EMB_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Public embedding zip not found: {zp}")

    clip: dict[str, list[np.ndarray]] = defaultdict(list)
    dino: dict[str, list[np.ndarray]] = defaultdict(list)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            rows = list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
        for row in rows:
            cat = row["category"].strip().lower()
            clip[cat].append(_load_npy(zf, row["clip_npy"]))
            dino[cat].append(_load_npy(zf, row["dinov3_npy"]))
    return (
        {c: np.stack(v, axis=0) for c, v in sorted(clip.items())},
        {c: np.stack(v, axis=0) for c, v in sorted(dino.items())},
    )


def load_public_crops(zip_path: Path | None = None) -> dict[str, list[Image.Image]]:
    zp = Path(zip_path or DEFAULT_CROPS_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Public crop zip not found: {zp}")

    by_cat: dict[str, list[Image.Image]] = defaultdict(list)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            rows = list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
        for row in rows:
            cat = row["category"].strip().lower()
            with zf.open(row["jpeg_path"]) as f:
                by_cat[cat].append(Image.open(BytesIO(f.read())).convert("RGB"))
    return dict(sorted(by_cat.items()))
