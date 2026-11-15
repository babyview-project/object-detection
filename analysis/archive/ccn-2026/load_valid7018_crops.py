"""Load all 7,018 valid7018 crop JPEGs from the git-shared zip (no cluster paths).

Usage::

    from load_valid7018_crops import load_valid7018_crops_from_zip

    crops_by_cat = load_valid7018_crops_from_zip()
    # crops_by_cat["apple"] -> list of PIL.Image (order matches embedding zip manifest)
"""
from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

CCN_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = (
    CCN_DIR.parent.parent
    / "data"
    / "shared_data_ccn_2026"
    / "crops"
    / "valid7018_crops.zip"
)


def load_valid7018_crops_from_zip(
    zip_path: Path | None = None,
) -> dict[str, list[Image.Image]]:
    """Return per-category PIL images in manifest order (same as embedding zip)."""
    zp = Path(zip_path or DEFAULT_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Crop zip not found: {zp}")

    by_cat: dict[str, list[Image.Image]] = defaultdict(list)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            rows = list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
        for row in rows:
            cat = row["category"].strip().lower()
            with zf.open(row["jpeg_path"]) as f:
                img = Image.open(BytesIO(f.read())).convert("RGB")
            by_cat[cat].append(img)
    return dict(sorted(by_cat.items()))


def load_valid7018_crop_paths_from_zip(
    zip_path: Path | None = None,
) -> list[dict[str, str]]:
    """Return manifest rows: category, stem, jpeg_path (paths inside the zip)."""
    zp = Path(zip_path or DEFAULT_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Crop zip not found: {zp}")
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            return list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
