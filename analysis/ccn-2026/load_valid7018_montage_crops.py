"""Load abstract montage crop JPEGs from the git-shared zip (no cluster paths)."""
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
    / "montages"
    / "valid7018_montage_crops.zip"
)


def load_montage_crops_from_zip(
    zip_path: Path | None = None,
) -> dict[str, list[Image.Image]]:
    """Return per-category PIL images ordered by slot (low→high confidence proxy)."""
    zp = Path(zip_path or DEFAULT_ZIP).expanduser()
    if not zp.is_file():
        raise FileNotFoundError(f"Montage crop zip not found: {zp}")

    by_cat: dict[str, list[tuple[int, Image.Image]]] = defaultdict(list)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("manifest.csv") as mf:
            rows = list(csv.DictReader(mf.read().decode("utf-8").splitlines()))
        for row in rows:
            cat = row["category"].strip().lower()
            slot = int(row["slot"])
            with zf.open(row["jpeg_path"]) as f:
                img = Image.open(BytesIO(f.read())).convert("RGB")
            by_cat[cat].append((slot, img))

    return {c: [img for _, img in sorted(v, key=lambda x: x[0])] for c, v in by_cat.items()}
