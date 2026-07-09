#!/usr/bin/env python3
"""Package anonymized montage crop JPEGs for abstract Figure 1A.

Creates ``data/shared_data_ccn_2026/montages/valid7018_montage_crops.zip`` with
25 JPEGs per montage category (default: clock, plant, blanket, pajamas, book).
The manifest lists only category + slot (no absolute crop paths or subject ids).

Requires cluster crop paths locally (one-time build). Regenerate montages on
clone via ``generate_valid7018_paper_figures.py --from-zip``.

Run from repo root::
  python analysis/ccn-2026/scripts/build_valid7018_montage_crops_zip.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

CCN_DIR = Path(__file__).resolve().parent.parent
CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_DIR.parent.parent
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
DEFAULT_ZIP = REPO_ROOT / "data" / "shared_data_ccn_2026" / "montages" / "valid7018_montage_crops.zip"
DEFAULT_STATS = CCN_DIR / "valid7018" / "valid7018_paper_stats.json"
DEFAULT_CATS = ["clock", "plant", "blanket", "pajamas", "book"]

for p in (CCN_SCRIPTS, MANUSCRIPT_SCRIPTS, str(CCN_DIR)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exemplar_set_zscore_embeddings import (  # noqa: E402
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    build_valid85_sampled_exemplar_table,
    load_config,
)
from generate_valid7018_paper_figures import (  # noqa: E402
    build_exemplar_crop_index,
    load_crop_images,
    parse_confidence,
)


def _montage_categories(stats_path: Path) -> list[str]:
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        cats = stats.get("montage_categories_low_to_high_global")
        if cats:
            return [str(c).strip().lower() for c in cats]
    return DEFAULT_CATS


def build(
    zip_path: Path = DEFAULT_ZIP,
    stats_path: Path = DEFAULT_STATS,
    n_exemplars: int = 25,
    cell_size: tuple[int, int] = (128, 128),
) -> Path:
    cfg = load_config()
    exemplar_df = build_valid85_sampled_exemplar_table(
        CATEGORY_FILES["valid85"],
        PER_CLASS_PRECISION_CSV,
        PER_FILE_PRECISION_CSV,
        SAMPLED_EXEMPLAR_CSV,
        cfg["precision_threshold"],
    )
    crop_index = build_exemplar_crop_index(exemplar_df)
    categories = _montage_categories(stats_path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    n_written = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme = (
            "BabyView CCN 2026 — valid7018 montage crop archive\n"
            "===================================================\n\n"
            "JPEG thumbnails for abstract Figure 1A (25 per category).\n"
            "manifest.csv columns: category, slot, jpeg_path\n"
            "No subject_id or absolute filesystem paths are stored.\n"
        )
        zf.writestr("README.txt", readme)

        for cat in categories:
            imgs = load_crop_images(crop_index, cat, n_exemplars)
            if len(imgs) < n_exemplars:
                raise RuntimeError(f"{cat}: only {len(imgs)}/{n_exemplars} readable crops")
            for slot, img in enumerate(imgs[:n_exemplars]):
                if img.size != cell_size:
                    img = img.resize(cell_size, Image.Resampling.LANCZOS)
                member = str(PurePosixPath("crops") / cat / f"slot_{slot:02d}.jpg")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=88)
                zf.writestr(member, buf.getvalue())
                manifest_rows.append({"category": cat, "slot": str(slot), "jpeg_path": member})
                n_written += 1

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["category", "slot", "jpeg_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        zf.writestr("manifest.csv", buf.getvalue())

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_categories": len(categories),
        "n_exemplars_per_category": n_exemplars,
        "categories": categories,
        "n_jpegs": n_written,
        "cell_size": list(cell_size),
    }
    meta_path = zip_path.with_name(zip_path.stem + "_zip.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {zip_path.relative_to(REPO_ROOT)} ({n_written} JPEGs)")
    print(f"Wrote {meta_path.relative_to(REPO_ROOT)}")
    return zip_path


if __name__ == "__main__":
    build()
