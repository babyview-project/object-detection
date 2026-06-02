#!/usr/bin/env python3
"""
Build anonymized, public-shareable intermediate data for the CCN 2026 poster.

This script exports only *category-level* aggregates (Tier A) into:
    data/shared_data_ccn_2026/

It intentionally excludes:
- per-exemplar tables (contain `subject_id` and `stem` / file-stem-like fields)
- images / montages / thumbnails / t-SNE PNG/PDFs

Run from the repo root (object-detection/):
    python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SHARED_DIR = DATA_DIR / "shared_data_ccn_2026"

CCN_DIR = PROJECT_ROOT / "analysis" / "ccn-2026"


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)


def _assert_columns_safe_csv(path: Path) -> None:
    """
    Hard safety check for accidentally exporting per-exemplar tables.

    If a future CCN output changes naming or layout, this makes sure we fail
    fast instead of producing an unsafe public bundle.
    """
    # Only read headers + small prefix; we only need to check column names.
    df = pd.read_csv(path, nrows=3)
    lowered = {c.lower() for c in df.columns}
    forbidden = {"subject_id", "stem", "processed_id", "file_stem"}
    if lowered & forbidden:
        raise ValueError(
            f"Refusing to export potentially unsafe per-exemplar CSV: {path}\n"
            f"Found forbidden columns: {sorted(list(lowered & forbidden))}"
        )


def _export_csv_bundle(expected_srcs: list[Path], rel_dir: str) -> list[str]:
    written: list[str] = []
    for src in expected_srcs:
        if not src.exists():
            raise FileNotFoundError(f"Missing source CSV for CCN bundle: {src}")
        if src.suffix.lower() != ".csv":
            raise ValueError(f"Expected .csv: {src}")
        _assert_columns_safe_csv(src)
        dst = SHARED_DIR / rel_dir / src.name
        _copy_file(src, dst)
        written.append(f"{rel_dir}/{src.name}")
    return written


def _export_glob_csv(glob_root: Path, pattern: str, rel_dir: str) -> list[str]:
    if not glob_root.exists():
        raise FileNotFoundError(f"Missing glob root: {glob_root}")
    srcs = sorted(glob_root.glob(pattern))
    if not srcs:
        raise FileNotFoundError(f"No files matched {pattern} under {glob_root}")
    written: list[str] = []
    for src in srcs:
        if not src.exists():
            raise FileNotFoundError(src)
        _assert_columns_safe_csv(src)
        dst = SHARED_DIR / rel_dir / src.name
        _copy_file(src, dst)
        written.append(f"{rel_dir}/{src.name}")
    return written


def build() -> None:
    if SHARED_DIR.exists():
        shutil.rmtree(SHARED_DIR)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Tier A: category-level summaries (Plot B and Plot C) ---
    plotb_dir = (
        CCN_DIR / "plotB_tsne_distance_to_centroid_outputs_20260402"
    )  # preferred newer run
    plotc_dir = CCN_DIR / "plotC_knn_diversity_outputs"

    written: list[str] = []

    written.append(
        "README.md"
    )

    plotb_rel = "plotB_tsne_distance_to_centroid_outputs_20260402"
    written.extend(
        _export_csv_bundle(
            expected_srcs=[
                plotb_dir / "bv-to-bv-centroid_distance_clip_summary.csv",
                plotb_dir / "bv-to-bv-centroid_distance_dinov3_summary.csv",
                plotb_dir / "bv-to-bv-centroid_distance_clip_vs_dinov3_comparison.csv",
            ],
            rel_dir=plotb_rel,
        )
    )

    plotc_rel = "plotC_knn_diversity_outputs"
    written.extend(
        _export_csv_bundle(
            expected_srcs=[
                plotc_dir / "bv_within_category_knn_clip_k5_summary.csv",
                plotc_dir / "bv_within_category_knn_dinov3_k5_summary.csv",
                plotc_dir / "bv_within_category_knn_clip_vs_dinov3_k5_comparison.csv",
                # CCN extremes & overlap lists (category name only)
                plotc_dir / "ccn2026_local_global_extreme_categories_clip_dino.csv",
                plotc_dir / "ccn2026_local_global_extreme_categories_things_clip_dino.csv",
                plotc_dir
                / "ccn2026_local_global_overlap_categories_used_for_tsne_2x2.csv",
                # THINGS-vs-embedding availability counts (category-level)
                CCN_DIR / "things_image_vs_embedding_counts.csv",
            ],
            rel_dir=plotc_rel,
        )
    )

    # THINGS/BV metric tables from the newer embedding rerun.
    things_metrics_dir = plotc_dir / "new_things_embeddings_20260428"
    written.extend(
        _export_glob_csv(
            glob_root=things_metrics_dir,
            pattern="*.csv",
            rel_dir=f"{plotc_rel}/new_things_embeddings_20260428",
        )
    )

    # --- Tier A: category sets and class precision (category only) ---
    cat_rel = "inputs"
    included_129 = PROJECT_ROOT / "data" / "included_categories_valid129.txt"
    included_85 = PROJECT_ROOT / "data" / "included_categories_valid85.txt"
    per_class_validation = PROJECT_ROOT / "annotation" / "per_class_validation_data.csv"

    if not included_129.exists():
        raise FileNotFoundError(f"Missing included categories: {included_129}")
    if not included_85.exists():
        raise FileNotFoundError(f"Missing included categories: {included_85}")
    if not per_class_validation.exists():
        raise FileNotFoundError(f"Missing per-class validation CSV: {per_class_validation}")

    _assert_columns_safe_csv(per_class_validation)
    _copy_file(included_129, SHARED_DIR / cat_rel / included_129.name)
    _copy_file(included_85, SHARED_DIR / cat_rel / included_85.name)
    _copy_file(per_class_validation, SHARED_DIR / cat_rel / per_class_validation.name)

    written.extend(
        [
            f"{cat_rel}/{included_129.name}",
            f"{cat_rel}/{included_85.name}",
            f"{cat_rel}/{per_class_validation.name}",
        ]
    )

    # --- Manifest + README ---
    generated_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_utc": generated_utc,
        "description": (
            "CCN 2026 public-shareable intermediate tables exported from this "
            "repository. This bundle contains only Tier A category-level aggregates "
            "(no per-exemplar subject_id/stem fields)."
        ),
        "notes": [
            "Do not redistribute Plot A/B/C images, per-exemplar CSVs, or thumbnails.",
            "Per-exemplar tables contain subject/file-stem-like identifiers and are excluded.",
        ],
        "files": sorted(set(written)),
    }
    (SHARED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = """\
# CCN 2026 — shared public data (category-level aggregates)

This directory contains Tier A intermediate tables for the CCN 2026 poster analyses
(exemplar variability).

What’s included:
- Plot B: BV→BV centroid distance *summary* CSVs (category-level)
- Plot C: within-category kNN diversity *summary* CSVs and CLIP/DINO comparisons
- CCN local/global extremes and overlap lists (category names)
- THINGS/BV metric tables (category-level)
- Category sets and per-class validation precision table

What’s intentionally excluded:
- Plot B/C per-exemplar CSVs (these include `subject_id` and `stem`-like fields)
- any images, montages, or thumbnails

Generated by:
`python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py`
"""
    _write_text(SHARED_DIR / "README.md", readme)

    # Update MANIFEST after README creation.
    manifest["files"] = sorted(set(manifest["files"] + ["README.md"]))
    (SHARED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote CCN shared bundle to: {SHARED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"MANIFEST: {(SHARED_DIR / 'MANIFEST.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    build()

