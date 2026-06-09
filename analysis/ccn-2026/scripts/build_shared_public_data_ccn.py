#!/usr/bin/env python3
"""
Build anonymized, public-shareable intermediate data for the CCN 2026 poster.

This script exports category-level aggregates (Tier A) into:
    data/shared_data_ccn_2026/

Primary cohort: valid7018/ (7,018 rater-validated crops, same global+local pool).

Legacy poster-pool Plot B/C tables are included only when present locally.

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
    """Fail fast if a CSV looks like a per-exemplar export."""
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


def _export_json_files(src_dir: Path, names: list[str], rel_dir: str) -> list[str]:
    written: list[str] = []
    for name in names:
        src = src_dir / name
        if not src.is_file():
            continue
        dst = SHARED_DIR / rel_dir / name
        _copy_file(src, dst)
        written.append(f"{rel_dir}/{name}")
    return written


def build() -> None:
    binary_stash: Path | None = None
    if SHARED_DIR.exists():
        for sub in ("embeddings", "montages"):
            src = SHARED_DIR / sub
            if src.is_dir() and any(src.iterdir()):
                if binary_stash is None:
                    binary_stash = SHARED_DIR.parent / ".shared_data_ccn_2026_binary_stash"
                    if binary_stash.exists():
                        shutil.rmtree(binary_stash)
                    binary_stash.mkdir(parents=True)
                shutil.copytree(src, binary_stash / sub)
        shutil.rmtree(SHARED_DIR)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    if binary_stash is not None and binary_stash.is_dir():
        for sub in ("embeddings", "montages"):
            src = binary_stash / sub
            if src.is_dir():
                shutil.copytree(src, SHARED_DIR / sub)
        shutil.rmtree(binary_stash)

    valid7018_dir = CCN_DIR / "valid7018"
    if not valid7018_dir.is_dir():
        raise FileNotFoundError(
            "Missing valid7018/ — run compute_valid7018_local_global.py first."
        )

    written: list[str] = ["README.md"]

    # --- Primary: valid7018 same-cohort metrics ---
    valid7018_rel = "valid7018"
    written.extend(
        _export_glob_csv(
            glob_root=valid7018_dir,
            pattern="*.csv",
            rel_dir=valid7018_rel,
        )
    )
    written.extend(
        _export_json_files(
            valid7018_dir,
            ["valid7018_paper_stats.json", "valid7018_run.json"],
            valid7018_rel,
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

    # Frequency + semantic inputs for abstract frequency plots (category-level only).
    freq_exports: list[tuple[Path, str]] = [
        (
            DATA_DIR / "long_tailed_dist_prop_included_categories_valid85.csv",
            "long_tailed_dist_prop_included_categories_valid85.csv",
        ),
        (
            DATA_DIR / "shared_data_manuscript_2026" / "inputs" / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
            "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
        ),
        (
            DATA_DIR / "shared_data_manuscript_2026" / "inputs" / "long_tailed_dist_prop_included_categories_valid129.csv",
            "long_tailed_dist_prop_included_categories_valid129.csv",
        ),
    ]
    for src, name in freq_exports:
        if not src.is_file():
            raise FileNotFoundError(f"Missing frequency table for CCN bundle: {src}")
        _assert_columns_safe_csv(src)
        _copy_file(src, SHARED_DIR / cat_rel / name)
        written.append(f"{cat_rel}/{name}")

    sem_src = DATA_DIR / "long_tailed_dist_prop_included_categories.csv"
    if sem_src.is_file():
        sem_df = pd.read_csv(sem_src, usecols=["category", "cdi_semantic"])
        sem_dst = SHARED_DIR / cat_rel / "category_cdi_semantic_map.csv"
        sem_dst.parent.mkdir(parents=True, exist_ok=True)
        sem_df.to_csv(sem_dst, index=False)
        written.append(f"{cat_rel}/category_cdi_semantic_map.csv")

    fig_sel = valid7018_dir / "figures" / "valid7018_figure_category_selection.csv"
    if fig_sel.is_file():
        _copy_file(fig_sel, SHARED_DIR / valid7018_rel / "figures" / fig_sel.name)
        written.append(f"{valid7018_rel}/figures/{fig_sel.name}")

    # Optional: embedding + montage zips (built separately).
    for rel in (
        "embeddings/valid7018_bv_embeddings.zip",
        "embeddings/valid7018_bv_embeddings_zip.json",
        "montages/valid7018_montage_crops.zip",
        "montages/valid7018_montage_crops_zip.json",
    ):
        if (SHARED_DIR / rel).is_file() and rel not in written:
            written.append(rel)

    generated_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_utc": generated_utc,
        "description": (
            "CCN 2026 public-shareable intermediate tables exported from this "
            "repository. Primary cohort: valid7018 (7,018 rater-validated "
            "crops). Category-level aggregates only (no per-exemplar subject_id/stem)."
        ),
        "notes": [
            "Do not redistribute Plot A/B/C images, per-exemplar CSVs, or thumbnails.",
            "Per-exemplar tables contain subject/file-stem-like identifiers and are excluded.",
            "Legacy poster-pool Plot B/C tables are local-only under analysis/ccn-2026/ (not in this bundle).",
        ],
        "files": sorted(set(written)),
    }
    (SHARED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    readme = """\
# CCN 2026 — shared public data (category-level aggregates)

This directory contains Tier A intermediate tables for the CCN 2026 exemplar
variability analyses.

Primary cohort (Methods-aligned):
- `valid7018/` — global + local metrics on
  the same 7,018 rater-validated crops (`valid7018_paper_stats.json` for headline ρ)
- `embeddings/valid7018_bv_embeddings.zip` — paired CLIP+DINO `.npy` (~20 MB;
  run `build_valid7018_embeddings_zip.py`)

Also included:
- Category sets, per-class validation precision, frequency tables, CDI semantic map (`inputs/`)
- `montages/valid7018_montage_crops.zip` — JPEG thumbnails for abstract Figure 1A

Excluded:
- Plot B/C per-exemplar CSVs (`subject_id`, `stem`-like fields)
- Full-resolution crop paths; only anonymized montage JPEGs are bundled

Generated by:
`python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py`
"""
    _write_text(SHARED_DIR / "README.md", readme)

    manifest["files"] = sorted(set(manifest["files"] + ["README.md"]))
    (SHARED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote CCN shared bundle to: {SHARED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"MANIFEST: {(SHARED_DIR / 'MANIFEST.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    build()
