#!/usr/bin/env python3
"""Compute Clerkin-style frame prevalence tables for CCN / manuscript figures.

Frame prevalence (#1): unique frames with >=1 detection of category / total unique
frames in the pool. This replaces detection-instance proportion (#2) in frequency CSVs.

Pools:
  - full infant-view (valid129 / valid85 category lists): all frames in
    ``merged_frame_detections_with_metadata_filtered-0.27.csv``.
  - annotation / VQA pool (valid85): frames that contain at least one
    rater-validated exemplar crop (7,018-crop cohort); detections counted on
    those frames only.

Run from repo root::

  python analysis/ccn-2026/scripts/compute_frame_prevalence_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_SCRIPTS.parents[2]
DATA_DIR = REPO_ROOT / "data"
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
SHARED_MS_VQA = DATA_DIR / "shared_data_manuscript_2026" / "vqa_detections"

FRAME_DATA_CSV = REPO_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
SEMANTIC_CSV = DATA_DIR / "long_tailed_dist_prop_included_categories.csv"

sys.path.insert(0, str(MANUSCRIPT_SCRIPTS))
from frame_prevalence import (  # noqa: E402
    build_annotation_pool_table,
    build_full_dataset_table,
    build_valid85_sampled_exemplar_table,
    load_detections,
    load_included_categories,
    mirror_csv,
    sync_public_frequency_tables,
)
from exemplar_set_zscore_embeddings import (  # noqa: E402
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    load_config,
)


def load_semantic_map() -> dict[str, str]:
    if not SEMANTIC_CSV.is_file():
        return {}
    df = pd.read_csv(SEMANTIC_CSV, usecols=["category", "cdi_semantic"]).dropna()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["cdi_semantic"] = df["cdi_semantic"].astype(str).str.strip().str.lower()
    return dict(zip(df["category"], df["cdi_semantic"]))


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} categories)")


def main() -> int:
    semantic_map = load_semantic_map()
    det = load_detections(FRAME_DATA_CSV)

    valid129 = load_included_categories("valid129")
    valid85 = load_included_categories("valid85")

    tbl129 = build_full_dataset_table(det, valid129, semantic_map)

    exemplar_counts = (
        build_valid85_sampled_exemplar_table(
            CATEGORY_FILES["valid85"],
            PER_CLASS_PRECISION_CSV,
            PER_FILE_PRECISION_CSV,
            SAMPLED_EXEMPLAR_CSV,
            load_config()["precision_threshold"],
        )
        .groupby("category")
        .size()
    )
    tbl85_pool = build_annotation_pool_table(det, valid85, semantic_map, exemplar_counts)

    sync_public_frequency_tables(tbl129, "valid129", DATA_DIR)
    sync_public_frequency_tables(tbl85_pool, "valid85", DATA_DIR)

    write_table(
        tbl129[["category", "count_frames", "count_instances", "proportion", "cdi_semantic"]],
        DATA_DIR / "long_tailed_dist_prop_included_categories_valid129.csv",
    )
    for dst in (
        DATA_DIR / "shared_data_manuscript_2026" / "inputs" / "long_tailed_dist_prop_included_categories_valid129.csv",
        DATA_DIR / "shared_data_ccn_2026" / "inputs" / "long_tailed_dist_prop_included_categories_valid129.csv",
    ):
        mirror_csv(dst, DATA_DIR / "long_tailed_dist_prop_included_categories_valid129.csv")

    write_table(
        tbl129[["category", "count_frames", "count_instances", "proportion", "cdi_semantic"]],
        SHARED_MS_VQA / "overall_category_distribution_129.csv",
    )

    chair = tbl129.loc[tbl129.category == "chair"].iloc[0]
    print(
        f"Full dataset: total_frames={chair.total_frames}, "
        f"chair frame_prev={chair.proportion:.4f} (was ~0.102 detection prop), "
        f"chair count_frames={chair.count_frames}"
    )
    chair_pool = tbl85_pool.loc[tbl85_pool.category == "chair"].iloc[0]
    print(
        f"Annotation pool: total_frames={chair_pool.total_frames}, "
        f"chair frame_prev={chair_pool.proportion:.4f}, "
        f"chair count_instances={chair_pool.count_instances}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
