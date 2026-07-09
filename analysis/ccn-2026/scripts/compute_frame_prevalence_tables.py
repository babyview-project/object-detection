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

import shutil
import sys
from pathlib import Path

import pandas as pd

CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_SCRIPTS.parents[2]
DATA_DIR = REPO_ROOT / "data"
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
SHARED_CCN_INPUTS = DATA_DIR / "shared_data_ccn_2026" / "inputs"
SHARED_MS_INPUTS = DATA_DIR / "shared_data_manuscript_2026" / "inputs"
SHARED_MS_VQA = DATA_DIR / "shared_data_manuscript_2026" / "vqa_detections"

FRAME_DATA_CSV = REPO_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
SEMANTIC_CSV = DATA_DIR / "long_tailed_dist_prop_included_categories.csv"

sys.path.insert(0, str(MANUSCRIPT_SCRIPTS))
from exemplar_set_zscore_embeddings import (  # noqa: E402
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    build_valid85_sampled_exemplar_table,
    load_clip_filter_pair_set,
    load_config,
)

USECOLS = ["class_name", "original_frame_path"]


def load_semantic_map() -> dict[str, str]:
    if not SEMANTIC_CSV.is_file():
        return {}
    df = pd.read_csv(SEMANTIC_CSV, usecols=["category", "cdi_semantic"]).dropna()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["cdi_semantic"] = df["cdi_semantic"].astype(str).str.strip().str.lower()
    return dict(zip(df["category"], df["cdi_semantic"]))


def load_included_categories(name: str) -> list[str]:
    path = CATEGORY_FILES[name]
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def stem_to_frame_key(stem: str) -> str | None:
    parts = str(stem).split("_")
    if len(parts) >= 8 and parts[6] == "processed":
        return f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_processed/{parts[7]}"
    return None


def path_to_frame_key(frame_path: str) -> str:
    p = Path(frame_path)
    return f"{p.parent.name}/{p.stem}"


def load_detections() -> pd.DataFrame:
    df = pd.read_csv(FRAME_DATA_CSV, usecols=USECOLS)
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df["frame_key"] = df["original_frame_path"].map(path_to_frame_key)
    return df


def build_full_dataset_table(
    det: pd.DataFrame,
    included: list[str],
    semantic_map: dict[str, str],
) -> pd.DataFrame:
    inc = set(included)
    pool = det.copy()
    total_frames = int(pool["frame_key"].nunique())

    inst = pool[pool["class_name"].isin(inc)].groupby("class_name").size().rename("count_instances")
    frames = (
        pool[pool["class_name"].isin(inc)]
        .groupby("class_name")["frame_key"]
        .nunique()
        .rename("count_frames")
    )

    rows = []
    for cat in included:
        rows.append(
            {
                "category": cat,
                "count_frames": int(frames.get(cat, 0)),
                "count_instances": int(inst.get(cat, 0)),
                "total_frames": total_frames,
                "proportion": float(frames.get(cat, 0)) / total_frames if total_frames else 0.0,
                "cdi_semantic": semantic_map.get(cat, "other"),
            }
        )
    out = pd.DataFrame(rows).sort_values("proportion", ascending=False).reset_index(drop=True)
    return out


def annotation_pool_frame_keys() -> set[str]:
    cfg = load_config()
    included = set(load_included_categories("valid85"))
    exemplars = build_valid85_sampled_exemplar_table(
        CATEGORY_FILES["valid85"],
        PER_CLASS_PRECISION_CSV,
        PER_FILE_PRECISION_CSV,
        SAMPLED_EXEMPLAR_CSV,
        cfg["precision_threshold"],
    )
    clip_pairs = load_clip_filter_pair_set(cfg["clip_filter_list_path"], included)
    exemplars = exemplars[
        exemplars.apply(lambda r: (r["category"], r["stem"]) in clip_pairs, axis=1)
    ]
    keys = {stem_to_frame_key(s) for s in exemplars["stem"] if stem_to_frame_key(s)}
    if not keys:
        raise RuntimeError("No annotation-pool frame keys resolved from validated exemplars")
    return keys


def build_annotation_pool_table(
    det: pd.DataFrame,
    included: list[str],
    semantic_map: dict[str, str],
    exemplar_counts: pd.Series | None = None,
) -> pd.DataFrame:
    inc = set(included)
    frame_keys = annotation_pool_frame_keys()
    total_frames = len(frame_keys)
    pool = det[det["frame_key"].isin(frame_keys) & det["class_name"].isin(inc)]

    inst = pool.groupby("class_name").size().rename("count_instances")
    frames = pool.groupby("class_name")["frame_key"].nunique().rename("count_frames")

    rows = []
    for cat in included:
        rows.append(
            {
                "category": cat,
                "count_frames": int(frames.get(cat, 0)),
                "count_instances": int(inst.get(cat, 0)),
                "count_exemplar_crops": int(exemplar_counts.get(cat, 0)) if exemplar_counts is not None else None,
                "total_frames": total_frames,
                "proportion": float(frames.get(cat, 0)) / total_frames if total_frames else 0.0,
                "cdi_semantic": semantic_map.get(cat, "other"),
            }
        )
    out = pd.DataFrame(rows).sort_values("proportion", ascending=False).reset_index(drop=True)
    return out


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} categories)")


def mirror(dst: Path, src: Path) -> None:
    if dst.resolve() == src.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Mirrored -> {dst}")


def main() -> int:
    semantic_map = load_semantic_map()
    det = load_detections()

    valid129 = load_included_categories("valid129")
    valid85 = load_included_categories("valid85")

    tbl129 = build_full_dataset_table(det, valid129, semantic_map)
    tbl85_full = build_full_dataset_table(det, valid85, semantic_map)

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

    outputs: list[tuple[pd.DataFrame, Path, list[Path]]] = [
        (
            tbl129[["category", "count_frames", "count_instances", "total_frames", "proportion", "cdi_semantic"]],
            DATA_DIR / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
            [
                SHARED_CCN_INPUTS / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
                SHARED_MS_INPUTS / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
            ],
        ),
        (
            tbl129[["category", "count_frames", "count_instances", "proportion", "cdi_semantic"]],
            DATA_DIR / "long_tailed_dist_prop_included_categories_valid129.csv",
            [
                SHARED_CCN_INPUTS / "long_tailed_dist_prop_included_categories_valid129.csv",
                SHARED_MS_INPUTS / "long_tailed_dist_prop_included_categories_valid129.csv",
            ],
        ),
        (
            tbl85_pool,
            DATA_DIR / "long_tailed_dist_prop_included_categories_valid85.csv",
            [
                SHARED_CCN_INPUTS / "long_tailed_dist_prop_included_categories_valid85.csv",
                SHARED_MS_INPUTS / "long_tailed_dist_prop_included_categories_valid85.csv",
                SHARED_MS_VQA / "overall_category_distribution_85.csv",
            ],
        ),
        (
            tbl85_full[["category", "count_frames", "count_instances", "proportion", "cdi_semantic"]],
            SHARED_MS_VQA / "overall_category_distribution_129.csv",
            [],
        ),
    ]

    for df, primary, mirrors in outputs:
        write_table(df, primary)
        for m in mirrors:
            mirror(m, primary)

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
