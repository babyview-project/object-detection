"""Clerkin-style frame prevalence tables for manuscript long-tail analyses.

Frame prevalence: unique frames with >=1 detection of category / total unique
frames in the pool.

Pools:
  - valid129 (and valid85 when using the full-dataset builder): all frames in
    ``merged_frame_detections_with_metadata_filtered-0.27.csv``.
  - valid85 supplemental: annotation / VQA frame pool (rater-validated exemplar
    frames from the 7,018-crop cohort).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from exemplar_set_zscore_embeddings import (
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    build_valid85_sampled_exemplar_table,
    load_clip_filter_pair_set,
    load_config,
)

USECOLS = ["class_name", "original_frame_path"]


def path_to_frame_key(frame_path: str) -> str:
    p = Path(frame_path)
    return f"{p.parent.name}/{p.stem}"


def stem_to_frame_key(stem: str) -> str | None:
    parts = str(stem).split("_")
    if len(parts) >= 8 and parts[6] == "processed":
        return f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_processed/{parts[7]}"
    return None


def load_included_categories(name: str) -> list[str]:
    path = CATEGORY_FILES[name]
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def load_detections(frame_data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(frame_data_csv, usecols=USECOLS)
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
    return pd.DataFrame(rows).sort_values("proportion", ascending=False).reset_index(drop=True)


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
    return pd.DataFrame(rows).sort_values("proportion", ascending=False).reset_index(drop=True)


def build_frame_prevalence_table(
    det: pd.DataFrame,
    category_set: str,
    included: list[str],
    semantic_map: dict[str, str],
) -> tuple[pd.DataFrame, str]:
    """Return (table, pool_label) for valid129 or valid85."""
    if category_set == "valid129":
        return build_full_dataset_table(det, included, semantic_map), "full infant-view dataset"
    if category_set == "valid85":
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
        return (
            build_annotation_pool_table(det, included, semantic_map, exemplar_counts),
            "annotation / VQA frame pool",
        )
    raise ValueError(f"Unsupported category_set: {category_set!r}")


def mirror_csv(dst: Path, src: Path) -> None:
    if dst.resolve() == src.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def public_frequency_destinations(
    category_set: str,
    data_dir: Path,
    threshold_token: str = "0.27",
) -> list[Path]:
    if category_set == "valid129":
        name = f"long_tailed_dist_prop_included_categories_filtered-{threshold_token}_valid129.csv"
        return [
            data_dir / name,
            data_dir / "shared_data_manuscript_2026" / "inputs" / name,
            data_dir / "shared_data_ccn_2026" / "inputs" / name,
        ]
    if category_set == "valid85":
        name = "long_tailed_dist_prop_included_categories_valid85.csv"
        return [
            data_dir / name,
            data_dir / "shared_data_manuscript_2026" / "inputs" / name,
            data_dir / "shared_data_ccn_2026" / "inputs" / name,
            data_dir / "shared_data_manuscript_2026" / "vqa_detections" / "overall_category_distribution_85.csv",
        ]
    raise ValueError(f"Unsupported category_set: {category_set!r}")


def sync_public_frequency_tables(
    df: pd.DataFrame,
    category_set: str,
    data_dir: Path,
    *,
    threshold_token: str = "0.27",
) -> None:
    if category_set == "valid129":
        out_df = df[
            ["category", "count_frames", "count_instances", "total_frames", "proportion", "cdi_semantic"]
        ]
    else:
        out_df = df
    for dst in public_frequency_destinations(category_set, data_dir, threshold_token=threshold_token):
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(dst, index=False)
        print(f"Mirrored frequency table -> {dst}")
