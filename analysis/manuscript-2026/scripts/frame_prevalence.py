"""Clerkin-style frame prevalence tables for manuscript long-tail analyses.

Frame prevalence: unique frames with >=1 detection of category / total unique
frames in the pool.

Pools:
  - valid129 (and valid85 when using the full-dataset builder): all frames in
    ``merged_frame_detections_with_metadata_filtered-0.27.csv``, or CLIP-passers
    in a 2025.2 packed ``crops_index.csv.gz``.
  - valid85 supplemental (868-hour only): annotation / VQA frame pool
    (rater-validated exemplar frames from the 7,018-crop cohort). On 2025.2
    packed stores, valid85 is the same full CLIP-passer pool restricted to the
    85-class inclusion list (868-hour annotation stems do not match).
"""
from __future__ import annotations

import os
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


def load_detections_from_packed_index(
    emb_root: Path | str,
    *,
    min_sim: float = 0.27,
    exclude_subjects: tuple[str, ...] = ("00270001",),
    exclude_classes: frozenset[str] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """CLIP-passers from 2025.2 ``crops_index.csv.gz`` as ``class_name`` + ``frame_key``.

    ``frame_key`` is ``{video_id}/{frame_number:05d}``. Matches Stage 0: cosine
    >= ``min_sim``, drop ``person`` / ``picture``, drop excluded subjects.
    """
    from packed_crop_embeddings import subject_id_from_video_id

    index_path = Path(emb_root) / "crops_index.csv.gz"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if exclude_classes is None:
        exclude_classes = frozenset({"person", "picture"})
    excl_subj = {str(s).zfill(8) for s in exclude_subjects if s}
    usecols = ["video_id", "frame_number", "class_name", "clip_sim", "passed"]
    write_time_threshold = abs(float(min_sim) - 0.27) < 1e-9
    parts: list[pd.DataFrame] = []
    from packed_crop_embeddings import frame_keep, frame_sampling

    for chunk in pd.read_csv(index_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[frame_keep(chunk["frame_number"].to_numpy())]  # BV_FRAME_STRIDE (default: all)
        chunk["class_name"] = chunk["class_name"].astype(str).str.strip().str.lower()
        if write_time_threshold and "passed" in chunk.columns:
            chunk = chunk[chunk["passed"].astype(bool)]
        else:
            chunk = chunk[pd.to_numeric(chunk["clip_sim"], errors="coerce") >= float(min_sim)]
        if exclude_classes:
            chunk = chunk[~chunk["class_name"].isin(exclude_classes)]
        if excl_subj:
            sid = chunk["video_id"].map(subject_id_from_video_id)
            chunk = chunk[~sid.isin(excl_subj)]
        if chunk.empty:
            continue
        frame_no = chunk["frame_number"].astype(int).astype(str).str.zfill(5)
        chunk = chunk.assign(frame_key=chunk["video_id"].astype(str) + "/" + frame_no)
        parts.append(chunk[["class_name", "frame_key"]])
    if not parts:
        return pd.DataFrame(columns=["class_name", "frame_key"])
    out = pd.concat(parts, ignore_index=True)
    print(
        f"Packed detections: {len(out):,} CLIP-pass boxes from {index_path} "
        f"(min_sim={min_sim:.2f}, frames={out['frame_key'].nunique():,}, "
        f"frame stride/offset={frame_sampling()})"
    )
    return out


def resolve_detections() -> tuple[pd.DataFrame, bool]:
    """Return (detections, is_packed). Packed if ``BV_EMBEDDINGS_BASE`` is a shard store."""
    from packed_crop_embeddings import is_packed_store, packed_root_from_path

    emb = os.environ.get("BV_EMBEDDINGS_BASE", "").strip()
    if emb:
        root = packed_root_from_path(Path(emb).expanduser()) or Path(emb).expanduser()
        if is_packed_store(root):
            min_sim = float(os.environ.get("BV_CLIP_FILTER_LIST_THRESHOLD", "0.27"))
            return load_detections_from_packed_index(root, min_sim=min_sim), True
    from _bootstrap import PROJECT_ROOT

    csv = PROJECT_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
    return load_detections(csv), False


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
    *,
    full_pool_for_valid85: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return (table, pool_label) for valid129 or valid85."""
    if category_set == "valid129":
        return build_full_dataset_table(det, included, semantic_map), "full infant-view dataset"
    if category_set == "valid85" and full_pool_for_valid85:
        return (
            build_full_dataset_table(det, included, semantic_map),
            "full 2025.2 infant-view dataset (valid85 inclusion list)",
        )
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
