#!/usr/bin/env python3
"""Headless runner for 06_exemplar_set_zscore_embeddings.ipynb (tmux / nohup friendly).

BabyDINOv3 only: exemplar .npy → per-category mean → z-score within category set.
(CLIP / facebook DINOv3 paths are skipped; see notebook 06 for commented blocks.)

Examples:
  cd analysis/manuscript-2026
  python scripts/exemplar_set_zscore_embeddings.py --all-category-sets

  BV_INCLUDE_BABYDINOV3=1 BV_CATEGORY_SET=valid129 python scripts/exemplar_set_zscore_embeddings.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
OUT_ROOT = PREPRINT_DIR / "exemplar_set_embeddings"

CATEGORY_FILES = {
    "valid85": DATA_DIR / "included_categories_valid85.txt",
    "valid129": DATA_DIR / "included_categories_valid129.txt",
}

PER_CLASS_PRECISION_CSV = ANNOTATION_DIR / "per_class_validation_data.csv"
PER_FILE_PRECISION_CSV = ANNOTATION_DIR / "per_file_precision_data.csv"
SAMPLED_EXEMPLAR_CSV = ANNOTATION_DIR / (
    "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--category-set",
        choices=tuple(CATEGORY_FILES),
        default=os.environ.get("BV_CATEGORY_SET", "valid129"),
        help="valid129 or valid85 (default: BV_CATEGORY_SET or valid129)",
    )
    p.add_argument(
        "--all-category-sets",
        action="store_true",
        help="Run valid129 then valid85 (overrides --category-set)",
    )
    return p.parse_args()


def load_config() -> dict:
    precision_threshold = float(os.environ.get("BV_PRECISION_THRESHOLD", "0.6"))
    crop_prefix = os.environ.get("BV_CROP_PATH_PREFIX", "").strip()
    crop_prefix_new = os.environ.get("BV_CROP_PATH_PREFIX_NEW", "").strip()
    emb_base = Path(
        os.environ.get(
            "BV_EMBEDDINGS_BASE",
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings",
        )
    ).expanduser()
    clip_embeddings_dir = Path(
        os.environ.get("BV_CLIP_EMBEDDINGS_DIR", str(emb_base / "clip_embeddings_new"))
    ).expanduser()
    dinov3_embeddings_dir = Path(
        os.environ.get(
            "BV_DINOV3_EMBEDDINGS_DIR",
            str(emb_base / "facebook_dinov3-vitb16-pretrain-lvd1689m"),
        )
    ).expanduser()
    # BabyDINOv3-only run (CLIP/DINOv3 disabled in run_category_set).
    include_babydinov3 = True
    babydinov3_checkpoint_step = os.environ.get("BV_BABYDINOV3_CHECKPOINT_STEP", "119999").strip()
    default_babydinov3_dir = emb_base / "babydinov3_grad_accum_1" / f"step_{babydinov3_checkpoint_step}"
    babydinov3_embeddings_dir = Path(
        os.environ.get("BV_BABYDINOV3_EMBEDDINGS_DIR", str(default_babydinov3_dir))
    ).expanduser()
    clip_filter_list_threshold = os.environ.get("BV_CLIP_FILTER_LIST_THRESHOLD", "0.27").strip()
    clip_filter_list_path = Path(
        os.environ.get(
            "BV_CLIP_FILTER_LIST",
            str(
                emb_base
                / f"clip_image_embeddings_filtered-by-clip-{clip_filter_list_threshold}_exclude-people_exclude-subject-00270001.txt"
            ),
        )
    ).expanduser()
    return {
        "precision_threshold": precision_threshold,
        "crop_prefix": crop_prefix,
        "crop_prefix_new": crop_prefix_new,
        "clip_embeddings_dir": clip_embeddings_dir,
        "dinov3_embeddings_dir": dinov3_embeddings_dir,
        "include_babydinov3": include_babydinov3,
        "babydinov3_checkpoint_step": babydinov3_checkpoint_step,
        "babydinov3_embeddings_dir": babydinov3_embeddings_dir,
        "clip_filter_list_path": clip_filter_list_path,
        "clip_filter_list_threshold": clip_filter_list_threshold,
    }


def resolve_category_subdir(embed_root: Path, cat: str) -> Path | None:
    direct = embed_root / cat
    if direct.is_dir():
        return direct
    for p in embed_root.iterdir():
        if p.is_dir() and p.name.lower() == cat.lower():
            return p
    return None


def build_category_subdir_cache(
    embed_root: Path,
    allowed_categories: set[str],
    label: str = "embed",
) -> dict[str, Path]:
    cache: dict[str, Path] = {}
    for cat in tqdm(sorted(allowed_categories), desc=f"Index {label} category dirs"):
        sub = resolve_category_subdir(embed_root, cat)
        if sub is not None:
            cache[cat] = sub
    return cache


def load_included_categories(txt_path: Path) -> list[str]:
    return [line.strip().lower() for line in txt_path.read_text().splitlines() if line.strip()]


def load_valid_classes(per_class_csv: Path, threshold: float) -> set[str]:
    df = pd.read_csv(per_class_csv, usecols=["class", "precision"])
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    return set(df.loc[df["precision"] > threshold, "class"])


def load_valid_pairs(per_file_csv: Path, threshold: float) -> set[tuple[str, str]]:
    df = pd.read_csv(per_file_csv, usecols=["filename", "class", "precision"])
    df = df[df["precision"] > threshold].copy()
    df["class_norm"] = df["class"].astype(str).str.strip().str.lower()
    df["stem"] = (
        df["filename"]
        .astype(str)
        .str.strip()
        .str.rsplit("/", n=1)
        .str[-1]
        .str.rsplit(".", n=1)
        .str[0]
        .str.lower()
    )
    return set(zip(df["class_norm"], df["stem"]))


def remap_absolute_path(p: Path, crop_prefix: str, crop_prefix_new: str) -> Path:
    s = str(p)
    if crop_prefix and crop_prefix_new and s.startswith(crop_prefix):
        return Path(crop_prefix_new + s[len(crop_prefix) :])
    return p


def load_clip_filter_pair_set(filter_list_path: Path, categories_lower: set[str]) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    if not filter_list_path.is_file():
        raise FileNotFoundError(f"CLIP filter list not found: {filter_list_path}")
    with filter_list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            stem = p.stem if p.suffix.lower() == ".npy" else p.name
            category = p.parent.name.strip().lower()
            if category in categories_lower:
                allowed.add((category, stem.lower()))
    return allowed


def load_clip_filter_npy_fname_map(
    filter_list_path: Path, categories_lower: set[str]
) -> dict[tuple[str, str], str]:
    """Map (category_lower, stem_lower) -> on-disk ``.npy`` basename (preserves case)."""
    m: dict[tuple[str, str], str] = {}
    if not filter_list_path.is_file():
        raise FileNotFoundError(f"CLIP filter list not found: {filter_list_path}")
    with filter_list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.suffix.lower() != ".npy":
                continue
            category = p.parent.name.strip().lower()
            if category not in categories_lower:
                continue
            m[(category, p.stem.lower())] = p.name
    return m


def resolve_npy_path(
    embed_root: Path,
    cat: str,
    stem_lower: str,
    npy_fname_map: dict[tuple[str, str], str] | None,
    crop_prefix: str,
    crop_prefix_new: str,
) -> Path:
    fname = npy_fname_map.get((cat, stem_lower)) if npy_fname_map else None
    if fname is None:
        fname = f"{stem_lower}.npy"
    subdir = resolve_category_subdir(embed_root, cat)
    if subdir is not None:
        pn = subdir / fname
        if pn.is_file():
            return remap_absolute_path(pn, crop_prefix, crop_prefix_new)
        for npy in subdir.glob("*.npy"):
            if npy.stem.lower() == stem_lower:
                return remap_absolute_path(npy, crop_prefix, crop_prefix_new)
    return remap_absolute_path(embed_root / cat / fname, crop_prefix, crop_prefix_new)


def collect_paired_npy_paths_from_clip_filter_list(
    filter_list_path: Path,
    clip_embed_root: Path,
    dino_embed_root: Path,
    allowed_categories: set[str],
    crop_prefix: str,
    crop_prefix_new: str,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict]:
    clip_out: dict[str, list[Path]] = {}
    dino_out: dict[str, list[Path]] = {}
    stats = {
        "n_lines_in_clip_filter_list": 0,
        "n_lines_in_allowed_categories": 0,
        "n_pairs_both_models_on_disk": 0,
        "n_lines_missing_clip_or_dino_file": 0,
        "categories_with_subdir_clip": 0,
        "categories_with_subdir_dino": 0,
    }
    seen: dict[str, set[str]] = defaultdict(set)
    clip_subdirs = build_category_subdir_cache(clip_embed_root, allowed_categories, "CLIP")
    dino_subdirs = build_category_subdir_cache(dino_embed_root, allowed_categories, "DINOv3")

    with filter_list_path.open("r") as f:
        for line in tqdm(f, desc="Scan CLIP filter list (CLIP+DINO pair)", unit=" lines", mininterval=2.0):
            line = line.strip()
            if not line:
                continue
            stats["n_lines_in_clip_filter_list"] += 1
            p = Path(line)
            if p.suffix.lower() != ".npy":
                continue
            fname = p.name
            cat = p.parent.name.strip().lower()
            if cat not in allowed_categories:
                continue
            stats["n_lines_in_allowed_categories"] += 1
            key = fname.lower()
            if key in seen[cat]:
                continue
            seen[cat].add(key)

            cdir_c = clip_subdirs.get(cat)
            cdir_d = dino_subdirs.get(cat)
            pc = cdir_c / fname if cdir_c is not None else None
            pd = cdir_d / fname if cdir_d is not None else None
            if cdir_c is None or cdir_d is None or pc is None or pd is None:
                stats["n_lines_missing_clip_or_dino_file"] += 1
                continue
            if not pc.is_file() or not pd.is_file():
                stats["n_lines_missing_clip_or_dino_file"] += 1
                continue
            if cat not in clip_out:
                clip_out[cat] = []
                dino_out[cat] = []
            clip_out[cat].append(remap_absolute_path(pc, crop_prefix, crop_prefix_new))
            dino_out[cat].append(remap_absolute_path(pd, crop_prefix, crop_prefix_new))
            stats["n_pairs_both_models_on_disk"] += 1

    stats["categories_with_subdir_clip"] = len(clip_subdirs)
    stats["categories_with_subdir_dino"] = len(dino_subdirs)

    for cat in sorted(allowed_categories):
        clip_out.setdefault(cat, [])
        dino_out.setdefault(cat, [])
        clip_out[cat] = sorted(dict.fromkeys(clip_out[cat]), key=lambda x: str(x))
        dino_out[cat] = sorted(dict.fromkeys(dino_out[cat]), key=lambda x: str(x))

    print(
        f"  [CLIP filter list] lines={stats['n_lines_in_clip_filter_list']:,} "
        f"in_allowed_cats={stats['n_lines_in_allowed_categories']:,} "
        f"paired_on_disk={stats['n_pairs_both_models_on_disk']:,} "
        f"missing_file_or_dir={stats['n_lines_missing_clip_or_dino_file']:,}"
    )
    return clip_out, dino_out, stats


def collect_npy_paths_from_clip_filter_list(
    filter_list_path: Path,
    embed_root: Path,
    allowed_categories: set[str],
    model_label: str,
    crop_prefix: str,
    crop_prefix_new: str,
) -> tuple[dict[str, list[Path]], dict]:
    out: dict[str, list[Path]] = {}
    stats = {
        "model_label": model_label,
        "n_lines_in_clip_filter_list": 0,
        "n_lines_in_allowed_categories": 0,
        "n_paths_on_disk": 0,
        "n_lines_missing_file_or_dir": 0,
        "categories_with_subdir": 0,
    }
    seen: dict[str, set[str]] = defaultdict(set)
    subdirs = build_category_subdir_cache(embed_root, allowed_categories, model_label)

    with filter_list_path.open("r") as f:
        for line in tqdm(
            f,
            desc=f"Scan CLIP filter list ({model_label})",
            unit=" lines",
            mininterval=2.0,
        ):
            line = line.strip()
            if not line:
                continue
            stats["n_lines_in_clip_filter_list"] += 1
            p = Path(line)
            if p.suffix.lower() != ".npy":
                continue
            fname = p.name
            cat = p.parent.name.strip().lower()
            if cat not in allowed_categories:
                continue
            stats["n_lines_in_allowed_categories"] += 1
            key = fname.lower()
            if key in seen[cat]:
                continue
            seen[cat].add(key)

            cdir = subdirs.get(cat)
            pn = cdir / fname if cdir is not None else None
            if cdir is None or pn is None or not pn.is_file():
                stats["n_lines_missing_file_or_dir"] += 1
                continue
            out.setdefault(cat, []).append(remap_absolute_path(pn, crop_prefix, crop_prefix_new))
            stats["n_paths_on_disk"] += 1

    stats["categories_with_subdir"] = len(subdirs)
    for cat in sorted(allowed_categories):
        out.setdefault(cat, [])
        out[cat] = sorted(dict.fromkeys(out[cat]), key=lambda x: str(x))

    print(
        f"  [CLIP filter list → {model_label}] lines={stats['n_lines_in_clip_filter_list']:,} "
        f"in_allowed_cats={stats['n_lines_in_allowed_categories']:,} "
        f"on_disk={stats['n_paths_on_disk']:,} "
        f"missing_file_or_dir={stats['n_lines_missing_file_or_dir']:,}"
    )
    return out, stats


def valid85_npy_paths_by_category_from_manifest(
    df: pd.DataFrame,
    embed_root: Path,
    clip_filter_pairs: set[tuple[str, str]],
    crop_prefix: str,
    crop_prefix_new: str,
    npy_fname_map: dict[tuple[str, str], str] | None = None,
) -> dict[str, list[Path]]:
    d: dict[str, list[Path]] = defaultdict(list)
    for _, row in df.iterrows():
        cat = str(row["category"]).strip().lower()
        stem = str(row["stem"]).strip().lower()
        if (cat, stem) not in clip_filter_pairs:
            continue
        npy = resolve_npy_path(
            embed_root, cat, stem, npy_fname_map, crop_prefix, crop_prefix_new
        )
        d[cat].append(npy)
    return {k: list(dict.fromkeys(v)) for k, v in sorted(d.items())}


def build_valid85_sampled_exemplar_table(
    included_txt: Path,
    per_class_csv: Path,
    per_file_csv: Path,
    sampled_csv: Path,
    precision_threshold: float,
) -> pd.DataFrame:
    included = set(load_included_categories(included_txt))
    valid_classes = load_valid_classes(per_class_csv, precision_threshold)
    eligible_cats = included & valid_classes
    valid_pairs = load_valid_pairs(per_file_csv, precision_threshold)

    sampled = pd.read_csv(sampled_csv)
    sampled = sampled[sampled["trial_type"] == "regular"].copy()
    sampled["category"] = sampled["category"].astype(str).str.strip().str.lower()
    sampled["stem"] = sampled["stem"].astype(str).str.strip().str.lower()

    mask = sampled.apply(lambda r: (r["category"], r["stem"]) in valid_pairs, axis=1)
    sampled = sampled.loc[mask].copy()
    sampled = sampled[sampled["category"].isin(eligible_cats)].copy()
    return sampled[["category", "path", "stem"]].reset_index(drop=True)


def zscore_rows(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    return (X - mu) / (sig + eps)


def load_normalized_npy_vector(path: Path, crop_prefix: str, crop_prefix_new: str) -> np.ndarray | None:
    p = remap_absolute_path(path, crop_prefix, crop_prefix_new)
    if not p.is_file():
        return None
    v = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def average_precomputed_embeddings_for_categories(
    paths_by_cat: dict[str, list[Path]],
    model_label: str,
    crop_prefix: str,
    crop_prefix_new: str,
) -> tuple[list[str], np.ndarray, dict]:
    categories = sorted(paths_by_cat.keys())
    diag: dict = {"missing_paths": [], "n_paths": {}, "skipped_categories": []}
    cat_vectors: list[np.ndarray] = []
    categories_out: list[str] = []

    for cat in tqdm(categories, desc=f"Average {model_label} per category"):
        paths_in_order = list(dict.fromkeys(paths_by_cat[cat]))
        vecs: list[np.ndarray] = []
        miss: list[str] = []
        path_iter = tqdm(
            paths_in_order,
            desc=f"{cat} ({len(paths_in_order):,} crops)",
            leave=False,
            disable=len(paths_in_order) < 200,
        )
        for p in path_iter:
            v = load_normalized_npy_vector(p, crop_prefix, crop_prefix_new)
            if v is None:
                miss.append(str(remap_absolute_path(p, crop_prefix, crop_prefix_new)))
            else:
                vecs.append(v)
        diag["missing_paths"].extend(miss[:20])
        diag["n_paths"][cat] = {
            "total": len(paths_in_order),
            "unique_paths": len(paths_in_order),
            "found": len(vecs),
        }
        if not vecs:
            diag["skipped_categories"].append(cat)
            continue
        cat_vectors.append(np.mean(np.stack(vecs, axis=0), axis=0))
        categories_out.append(cat)

    X = np.stack(cat_vectors, axis=0) if cat_vectors else np.zeros((0, 0))
    return categories_out, X, diag


def save_cat_emb(csv_path: Path, cats: list[str], X: np.ndarray, dim_cols: list[str]) -> None:
    df = pd.DataFrame(X, columns=dim_cols)
    df.insert(0, "category", cats)
    df.to_csv(csv_path, index=False)


def run_category_set(category_set: str, cfg: dict) -> None:
    inc_path = CATEGORY_FILES[category_set]
    clip_filter_list_path = cfg["clip_filter_list_path"]
    crop_prefix = cfg["crop_prefix"]
    crop_prefix_new = cfg["crop_prefix_new"]
    babydinov3_paths_by_cat = None
    babydinov3_scan_stats = None

    if category_set == "valid129":
        if not clip_filter_list_path.is_file():
            raise FileNotFoundError(f"Missing CLIP filter list: {clip_filter_list_path}")
        allowed = set(load_included_categories(inc_path))
        inc_all = sorted(allowed)
        if not cfg["babydinov3_embeddings_dir"].is_dir():
            raise FileNotFoundError(
                f"BabyDINOv3 embeddings dir not found: {cfg['babydinov3_embeddings_dir']}"
            )
        babydinov3_paths_by_cat, babydinov3_scan_stats = collect_npy_paths_from_clip_filter_list(
            clip_filter_list_path,
            cfg["babydinov3_embeddings_dir"],
            allowed,
            "babydinov3",
            crop_prefix,
            crop_prefix_new,
        )
        babydinov3_paths_by_cat = {c: babydinov3_paths_by_cat.get(c, []) for c in inc_all}
        n_paths_total = babydinov3_scan_stats["n_paths_on_disk"]
        n_cat = len(inc_all)
        n_clip_csv_cats = sum(1 for c in inc_all if babydinov3_paths_by_cat[c])
        clip_filter_list_stats = babydinov3_scan_stats
        exemplar_source = "flat_babydinov3_npy_clip_filter_list"
    elif category_set == "valid85":
        exemplar_df = build_valid85_sampled_exemplar_table(
            inc_path,
            PER_CLASS_PRECISION_CSV,
            PER_FILE_PRECISION_CSV,
            SAMPLED_EXEMPLAR_CSV,
            cfg["precision_threshold"],
        )
        elig_cats = set(exemplar_df["category"].astype(str).str.strip().str.lower())
        clip_filter_pairs = load_clip_filter_pair_set(clip_filter_list_path, elig_cats)
        npy_fname_map = load_clip_filter_npy_fname_map(clip_filter_list_path, elig_cats)
        if not cfg["babydinov3_embeddings_dir"].is_dir():
            raise FileNotFoundError(
                f"BabyDINOv3 embeddings dir not found: {cfg['babydinov3_embeddings_dir']}"
            )
        babydinov3_paths_by_cat = valid85_npy_paths_by_category_from_manifest(
            exemplar_df,
            cfg["babydinov3_embeddings_dir"],
            clip_filter_pairs,
            crop_prefix,
            crop_prefix_new,
            npy_fname_map,
        )
        n_paths_total = sum(len(v) for v in babydinov3_paths_by_cat.values())
        n_cat = len(babydinov3_paths_by_cat)
        n_clip_csv_cats = sum(1 for v in babydinov3_paths_by_cat.values() if v)
        clip_filter_list_stats = {
            "n_category_stem_pairs_in_clip_filter_list_for_eligible_cats": len(clip_filter_pairs),
        }
        exemplar_source = "sampled_csv_validated_clip_filter_list_babydinov3_npy"
    else:
        raise ValueError(category_set)

    print(f"\n=== {category_set} (BabyDINOv3 only) ===")
    out_dir = OUT_ROOT / category_set
    out_dir.mkdir(parents=True, exist_ok=True)

    if babydinov3_paths_by_cat is None:
        raise RuntimeError("babydinov3_paths_by_cat was not built")

    cats_bd, Xbd_raw, diag_bd = average_precomputed_embeddings_for_categories(
        babydinov3_paths_by_cat, "BabyDINOv3", crop_prefix, crop_prefix_new
    )
    dim_cols_bd = [f"dim_{i}" for i in range(Xbd_raw.shape[1])]
    save_cat_emb(
        out_dir / f"bv_babydinov3_exemplar_avg_zscore_within_{category_set}.csv",
        cats_bd,
        zscore_rows(Xbd_raw),
        dim_cols_bd,
    )
    save_cat_emb(
        out_dir / f"bv_babydinov3_exemplar_avg_raw_within_{category_set}.csv",
        cats_bd,
        Xbd_raw,
        dim_cols_bd,
    )

    meta = {
        "category_set": category_set,
        "run_mode": "babydinov3_only",
        "exemplar_source": exemplar_source,
        "included_categories_txt": str(inc_path),
        "n_categories": len(cats_bd),
        "n_unique_npy_paths": n_paths_total,
        "n_inclusion_categories": n_cat,
        "n_categories_with_at_least_one_npy": n_clip_csv_cats,
        "precision_threshold": cfg["precision_threshold"],
        "clip_filter_list_path": str(clip_filter_list_path),
        "clip_filter_list_stats": clip_filter_list_stats,
        "babydinov3_checkpoint_step": cfg["babydinov3_checkpoint_step"],
        "babydinov3_embeddings_dir": str(cfg["babydinov3_embeddings_dir"]),
        "categories_alphabetical": cats_bd,
        "diagnostics_babydinov3": {k: v for k, v in diag_bd.items() if k != "missing_paths"},
        "n_missing_paths_reported_babydinov3": len(diag_bd.get("missing_paths", [])),
    }
    (out_dir / "exemplar_embedding_run.json").write_text(json.dumps(meta, indent=2))
    print("Wrote:", out_dir / f"bv_babydinov3_exemplar_avg_zscore_within_{category_set}.csv")
    print("Wrote:", out_dir / f"bv_babydinov3_exemplar_avg_raw_within_{category_set}.csv")
    print("Wrote:", out_dir / "exemplar_embedding_run.json")


def main() -> int:
    args = parse_args()
    cfg = load_config()
    category_sets = ["valid129", "valid85"] if args.all_category_sets else [args.category_set]

    print("[exemplar_set_zscore_embeddings] PROJECT_ROOT", PROJECT_ROOT)
    print("category_sets", category_sets)
    print("run_mode: babydinov3_only")
    print("CLIP_FILTER_LIST_PATH", cfg["clip_filter_list_path"], "exists:", cfg["clip_filter_list_path"].is_file())
    print("BABYDINOV3_EMBEDDINGS_DIR", cfg["babydinov3_embeddings_dir"], "exists:", cfg["babydinov3_embeddings_dir"].is_dir())

    for category_set in category_sets:
        run_category_set(category_set, cfg)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
