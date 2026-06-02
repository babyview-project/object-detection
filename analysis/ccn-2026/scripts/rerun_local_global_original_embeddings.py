from __future__ import annotations

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent

import sys

if str(CCN_DIR) not in sys.path:
    sys.path.insert(0, str(CCN_DIR))

from load_things_embeddings import (
    THINGS_CLIP_NPY_DIR,
    THINGS_DINOV3_DIR,
    load_things_clip_from_dir,
    load_things_dinov3_from_dir,
)

FILTER_THRESHOLD = 0.27
GROUPED_EMBEDDINGS_BASE = Path(
    os.getenv("BV_EMBEDDINGS_BASE", "SET_BV_EMBEDDINGS_BASE")
).expanduser()
GROUPED_EMBEDDINGS_DIRS = {
    "clip": GROUPED_EMBEDDINGS_BASE / f"clip_embeddings_grouped_by_age-mo_filtered-{FILTER_THRESHOLD}_normalized",
    "dinov3": GROUPED_EMBEDDINGS_BASE / f"dinov3_embeddings_grouped_by_age-mo_filtered-{FILTER_THRESHOLD}_normalized",
}
EXCLUDED_SUBJECT = "00270001"
DEFAULT_OUT_DIR = CCN_DIR / "plotC_knn_diversity_outputs" / "original_embeddings_rerun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local coherence/global dispersion from original per-exemplar embeddings."
    )
    parser.add_argument("--category-set", default="valid129", choices=["valid85", "valid129"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-exemplars", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_allowed_categories(category_set: str) -> set[str]:
    path = REPO_ROOT / "data" / f"included_categories_{category_set}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing categories file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_bv_embeddings(
    embedding_type: str,
    allowed_categories: set[str] | None,
    excluded_subject: str | None,
    min_exemplars: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[str, int | None]]]]:
    grouped_dir = GROUPED_EMBEDDINGS_DIRS[embedding_type]
    if not grouped_dir.exists():
        raise FileNotFoundError(f"BV grouped dir not found: {grouped_dir}")

    category_embeddings: dict[str, np.ndarray] = {}
    category_exemplar_ids: dict[str, list[tuple[str, int | None]]] = {}
    for cat_folder in sorted(grouped_dir.iterdir()):
        if not cat_folder.is_dir():
            continue
        cat_name = cat_folder.name.strip().lower()
        if allowed_categories is not None and cat_name not in allowed_categories:
            continue
        embs: list[np.ndarray] = []
        ids: list[tuple[str, int | None]] = []
        for f in sorted(cat_folder.glob("*.npy")):
            stem = f.stem
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject_id = parts[0]
            try:
                age_mo = int(parts[1])
            except ValueError:
                age_mo = None
            if excluded_subject and subject_id == excluded_subject:
                continue
            try:
                e = np.load(f)
                e = np.asarray(e, dtype=np.float64).flatten()
                embs.append(e)
                ids.append((subject_id, age_mo))
            except Exception:
                continue
        if len(embs) >= min_exemplars:
            category_embeddings[cat_name] = np.array(embs)
            category_exemplar_ids[cat_name] = ids
    return category_embeddings, category_exemplar_ids


def compute_category_metrics(
    category_embeddings: dict[str, np.ndarray], k: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for cat in sorted(category_embeddings.keys()):
        x = category_embeddings[cat]
        n = x.shape[0]
        effective_k = min(k, n - 1)
        if effective_k < 1:
            rows.append(
                {
                    "category": cat,
                    "n_exemplars": n,
                    "k": k,
                    "effective_k": effective_k,
                    "mean_knn_dist": np.nan,
                    "mean_pairwise_dist": np.nan,
                    "local_coherence": np.nan,
                    "global_dispersion": np.nan,
                    "local_over_global": np.nan,
                }
            )
            continue

        nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        nn.fit(x)
        dists, _ = nn.kneighbors(x)
        mean_knn = np.mean(dists[:, 1:], axis=1)
        pw = pairwise_distances(x, metric="euclidean")
        iu = np.triu_indices(n, k=1)
        mean_pairwise_dist = float(np.mean(pw[iu])) if len(iu[0]) > 0 else np.nan
        centroid = np.mean(x, axis=0, keepdims=True)
        centroid_dist = np.linalg.norm(x - centroid, axis=1)

        mean_knn_dist = float(np.mean(mean_knn))
        local_coherence = float(1.0 / mean_knn_dist) if mean_knn_dist > 0 else np.nan
        global_dispersion = float(np.mean(centroid_dist))
        local_over_global = float(local_coherence / global_dispersion) if global_dispersion > 0 else np.nan
        rows.append(
            {
                "category": cat,
                "n_exemplars": n,
                "k": k,
                "effective_k": effective_k,
                "mean_knn_dist": mean_knn_dist,
                "mean_pairwise_dist": mean_pairwise_dist,
                "local_coherence": local_coherence,
                "global_dispersion": global_dispersion,
                "local_over_global": local_over_global,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("local_over_global", ascending=False).reset_index(drop=True)


def compare_things_vs_bv(things_df: pd.DataFrame, bv_df: pd.DataFrame) -> pd.DataFrame:
    t = things_df.rename(
        columns={
            "n_exemplars": "things_n_exemplars",
            "mean_knn_dist": "things_mean_knn_dist",
            "mean_pairwise_dist": "things_mean_pairwise_dist",
            "local_coherence": "things_local_coherence",
            "global_dispersion": "things_global_dispersion",
            "local_over_global": "things_local_over_global",
        }
    )
    b = bv_df.rename(
        columns={
            "n_exemplars": "bv_n_exemplars",
            "mean_knn_dist": "bv_mean_knn_dist",
            "mean_pairwise_dist": "bv_mean_pairwise_dist",
            "local_coherence": "bv_local_coherence",
            "global_dispersion": "bv_global_dispersion",
            "local_over_global": "bv_local_over_global",
        }
    )
    # Compare on common categories only; effective_k can differ across datasets.
    m = t.merge(b, on=["category"], how="inner")
    m["delta_local_over_global_bv_minus_things"] = m["bv_local_over_global"] - m["things_local_over_global"]
    m["delta_local_coherence_bv_minus_things"] = m["bv_local_coherence"] - m["things_local_coherence"]
    m["delta_pairwise_dist_bv_minus_things"] = m["bv_mean_pairwise_dist"] - m["things_mean_pairwise_dist"]
    m["delta_global_dispersion_bv_minus_things"] = m["bv_global_dispersion"] - m["things_global_dispersion"]
    return m.sort_values("delta_local_over_global_bv_minus_things", ascending=False).reset_index(drop=True)


def run(category_set: str, k: int, min_exemplars: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_categories = load_allowed_categories(category_set)

    bv_by_embedding: dict[str, dict[str, np.ndarray]] = {}
    things_by_embedding: dict[str, dict[str, np.ndarray]] = {}
    common_by_embedding: dict[str, set[str]] = {}
    for embedding in ["clip", "dinov3"]:
        bv_embeddings, _ = load_bv_embeddings(
            embedding_type=embedding,
            allowed_categories=allowed_categories,
            excluded_subject=EXCLUDED_SUBJECT,
            min_exemplars=min_exemplars,
        )
        if embedding == "clip":
            things_embeddings, _ = load_things_clip_from_dir(
                THINGS_CLIP_NPY_DIR, allowed_categories=allowed_categories, min_exemplars=min_exemplars
            )
        else:
            things_embeddings, _ = load_things_dinov3_from_dir(
                THINGS_DINOV3_DIR, allowed_categories=allowed_categories, min_exemplars=min_exemplars
            )
        bv_by_embedding[embedding] = bv_embeddings
        things_by_embedding[embedding] = things_embeddings
        common_by_embedding[embedding] = set(bv_embeddings) & set(things_embeddings)

    common_all_embeddings = sorted(common_by_embedding["clip"] & common_by_embedding["dinov3"])
    (out_dir / f"common_categories_all_embeddings_{category_set}.txt").write_text(
        "\n".join(common_all_embeddings) + "\n",
        encoding="utf-8",
    )
    print(f"[all-models] shared categories across BV+THINGS and CLIP+DINOv3: {len(common_all_embeddings)}")

    for embedding in ["clip", "dinov3"]:
        bv_embeddings = {
            c: bv_by_embedding[embedding][c]
            for c in common_all_embeddings
            if c in bv_by_embedding[embedding]
        }
        things_embeddings = {
            c: things_by_embedding[embedding][c]
            for c in common_all_embeddings
            if c in things_by_embedding[embedding]
        }

        things_df = compute_category_metrics(things_embeddings, k)
        bv_df = compute_category_metrics(bv_embeddings, k)
        compare_df = compare_things_vs_bv(things_df, bv_df)

        things_path = out_dir / f"things_{embedding}_local_global_k{k}_{category_set}.csv"
        bv_path = out_dir / f"bv_{embedding}_local_global_k{k}_{category_set}.csv"
        compare_path = out_dir / f"things_vs_bv_{embedding}_local_global_k{k}_{category_set}.csv"
        things_df.to_csv(things_path, index=False)
        bv_df.to_csv(bv_path, index=False)
        compare_df.to_csv(compare_path, index=False)

        print(f"[{embedding}] THINGS categories after shared filter: {len(things_df)}")
        print(f"[{embedding}] BabyView categories after shared filter: {len(bv_df)}")
        print(f"[{embedding}] overlap categories after shared filter: {len(compare_df)}")
        if len(compare_df) > 1:
            corr = compare_df[["things_local_over_global", "bv_local_over_global"]].corr().iloc[0, 1]
            print(f"[{embedding}] corr(local_over_global THINGS vs BV): {corr:.4f}")
        print(f"[{embedding}] wrote: {compare_path}")


if __name__ == "__main__":
    args = parse_args()
    run(args.category_set, args.k, args.min_exemplars, args.out_dir)
