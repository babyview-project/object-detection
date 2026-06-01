from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent
PREPRINT_EMBED_DIR = REPO_ROOT / "analysis" / "manuscript-2026" / "exemplar_set_embeddings"
DEFAULT_OUT_DIR = CCN_DIR / "plotC_knn_diversity_outputs" / "zscore_rerun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute local coherence/global dispersion from z-scored exemplar embeddings "
            "and compare THINGS against BabyView."
        )
    )
    parser.add_argument("--valid-set", default="valid129", choices=["valid85", "valid129"])
    parser.add_argument("--k", type=int, default=5, help="k for within-category kNN.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_embedding_csv(valid_set: str, dataset: str, embedding: str) -> pd.DataFrame:
    csv_path = PREPRINT_EMBED_DIR / valid_set / f"{dataset}_{embedding}_exemplar_avg_zscore_within_{valid_set}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing expected input: {csv_path}")
    return pd.read_csv(csv_path)


def embedding_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("No embedding dimension columns found (expected dim_*)")
    x = df[dim_cols].to_numpy(dtype=np.float64)
    categories = df["category"].astype(str).str.strip().str.lower().tolist()
    return x, categories


def per_category_metrics(df: pd.DataFrame, k: int) -> pd.DataFrame:
    x, categories = embedding_matrix(df)
    cat_arr = np.array(categories)
    rows: list[dict] = []
    for cat in sorted(set(categories)):
        idx = np.where(cat_arr == cat)[0]
        x_cat = x[idx]
        n = x_cat.shape[0]
        if n < 2:
            rows.append(
                {
                    "category": cat,
                    "n_exemplars": n,
                    "k": k,
                    "effective_k": np.nan,
                    "mean_knn_dist": np.nan,
                    "local_coherence": np.nan,
                    "global_dispersion": np.nan,
                    "local_over_global": np.nan,
                }
            )
            continue

        effective_k = min(k, n - 1)
        nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        nn.fit(x_cat)
        dists, _ = nn.kneighbors(x_cat)
        mean_knn = dists[:, 1:].mean(axis=1)
        centroid = x_cat.mean(axis=0, keepdims=True)
        centroid_dists = np.linalg.norm(x_cat - centroid, axis=1)

        mean_knn_dist = float(np.mean(mean_knn))
        global_dispersion = float(np.mean(centroid_dists))
        local_coherence = float(1.0 / mean_knn_dist) if mean_knn_dist > 0 else np.nan
        local_over_global = float(local_coherence / global_dispersion) if global_dispersion > 0 else np.nan

        rows.append(
            {
                "category": cat,
                "n_exemplars": n,
                "k": k,
                "effective_k": effective_k,
                "mean_knn_dist": mean_knn_dist,
                "local_coherence": local_coherence,
                "global_dispersion": global_dispersion,
                "local_over_global": local_over_global,
            }
        )

    out = pd.DataFrame(rows)
    out["local_rank_high_to_low"] = out["local_coherence"].rank(ascending=False, method="average")
    out["global_rank_high_to_low"] = out["global_dispersion"].rank(ascending=False, method="average")
    out["ratio_rank_high_to_low"] = out["local_over_global"].rank(ascending=False, method="average")
    return out.sort_values("local_over_global", ascending=False).reset_index(drop=True)


def compare_things_vs_bv(things_df: pd.DataFrame, bv_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["category", "n_exemplars", "local_coherence", "global_dispersion", "local_over_global"]
    things = things_df[cols].rename(
        columns={
            "n_exemplars": "things_n_exemplars",
            "local_coherence": "things_local_coherence",
            "global_dispersion": "things_global_dispersion",
            "local_over_global": "things_local_over_global",
        }
    )
    bv = bv_df[cols].rename(
        columns={
            "n_exemplars": "bv_n_exemplars",
            "local_coherence": "bv_local_coherence",
            "global_dispersion": "bv_global_dispersion",
            "local_over_global": "bv_local_over_global",
        }
    )
    merged = things.merge(bv, on="category", how="inner")
    merged["delta_local_over_global_bv_minus_things"] = (
        merged["bv_local_over_global"] - merged["things_local_over_global"]
    )
    merged["delta_local_coherence_bv_minus_things"] = (
        merged["bv_local_coherence"] - merged["things_local_coherence"]
    )
    merged["delta_global_dispersion_bv_minus_things"] = (
        merged["bv_global_dispersion"] - merged["things_global_dispersion"]
    )
    return merged.sort_values("delta_local_over_global_bv_minus_things", ascending=False).reset_index(drop=True)


def run(valid_set: str, k: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for embedding in ["clip", "dinov3"]:
        things_src = load_embedding_csv(valid_set, "things", embedding)
        bv_src = load_embedding_csv(valid_set, "bv", embedding)

        things_metrics = per_category_metrics(things_src, k=k)
        bv_metrics = per_category_metrics(bv_src, k=k)
        comparison = compare_things_vs_bv(things_metrics, bv_metrics)

        things_path = out_dir / f"things_{embedding}_local_global_k{k}_{valid_set}.csv"
        bv_path = out_dir / f"bv_{embedding}_local_global_k{k}_{valid_set}.csv"
        cmp_path = out_dir / f"things_vs_bv_{embedding}_local_global_k{k}_{valid_set}.csv"

        things_metrics.to_csv(things_path, index=False)
        bv_metrics.to_csv(bv_path, index=False)
        comparison.to_csv(cmp_path, index=False)

        print(f"[{embedding}] saved THINGS metrics: {things_path}")
        print(f"[{embedding}] saved BabyView metrics: {bv_path}")
        print(f"[{embedding}] saved THINGS vs BabyView comparison: {cmp_path}")
        print(f"[{embedding}] overlap categories: {len(comparison)}")
        if len(comparison) > 1:
            corr = comparison[["things_local_over_global", "bv_local_over_global"]].corr().iloc[0, 1]
            print(f"[{embedding}] corr(things_ratio, bv_ratio): {corr:.4f}")


if __name__ == "__main__":
    args = parse_args()
    run(valid_set=args.valid_set, k=args.k, out_dir=args.out_dir)
