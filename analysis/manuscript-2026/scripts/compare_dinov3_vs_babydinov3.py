#!/usr/bin/env python3
"""Compare facebook DINOv3 vs BabyDINOv3 exemplar embeddings (BV and THINGS).

DINOv3 (768-d) and BabyDINOv3 (1024-d) live in different spaces, so we compare:
  - Category RDMs (cosine distance, lower triangle)
  - Per-category BV vs THINGS centroid cosine (within each backbone)

Outputs:
  - dinov3_vs_babydinov3_summary_{valid129|valid85}.csv
  - dinov3_vs_babydinov3_per_category_{valid129|valid85}.csv
  - figures/dinov3_vs_babydinov3_scatter_{valid129|valid85}.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
CATEGORY_SET = os.environ.get("CATEGORY_SET", "valid129").strip()

if CATEGORY_SET == "valid85":
    OUTPUT_RUN_ROOT = PREPRINT_DIR / "supplemental_results_valid85cats_04302026"
else:
    OUTPUT_RUN_ROOT = SCRIPT_DIR / "main_results_valid129s_04302026"

RESULTS_DIR = OUTPUT_RUN_ROOT / "results"
FIGURES_DIR = OUTPUT_RUN_ROOT / "figures"
EXEMPLAR_EMBED_DIR = PREPRINT_DIR / "exemplar_set_embeddings" / CATEGORY_SET
INCLUDED_CATEGORIES_TXT = PROJECT_ROOT / "data" / f"included_categories_{CATEGORY_SET}.txt"


def load_included_categories(path: Path) -> list[str]:
    return [ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()]


def load_embedding_csv(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "category" in df.columns:
        categories = df["category"].astype(str).str.strip().str.lower().tolist()
        embedding_df = df.drop(columns=["category"])
    elif "Unnamed: 0" in df.columns:
        categories = df["Unnamed: 0"].astype(str).str.strip().str.lower().tolist()
        embedding_df = df.drop(columns=["Unnamed: 0"])
    else:
        first_col = df.columns[0]
        if pd.api.types.is_object_dtype(df[first_col]):
            categories = df[first_col].astype(str).str.strip().str.lower().tolist()
            embedding_df = df.drop(columns=[first_col])
        else:
            categories = [str(c).strip().lower() for c in df.index]
            embedding_df = df.copy()
    embedding_df = embedding_df.select_dtypes(include=[np.number])
    emb = embedding_df.to_numpy(dtype=np.float64)
    if len(categories) != emb.shape[0]:
        raise ValueError(f"Row mismatch in {path}: {len(categories)} vs {emb.shape[0]}")
    return categories, emb


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings, metric="cosine"))


def vectorize_lower_triangle(rdm: np.ndarray) -> np.ndarray:
    il = np.tril_indices_from(rdm, k=-1)
    return rdm[il]


def compare_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> dict[str, float]:
    pr, pp = pearsonr(vec_a, vec_b)
    sr, sp = spearmanr(vec_a, vec_b)
    kr, kp = kendalltau(vec_a, vec_b)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "kendall_r": float(kr),
        "kendall_p": float(kp),
        "mae": float(np.mean(np.abs(vec_a - vec_b))),
        "rmse": float(np.sqrt(np.mean((vec_a - vec_b) ** 2))),
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum((a / na) * (b / nb), axis=1)


def align_pair(
    cats_order: list[str],
    cats_a: list[str],
    emb_a: np.ndarray,
    cats_b: list[str],
    emb_b: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx_a = {c: i for i, c in enumerate(cats_a)}
    idx_b = {c: i for i, c in enumerate(cats_b)}
    common = [c for c in cats_order if c in idx_a and c in idx_b]
    al_a = np.stack([emb_a[idx_a[c]] for c in common], axis=0)
    al_b = np.stack([emb_b[idx_b[c]] for c in common], axis=0)
    return common, al_a, al_b


def main() -> None:
    included = load_included_categories(INCLUDED_CATEGORIES_TXT)

    paths = {
        "bv": {
            "dinov3": EXEMPLAR_EMBED_DIR / f"bv_dinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv",
            "babydinov3": EXEMPLAR_EMBED_DIR
            / f"bv_babydinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv",
        },
        "things": {
            "dinov3": EXEMPLAR_EMBED_DIR / f"things_dinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv",
            "babydinov3": EXEMPLAR_EMBED_DIR
            / f"things_babydinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv",
        },
    }

    d_cats, d_emb = load_embedding_csv(paths["bv"]["dinov3"])
    b_cats, b_emb = load_embedding_csv(paths["bv"]["babydinov3"])
    th_d_cats, th_d_emb = load_embedding_csv(paths["things"]["dinov3"])
    th_b_cats, th_b_emb = load_embedding_csv(paths["things"]["babydinov3"])

    common, bv_d, bv_b = align_pair(included, d_cats, d_emb, b_cats, b_emb)
    _, th_d, th_b = align_pair(common, th_d_cats, th_d_emb, th_b_cats, th_b_emb)

    summary_rows: list[dict] = []
    scatter_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for dataset, d_al, b_al in (
        ("bv", bv_d, bv_b),
        ("things", th_d, th_b),
    ):
        vec_d = vectorize_lower_triangle(compute_rdm(d_al))
        vec_b = vectorize_lower_triangle(compute_rdm(b_al))
        scatter_data[dataset] = (vec_d, vec_b)
        summary_rows.append(
            {
                "dataset": dataset,
                "comparison": "rdm_lower_triangle",
                "n_categories": len(common),
                "n_pairs": int(len(vec_d)),
                **compare_vectors(vec_d, vec_b),
            }
        )

    bv_th_dino = cosine_rows(bv_d, th_d)
    bv_th_baby = cosine_rows(bv_b, th_b)
    pr_bvth, pp_bvth = pearsonr(bv_th_dino, bv_th_baby)
    sr_bvth, sp_bvth = spearmanr(bv_th_dino, bv_th_baby)
    kt_bvth = kendalltau(bv_th_dino, bv_th_baby)
    summary_rows.append(
        {
            "dataset": "per_category",
            "comparison": "bv_vs_things_cosine_dinov3_vs_babydinov3",
            "n_categories": len(common),
            "n_pairs": len(common),
            "pearson_r": float(pr_bvth),
            "pearson_p": float(pp_bvth),
            "spearman_r": float(sr_bvth),
            "spearman_p": float(sp_bvth),
            "kendall_r": float(kt_bvth.statistic),
            "kendall_p": float(kt_bvth.pvalue),
            "mae": float(np.mean(np.abs(bv_th_dino - bv_th_baby))),
            "rmse": float(np.sqrt(np.mean((bv_th_dino - bv_th_baby) ** 2))),
        }
    )

    per_cat_df = pd.DataFrame(
        {
            "category": common,
            "bv_vs_things_cosine_dinov3": bv_th_dino,
            "bv_vs_things_cosine_babydinov3": bv_th_baby,
            "bv_things_cosine_delta_baby_minus_dino": bv_th_baby - bv_th_dino,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    out_summary = RESULTS_DIR / f"dinov3_vs_babydinov3_summary_{CATEGORY_SET}.csv"
    out_per_cat = RESULTS_DIR / f"dinov3_vs_babydinov3_per_category_{CATEGORY_SET}.csv"
    summary_df.to_csv(out_summary, index=False)
    per_cat_df.to_csv(out_per_cat, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panels = [
        ("bv", "BabyView RDM", scatter_data["bv"]),
        ("things", "THINGS RDM", scatter_data["things"]),
    ]
    for ax, _key, title, (x, y) in zip(axes, [p[0] for p in panels], [p[1] for p in panels], [p[2] for p in panels]):
        row = summary_df[
            (summary_df["dataset"] == _key) & (summary_df["comparison"] == "rdm_lower_triangle")
        ].iloc[0]
        ax.scatter(x, y, s=8, alpha=0.25, c="#455a64", edgecolors="none")
        lim = max(float(x.max()), float(y.max()))
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("DINOv3 cosine distance")
        ax.set_ylabel("BabyDINOv3 cosine distance")
        ax.set_title(f"{title}\nPearson r={row['pearson_r']:.3f}, Spearman ρ={row['spearman_r']:.3f}")
        ax.set_aspect("equal", adjustable="box")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(bv_th_dino, bv_th_baby, s=36, alpha=0.7, c="#2e7d32", edgecolors="white", linewidths=0.3)
    lim = min(1.0, max(float(bv_th_dino.max()), float(bv_th_baby.max())) + 0.05)
    ax2.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("BV vs THINGS cosine (DINOv3)")
    ax2.set_ylabel("BV vs THINGS cosine (BabyDINOv3)")
    ax2.set_title(f"Per-category alignment\nPearson r={pr_bvth:.3f}, n={len(common)}")
    ax2.set_aspect("equal", adjustable="box")
    fig2.tight_layout()
    fig2_path = FIGURES_DIR / f"dinov3_vs_babydinov3_bv_things_cosine_{CATEGORY_SET}.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig.suptitle(f"DINOv3 vs BabyDINOv3 RDMs ({CATEGORY_SET}, n={len(common)})", y=1.02)
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"dinov3_vs_babydinov3_scatter_{CATEGORY_SET}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_per_cat}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig2_path}")
    print("\nRDM correlation (dinov3 vs babydinov3, lower triangle):")
    print(
        summary_df[summary_df["comparison"] == "rdm_lower_triangle"][
            ["dataset", "pearson_r", "spearman_r", "mae", "rmse"]
        ].to_string(index=False)
    )
    print(f"\nBV vs THINGS cosine (per category): dinov3 vs babydinov3 r={pr_bvth:.3f}")


if __name__ == "__main__":
    main()
