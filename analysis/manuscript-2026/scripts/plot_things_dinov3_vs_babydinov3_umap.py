#!/usr/bin/env python3
"""THINGS-only 2D UMAP: pretrained DINOv3 vs BabyDINOv3 category embeddings (CDI-colored).

Uses the same exemplar-set z-scored tables as the preprint RDM pipeline
(``things_{dinov3,babydinov3}_exemplar_avg_zscore_within_{valid129|valid85}.csv``).

Outputs (under main/supplemental results ``figures/``):
  - things_dinov3_vs_babydinov3_umap_cdi_{CATEGORY_SET}.png|.pdf
  - things_dinov3_vs_babydinov3_umap_cdi_coords_{CATEGORY_SET}.csv
  - things_dinov3_vs_babydinov3_umap_cdi_summary_{CATEGORY_SET}.txt

Example:
  CATEGORY_SET=valid129 python plot_things_dinov3_vs_babydinov3_umap.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
CATEGORY_SET = os.environ.get("CATEGORY_SET", "valid129").strip()

if CATEGORY_SET == "valid85":
    FIGURES_DIR = PREPRINT_DIR / "supplemental_results_valid85cats_04302026" / "figures"
    RESULTS_DIR = PREPRINT_DIR / "supplemental_results_valid85cats_04302026" / "results"
else:
    FIGURES_DIR = SCRIPT_DIR / "main_results_valid129s_04302026" / "figures"
    RESULTS_DIR = SCRIPT_DIR / "main_results_valid129s_04302026" / "results"

EXEMPLAR_EMBED_DIR = PREPRINT_DIR / "exemplar_set_embeddings" / CATEGORY_SET
INCLUDED_CATEGORIES_TXT = PROJECT_ROOT / "data" / f"included_categories_{CATEGORY_SET}.txt"
CDI_SEMANTIC_CSV = PROJECT_ROOT / "data" / f"long_tailed_dist_prop_included_categories_{CATEGORY_SET}.csv"

THINGS_DINOV3_CSV = EXEMPLAR_EMBED_DIR / f"things_dinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv"
THINGS_BABYDINOV3_CSV = EXEMPLAR_EMBED_DIR / f"things_babydinov3_exemplar_avg_zscore_within_{CATEGORY_SET}.csv"

UMAP_SEED = int(os.environ.get("UMAP_SEED", "42"))
UMAP_N_NEIGHBORS = int(os.environ.get("UMAP_N_NEIGHBORS", "15"))
UMAP_MIN_DIST = float(os.environ.get("UMAP_MIN_DIST", "0.25"))

CDI_SEMANTIC_ORDER = [
    "animals",
    "body_parts",
    "clothing",
    "food_drink",
    "furniture_rooms",
    "household",
    "outside",
    "toys",
    "vehicles",
    "other",
]
CDI_SEMANTIC_COLORS = {
    "animals": "#4DB8A8",
    "body_parts": "#E87A5F",
    "clothing": "#9B7EC8",
    "food_drink": "#E8A54C",
    "furniture_rooms": "#6BAB7A",
    "household": "#D97B9E",
    "outside": "#5B9BD5",
    "toys": "#B07CC8",
    "vehicles": "#6BA3D5",
    "other": "#8B9A9E",
}


def load_included_categories(path: Path) -> list[str]:
    return [ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()]


def load_cdi_semantic_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    col = "category" if "category" in df.columns else df.columns[0]
    sem_col = "cdi_semantic" if "cdi_semantic" in df.columns else "semantic"
    return {
        str(r[col]).strip().lower(): str(r[sem_col]).strip().lower()
        for _, r in df.iterrows()
    }


def load_embedding_matrix(
    path: Path, cat_order: list[str]
) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    df.index = [str(c).strip().lower() for c in df.index]
    missing = [c for c in cat_order if c not in df.index]
    if missing:
        raise ValueError(f"{path.name}: missing {len(missing)} categories (e.g. {missing[:3]})")
    mat = df.loc[cat_order].values.astype(np.float64)
    return cat_order, mat


def run_umap_2d(X: np.ndarray) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_SEED,
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)


def rdm_upper(vec: np.ndarray) -> np.ndarray:
    return squareform(pdist(vec, metric="cosine"))


def knn_label_agreement(X_a: np.ndarray, X_b: np.ndarray, k: int = 5) -> float:
    """Fraction of categories whose k nearest neighbors (by cosine) share the same set."""
    n = X_a.shape[0]
    k = min(k, n - 1)
    Da = squareform(pdist(X_a, metric="cosine"))
    Db = squareform(pdist(X_b, metric="cosine"))
    agree = 0
    for i in range(n):
        na = set(np.argsort(Da[i])[1 : k + 1])
        nb = set(np.argsort(Db[i])[1 : k + 1])
        agree += int(na == nb)
    return agree / n


def plot_panel(ax, coords: np.ndarray, semantics: list[str], title: str) -> None:
    for sem in CDI_SEMANTIC_ORDER:
        mask = [s == sem for s in semantics]
        if not any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"]),
            label=sem.replace("_", " "),
            s=42,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2, linewidth=0.5)


def main() -> None:
    for p in (THINGS_DINOV3_CSV, THINGS_BABYDINOV3_CSV, CDI_SEMANTIC_CSV, INCLUDED_CATEGORIES_TXT):
        if not p.exists():
            raise FileNotFoundError(p)

    cat_order = load_included_categories(INCLUDED_CATEGORIES_TXT)
    cdi_map = load_cdi_semantic_map(CDI_SEMANTIC_CSV)
    semantics = [cdi_map.get(c, "other") for c in cat_order]

    _, X_dino = load_embedding_matrix(THINGS_DINOV3_CSV, cat_order)
    _, X_baby = load_embedding_matrix(THINGS_BABYDINOV3_CSV, cat_order)

    rdm_dino = rdm_upper(X_dino)
    rdm_baby = rdm_upper(X_baby)
    spr_rdm, _ = spearmanr(rdm_dino[np.triu_indices(len(cat_order), k=1)], rdm_baby[np.triu_indices(len(cat_order), k=1)])

    coords_dino = run_umap_2d(X_dino)
    coords_baby = run_umap_2d(X_baby)

    knn5 = knn_label_agreement(X_dino, X_baby, k=5)
    knn10 = knn_label_agreement(X_dino, X_baby, k=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"things_dinov3_vs_babydinov3_umap_cdi_{CATEGORY_SET}"
    coords_df = pd.DataFrame(
        {
            "category": cat_order,
            "cdi_semantic": semantics,
            "umap1_dinov3": coords_dino[:, 0],
            "umap2_dinov3": coords_dino[:, 1],
            "umap1_babydinov3": coords_baby[:, 0],
            "umap2_babydinov3": coords_baby[:, 1],
        }
    )
    coords_df.to_csv(RESULTS_DIR / f"{stem}_coords.csv", index=False)

    summary_lines = [
        f"THINGS UMAP comparison ({CATEGORY_SET})",
        f"n_categories: {len(cat_order)}",
        f"UMAP: seed={UMAP_SEED}, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric=cosine",
        f"Inputs: {THINGS_DINOV3_CSV.name} vs {THINGS_BABYDINOV3_CSV.name}",
        f"RDM Spearman (DINOv3 vs BabyDINOv3): {spr_rdm:.6f}",
        f"kNN set agreement (k=5): {knn5:.4f}",
        f"kNN set agreement (k=10): {knn10:.4f}",
        "",
        "Interpretation note: separate UMAP fits per backbone (geometry change is visible).",
        "CDI colors are superordinate bins from long_tailed_dist_prop CSV.",
    ]
    summary_path = RESULTS_DIR / f"{stem}_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    plot_panel(
        axes[0],
        coords_dino,
        semantics,
        f"THINGS · pretrained DINOv3 (n={len(cat_order)})",
    )
    plot_panel(
        axes[1],
        coords_baby,
        semantics,
        f"THINGS · BabyDINOv3 (n={len(cat_order)})",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    fig.suptitle(
        f"Category embedding layout (UMAP, cosine metric) · RDM Spearman ρ={spr_rdm:.2f}",
        fontsize=13,
        y=1.02,
    )

    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")

    plt.close(fig)
    print(f"Wrote: {coords_df.shape[0]} rows -> {RESULTS_DIR / f'{stem}_coords.csv'}")
    print(f"Wrote: {summary_path}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
