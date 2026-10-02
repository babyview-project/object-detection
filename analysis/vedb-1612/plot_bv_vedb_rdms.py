#!/usr/bin/env python3
"""Cosine-distance RDM figures for BV vs VEDB, plotted exactly like plot_rdms.py
(which does VEDB vs THINGS) -- same heatmap style, CDI-semantic ordering, and
upper-triangle scatter comparison, but the second dataset is the real BV
exemplar table instead of THINGS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vedb")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr

VEDB_DIR = Path(__file__).resolve().parent
REPO = VEDB_DIR.parents[1]
MANUSCRIPT = REPO / "analysis" / "manuscript-2026"
sys.path.insert(0, str(MANUSCRIPT))

from manuscript_config import load_cdi_semantic_map  # noqa: E402

VEDB_EXEMPLAR = VEDB_DIR / "exemplar_set_embeddings"
BV_EXEMPLAR = MANUSCRIPT / "exemplar_set_embeddings"
ANALYSES = VEDB_DIR / "analyses"
MODELS = ("clip", "dinov3")
SETS = ("valid129", "valid85")
CDI_SEMANTIC_ORDER = [
    "animals",
    "body_parts",
    "clothing",
    "food_drink",
    "furniture_rooms",
    "household",
    "outside",
    "people",
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
    "people": "#E8C44C",
    "toys": "#B07CC8",
    "vehicles": "#6BA3D5",
    "other": "#8B9A9E",
}

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


def load_matrix(path: Path):
    df = pd.read_csv(path)
    cats = df["category"].astype(str).str.strip().str.lower().tolist()
    X = df.drop(columns=["category"]).to_numpy(dtype="float64")
    return cats, X


def aligned(category_set: str, model: str):
    inc = [
        ln.strip().lower()
        for ln in (REPO / "data" / f"included_categories_{category_set}.txt").read_text().splitlines()
        if ln.strip()
    ]
    vedb_cats, vedb_X = load_matrix(
        VEDB_EXEMPLAR / category_set / f"bv_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    bv_cats, bv_X = load_matrix(
        BV_EXEMPLAR / category_set / f"bv_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    vedb_i = {c: i for i, c in enumerate(vedb_cats)}
    bv_i = {c: i for i, c in enumerate(bv_cats)}
    cats = [c for c in inc if c in vedb_i and c in bv_i]
    V = np.stack([vedb_X[vedb_i[c]] for c in cats])
    B = np.stack([bv_X[bv_i[c]] for c in cats])
    return cats, V, B


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings, metric="cosine"))


def semantic_order(categories: list[str], sem_map: dict[str, str]):
    rank = {s: i for i, s in enumerate(CDI_SEMANTIC_ORDER)}
    annotated = []
    for i, cat in enumerate(categories):
        sem = sem_map.get(cat, "other")
        annotated.append((i, cat, sem, rank.get(sem, rank["other"])))
    annotated.sort(key=lambda x: (x[3], x[0]))
    idx = [x[0] for x in annotated]
    cats = [x[1] for x in annotated]
    sems = [x[2] for x in annotated]
    boundaries = [i for i, sem in enumerate(sems) if i > 0 and sem != sems[i - 1]]
    return idx, cats, sems, boundaries


def clustered_semantic_order(categories: list[str], sem_map: dict[str, str], ref_rdm: np.ndarray):
    """CDI-domain blocks as in semantic_order; within each domain, categories are ordered by
    average-linkage clustering of ref_rdm's within-domain block (optimal leaf ordering)."""
    idx, _, sems, _ = semantic_order(categories, sem_map)
    out: list[int] = []
    for dom in dict.fromkeys(sems):
        members = [i for i, s_ in zip(idx, sems) if s_ == dom]
        if len(members) > 2:
            d = squareform(ref_rdm[np.ix_(members, members)], checks=False)
            members = [members[k] for k in leaves_list(optimal_leaf_ordering(linkage(d, method="average"), d))]
        out.extend(members)
    ordered_sems = [sem_map.get(categories[i], "other") for i in out]
    boundaries = [i for i, sem in enumerate(ordered_sems) if i > 0 and sem != ordered_sems[i - 1]]
    return out, [categories[i] for i in out], ordered_sems, boundaries


def plot_clustered_pair(vedb_rdm, bv_rdm, cats, sem_map, ref, model, category_set, metrics, figures):
    """VEDB | BV heatmaps in one within-domain clustered order (reference = 'bv' or 'vedb'),
    with category names colored by CDI domain."""
    ref_rdm = bv_rdm if ref == "bv" else vedb_rdm
    idx, ordered_cats, ordered_sems, boundaries = clustered_semantic_order(cats, sem_map, ref_rdm)
    blocks = cluster_blocks(ordered_sems)
    n = len(ordered_cats)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), constrained_layout=True)
    for ax, rdm, title in zip(axes, (vedb_rdm, bv_rdm), ("VEDB", "BV")):
        im = plot_heatmap(ax, reorder(rdm, idx), f"{title} ({model.upper()})", boundaries, blocks)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(ordered_cats, fontsize=4.2, fontweight="normal")
        ax.tick_params(axis="y", length=0, pad=1)
        for tick, sem in zip(ax.get_yticklabels(), ordered_sems):
            tick.set_color(CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"]))
        ax.set_xticks([0.5 * (s_ + e_) for _, s_, e_ in blocks])
        ax.set_xticklabels([lab.replace("_", " ") for lab, _, _ in blocks], rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="x", length=0)
        for tick, (lab, _, _) in zip(ax.get_xticklabels(), blocks):
            tick.set_color(CDI_SEMANTIC_COLORS.get(lab, CDI_SEMANTIC_COLORS["other"]))
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="cosine distance")
    ref_label = "BV" if ref == "bv" else "VEDB"
    fig.suptitle(
        f"VEDB vs BV RDMs ({model.upper()}, {category_set}, n={n}): CDI domains, clustered on {ref_label} RDM\n"
        f"within domain. Spearman={metrics['spearman_r']:.3f}, Pearson={metrics['pearson_r']:.3f}",
        fontsize=12,
    )
    stem = figures / f"bv_vedb_rdm_comparison_{model}_clustered_{ref}_filtered-0.27_{category_set}"
    fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({"category": ordered_cats, "cdi_semantic": ordered_sems, "order_index": range(n)}).to_csv(
        figures.parent / "results" / f"bv_vedb_rdm_order_clustered_{ref}_{model}_filtered-0.27_{category_set}.csv",
        index=False,
    )


def cluster_blocks(labels: list[str]):
    blocks = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            blocks.append((cur, start, i - 1))
            start = i
            cur = labels[i]
    blocks.append((cur, start, len(labels) - 1))
    return blocks


def reorder(rdm: np.ndarray, idx: list[int]) -> np.ndarray:
    ii = np.array(idx, dtype=int)
    return rdm[np.ix_(ii, ii)]


def plot_heatmap(ax, matrix, title, boundaries, block_labels):
    im = ax.imshow(matrix, cmap="viridis", aspect="equal", vmin=0, vmax=2)
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
    ax.set_box_aspect(1)
    mids = [0.5 * (s + e) for _, s, e in block_labels]
    labs = [lab.replace("_", " ") for lab, _, _ in block_labels]
    ax.set_yticks(mids)
    ax.set_yticklabels(labs, fontsize=11, fontweight="bold")
    for tick, (lab, _, _) in zip(ax.get_yticklabels(), block_labels):
        tick.set_color(CDI_SEMANTIC_COLORS.get(lab, CDI_SEMANTIC_COLORS["other"]))
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    return im


def save_rdm_csv(path: Path, categories: list[str], rdm: np.ndarray) -> None:
    df = pd.DataFrame(rdm, index=categories, columns=categories)
    df.index.name = "category"
    df.to_csv(path)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    iu = np.triu_indices_from(a, k=1)
    va, vb = a[iu], b[iu]
    pr, pp = pearsonr(va, vb)
    sr, sp = spearmanr(va, vb)
    kr, kp = kendalltau(va, vb)
    return {
        "n_categories": int(a.shape[0]),
        "n_pairs": int(iu[0].size),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "kendall_tau": float(kr),
        "kendall_p": float(kp),
    }


def main() -> None:
    sem_map = load_cdi_semantic_map()
    for category_set in SETS:
        results = ANALYSES / category_set / "results"
        figures = ANALYSES / category_set / "figures"
        results.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        panels = {}
        rows = []
        for model in MODELS:
            cats, V, B = aligned(category_set, model)
            vedb_rdm = compute_rdm(V)
            bv_rdm = compute_rdm(B)
            idx, ordered_cats, ordered_sems, boundaries = semantic_order(cats, sem_map)
            vedb_ord = reorder(vedb_rdm, idx)
            bv_ord = reorder(bv_rdm, idx)
            blocks = cluster_blocks(ordered_sems)
            metrics = compare(vedb_ord, bv_ord)
            metrics.update({"category_set": category_set, "model": model, "order": "vedb_cdi_semantic"})
            rows.append(metrics)

            save_rdm_csv(
                results / f"bv_rdm_{model}_vedb_order_filtered-0.27_{category_set}.csv",
                ordered_cats,
                bv_ord,
            )
            pd.DataFrame(
                {"category": ordered_cats, "cdi_semantic": ordered_sems, "order_index": range(len(ordered_cats))}
            ).to_csv(
                results / f"bv_vedb_rdm_order_cdi_semantic_{model}_filtered-0.27_{category_set}.csv",
                index=False,
            )

            fig = plt.figure(figsize=(19, 6))
            ax1 = fig.add_subplot(1, 3, 1)
            im1 = plot_heatmap(ax1, vedb_ord, f"VEDB RDM ({model.upper()}, CDI order)", boundaries, blocks)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = plot_heatmap(ax2, bv_ord, f"BV RDM ({model.upper()}, same order)", boundaries, blocks)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            ax3 = fig.add_subplot(1, 3, 3)
            iu = np.triu_indices_from(vedb_ord, k=1)
            ax3.scatter(vedb_ord[iu], bv_ord[iu], s=14, alpha=0.35)
            lo = float(min(vedb_ord[iu].min(), bv_ord[iu].min()))
            hi = float(max(vedb_ord[iu].max(), bv_ord[iu].max()))
            ax3.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax3.set_xlabel("VEDB upper-triangle dissimilarity")
            ax3.set_ylabel("BV upper-triangle dissimilarity")
            ax3.set_title(
                f"Upper-triangle comparison\nPearson={metrics['pearson_r']:.3f}, Spearman={metrics['spearman_r']:.3f}"
            )
            fig.tight_layout()
            stem = figures / f"bv_vedb_rdm_comparison_{model}_filtered-0.27_{category_set}"
            fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
            fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)

            # Within-domain clustered versions (unordered RDMs; order derived from BV or VEDB).
            for ref in ("bv", "vedb"):
                plot_clustered_pair(vedb_rdm, bv_rdm, cats, sem_map, ref, model, category_set, metrics, figures)

            panels[model] = {
                "vedb_rdm": vedb_ord,
                "bv_rdm": bv_ord,
                "boundaries": boundaries,
                "block_labels": blocks,
            }
            print(
                f"{category_set} {model}: n={metrics['n_categories']} pearson={metrics['pearson_r']:.3f}",
                flush=True,
            )

        fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
        last_im = None
        for r, (model, label) in enumerate((("clip", "CLIP"), ("dinov3", "DINOv3"))):
            panel = panels[model]
            last_im = plot_heatmap(
                axes[r, 0],
                panel["vedb_rdm"],
                f"VEDB ({label}, CDI order)",
                panel["boundaries"],
                panel["block_labels"],
            )
            plot_heatmap(
                axes[r, 1],
                panel["bv_rdm"],
                f"BV ({label}, same order)",
                panel["boundaries"],
                panel["block_labels"],
            )
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.01)
        fig.suptitle(
            f"VEDB vs BV RDMs (CDI order, filtered-0.27, {category_set})",
            fontsize=16,
            fontweight="bold",
        )
        stem = figures / f"bv_vedb_rdm_comparison_2x2_clip_dinov3_filtered-0.27_{category_set}"
        fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(rows).to_csv(
            results / f"bv_vedb_rdm_comparison_summary_filtered-0.27_{category_set}.csv",
            index=False,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
