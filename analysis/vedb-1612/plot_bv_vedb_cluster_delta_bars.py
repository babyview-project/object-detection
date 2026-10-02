#!/usr/bin/env python3
"""Per-CDI-domain cluster-strength bar chart, VEDB vs BV, matching
analysis/manuscript-2026/05_within_between_cdi_cluster_correlation.ipynb's
_draw_perdomain_cluster_delta_bars (the bv_vs_things_cluster_delta_bars
figure) -- same grouped-bar + shuffle-null-ribbon style, but VEDB vs real BV
instead of BV vs THINGS. Adds significance stars (FDR q, one-sided) on
domains where the VEDB-BV difference itself is significant.

Reads bv_vedb_comparison.py's output:
  bv_vedb_cdi_cluster_within_between_{model}_filtered-0.27_{set}.csv
  bv_vedb_cdi_cluster_shuffle_null_bands_{model}_filtered-0.27_{set}.csv
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vedb")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

VEDB_DIR = Path(__file__).resolve().parent
ANALYSES = VEDB_DIR / "analyses"
MODELS = ("clip", "dinov3")
SETS = ("valid129", "valid85")

CDI_SEMANTIC_ORDER = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]
CDI_SEMANTIC_COLORS = {
    "animals": "#4DB8A8", "body_parts": "#E87A5F", "clothing": "#9B7EC8",
    "food_drink": "#E8A54C", "furniture_rooms": "#6BAB7A", "household": "#D97B9E",
    "outside": "#5B9BD5", "people": "#E8C44C", "toys": "#B07CC8",
    "vehicles": "#6BA3D5", "other": "#8B9A9E",
}

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


def q_to_sig_star(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q <= 0.001:
        return "***"
    if q <= 0.01:
        return "**"
    if q <= 0.05:
        return "*"
    return ""


def draw_bars(ax, bar_order, merged, null_bv, null_bv_real, *, title, show_ylabel, show_legend):
    """Bars = VEDB vs BV; x-tick colors = CDI domain; ribbons = shuffle null (2.5-97.5%)."""
    vedb_bar = "#0d47a1"
    bv_bar = "#90caf9"
    merged_ix = merged.set_index("cdi_semantic")

    x = np.arange(len(bar_order))
    w = 0.36
    gap = 0.02
    vedb_x = x - w / 2 - gap / 2
    bv_x = x + w / 2 + gap / 2
    ax.grid(False)

    ys: list[float] = []
    for k, sem in enumerate(bar_order):
        lo_v, hi_v = null_bv[k, 0], null_bv[k, 2]
        lo_b, hi_b = null_bv_real[k, 0], null_bv_real[k, 2]
        if np.isfinite(lo_v) and np.isfinite(hi_v):
            ax.add_patch(Rectangle((vedb_x[k] - w / 2, lo_v), w, hi_v - lo_v,
                                    facecolor="0.42", edgecolor="none", alpha=0.24, zorder=0))
            ys.extend([lo_v, hi_v])
        if np.isfinite(lo_b) and np.isfinite(hi_b):
            ax.add_patch(Rectangle((bv_x[k] - w / 2, lo_b), w, hi_b - lo_b,
                                    facecolor="0.62", edgecolor="none", alpha=0.2, zorder=0))
            ys.extend([lo_b, hi_b])

        row = merged_ix.loc[sem]
        dv = float(row["delta_between_minus_within_vedb"])
        db = float(row["delta_between_minus_within_bv"])
        ax.bar(vedb_x[k], dv, width=w, color=vedb_bar, edgecolor="white", linewidth=0.9, zorder=2)
        ax.bar(bv_x[k], db, width=w, color=bv_bar, edgecolor="white", linewidth=0.9, zorder=2)
        ys.extend([dv, db])

        star = q_to_sig_star(float(row["vedb_vs_bv_shuffle_q_one_sided_gt_fdr_bh"]))
        if star:
            top = max(dv, db) + 0.02
            ax.text(x[k], top, star, ha="center", va="bottom", fontsize=11, fontweight="bold")
            ys.append(top + 0.02)

    ax.axhline(0, color="0.35", linewidth=0.85, zorder=0)
    ax.set_xticks(x)
    labels = [s.replace("_", " ").title() for s in bar_order]
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=10)
    for tick, sem in zip(ax.get_xticklabels(), bar_order):
        tick.set_color(CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"]))

    if show_ylabel:
        ax.set_ylabel(r"$\overline{d}_{between}-\overline{d}_{within}$ (cosine distance)", fontsize=11)
    ax.set_title(title, fontsize=12)
    y_min, y_max = min(ys), max(ys)
    pad = max(0.02, 0.08 * (y_max - y_min + 1e-9))
    ax.set_ylim(y_min - pad, y_max + pad)

    if show_legend:
        leg = [
            Patch(facecolor="0.42", alpha=0.45, edgecolor="none", label="Shuffle null (VEDB, 2.5-97.5%)"),
            Patch(facecolor="0.62", alpha=0.4, edgecolor="none", label="Shuffle null (BV, 2.5-97.5%)"),
            Patch(facecolor=vedb_bar, edgecolor="none", label="VEDB"),
            Patch(facecolor=bv_bar, edgecolor="none", label="BV"),
        ]
        ax.legend(handles=leg, loc="upper right", fontsize=8.5, frameon=False)


def main() -> None:
    for category_set in SETS:
        results = ANALYSES / category_set / "results"
        figures = ANALYSES / category_set / "figures"

        for model in MODELS:
            merged = pd.read_csv(
                results / f"bv_vedb_cdi_cluster_within_between_{model}_filtered-0.27_{category_set}.csv"
            )
            nulls = pd.read_csv(
                results / f"bv_vedb_cdi_cluster_shuffle_null_bands_{model}_filtered-0.27_{category_set}.csv"
            )
            present = set(merged["cdi_semantic"].tolist())
            bar_order = [d for d in CDI_SEMANTIC_ORDER if d in present]
            nulls_ix = nulls.set_index("cdi_semantic").loc[bar_order]
            null_vedb = nulls_ix[["vedb_null_q025", "vedb_null_q500", "vedb_null_q975"]].to_numpy()
            null_bv = nulls_ix[["bv_null_q025", "bv_null_q500", "bv_null_q975"]].to_numpy()

            w = max(10.0, 0.42 * len(bar_order))
            fig, ax = plt.subplots(figsize=(w, 5.8), constrained_layout=True)
            draw_bars(
                ax, bar_order, merged, null_vedb, null_bv,
                title=f"CDI domain cluster strength: VEDB vs BV ({model.upper()}, {category_set})",
                show_ylabel=True, show_legend=True,
            )
            fig.text(
                0.5, -0.02,
                r"Ribbons: per-domain $\Delta$ under parallel label shuffle (n_perm=2000; 2.5-97.5%). "
                r"* / ** / *** = FDR-BH q < .05 / .01 / .001 for the VEDB-BV difference itself. "
                r"X-axis label colors = CDI domain.",
                fontsize=9, color="0.35", ha="center", va="top", transform=fig.transFigure,
            )
            stem = figures / f"bv_vedb_cluster_delta_bars_{model}_filtered-0.27_{category_set}"
            fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
            fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {stem}.png", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
