#!/usr/bin/env python3
"""Sorted bar chart of BV-vs-VEDB category-wise cosine similarity, using the
tables bv_vedb_comparison.py already wrote
(bv_vedb_category_wise_cosine_similarity_{model}_filtered-0.27_{set}.csv).
Same CDI-semantic color coding as plot_bv_vedb_rdms.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vedb")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

VEDB_DIR = Path(__file__).resolve().parent
REPO = VEDB_DIR.parents[1]
MANUSCRIPT = REPO / "analysis" / "manuscript-2026"
ANALYSES = VEDB_DIR / "analyses"
MODELS = ("clip", "dinov3")
SETS = ("valid129", "valid85")

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


def main() -> None:
    for category_set in SETS:
        results = ANALYSES / category_set / "results"
        figures = ANALYSES / category_set / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        for model in MODELS:
            df = pd.read_csv(
                results / f"bv_vedb_category_wise_cosine_similarity_{model}_filtered-0.27_{category_set}.csv"
            ).sort_values("cosine_similarity", ascending=True)
            colors = [CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]) for s in df["cdi_semantic"]]

            height = max(6, 0.17 * len(df))
            fig, ax = plt.subplots(figsize=(8, height))
            ax.barh(df["category"], df["cosine_similarity"], color=colors)
            ax.set_xlabel("BV-VEDB cosine similarity (category mean embeddings)")
            ax.set_title(
                f"BV vs VEDB category-wise cosine similarity ({model.upper()}, {category_set}, n={len(df)})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.tick_params(axis="y", labelsize=7)
            handles = [
                plt.Rectangle((0, 0), 1, 1, color=c)
                for c in CDI_SEMANTIC_COLORS.values()
            ]
            ax.legend(
                handles,
                CDI_SEMANTIC_COLORS.keys(),
                loc="lower right",
                fontsize=7,
                ncol=2,
                title="CDI semantic cluster",
                title_fontsize=7,
            )
            fig.tight_layout()
            stem = figures / f"bv_vedb_category_wise_cosine_similarity_{model}_filtered-0.27_{category_set}"
            fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
            fig.savefig(f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {stem}.png", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
