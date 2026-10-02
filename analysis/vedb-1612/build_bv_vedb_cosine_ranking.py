#!/usr/bin/env python3
"""Merge the per-model BV-vs-VEDB category cosine similarity tables
(bv_vedb_comparison.py's output) into one ranked CSV per category set:
rank, category, cdi_semantic, clip_cosine_similarity, dinov3_cosine_similarity
-- sorted by CLIP similarity, high to low.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ANALYSES = Path(__file__).resolve().parent / "analyses"
SETS = ("valid129", "valid85")


def main() -> None:
    for category_set in SETS:
        results = ANALYSES / category_set / "results"
        clip_df = pd.read_csv(results / f"bv_vedb_category_wise_cosine_similarity_clip_filtered-0.27_{category_set}.csv")
        dino_df = pd.read_csv(results / f"bv_vedb_category_wise_cosine_similarity_dinov3_filtered-0.27_{category_set}.csv")
        merged = (
            clip_df[["category", "cdi_semantic", "cosine_similarity"]]
            .rename(columns={"cosine_similarity": "clip_cosine_similarity"})
            .merge(
                dino_df[["category", "cosine_similarity"]].rename(
                    columns={"cosine_similarity": "dinov3_cosine_similarity"}
                ),
                on="category",
            )
            .sort_values("clip_cosine_similarity", ascending=False)
            .reset_index(drop=True)
        )
        merged.insert(0, "rank", range(1, len(merged) + 1))
        out = results / f"bv_vedb_category_wise_cosine_similarity_ranked_filtered-0.27_{category_set}.csv"
        merged.to_csv(out, index=False)
        print(f"wrote {out}  ({len(merged)} categories)")


if __name__ == "__main__":
    main()
