#!/usr/bin/env python3
"""Frame x category presence matrix for the co-occurrence analyses (cooccurrence_analysis.ipynb), BabyView and VEDB,
from the two Qwen box stores.

Category: Qwen's basic-level name (build_open_means.basic_of), Tier 1 list (data/included_categories_qwen_open_tier1.txt).
Frames: every frame with at least one box (any category), so frames with no Tier 1 object still count in the totals.
Unit, age bin and home-scene flag as in build_bv_vedb_data.py.

Outputs (out/private/cooccurrence/, private: per-frame table):
  presence.npz      X: sparse CSR, frames x categories, 1 = category present in the frame (any number of boxes)
  depicted.npz      same shape, 1 = present only as a depiction (toy / photo / drawing / screen / picture)
  frames.csv.gz     row order of X: dataset, unit, video_id, frame_number, age_months, age_bin, home
  categories.txt    column order of X
  python build_cooccurrence_data.py       # analysis env (WordNet for basic_of), ~10 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
import variant as V
from build_bv_vedb_data import DEPICTED, HOME, STORES, age_bin, age_table, unit_of

OUT = V.out_dir(HERE / "out" / "private" / "cooccurrence")


def main():
    from build_open_means import basic_of
    OUT.mkdir(parents=True, exist_ok=True)
    tier1 = sorted({l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()})
    ci = {c: i for i, c in enumerate(tier1)}
    home = HOME & set(tier1)
    frames_all, X_all, Xd_all = [], [], []
    for ds in ("bv", "vedb"):
        ix = V.read_index(STORES[ds], usecols=["video_id", "frame_number", "label", "format"],
                          dtype={"video_id": "category", "format": "category"})
        ix["label"] = ix.label.astype(str)
        cmap = basic_of(ix.label.unique())
        ix["cat"] = ix.label.map(cmap).fillna("")
        ix["depicted"] = ix.format.astype(str).isin(DEPICTED)
        fk = ix.groupby(["video_id", "frame_number"], observed=True, sort=True).ngroup().values
        fr = ix[["video_id", "frame_number"]].drop_duplicates().sort_values(["video_id", "frame_number"]).reset_index(drop=True)
        assert len(fr) == fk.max() + 1
        fr["video_id"] = fr.video_id.astype(str)
        fr["unit"] = fr.video_id.map(unit_of)
        if ds == "bv":
            ages = age_table(fr.video_id.unique())
            fr["age_months"] = fr.video_id.map(ages)
            fr["age_bin"] = fr.age_months.map(age_bin)
        else:
            fr["age_months"] = np.nan; fr["age_bin"] = "adult"
        col = ix.cat.map(ci).fillna(-1).astype(int).values
        k = col >= 0
        X = sp.csr_matrix((np.ones(k.sum(), np.int32), (fk[k], col[k])), shape=(len(fr), len(tier1)))
        Xr = sp.csr_matrix((np.ones((k & ~ix.depicted.values).sum(), np.int32),
                            (fk[k & ~ix.depicted.values], col[k & ~ix.depicted.values])), shape=X.shape)
        X.data[:] = 1; Xr.data[:] = 1                       # duplicates summed on construction; presence only
        Xd = (X - X.multiply(Xr)).tocsr(); Xd.eliminate_zeros()      # present, but never as a real object
        fr["home"] = np.asarray(X[:, [ci[c] for c in sorted(home)]].sum(1)).ravel() > 0
        fr["dataset"] = ds
        frames_all.append(fr); X_all.append(X.astype(np.int8)); Xd_all.append(Xd.astype(np.int8))
        print(f"{ds}: {len(ix):,} boxes, {len(fr):,} frames, {X.nnz:,} frame-category pairs", flush=True)
        del ix
    fr = pd.concat(frames_all, ignore_index=True)
    fr[["dataset", "unit", "video_id", "frame_number", "age_months", "age_bin", "home"]].to_csv(OUT / "frames.csv.gz", index=False)
    sp.save_npz(OUT / "presence.npz", sp.vstack(X_all).tocsr())
    sp.save_npz(OUT / "depicted.npz", sp.vstack(Xd_all).tocsr())
    (OUT / "categories.txt").write_text("\n".join(tier1) + "\n")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
