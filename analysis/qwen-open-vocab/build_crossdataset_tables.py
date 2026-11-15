#!/usr/bin/env python3
"""Tier 1 category embedding tables for every dataset (BabyView, VEDB, THINGS, LVIS, ImageNet), CLIP and DINOv3.

Category vector = mean of L2-normalized exemplar vectors over every dataset item mapped to the category
(mapping/map_*.csv, build_crossdataset_mapping.py), as analysis/comparison-datasets builds category means; z-scoring
across the categories compared happens at analysis time.

  BabyView   Qwen boxes (out/private/open/sums_*.npz, build_open_means.py): "bv" all formats, "bv_real" real only
  VEDB       Qwen boxes of the VEDB run (out/private/open_vedb/sums_*.npz, build_open_means.py --dataset vedb) if present
  THINGS     all images of each mapped concept (embed_things.py concept means x n_images)
  ImageNet   comparison-datasets per-class sums of L2-normalized whole-image embeddings (train + val)
  LVIS       comparison-datasets per-category sums of L2-normalized box-crop embeddings (categories that have crop
             embeddings; the rest after their crops are embedded)

Output: out/crossdataset/tables_{clip,dinov3}.npz (dataset -> categories, means, counts) and coverage.csv
  python build_crossdataset_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CD = HERE.parents[1] / "analysis" / "comparison-datasets" / "outputs" / "mapping"
THINGS = Path("/data2/dataset/babyview/related_data/things/embeddings")
MAP = HERE / "mapping"
import variant as V                                     # QOV_LABELS=renamed: BabyView / VEDB from the renamed means
OUT0 = HERE / "out" / "crossdataset"                    # LVIS extra embeddings live here in every variant
OUT = V.out_dir(OUT0)
MODELS = ("clip", "dinov3")


def pool(items: pd.DataFrame, sums: dict, counts: dict, cats: list[str]):
    """items: id -> category; sums/counts by item id -> category means (mean of normalized vectors) and exemplar counts."""
    S, N = {}, {}
    for r in items.itertuples():
        if r.id in sums and counts.get(r.id, 0) > 0:
            S[r.category] = S.get(r.category, 0) + sums[r.id]
            N[r.category] = N.get(r.category, 0) + counts[r.id]
    keep = [c for c in cats if c in S]
    return keep, np.stack([S[c] / N[c] for c in keep]) if keep else np.zeros((0, 1)), np.array([N[c] for c in keep])


def bv_tables(model: str, part: str, root: Path):
    p = root / f"sums_{model}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    s, n = z[f"sum_{part}"].sum(0), z[f"n_{part}"].sum(0)
    keep = n >= 30
    return list(z["categories"][keep]), s[keep] / n[keep, None], n[keep]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()]
    cov = pd.DataFrame(index=cats)
    for model in MODELS:
        tables = {}
        P = HERE / "out" / "private"
        for name, root, part in (("bv", V.out_dir(P / "open"), "all"), ("bv_real", V.out_dir(P / "open"), "real"),
                                 ("vedb", V.out_dir(P / "open_vedb"), "all"), ("vedb_real", V.out_dir(P / "open_vedb"), "real")):
            t = bv_tables(model, part, root)
            if t is not None:
                tables[name] = t
        # THINGS: concept means of normalized vectors x image counts
        th = pd.read_csv(THINGS / f"things_{model}_concept_mean_raw.csv").set_index("concept")
        n = th.pop("n_images")
        tables["things"] = pool(pd.read_csv(MAP / "map_things.csv"), {c: v * n[c] for c, v in zip(th.index, th.values)},
                                n.to_dict(), cats)
        # ImageNet: per-class sums of normalized vectors
        z = np.load(CD / f"imagenet_class_embeddings_{model}.npz", allow_pickle=True)
        tables["imagenet"] = pool(pd.read_csv(MAP / "map_imagenet.csv"), dict(zip(z["wnids"], z["sums"])),
                                  dict(zip(z["wnids"], z["counts"])), cats)
        # LVIS: per-category sums of normalized crop vectors (categories with crop embeddings)
        z = np.load(CD / f"lvis_category_embeddings_{model}.npz", allow_pickle=True)
        extra = OUT0 / f"lvis_extra_category_embeddings_{model}.npz"          # categories embedded later (if any)
        sums, counts = dict(zip(z["category_ids"].tolist(), z["sums"])), dict(zip(z["category_ids"].tolist(), z["counts"]))
        if extra.exists():
            e = np.load(extra, allow_pickle=True)
            sums.update(zip(e["category_ids"].tolist(), e["sums"])); counts.update(zip(e["category_ids"].tolist(), e["counts"]))
        lv = pd.read_csv(MAP / "map_lvis.csv")
        tables["lvis"] = pool(lv, sums, counts, cats)
        np.savez(OUT / f"tables_{model}.npz", **{f"{k}__categories": np.array(v[0]) for k, v in tables.items()},
                 **{f"{k}__means": v[1] for k, v in tables.items()}, **{f"{k}__counts": v[2] for k, v in tables.items()})
        if model == "clip":
            for k, v in tables.items():
                cov[k] = pd.Series(v[2], index=v[0]).reindex(cats).fillna(0).astype(int)
            lv_mapped = lv.groupby("category").id.apply(set)
            cov["lvis_mapped_items"] = lv_mapped.reindex(cats).apply(lambda s: len(s) if isinstance(s, set) else 0)
            cov["lvis_items_without_embeddings"] = lv_mapped.reindex(cats).apply(
                lambda s: len([i for i in s if i not in sums]) if isinstance(s, set) else 0)
        print(model, {k: len(v[0]) for k, v in tables.items()})
    cov.to_csv(OUT / "coverage.csv")
    have = set(np.load(CD / "lvis_category_embeddings_clip.npz", allow_pickle=True)["category_ids"].tolist())
    if V.LABELS != "original":
        return
    if (OUT / "lvis_extra_category_embeddings_clip.npz").exists():
        have |= set(np.load(OUT / "lvis_extra_category_embeddings_clip.npz", allow_pickle=True)["category_ids"].tolist())
    need = sorted({int(i) for i in pd.read_csv(MAP / "map_lvis.csv").id} - have)
    if not need and (OUT / "lvis_categories_to_embed.csv").exists():
        return                                                   # keep the list embed_lvis_extra.py used
    pd.Series(need, name="lvis_category_id").to_csv(OUT / "lvis_categories_to_embed.csv", index=False)
    print(f"LVIS categories mapped to Tier 1 without crop embeddings: {len(need)} -> {OUT / 'lvis_categories_to_embed.csv'}")


if __name__ == "__main__":
    main()
