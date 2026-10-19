#!/usr/bin/env python3
"""Export the CLIP-band sample in the objects-gallerize local_csv format.

objects-gallerize (github.com/babyview-project/objects-gallerize) shows one page per
category: up to `num` (25) crops plus a non-object attention-check tile; raters click
the invalid ones. Each worker is assigned one bucket (`shuffled_index`) and sees every
category once from that bucket. store.js caps each bucket at 3 workers, so every item
is rated by 3 people.

This script splits each category's ~56 items into N_BUCKETS pages (default 3, so ~19
per page, under the 25 cap), balanced across CLIP bands so no page is all low- or all
high-score crops. Set NUM_BUCKETS in server/store.js to the same value.

Image file names are opaque item ids ("<class>/<item_id>.jpg"): no subject ids, video
ids or CLIP scores reach the browser (the 868 h study used stems that contained both).

Outputs (out/private/gallerize/, git-ignored)
  annotations.csv            DATA_CSV_PATH for store.js (shuffled_index, class, filename, ...)
  img/<class>/<item_id>.jpg  IMG_DIR for store.js: the crops (phase 1, crop only)
  img_frame/<class>/<item_id>.jpg  frames with the box drawn (phase 2, needs the
                                   context patch to Trial.js; see README)
  bucket_summary.csv         items per bucket x band (balance check)

Usage
  python export_gallerize.py [--n-buckets 3]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
IMG = OUT / "images"
GAL = OUT / "gallerize"
SEED = 2027


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-buckets", type=int, default=3)
    ap.add_argument("--max-per-page", type=int, default=25, help="Trial num in study.js")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    s = pd.read_csv(OUT / "items.csv")

    # Within each category, deal items band by band so each bucket gets a similar band mix.
    s["shuffled_index"] = 0
    for _, g in s.groupby("class_name"):
        offset = int(rng.integers(args.n_buckets))
        idx = []
        for _, gb in g.groupby("band"):
            idx.extend(rng.permutation(gb.index.to_numpy()))
        for j, i in enumerate(idx):
            s.at[i, "shuffled_index"] = (offset + j) % args.n_buckets + 1
    per_page = s.groupby(["class_name", "shuffled_index"]).size()
    assert per_page.max() <= args.max_per_page, per_page.max()

    s["class"] = s["class_name"]
    s["stem"] = s["item_id"]
    s["filename"] = s["class"] + "/" + s["item_id"] + ".jpg"
    s["trial_type"] = "regular"
    s = s.sort_values(["shuffled_index", "class", "item_id"]).reset_index(drop=True)
    s["bucket_trial_index"] = s.groupby("shuffled_index").cumcount() + 1
    # store.js parses the CSV by splitting on commas: keep only comma-free fields.
    cols = ["shuffled_index", "class", "filename", "stem", "trial_type", "bucket_trial_index"]
    assert not s[cols].astype(str).apply(lambda c: c.str.contains(",")).any().any()
    GAL.mkdir(parents=True, exist_ok=True)
    s[cols].to_csv(GAL / "annotations.csv", index=False)

    for r in s.itertuples(index=False):
        link(IMG / f"{r.item_id}_crop.jpg", GAL / "img" / r.filename)
        link(IMG / f"{r.item_id}_frame.jpg", GAL / "img_frame" / r.filename)

    bal = s.pivot_table(index="shuffled_index", columns="band", values="item_id", aggfunc="count")
    bal["total"] = bal.sum(axis=1)
    bal.to_csv(GAL / "bucket_summary.csv")
    print(bal.to_string())
    print(f"\npages per bucket: {s.groupby('shuffled_index')['class'].nunique().to_dict()}"
          f" · crops per page: {per_page.min()}-{per_page.max()}")
    print(f"wrote {GAL / 'annotations.csv'} ({len(s)} rows); set NUM_BUCKETS = {args.n_buckets} in store.js")


if __name__ == "__main__":
    main()
