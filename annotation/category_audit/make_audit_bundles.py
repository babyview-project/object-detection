#!/usr/bin/env python3
"""Category audit, step 2: one bundle per RA for the web page (web/index.html).

Per category, the items of both datasets are pooled (the page never shows the dataset). SHARED of them (stratified by
dataset) go to every RA for agreement; the rest are split evenly. Pages hold one category, about PAGE items, shuffled
with a fixed seed; page order is shuffled too, so categories and datasets are interleaved over the session.

Output (private): out/private/bundles/bundle_<RA>/ (manifest.json, crop/<id>.jpg, frame/<id>.jpg) [+ .zip]. Share
bundles only through IRB-approved storage (frames of children's homes and families; never GitHub, Slack or email).

  /data/j7yang/anaconda3/bin/python make_audit_bundles.py --raters Mira Emma [--shared 0.2] [--zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
PAGE = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", nargs="+", required=True)
    ap.add_argument("--shared", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    it = pd.read_csv(OUT / "items.csv")
    rng = np.random.default_rng(a.seed)
    assign = {r: [] for r in a.raters}; rows = []
    for cat, g in it.groupby("category"):
        shared = pd.concat([d.sample(frac=a.shared, random_state=int(rng.integers(1 << 30))) for _, d in g.groupby("dataset")])
        rest = g.drop(shared.index).sample(frac=1, random_state=int(rng.integers(1 << 30)))
        for r in a.raters:
            assign[r] += [(cat, i) for i in shared.item_id]
        for k, i in enumerate(rest.item_id):
            assign[a.raters[k % len(a.raters)]].append((cat, i))
        rows += [dict(item_id=i, shared=True) for i in shared.item_id] + [dict(item_id=i, shared=False) for i in rest.item_id]
    pd.DataFrame(rows).to_csv(OUT / "assignment.csv", index=False)
    for r in a.raters:
        df = pd.DataFrame(assign[r], columns=["category", "item_id"])
        pages = []
        for cat, g in df.groupby("category"):
            ids = list(g.item_id.sample(frac=1, random_state=int(rng.integers(1 << 30))))
            k = max(1, round(len(ids) / PAGE))
            pages += [{"category": cat, "items": list(c)} for c in np.array_split(ids, k)]
        pages = [pages[i] for i in rng.permutation(len(pages))]
        d = OUT / "bundles" / f"bundle_{r}"
        if d.exists():
            shutil.rmtree(d)
        (d / "crop").mkdir(parents=True); (d / "frame").mkdir()
        for p in pages:
            for i in p["items"]:
                shutil.copyfile(OUT / "img" / f"{i}_crop.jpg", d / "crop" / f"{i}.jpg")
                shutil.copyfile(OUT / "img" / f"{i}_frame.jpg", d / "frame" / f"{i}.jpg")
        bundle_id = hashlib.sha256(json.dumps(pages).encode()).hexdigest()[:10]
        m = {"study": "category-audit", "version": 1, "rater": r, "bundle_id": bundle_id,
             "n_items": int(sum(len(p["items"]) for p in pages)), "pages": pages}
        (d / "manifest.json").write_text(json.dumps(m, indent=1))
        if a.zip:
            shutil.make_archive(str(d), "zip", d.parent, d.name)
        print(f"{r}: {m['n_items']} items on {len(pages)} pages -> {d}")


if __name__ == "__main__":
    main()
