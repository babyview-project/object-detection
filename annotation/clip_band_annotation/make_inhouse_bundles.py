#!/usr/bin/env python3
"""Split the CLIP-band sample across in-house RAs and write one private bundle per RA.

Design: every item is labelled by one RA; a shared set (--shared-frac, default 15%,
stratified by category x band) is labelled by every RA for inter-rater agreement. Each
RA's items are grouped into one page per category (shared items mixed in, not flagged),
pages in a per-RA random order, items shuffled within a page.

Each bundle is a self-contained folder the RA opens in the web task (web/index.html,
hosted on GitHub Pages) with a folder picker; images never leave the RA's computer:

  out/private/inhouse/bundle_<rater>/
    manifest.json            rater, bundle id, pages [{category, items: [item_id, ...]}]
    crop/<item_id>.jpg       the crop (exact YOLOE box)
    frame/<item_id>.jpg      the full frame with the box outlined in magenta
  out/private/inhouse/assignments.csv   item_id, rater, shared (all bundles)
  out/private/inhouse/bundle_<rater>.zip  (with --zip) for sharing via the lab drive

File names are opaque item ids: no subject ids, video ids, CLIP scores or bands reach
the bundle. Bundles hold frames of children's homes and family members: share them
only through IRB-approved storage, never GitHub, Slack or email.

Usage
  python make_inhouse_bundles.py --raters RA1 RA2 RA3 [--zip]
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
IMG = OUT / "images"
INH = OUT / "inhouse"
SEED = 2028


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", nargs="+", required=True, help="short ids, e.g. RA1 RA2 RA3")
    ap.add_argument("--shared-frac", type=float, default=0.15)
    ap.add_argument("--zip", action="store_true", help="also write bundle_<rater>.zip")
    args = ap.parse_args()
    raters = args.raters
    assert len(set(raters)) == len(raters) and all(r.isalnum() for r in raters), "rater ids: unique, alphanumeric"
    rng = np.random.default_rng(SEED)

    s = pd.read_csv(OUT / "items.csv")[["item_id", "class_name", "band"]]
    # Shared set: round(frac x n) items per category, spread over its bands (one per band
    # first, in random band order, then a second round), so agreement covers every band.
    s["u"] = rng.random(len(s))
    s["rank"] = s.groupby(["class_name", "band"]).u.rank(method="first") - 1
    s["shared"] = False
    for _, g in s.groupby("class_name"):
        k = max(1, round(len(g) * args.shared_frac))
        s.loc[g.sort_values(["rank", "u"]).index[:k], "shared"] = True
    # Remaining items: dealt round-robin within each category (band by band) so every RA
    # gets a similar mix of categories and bands; the offset rotates per category.
    rows = []
    for k, (cat, g) in enumerate(s.groupby("class_name")):
        rest = g[~g.shared].sort_values(["band", "u"])
        for j, item in enumerate(rest.item_id):
            rows.append((item, raters[(j + k) % len(raters)], False))
        for item in g[g.shared].item_id:
            rows.extend((item, r, True) for r in raters)
    a = pd.DataFrame(rows, columns=["item_id", "rater", "shared"])
    assert a[~a.shared].item_id.is_unique and set(a.item_id) == set(s.item_id)

    if INH.exists():
        shutil.rmtree(INH)
    INH.mkdir(parents=True)
    a.to_csv(INH / "assignments.csv", index=False)
    cats = s.set_index("item_id").class_name
    for r in raters:
        items = a[a.rater == r].item_id
        d = INH / f"bundle_{r}"
        (d / "crop").mkdir(parents=True)
        (d / "frame").mkdir()
        pages = []
        order = rng.permutation(sorted(cats[items].unique()))
        for cat in order:
            its = [i for i in items if cats[i] == cat]
            pages.append({"category": str(cat), "items": [str(i) for i in rng.permutation(its)]})
        for i in items:
            shutil.copyfile(IMG / f"{i}_crop.jpg", d / "crop" / f"{i}.jpg")
            shutil.copyfile(IMG / f"{i}_frame.jpg", d / "frame" / f"{i}.jpg")
        bundle_id = hashlib.sha256(json.dumps(pages).encode()).hexdigest()[:10]
        manifest = {"study": "clip-band-context", "version": 1, "rater": r, "bundle_id": bundle_id,
                    "n_items": int(len(items)), "pages": pages}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if args.zip:
            shutil.make_archive(str(INH / f"bundle_{r}"), "zip", INH, f"bundle_{r}")
        mb = sum(f.stat().st_size for f in d.rglob("*.jpg")) / 1e6
        sizes = [len(p["items"]) for p in pages]
        print(f"{r}: {len(items)} items ({int(a[(a.rater == r) & a.shared].shape[0])} shared), "
              f"{len(pages)} pages of {min(sizes)}-{max(sizes)}, {mb:.0f} MB, bundle {bundle_id}")
    print(f"shared: {s.shared.sum()} items ({s.shared.mean():.1%}); written to {INH}")


if __name__ == "__main__":
    main()
