#!/usr/bin/env python3
"""Sample BabyView 2025.2 YOLOE crops by CLIP-score band for human annotation.

Why: the CLIP filter (cosine of "a photo of a {label}" vs the crop, ViT-B/32) is a
single cutoff inside a narrow score range (middle 90% of 13.2M crops: 0.219-0.299).
We tried 0.26, 0.27 and 0.28. Labelling crops across score bands gives precision and
false-rejection rates at any threshold, so the choice can be justified from data.

Design
  - Population: every scored YOLOE crop in 2025.2 (valid129 categories, person/picture
    and subject 00270001 already excluded upstream): crops_index.csv.gz.
  - Strata: category x CLIP band. Band edges fall on every threshold we used.
  - Per stratum: N_PER crops (default 8), spread over children first, then videos,
    then frames (round-robin), so no single child dominates a cell.
  - Reweighting: band_population.csv holds the population count of every stratum;
    estimates at a threshold must weight each band by its population share.
  - Double coding: DOUBLE_FRAC of items (stratified by band) go to every rater (kappa).
  - Blinding: items get opaque ids and a shuffled order; annotators never see the
    CLIP score or band.

Outputs (out/private/, git-ignored: holds subject ids and frame paths)
  items.csv             one row per sampled crop, with box, scores and frame path
  band_population.csv   category x band population counts (for weights)
  assignments_<rater>.csv  item ids and order for each rater
  sample_summary.json

Usage
  nice -n 19 python sample_clip_bands.py --raters A,B
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
INDEX = Path("/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/crops_index.csv.gz")
FRAMES_ROOT = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
CATS_FILE = HERE.parents[1] / "data" / "included_categories_valid129.txt"

# Left-closed bands; edges include every threshold tried (0.26, 0.27, 0.28).
BAND_EDGES = [-np.inf, 0.22, 0.24, 0.26, 0.27, 0.28, 0.30, np.inf]
BAND_LABELS = ["lt0.22", "0.22-0.24", "0.24-0.26", "0.26-0.27", "0.27-0.28", "0.28-0.30", "ge0.30"]
THRESHOLDS = [0.26, 0.27, 0.28]

N_PER = 8
DOUBLE_FRAC = 0.10
SEED = 2026


def subject_id(video_id: str) -> str:
    head = str(video_id).split("_", 1)[0]
    return head[1:].zfill(8) if head.startswith("S") else head.zfill(8)


def frame_path(video_id: str, frame_number: int) -> Path | None:
    d = FRAMES_ROOT / video_id
    for name in (f"{frame_number:05d}.jpg", f"{frame_number:08d}.jpg", f"{frame_number}.jpg"):
        if (d / name).is_file():
            return d / name
    return None


def spread_sample(g: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take n rows, round-robin over subjects, then videos, then frames."""
    g = g.iloc[rng.permutation(len(g))]
    g = g.drop_duplicates(["video_id", "frame_number"])  # one crop per frame per cell
    r_subj = g.groupby("subject_id", sort=False, observed=True).cumcount()
    r_vid = g.groupby("video_id", sort=False, observed=True).cumcount()
    order = np.lexsort((r_vid.to_numpy(), r_subj.to_numpy()))
    return g.iloc[order[:n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--double-frac", type=float, default=DOUBLE_FRAC)
    ap.add_argument("--raters", default="A,B", help="comma-separated rater ids")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    cats = [c.strip() for c in CATS_FILE.read_text().splitlines() if c.strip()]
    cols = ["image_id", "video_id", "frame_number", "class_name", "xmin", "ymin", "xmax", "ymax",
            "yolo_conf", "clip_sim"]
    d = pd.read_csv(INDEX, usecols=cols, dtype={"class_name": "category", "video_id": "category"})
    n_all = len(d)
    d = d[d["class_name"].isin(cats)]
    d["band"] = pd.cut(d["clip_sim"], BAND_EDGES, right=False, labels=BAND_LABELS)
    vids = d["video_id"].cat.categories
    subj_map = pd.Series([subject_id(v) for v in vids], index=vids)
    d["subject_id"] = d["video_id"].map(subj_map).astype("category")

    pop = d.groupby(["class_name", "band"], observed=False).size().rename("n_population").reset_index()
    pop.to_csv(OUT / "band_population.csv", index=False)

    picks = []
    for (cat, band), g in d.groupby(["class_name", "band"], observed=True):
        picks.append(spread_sample(g, args.n_per, rng))
    s = pd.concat(picks, ignore_index=True)
    s["class_name"] = s["class_name"].astype(str)
    s["video_id"] = s["video_id"].astype(str)
    s["subject_id"] = s["subject_id"].astype(str)
    s["band"] = s["band"].astype(str)

    # Frame paths; drop rows whose frame is missing (reported below).
    s["frame_path"] = [str(p) if (p := frame_path(v, int(f))) else "" for v, f in zip(s.video_id, s.frame_number)]
    n_missing = int((s.frame_path == "").sum())
    s = s[s.frame_path != ""].copy()

    for c in ["xmin", "ymin", "xmax", "ymax"]:
        s[c] = s[c].round().astype(int)
    s["box_w"] = s.xmax - s.xmin
    s["box_h"] = s.ymax - s.ymin
    s["short_px"] = s[["box_w", "box_h"]].min(axis=1)
    for t in THRESHOLDS:
        s[f"pass_{t:.2f}"] = s.clip_sim >= t
    s["item_id"] = [hashlib.sha256(f"clipband:{i}".encode()).hexdigest()[:12] for i in s.image_id]
    assert s.item_id.is_unique

    # Double-coded subset, stratified by band.
    s["double_code"] = False
    for band, g in s.groupby("band"):
        k = int(round(args.double_frac * len(g)))
        s.loc[rng.choice(g.index, size=k, replace=False), "double_code"] = True

    # Split single-coded items across raters (balanced within band x category), then
    # give every rater the double-coded items; each rater gets an independent shuffle.
    s["rater"] = ""
    single = s[~s.double_code]
    for _, g in single.groupby(["band", "class_name"]):
        idx = rng.permutation(g.index.to_numpy())
        start = rng.integers(len(raters))
        for j, i in enumerate(idx):
            s.at[i, "rater"] = raters[(start + j) % len(raters)]
    s.loc[s.double_code, "rater"] = "all"

    s = s.sort_values(["class_name", "band", "item_id"]).reset_index(drop=True)
    s.to_csv(OUT / "items.csv", index=False)
    for r in raters:
        a = s[(s.rater == r) | s.double_code][["item_id", "class_name"]]
        a = a.iloc[rng.permutation(len(a))].reset_index(drop=True)
        a.insert(0, "order", np.arange(1, len(a) + 1))
        a.to_csv(OUT / f"assignments_{r}.csv", index=False)

    cell = s.groupby(["class_name", "band"]).size()
    summary = {
        "index": str(INDEX),
        "n_population_crops": int(n_all),
        "n_population_valid129": int(len(d)),
        "n_items": int(len(s)),
        "n_categories": int(s.class_name.nunique()),
        "n_subjects": int(s.subject_id.nunique()),
        "n_videos": int(s.video_id.nunique()),
        "n_missing_frames_dropped": n_missing,
        "n_per_target": args.n_per,
        "cells_total": int(len(cats) * len(BAND_LABELS)),
        "cells_filled": int(len(cell)),
        "cells_short": int((cell < args.n_per).sum()),
        "items_by_band": s.band.value_counts().reindex(BAND_LABELS).fillna(0).astype(int).to_dict(),
        "n_double_coded": int(s.double_code.sum()),
        "items_per_rater": {r: int(((s.rater == r) | s.double_code).sum()) for r in raters},
        "max_items_one_subject_in_cell": int(s.groupby(["class_name", "band", "subject_id"]).size().max()),
        "band_edges": BAND_LABELS,
        "seed": args.seed,
    }
    (OUT / "sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
