#!/usr/bin/env python3
"""Tables for the BabyView vs VEDB comparison (bv_vedb_analysis.ipynb): scene statistics, category presence and an
embedding sample, by child / session and age bin, from the two Qwen box stores (embed_full_run.py).

Unit: BabyView child (subject id); VEDB session (VEDB has no participant ids in our copy). Age bins (BabyView, months
at recording, from /ccn2a/dataset/babyview/demogs.csv): AGE_BINS; VEDB is one "adult" bin.
Category: Qwen's basic-level name (build_open_means.basic_of); Tier 1 list (data/included_categories_qwen_open_tier1.txt).
Home-scene frames: frames with at least one HOME object (bed, couch, crib, high chair, fridge, stove, ...); a
content-based stand-in for setting matching (VEDB setting labels exist for 80 of the 717 sessions only).

Outputs (out/private/bv_vedb/, private: per-frame / per-child tables):
  frames.csv.gz        dataset, unit, video_id, frame_number, age_months, age_bin, home, n_obj, n_tier1, n_distinct,
                       n_person, n_hand, n_face (face / eye / mouth / nose), n_screen (format), n_depicted (toy / photo /
                       drawing / screen), median_area (box area / frame area), mean_cy (box centre, 0 top - 1 bottom)
  presence.csv.gz      dataset, unit, age_bin, home, category, frames (frames with the category), boxes
  geometry.csv.gz      dataset, age_bin, category: boxes, median area, centre y, edge-touching share, depicted share
  sample_{clip,dinov3}.npz  up to K boxes per (dataset, age bin, Tier 1 category), random: dataset, unit, age_bin,
                       category, format, home, emb (float16, not normalized), video_id, frame_number, bbox (px; re-cut
                       crops for the blur control, build_blur_control.py)
  python build_bv_vedb_data.py [--k 150] [--workers 4]       # analysis env (WordNet for basic_of)
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
STORES = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
          "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults")}
FRAME_WH = {"bv": (512, 910), "vedb": (2048, 1536)}
DEMOGS = Path("/ccn2a/dataset/babyview/demogs.csv")
import variant as V                                     # QOV_LABELS=renamed: renamed index, out/private/bv_vedb_renamed
OUT = V.out_dir(HERE / "out" / "private" / "bv_vedb")
AGE_BINS = [(0, 12, "<12 mo"), (12, 18, "12-18 mo"), (18, 24, "18-24 mo"), (24, 48, "24+ mo")]
HOME = {"bed", "couch", "crib", "high chair", "refrigerator", "stove", "oven", "dishwasher", "bathtub", "toilet",
        "dresser", "nightstand", "changing table", "microwave", "bunk bed", "playpen", "bassinet", "sink"}
FACE = {"face", "eye", "mouth", "nose", "eyes"}
DEPICTED = {"toy", "photo", "drawing", "screen", "picture"}


def unit_of(video_id: str) -> str:
    return video_id.split("_")[1] if video_id.startswith("session_") else video_id.split("_")[0]


def age_table(videos) -> pd.Series:
    d = pd.read_csv(DEMOGS, encoding="utf-8-sig", usecols=["subject_id", "date_birth_rounded"])
    dob = pd.to_datetime(d.set_index("subject_id").date_birth_rounded, format="%m/%d/%Y")
    v = pd.Series(list(videos))
    date = pd.to_datetime(v.str.split("_").str[1], errors="coerce")
    return pd.Series(((date - v.str[:9].map(dob)).dt.days / 30.4375).values, index=v.values)


def age_bin(a: float) -> str:
    for lo, hi, name in AGE_BINS:
        if lo <= a < hi:
            return name
    return ""


def load_index(ds: str, cmap: dict | None = None):
    ix = V.read_index(STORES[ds], usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"],
                      dtype={"video_id": "category", "format": "category"})
    ix["label"] = ix.label.astype(str)
    ix["pos"] = ix.groupby("video_id", observed=True).cumcount()          # row within the video's shard
    return ix


def take(args):
    """Rows `pos` of one shard: (video_id, pos array, clip, dino)."""
    ds, vid, pos = args
    with np.load(STORES[ds] / "shards" / f"{vid}.npz", allow_pickle=True) as z:
        return vid, pos, z["clip_emb"][pos], z["dino_emb"][pos]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from build_open_means import basic_of
    OUT.mkdir(parents=True, exist_ok=True)
    tier1 = {l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()}
    home = HOME & tier1
    print("home-scene objects in Tier 1:", sorted(home), flush=True)
    rng = np.random.default_rng(0)
    frames_all, pres_all, geo_all, picks = [], [], [], []
    for ds in ("bv", "vedb"):
        ix = load_index(ds)
        cmap = basic_of(ix.label.unique())
        ix["cat"] = ix.label.map(cmap).fillna("")
        ix["unit"] = ix.video_id.astype(str).map(unit_of)
        if ds == "bv":
            ages = age_table(ix.video_id.cat.categories)
            ix["age_months"] = ix.video_id.astype(str).map(ages)
            ix["age_bin"] = ix.age_months.map(age_bin)
        else:
            ix["age_months"] = np.nan; ix["age_bin"] = "adult"
        W, H = FRAME_WH[ds]
        ix["area"] = (ix.xmax - ix.xmin) * (ix.ymax - ix.ymin) / (W * H)
        ix["cy"] = (ix.ymin + ix.ymax) / 2 / H
        ix["edge"] = (ix.xmin <= .01 * W) | (ix.ymin <= .01 * H) | (ix.xmax >= .99 * W) | (ix.ymax >= .99 * H)
        fmt = ix.format.astype(str)
        ix["depicted"] = fmt.isin(DEPICTED)
        f = ix.assign(is_t1=ix.cat.isin(tier1), person=ix.cat == "person", hand=ix.cat == "hand",
                      face=ix.label.str.lower().isin(FACE), screen=fmt == "screen", home_obj=ix.cat.isin(home))
        g = f.groupby(["video_id", "frame_number"], observed=True)
        fr = pd.DataFrame({"n_obj": g.size(), "n_tier1": g.is_t1.sum(), "n_distinct": g.cat.nunique(),
                           "n_person": g.person.sum(), "n_hand": g.hand.sum(), "n_face": g.face.sum(),
                           "n_screen": g.screen.sum(), "n_depicted": g.depicted.sum(), "median_area": g.area.median(),
                           "mean_cy": g.cy.mean(), "home": g.home_obj.any()}).reset_index()
        fr["dataset"] = ds; fr["unit"] = fr.video_id.astype(str).map(unit_of)
        meta = ix.drop_duplicates("video_id")[["video_id", "age_months", "age_bin"]].set_index("video_id")
        fr = fr.join(meta, on="video_id")
        frames_all.append(fr)
        ix = ix.merge(fr[["video_id", "frame_number", "home"]], on=["video_id", "frame_number"], how="left")
        t = ix[ix.cat.isin(tier1)]
        t = t.assign(fkey=t.video_id.cat.codes.astype(np.int64) * 100000 + t.frame_number)
        p = t.groupby(["unit", "age_bin", "home", "cat"], observed=True).agg(
            frames=("fkey", "nunique"), boxes=("fkey", "size")).reset_index().rename(columns={"cat": "category"})
        p["dataset"] = ds; pres_all.append(p)
        geo = t.groupby(["age_bin", "cat"], observed=True).agg(boxes=("area", "size"), median_area=("area", "median"),
                                                               mean_cy=("cy", "mean"), edge_share=("edge", "mean"),
                                                               depicted_share=("depicted", "mean")).reset_index()
        geo["dataset"] = ds; geo_all.append(geo.rename(columns={"cat": "category"}))
        # embedding sample: up to K random boxes per (age bin, category)
        for (b, c), gg in t.groupby(["age_bin", "cat"], observed=True):
            if not b:
                continue
            sel = gg.index.values if len(gg) <= args.k else rng.choice(gg.index.values, args.k, replace=False)
            s = ix.loc[sel, ["video_id", "pos", "frame_number", "xmin", "ymin", "xmax", "ymax", "unit", "age_bin", "cat", "format", "home"]].copy()
            s["dataset"] = ds; picks.append(s)
        print(f"{ds}: {len(ix):,} boxes, {len(fr):,} frames", flush=True)
        del ix, t, f
    pd.concat(frames_all).to_csv(OUT / "frames.csv.gz", index=False)
    pd.concat(pres_all).to_csv(OUT / "presence.csv.gz", index=False)
    pd.concat(geo_all).to_csv(OUT / "geometry.csv.gz", index=False)
    P = pd.concat(picks, ignore_index=True)
    P["video_id"] = P.video_id.astype(str)
    jobs = [(ds, v, g.pos.values) for (ds, v), g in P.groupby(["dataset", "video_id"])]
    order = {(ds, v): g.index.values for (ds, v), g in P.groupby(["dataset", "video_id"])}
    C = np.zeros((len(P), 512), np.float16); D = np.zeros((len(P), 768), np.float16)
    with Pool(args.workers) as pool:
        for i, (vid, pos, c, d) in enumerate(pool.imap_unordered(take, jobs, chunksize=8)):
            ds = "vedb" if vid.startswith("session_") else "bv"
            rows = order[(ds, vid)]
            C[rows], D[rows] = c, d
            if i % 1000 == 0:
                print(f"  {i:,}/{len(jobs):,} videos read", flush=True)
    for m, E in (("clip", C), ("dinov3", D)):
        np.savez(OUT / f"sample_{m}.npz", dataset=P.dataset.values.astype(str), unit=P.unit.values.astype(str),
                 age_bin=P.age_bin.values.astype(str), category=P.cat.values.astype(str),
                 format=P.format.astype(str).values, home=P.home.values.astype(bool), emb=E,
                 video_id=P.video_id.values.astype(str), frame_number=P.frame_number.values.astype(np.int32),
                 bbox=P[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32))
    print(f"done: {len(P):,} sampled boxes -> {OUT}")


if __name__ == "__main__":
    main()
