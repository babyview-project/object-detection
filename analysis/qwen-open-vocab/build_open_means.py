#!/usr/bin/env python3
"""Open-category (Qwen basic level) embedding means for the manuscript-style analyses beyond valid129.

Category = Qwen's basic-level name (qwen_data.norm_label, then the hierarchy table's basic level; CDI-convention
synonyms as build_qwen_packed.SYN). Open set (user choice 2026-10-27): >= 300 boxes from >= 10 children, excluding generic
labels (hierarchy `generic`, toy, animal, object, ...), person, and surfaces / places RAs do not box (floor, ceiling,
wall, carpet, sky, ground, road, doorway, staircase, stairs, room).

One pass over the Qwen box store (embed_full_run.py): for each box in the open set, its CLIP and DINOv3 vectors are
L2-normalized and summed per (child, category), separately for all formats and for format "real"; means follow the
manuscript (mean of L2-normalized vectors; z-scoring across the categories compared happens at analysis time).
Frame prevalence counts come from the box index.

Outputs (analysis/qwen-open-vocab/out/private/open/, private: per-child tables):
  categories.csv           category (Tier 1 list: also written to data/included_categories_qwen_open_tier1.txt), boxes, frames, children, domain, things_concept (exact name match or synonym), ...
  sums_{clip,dinov3}.npz   subjects, categories, sum_all / n_all / sum_real / n_real arrays (subject x category [x dim])
  frames_by_child.csv      frames per (child, category) with the category (all formats, and real only)
  frames_total.csv         frames per child

  python build_open_means.py [--workers 4]       # analysis env (needs WordNet for norm_label)

--dataset vedb: the VEDB 1612 Qwen store (annotation/qwen_pilot/embed_full_run.py --dataset vedb) on BabyView's Tier 1
list (data/included_categories_qwen_open_tier1.txt, not reselected); the unit is the session (VEDB has no child ids).
Output: out/private/open_vedb/ (same files; "subjects" are sessions).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QWEN_PILOT = HERE.parents[1] / "annotation" / "qwen_pilot"     # qwen_data, build_qwen_packed (label normalization)
sys.path.insert(0, str(QWEN_PILOT))
STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
THINGS = Path("/data2/dataset/babyview/related_data/things")
OUT = HERE / "out" / "private" / "open"
MIN_BOXES, MIN_CHILDREN = 300, 10
GENERIC = {"toy", "animal", "object", "item", "thing", "character", "food", "vehicle", "figure", "stuff"}
SURFACES = {"floor", "ceiling", "wall", "carpet", "sky", "ground", "road", "street", "doorway", "staircase", "stairs", "room"}
# Tier 1 merges (2026-10-28): same thing under two names (WordNet first sense or singular/plural) -> one name
MERGE = {"tub": "bathtub", "chalkboard": "blackboard", "shrub": "bush", "chest of drawers": "dresser",
         "garbage can": "trash can", "trash bin": "trash can", "handbag": "purse", "lightbulb": "light bulb",
         "mantelpiece": "mantel", "packet": "package", "photograph": "photo", "fabric": "cloth",
         "ornament": "decoration", "blind": "blinds", "legging": "leggings", "pajama": "pajamas"}
THINGS_SYN = {"sofa": "couch", "trousers": "pants", "trouser": "pants", "vacuum cleaner": "vacuum", "tv": "television",
              "cellphone": "cell phone", "phone": "telephone", "refrigerator": "refrigerator"}


def basic_of(labels) -> dict[str, str]:
    import qwen_data as qd
    from build_qwen_packed import SYN
    h = qd.hierarchy().qwen_basic.dropna().to_dict()
    out = {}
    for lab in labels:
        cat = qd.norm_label(lab)
        b = h.get(cat, cat)
        b = SYN.get(b, b)
        out[lab] = MERGE.get(b, b)
    return out


def things_match(cats) -> dict[str, str]:
    t = pd.read_csv(THINGS / "concepts-metadata_things.tsv", sep="\t", usecols=["Word", "uniqueID"])
    t["w"] = t.Word.astype(str).str.strip().str.lower()
    words = t.groupby("w").uniqueID.apply(list)          # homonyms (bat, mouse) list several concepts
    out = {}
    for c in cats:
        w = THINGS_SYN.get(c, c)
        if w in words.index and len(words[w]) == 1:
            out[c] = words[w][0]
    return out


def unit_of(stem: str) -> str:
    """BabyView: subject id; VEDB: session id."""
    return stem.split("_")[1] if stem.startswith("session_") else stem.split("_")[0]


def accumulate(args):
    stem, cmap, cats, store, labels = args
    ix = {c: i for i, c in enumerate(cats)}
    with np.load(store / "shards" / f"{stem}.npz", allow_pickle=True) as z:
        if len(z["image_id"]) == 0:
            return None
        lab = z["label"] if labels is None else labels          # renamed variant: labels from the renamed index
        b = np.array([cmap.get(str(l), "") for l in lab], dtype=object)
        keep = np.array([x in ix for x in b])
        if not keep.any():
            return None
        ci = np.array([ix[x] for x in b[keep]])
        real = (z["format"].astype(str)[keep] == "real")
        res = {"subject": unit_of(stem)}
        for model in ("clip", "dinov3"):
            X = z["clip_emb" if model == "clip" else "dino_emb"][keep].astype(np.float32)
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            S = np.zeros((len(cats), X.shape[1]), np.float64); np.add.at(S, ci, X)
            R = np.zeros_like(S); np.add.at(R, ci[real], X[real])
            res[model] = (S, R)
        res["n_all"] = np.bincount(ci, minlength=len(cats))
        res["n_real"] = np.bincount(ci[real], minlength=len(cats))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset", choices=["bv", "vedb"], default="bv")
    args = ap.parse_args()
    global STORE, OUT
    import variant as V
    OUT = V.out_dir(OUT)
    if args.dataset == "vedb":
        STORE = Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults")
        OUT = V.out_dir(HERE / "out" / "private" / "open_vedb")
    import qwen_data as qd
    OUT.mkdir(parents=True, exist_ok=True)
    ix = V.read_index(STORE, usecols=["video_id", "frame_number", "label", "format"])
    ix["label"] = ix.label.astype(str)
    cmap = basic_of(ix.label.unique())
    ix["cat"] = ix.label.map(cmap)
    ix["subject"] = ix.video_id.map(unit_of)
    ix["frame"] = ix.video_id + "/" + ix.frame_number.astype(str)
    h = qd.hierarchy()
    generic = GENERIC | set(h.index[h.qwen_generic.fillna(False).astype(bool)])
    g = ix.groupby("cat").agg(boxes=("label", "size"), frames=("frame", "nunique"), children=("subject", "nunique"))
    g["real_share"] = ix.assign(r=ix.format == "real").groupby("cat").r.mean()
    if args.dataset == "bv" and V.LABELS == "original":
        sel = g[(g.boxes >= MIN_BOXES) & (g.children >= MIN_CHILDREN) & ~g.index.isin(generic | SURFACES | {"person"})]
    else:                                      # BabyView's Tier 1 list, whatever VEDB has of it
        tier1 = [l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()]
        sel = g[g.index.isin(set(tier1))]
    cats = sorted(sel.index)
    dom = h.domain.dropna().to_dict() | qd.DOMAIN_OVERRIDES          # hand overrides win over Qwen's hierarchy
    sel = sel.loc[cats].assign(domain=[dom.get(c, "") for c in cats])
    # domain of a basic name that is not itself a hierarchy category: the most frequent category mapping to it
    miss = sel.index[sel.domain == ""]
    if len(miss):
        cd = ix[ix.cat.isin(miss)].assign(c2=lambda d: d.label.map(qd.norm_label)).groupby("cat").c2.agg(lambda s: s.value_counts().index[0])
        sel.loc[miss, "domain"] = [dom.get(cd.get(c, c), "other") for c in miss]
    tm = things_match(cats)
    sel["things_concept"] = [tm.get(c, "") for c in cats]
    sel.to_csv(OUT / "categories.csv")
    if args.dataset == "bv" and V.LABELS == "original":
        (HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt").write_text("\n".join(cats) + "\n")
    if args.dataset == "bv" and V.LABELS == "renamed":         # categories the renamed labels would add to Tier 1
        new = g[(g.boxes >= MIN_BOXES) & (g.children >= MIN_CHILDREN) & ~g.index.isin(generic | SURFACES | {"person"} | set(cats))]
        new.sort_values("boxes", ascending=False).to_csv(OUT / "tier1_candidates_from_renaming.csv")
        print(f"{len(new)} categories would newly qualify for Tier 1 with the renamed labels (not added)", flush=True)
    print(f"{len(cats)} open categories ({sel.boxes.sum():,} boxes, {sel.boxes.sum() / len(ix):.1%}); "
          f"{(sel.things_concept != '').sum()} match a THINGS concept", flush=True)

    lab_by_video = ({v: g_.label.values for v, g_ in ix.groupby("video_id", sort=False)} if V.LABELS == "renamed" else {})
    sub = ix[ix.cat.isin(set(cats))]
    fr = sub.groupby(["subject", "cat"]).frame.nunique().rename("frames_all").to_frame()
    fr["frames_real"] = sub[sub.format == "real"].groupby(["subject", "cat"]).frame.nunique()
    fr.fillna(0).astype(int).to_csv(OUT / "frames_by_child.csv")
    ix.groupby("subject").frame.nunique().rename("frames").to_csv(OUT / "frames_total.csv")
    del ix, sub

    subjects = sorted(fr.index.get_level_values(0).unique())
    si = {s: i for i, s in enumerate(subjects)}
    acc = {m: {"sum_all": np.zeros((len(subjects), len(cats), d)), "sum_real": np.zeros((len(subjects), len(cats), d))}
           for m, d in (("clip", 512), ("dinov3", 768))}
    n_all = np.zeros((len(subjects), len(cats)), np.int64); n_real = np.zeros_like(n_all)
    stems = sorted(p.stem for p in (STORE / "shards").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(accumulate, [(s, cmap, cats, STORE, lab_by_video.get(s)) for s in stems], chunksize=4)):
            if r is not None:
                k = si[r["subject"]]
                for m in ("clip", "dinov3"):
                    acc[m]["sum_all"][k] += r[m][0]; acc[m]["sum_real"][k] += r[m][1]
                n_all[k] += r["n_all"]; n_real[k] += r["n_real"]
            if i % 1000 == 0:
                print(f"  {i:,}/{len(stems):,} videos", flush=True)
    for m in acc:
        np.savez(OUT / f"sums_{m}.npz", subjects=np.array(subjects), categories=np.array(cats),
                 sum_all=acc[m]["sum_all"].astype(np.float32), sum_real=acc[m]["sum_real"].astype(np.float32),
                 n_all=n_all, n_real=n_real)
    print(f"done: {len(subjects)} children x {len(cats)} categories -> {OUT}")


if __name__ == "__main__":
    main()
