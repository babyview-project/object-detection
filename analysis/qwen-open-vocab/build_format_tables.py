#!/usr/bin/env python3
"""Per (dataset, Tier 1 category): share of exemplars that are real / toy / picture / screen.

  bv, vedb      Qwen's own format labels on every box (photo + drawing = picture), all formats counted
  things        format probe (format_classifier.py) on every THINGS image of the mapped concepts
  lvis          format probe on up to 300 crops per mapped LVIS category (extra categories: embed_lvis_extra.py sample;
                the others: comparison-datasets' per-crop embedding files on /ccn2a)
  imagenet      format probe on up to 300 images per mapped ImageNet class (per-image embedding files on /ccn2a)
Steps are independent and merge into out/crossdataset/format_by_category.csv (index dataset, category; columns real,
toy, picture, screen, n). The /ccn2a steps (lvis, imagenet) read many small files: run them when the VEDB run is done.

  python build_format_tables.py bv things            # analysis env
  python build_format_tables.py lvis imagenet vedb
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
import variant as V                                     # QOV_LABELS=renamed: Qwen rows from the renamed index
XD0 = HERE / "out" / "crossdataset"
XD = V.out_dir(XD0)
MAP = HERE / "mapping"
RES = HERE / "results"
OUTF = XD / "format_by_category.csv"
FMTS = ["real", "toy", "picture", "screen"]
CD_EMB = {"lvis": Path("/ccn2a/dataset/babyview/related_data/lvis/outputs/crop_embeddings"),
          "imagenet": Path("/ccn2a/dataset/babyview/related_data/imagenet_1k/outputs/image_embeddings")}
CLIP_DIR, DINO_DIR = "openai_clip-vit-base-patch32", "facebook_dinov3-vitb16-pretrain-lvd1689m"
PER_CAT = 300


def shares(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """df: category, format -> shares per category."""
    t = df.groupby("category").format.value_counts().unstack(fill_value=0).reindex(columns=FMTS, fill_value=0)
    n = t.sum(axis=1)
    out = t.div(n, axis=0).assign(n=n)
    out.index = pd.MultiIndex.from_product([[dataset], out.index], names=["dataset", "category"])
    return out


def qwen_boxes(store: Path, dataset: str) -> pd.DataFrame:
    from build_open_means import basic_of
    ix = V.read_index(store, usecols=["label", "format"])
    ix["label"] = ix.label.astype(str)
    b = basic_of(ix.label.unique())
    tier1 = {l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()}
    ix["category"] = ix.label.map(b)
    ix = ix[ix.category.isin(tier1)]
    ix["format"] = ix.format.replace({"photo": "picture", "drawing": "picture"})
    return shares(ix[ix.format.isin(FMTS)], dataset)


def probe_predict(clip: np.ndarray, dino: np.ndarray, labels: list[str]) -> np.ndarray:
    import format_classifier as fc
    model = pickle.load(open(RES / "format_classifier.pkl", "rb"))
    bank = fc.text_bank(sorted(set(labels)))
    F = fc.features(clip.astype(np.float32), labels, bank)
    def l2(A): A = A.astype(np.float32); return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return model["clf"].classes_[model["clf"].predict_proba(np.hstack([l2(clip), l2(dino), F / 100.0])).argmax(1)]


def things() -> pd.DataFrame:
    f = pd.read_csv(RES / "format_things.csv", usecols=["item", "format"])
    m = pd.read_csv(MAP / "map_things.csv").set_index("id").category
    f["category"] = f.item.map(m)
    return shares(f.dropna(subset=["category"]), "things")


def load_npy(paths):
    out = []
    for p in paths:
        try:
            out.append(np.load(p).astype(np.float32).ravel())
        except Exception:
            out.append(None)
    return out


def lvis() -> pd.DataFrame:
    import json
    m = pd.read_csv(MAP / "map_lvis.csv")
    names = {c["id"]: c["name"].replace("_", " ") for c in json.load(open("/data2/j7yang/lvis/annotations/lvis_v1_val.json"))["categories"]}
    parts = []
    ex = np.load(XD0 / "lvis_extra_crops_sample.npz", allow_pickle=True)          # newly embedded categories
    cid = ex["category_id"]
    lab = [names[int(c)] for c in cid]
    parts.append(pd.DataFrame({"category_id": cid, "format": probe_predict(ex["clip"], ex["dinov3"], lab)}))
    meta = pd.read_csv(HERE.parents[1] / "analysis" / "comparison-datasets" / "outputs" / "mapping" / "lvis_crop_metadata.csv.gz",
                       usecols=["crop_id", "category_id"])
    meta = meta[meta.category_id.isin(set(m.id)) & ~meta.category_id.isin(set(cid))]
    samp = meta.groupby("category_id", group_keys=False).apply(lambda g: g.sample(min(PER_CAT, len(g)), random_state=0))
    C = load_npy([CD_EMB["lvis"] / CLIP_DIR / f"{c}.npy" for c in samp.crop_id])
    D = load_npy([CD_EMB["lvis"] / DINO_DIR / f"{c}.npy" for c in samp.crop_id])
    ok = [i for i, (a, b) in enumerate(zip(C, D)) if a is not None and b is not None]
    s2 = samp.iloc[ok]
    parts.append(pd.DataFrame({"category_id": s2.category_id.values,
                               "format": probe_predict(np.stack([C[i] for i in ok]), np.stack([D[i] for i in ok]),
                                                       [names[int(c)] for c in s2.category_id])}))
    f = pd.concat(parts)
    f["category"] = f.category_id.map(m.set_index("id").category)
    return shares(f.dropna(subset=["category"]), "lvis")


def imagenet() -> pd.DataFrame:
    from nltk.corpus import wordnet as wn
    import os
    m = pd.read_csv(MAP / "map_imagenet.csv")
    by_wnid = {}                                   # flat folder: <split>_..._<wnid>.npy
    for f in os.listdir(CD_EMB["imagenet"] / CLIP_DIR):
        by_wnid.setdefault(f[:-4].rsplit("_", 1)[-1], []).append(f)
    rows = []
    for w in m.id:
        files = [CD_EMB["imagenet"] / CLIP_DIR / f for f in sorted(by_wnid.get(w, []))]
        if not files:
            continue
        rng = np.random.default_rng(0)
        pick = [files[i] for i in rng.choice(len(files), min(PER_CAT, len(files)), replace=False)]
        rows += [(w, p) for p in pick]
    if not rows:
        raise SystemExit(f"no per-image ImageNet embeddings found under {CD_EMB['imagenet'] / CLIP_DIR}/<wnid>/")
    df = pd.DataFrame(rows, columns=["wnid", "clip_path"])
    df["dino_path"] = [str(p).replace(CLIP_DIR, DINO_DIR) for p in df.clip_path]
    C, D = load_npy(df.clip_path), load_npy(df.dino_path)
    ok = [i for i, (a, b) in enumerate(zip(C, D)) if a is not None and b is not None]
    df = df.iloc[ok]
    lab = [wn.synset_from_pos_and_offset("n", int(w[1:])).lemma_names()[0].replace("_", " ") for w in df.wnid]
    f = pd.DataFrame({"wnid": df.wnid.values, "format": probe_predict(np.stack([C[i] for i in ok]), np.stack([D[i] for i in ok]), lab)})
    f["category"] = f.wnid.map(m.set_index("id").category)
    return shares(f.dropna(subset=["category"]), "imagenet")


def main():
    steps = {"bv": lambda: qwen_boxes(Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"), "bv"),
             "vedb": lambda: qwen_boxes(Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults"), "vedb"),
             "things": things, "lvis": lvis, "imagenet": imagenet}
    XD.mkdir(parents=True, exist_ok=True)
    src = OUTF if OUTF.exists() else XD0 / "format_by_category.csv"      # a new variant starts from the original table
    old = pd.read_csv(src, index_col=[0, 1]) if src.exists() else pd.DataFrame()
    for s in sys.argv[1:]:
        t = steps[s]()
        old = pd.concat([old[old.index.get_level_values(0) != s] if len(old) else old, t])
        old.to_csv(OUTF)
        print(f"{s}: {len(t)} categories, {int(t.n.sum()):,} exemplars; overall "
              f"{ {f: round(float((t[f] * t.n).sum() / t.n.sum()), 3) for f in FMTS} }", flush=True)


if __name__ == "__main__":
    main()
