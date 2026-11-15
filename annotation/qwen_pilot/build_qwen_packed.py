#!/usr/bin/env python3
"""Qwen box store (embed_full_run.py) -> the manuscript's packed-store layout, on the valid129 categories.

So the manuscript pipeline (analysis/manuscript-2026: exemplar_set_zscore_embeddings.py, notebooks 01-09) runs on Qwen's
boxes unchanged (BV_EMBEDDINGS_BASE = one of the output roots below).

Category mapping (Qwen open label -> valid129 class): the label's normalized category (qwen_data.norm_label: plurals,
exact synonyms, depiction words dropped since `format` carries them) or its basic-level name (hierarchy table: Qwen
judged the whole name) must BE a valid129 class, after a few CDI-convention synonyms (sofa -> couch, horse -> pony,
mug -> cup, ...). Head nouns alone are never used ("computer mouse" is not mouse, "trash can" is not can). person and
picture are left out, as in the YOLOE + CLIP store. Unmapped boxes are dropped from the packed copy (they stay in the
Qwen store).

No CLIP filter (user decision 2026-10-27): every mapped box counts, so `passed` = True and `clip_sim` = 1.0 (the pipeline
applies min_sim 0.27 to clip_sim); Qwen's own CLIP score (crop vs "a photo of a {Qwen label}") is kept as qwen_clip_sim.

Outputs (OUT_BASE = /data2/dataset/babyview/2025.2/outputs/object_detections/):
  qwen_packed_valid129_all/    every format (real, toy, photo, drawing, screen), like YOLOE + CLIP counted depictions
  qwen_packed_valid129_real/   format "real" only
  each: shards/<video_id>.npz (class_name, frame_number, image_id, bbox, passed, clip_sim, qwen_clip_sim, qwen_label,
        format, clip_emb, dino_emb, dino_idx), crops_index.csv.gz, label_map.csv, run_config.json

  python build_qwen_packed.py [--workers 4]        # analysis env (vislearnlabpy: needs WordNet for norm_label)
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OD_ROOT = HERE.parents[1]
STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
OUT_BASE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections")
VARIANTS = {"all": None, "real": {"real"}}
EXCLUDE = {"person", "picture"}
# CDI-convention synonyms onto valid129 class names (only where the meaning is the same)
SYN = {"sofa": "couch", "settee": "couch", "horse": "pony", "mug": "cup", "puppy": "dog", "kitten": "cat",
       "kitty": "cat", "stone": "rock", "pebble": "rock", "trouser": "pants", "trousers": "pants", "sneaker": "shoe",
       "teeth": "tooth", "fridge": "refrigerator", "shoes": "shoe", "t-shirt": "shirt", "tshirt": "shirt",
       "sunglasses": "glasses", "pant": "pants"}


def valid129() -> list[str]:
    return [l.strip() for l in open(OD_ROOT / "data" / "included_categories_valid129.txt") if l.strip()]


def label_map(labels) -> pd.DataFrame:
    import qwen_data as qd
    valid = set(valid129()) - EXCLUDE
    basic = qd.hierarchy().qwen_basic.dropna().to_dict()
    rows = []
    for lab in labels:
        cat = qd.norm_label(lab)
        b = basic.get(cat, cat)
        cls, via = "", ""
        for name, c in (("category", cat), ("basic level", b)):
            c2 = SYN.get(c, c)
            if c2 in valid:
                cls, via = c2, name + (" + synonym" if c2 != c else "")
                break
        rows.append({"label": lab, "cat": cat, "basic": b, "class_name": cls, "via": via,
                     "in_hierarchy": cat in basic})
    return pd.DataFrame(rows)


def convert(args):
    stem, lm = args
    with np.load(STORE / "shards" / f"{stem}.npz", allow_pickle=True) as z:
        if len(z["image_id"]) == 0:
            return {}
        cls = np.array([lm.get(str(l), "") for l in z["label"]], dtype=object)
        fmt = z["format"].astype(str)
        out = {}
        for v, keep_fmt in VARIANTS.items():
            m = cls != ""
            if keep_fmt is not None:
                m &= np.isin(fmt, list(keep_fmt))
            idx = np.flatnonzero(m)
            root = OUT_BASE / f"qwen_packed_valid129_{v}" / "shards"
            if idx.size == 0:
                out[v] = None
                continue
            n = idx.size
            np.savez(root / f"{stem}.npz", class_name=cls[idx], frame_number=z["frame_number"][idx],
                     image_id=z["image_id"][idx], bbox=z["bbox"][idx], passed=np.ones(n, bool),
                     clip_sim=np.ones(n, np.float32), qwen_clip_sim=z["clip_sim"][idx], qwen_label=z["label"][idx],
                     format=fmt[idx].astype(object), clip_emb=z["clip_emb"][idx], dino_emb=z["dino_emb"][idx],
                     dino_idx=np.arange(n, dtype=np.int32))
            b = z["bbox"][idx]
            out[v] = pd.DataFrame({"image_id": z["image_id"][idx], "video_id": stem, "frame_number": z["frame_number"][idx],
                                   "class_name": cls[idx], "xmin": b[:, 0], "ymin": b[:, 1], "xmax": b[:, 2], "ymax": b[:, 3],
                                   "clip_sim": 1.0, "passed": True, "has_dino": True, "qwen_label": z["label"][idx],
                                   "format": fmt[idx], "qwen_clip_sim": z["clip_sim"][idx]})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ix = pd.read_csv(STORE / "boxes_index.csv.gz", usecols=["label", "format"])
    labs = ix.label.astype(str).value_counts()
    lm = label_map(labs.index)
    lm["boxes"] = lm.label.map(labs)
    mapped = lm[lm.class_name != ""]
    print(f"{len(lm):,} distinct labels; {len(mapped):,} map to a valid129 class, covering {mapped.boxes.sum():,} of "
          f"{labs.sum():,} boxes ({mapped.boxes.sum() / labs.sum():.1%}); labels in the hierarchy table: "
          f"{lm.loc[lm.in_hierarchy, 'boxes'].sum() / labs.sum():.1%} of boxes", flush=True)
    for v in VARIANTS:
        (OUT_BASE / f"qwen_packed_valid129_{v}" / "shards").mkdir(parents=True, exist_ok=True)
        lm.to_csv(OUT_BASE / f"qwen_packed_valid129_{v}" / "label_map.csv", index=False)
    mp = dict(zip(mapped.label, mapped.class_name))
    stems = sorted(p.stem for p in (STORE / "shards").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    parts = {v: [] for v in VARIANTS}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(convert, [(s, mp) for s in stems], chunksize=8)):
            for v, d in res.items():
                if d is not None:
                    parts[v].append(d)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(stems):,} videos", flush=True)
    for v in VARIANTS:
        root = OUT_BASE / f"qwen_packed_valid129_{v}"
        d = pd.concat(parts[v], ignore_index=True)
        d.to_csv(root / "crops_index.csv.gz", index=False)
        (root / "run_config.json").write_text(json.dumps({
            "source_store": str(STORE), "variant": v, "formats": sorted(VARIANTS[v]) if VARIANTS[v] else "all",
            "category_set": "valid129 minus person, picture", "mapping": "norm_label or hierarchy basic level == class "
            "(after CDI synonyms); no head-noun fallback", "synonyms": SYN, "clip_filter": "none (passed=True, clip_sim=1.0; "
            "Qwen's own CLIP score in qwen_clip_sim)", "boxes": len(d), "videos": int(d.video_id.nunique()),
            "frames": "every 5th 1 fps frame (full Qwen run)"}, indent=2) + "\n")
        print(f"{v}: {len(d):,} boxes in {d.video_id.nunique():,} videos, {d.class_name.nunique()} classes -> {root}")


if __name__ == "__main__":
    main()
