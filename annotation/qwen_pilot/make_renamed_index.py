#!/usr/bin/env python3
"""Renamed copy of a full-run box index: generic labels replaced by the second-pass names (qwen_generic_crops.py),
and "mouse" split into the animal and the computer mouse. The original boxes_index.csv.gz is not changed.

label_renamed:
  - generic boxes (toy, animal, food, character, ...): the second-pass name, unless it is "unclear", "none", empty, or a
    generic label again (then the original label);
  - "mouse" (after normalization) with format "real" in a frame that also has a keyboard, laptop, computer, monitor or
    computer screen: "computer mouse";
  - everything else: the original label.
renamed_from: why it changed ("generic pass", "computer mouse rule" or "").

Output: <store>/boxes_index_renamed.csv.gz (the original columns + label_renamed, renamed_from)
  python make_renamed_index.py --store bv          # analysis env (norm_label needs WordNet)
  python make_renamed_index.py --store vedb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
STORES = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
          "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv")}
COMPUTER = {"keyboard", "laptop", "computer", "monitor", "computer monitor", "computer screen", "desktop computer"}
KEEP = {"unclear", "none", ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", choices=list(STORES), required=True)
    args = ap.parse_args()
    import qwen_data as qd
    from qwen_generic_crops import GENERIC
    root = STORES[args.store]
    ix = pd.read_csv(root / "boxes_index.csv.gz")
    ix["label"] = ix.label.astype(str)
    width = 8 if args.store == "vedb" else 5
    ix["unit_id"] = ix.video_id + "/" + ix.frame_number.map(lambda f: f"{f:0{width}d}")
    names = {}
    for l in open(PRIV / f"qwen_generic_crops_fullrun_{args.store}.jsonl"):
        r = json.loads(l)
        names[(r["unit_id"], int(r["obj"]))] = (r["name"] or "").strip().lower()
    new = [names.get((u, int(o))) for u, o in zip(ix.unit_id, ix.obj_idx)]
    nm = {l: qd.norm_label(l) for l in set(ix.label) | {n for n in new if n}}
    out, why = [], []
    for lab, n in zip(ix.label, new):
        if n is not None and n not in KEEP and nm.get(n, n) not in GENERIC:
            out.append(n); why.append("generic pass")
        else:
            out.append(lab); why.append("")
    ix["label_renamed"], ix["renamed_from"] = out, why
    # computer mouse: real "mouse" boxes in frames with a computer
    cat = ix.label_renamed.map(lambda l: nm.get(l) or qd.norm_label(l))
    has_pc = ix.assign(pc=cat.isin(COMPUTER)).groupby("unit_id").pc.transform("any")
    m = (cat == "mouse") & (ix.format.astype(str) == "real") & has_pc
    ix.loc[m, "label_renamed"] = "computer mouse"; ix.loc[m, "renamed_from"] = "computer mouse rule"
    ix.drop(columns="unit_id").to_csv(root / "boxes_index_renamed.csv.gz", index=False)
    print(f"{args.store}: {len(ix):,} boxes; renamed by the generic pass {(ix.renamed_from == 'generic pass').sum():,}, "
          f"computer mouse rule {m.sum():,} (of {(cat == 'mouse').sum():,} mouse boxes) -> {root / 'boxes_index_renamed.csv.gz'}")


if __name__ == "__main__":
    main()
