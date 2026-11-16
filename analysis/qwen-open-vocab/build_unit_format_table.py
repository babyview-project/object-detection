#!/usr/bin/env python3
"""Per child (BabyView) / per session (VEDB) x category box counts by format and frame-edge truncation, so the format and
truncation panels (Figs. 3A, 3C, 8A right) can show every child and session with a mean +- SE, not only pooled bars.

Same boxes and definitions as build_bv_vedb_data.py: Qwen's basic-level category (build_open_means.basic_of), Tier 1
categories, unit = child / session (adults only for VEDB), truncated = box within 1% of a frame edge (as
build_controls_data.py), formats real / toy / picture (photo + drawing) / screen.

Output (private, per unit): out/private/bv_vedb_renamed/unit_category_format.csv.gz
  columns dataset, unit, category, boxes, toy, picture, screen, edge
  QOV_LABELS=renamed /data/j7yang/anaconda3/bin/python build_unit_format_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_bv_vedb_data import FRAME_WH, OUT, load_index, unit_of  # noqa: E402


def main():
    from build_open_means import basic_of
    tier1 = {l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()}
    out = []
    for ds in ("bv", "vedb"):
        ix = load_index(ds)
        ix["cat"] = ix.label.map(basic_of(ix.label.unique())).fillna("")
        ix = ix[ix.cat.isin(tier1)]
        ix["unit"] = ix.video_id.astype(str).map(unit_of)
        W, H = FRAME_WH[ds]
        fmt = ix.format.astype(str)
        t = pd.DataFrame({"unit": ix.unit.values, "category": ix.cat.values, "boxes": 1,
                          "toy": (fmt == "toy").values.astype(int),
                          "picture": fmt.isin(["photo", "drawing", "picture"]).values.astype(int),
                          "screen": (fmt == "screen").values.astype(int),
                          "edge": ((ix.xmin <= .01 * W) | (ix.ymin <= .01 * H) | (ix.xmax >= .99 * W) | (ix.ymax >= .99 * H)).values.astype(int)})
        g = t.groupby(["unit", "category"], observed=True).sum().reset_index()
        g.insert(0, "dataset", ds); out.append(g)
        print(f"{ds}: {len(ix):,} Tier 1 boxes, {g.unit.nunique()} units", flush=True)
    pd.concat(out).to_csv(OUT / "unit_category_format.csv.gz", index=False)
    print("wrote", OUT / "unit_category_format.csv.gz")


if __name__ == "__main__":
    main()
