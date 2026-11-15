#!/usr/bin/env python3
"""Inputs for the THINGS-27 domain variant (QOV_DOMAINS=things27): copies of the renamed-label data folders with the
domain column of categories.csv replaced by data/category_domains_things27.csv `domain_analysis` (a THINGS group with
>= 5 core categories, else a singleton "other:<category>", so the category never forms within-domain pairs).

Copies, not links: a script that rewrites a file in a *_things27 folder must never write through to the CDI original.
Folders that the variant rebuilds itself (out/crossdataset_renamed_things27, from build_crossdataset_tables.py) are not
copied. Rerun after the domain table changes (only categories.csv is rewritten when a copy exists).

  QOV_LABELS=renamed python make_things27_inputs.py
"""
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P = HERE / "out" / "private"
FOLDERS = ["open_renamed", "open_vedb_renamed", "bv_vedb_renamed", "controls_renamed", "dev_within_child_renamed",
           "things_readout"]


def main():
    dom = pd.read_csv(HERE.parents[1] / "data" / "category_domains_things27.csv").set_index("category").domain_analysis
    for f in FOLDERS:
        src = P / f
        dst = P / (f + "_things27") if f.endswith("_renamed") else P / (f + "_renamed_things27")
        if not src.exists():
            print("missing", src); continue
        if not dst.exists():
            shutil.copytree(src, dst)
            print("copied", src.name, "->", dst.name, flush=True)
        c = dst / "categories.csv"
        if c.exists():
            t = pd.read_csv(src / "categories.csv", index_col=0)
            t["domain_cdi"] = t.domain
            t["domain"] = [dom.get(k, f"other:{k}") for k in t.index]
            t.to_csv(c)
            print(f"  {c.relative_to(HERE)}: {t.domain.str.startswith('other:').sum()} singleton, "
                  f"{t.domain.nunique()} distinct", flush=True)


if __name__ == "__main__":
    main()
