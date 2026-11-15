#!/usr/bin/env python3
"""Before / after the VEDB adults-only rerun: every results table of the Qwen pipeline vs the pre-exclusion snapshot.

Rows are matched on their non-numeric columns (model, dataset, group, ...), so tables whose category counts changed
(183 -> 182 core) still line up; `n_categories` / `categories` / `n` are treated as keys only when they are not the
quantity being compared. Writes vedb-1612/results/adults_rerun_changes.csv (every changed number) and prints, per
file, the largest absolute change among DINOv3 rows (the paper reports DINOv3 only) or all rows when there is no
model column.

  python compare_adults_rerun.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEW = HERE.parents[0] / "qwen-open-vocab" / "results"
OLD = Path("/data2/j7yang/object-detection-offload/backups/pre_vedb_adults_20261115/qwen-open-vocab/results")
DIRS = ["crossdataset_renamed_core", "bv_vedb_renamed_core", "controls_renamed_core", "bv_vedb_renamed",
        "controls_renamed", "crossdataset_renamed_things27_core", "bv_vedb_renamed_things27_core",
        "controls_renamed_things27_core", "open_renamed"]
COUNTS = {"n", "n_categories", "categories", "draws", "n_boot"}


def compare(a: pd.DataFrame, b: pd.DataFrame):
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    keys = [c for c in a.columns if c in b.columns and c not in num]
    if not keys:
        return None
    a = a.assign(_k=a.groupby(keys, dropna=False).cumcount()); b = b.assign(_k=b.groupby(keys, dropna=False).cumcount())
    m = a.merge(b, on=keys + ["_k"], suffixes=("_old", "_new"))
    rows = []
    for c in num:
        d = m[c + "_new"] - m[c + "_old"]
        ch = d.abs() > 1e-9
        for i in np.flatnonzero(ch.values):
            r = m.iloc[i]
            rows.append({**{k: r[k] for k in keys}, "column": c, "old": r[c + "_old"], "new": r[c + "_new"],
                         "change": d.iloc[i]})
    return pd.DataFrame(rows), len(m), len(a), len(b)


def main():
    out, summary = [], []
    for d in DIRS:
        for f in sorted((NEW / d).glob("*.csv")):
            o = OLD / d / f.name
            if not o.exists():
                summary.append((f"{d}/{f.name}", "new file", "")); continue
            try:
                r = compare(pd.read_csv(o), pd.read_csv(f))
            except Exception as e:                                   # noqa: BLE001
                summary.append((f"{d}/{f.name}", f"not compared ({type(e).__name__})", "")); continue
            if r is None:
                summary.append((f"{d}/{f.name}", "no key columns", "")); continue
            t, n_m, n_o, n_n = r
            if t.empty:
                summary.append((f"{d}/{f.name}", f"unchanged ({n_m} rows)", "")); continue
            t.insert(0, "file", f"{d}/{f.name}")
            out.append(t)
            v = t[~t.column.isin(COUNTS)]
            if "model" in v:
                v = v[v.model == "dinov3"] if (v.model == "dinov3").any() else v
            if v.empty:
                summary.append((f"{d}/{f.name}", "only counts changed", "")); continue
            top = v.loc[v.change.abs().idxmax()]
            summary.append((f"{d}/{f.name}", f"max |change| {abs(top.change):.4f}",
                            f"{top.column}: {top.old:.4g} -> {top.new:.4g}  ({', '.join(f'{k}={top[k]}' for k in t.columns[1:4] if k not in ('column', 'old', 'new', 'change'))})"))
    res = HERE / "results"; res.mkdir(exist_ok=True)
    if out:
        pd.concat(out).to_csv(res / "adults_rerun_changes.csv", index=False)
    for f, s, x in summary:
        print(f"{f:70s} {s:28s} {x}")


if __name__ == "__main__":
    main()
