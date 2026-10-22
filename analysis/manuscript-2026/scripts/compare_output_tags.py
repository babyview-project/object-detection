#!/usr/bin/env python3
"""Compare every result table between two output tags (e.g. 2025.2 vs 2025.2_5s).

For each CSV present under both main_results_valid129s_<tag>/ and
supplemental_results_valid85cats_<tag>/ (caches and per-subject files skipped), rows are
aligned on their text columns (file paths ignored) or, if that fails, by position.
Per numeric column it reports the largest and mean absolute change and the correlation
across rows. Columns that look like p-values are checked for crossing 0.05; columns that
look like correlations / effects for sign changes.

Output: <out>/tag_comparison_columns.csv (one row per table x column) and
        <out>/tag_comparison_flags.csv (p crossing 0.05, sign changes).

  python scripts/compare_output_tags.py --a 2025.2 --b 2025.2_5s
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MS = Path(__file__).resolve().parents[1]
ROOTS = ["main_results_valid129s", "supplemental_results_valid85cats"]
SKIP = re.compile(r"cache|subject_|all_subjects_.*pairwise|/figures/")
PATHLIKE = re.compile(r"path|png|pdf|file|dir|_at$|date|time", re.I)
PCOL = re.compile(r"(^p$|^p_|_p$|_p_|pval|p_value|^perm_p|boot_p)", re.I)
EFFECT = re.compile(r"(_r$|_r_|^r$|rho|tau|pearson|spearman|kendall|delta|diff|slope|beta|corr)", re.I)


def is_p(c):
    return bool(PCOL.search(c)) and not re.search(r"pairs|prop|pred|prev|precision|pass", c, re.I)


def align(a, b):
    keys = [c for c in a.columns if c in b.columns and not pd.api.types.is_numeric_dtype(a[c])
            and not PATHLIKE.search(c)]
    if keys and not a.duplicated(keys).any() and not b.duplicated(keys).any():
        m = a.merge(b, on=keys, suffixes=("__a", "__b"))
        if len(m) == len(a) == len(b):
            return m, "keys"
    if len(a) == len(b):
        return pd.concat([a.add_suffix("__a"), b.add_suffix("__b")], axis=1), "position"
    return None, f"rows differ ({len(a)} vs {len(b)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="2025.2")
    ap.add_argument("--b", default="2025.2_5s")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or MS / f"main_results_valid129s_{args.b}" / "results" / "comparison_vs_1s"
    out.mkdir(parents=True, exist_ok=True)
    rows, flags, skipped = [], [], []
    for root in ROOTS:
        da, db = MS / f"{root}_{args.a}", MS / f"{root}_{args.b}"
        for fa in sorted(da.rglob("*.csv")):
            rel = fa.relative_to(da).as_posix()
            if SKIP.search("/" + rel) or "comparison_vs" in rel:
                continue
            fb = db / rel
            if not fb.exists():
                skipped.append((root, rel, "missing in b")); continue
            a, b = pd.read_csv(fa), pd.read_csv(fb)
            m, how = align(a, b)
            if m is None:
                skipped.append((root, rel, how)); continue
            for c in a.columns:
                if c not in b.columns or not pd.api.types.is_numeric_dtype(a[c]) or PATHLIKE.search(c):
                    continue
                x, y = pd.to_numeric(m[c + "__a"], errors="coerce"), pd.to_numeric(m[c + "__b"], errors="coerce")
                ok = x.notna() & y.notna()
                if not ok.any():
                    continue
                d = (y[ok] - x[ok]).abs()
                r = x[ok].corr(y[ok]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
                rows.append({"set": root.split("_")[-1], "table": rel, "aligned_by": how, "column": c,
                             "n": int(ok.sum()), "max_abs_change": d.max(), "mean_abs_change": d.mean(),
                             "a_mean": x[ok].mean(), "b_mean": y[ok].mean(), "corr_a_b": r})
                if is_p(c):
                    flip = ok & ((x < 0.05) != (y < 0.05))
                    for i in np.flatnonzero(flip):
                        flags.append({"set": root.split("_")[-1], "table": rel, "column": c, "row": int(i),
                                      "kind": "p crosses 0.05", "a": x.iloc[i], "b": y.iloc[i]})
                elif EFFECT.search(c):
                    sign = ok & (np.sign(x) != np.sign(y)) & ((x.abs() > 0.05) | (y.abs() > 0.05))
                    for i in np.flatnonzero(sign):
                        flags.append({"set": root.split("_")[-1], "table": rel, "column": c, "row": int(i),
                                      "kind": "sign change", "a": x.iloc[i], "b": y.iloc[i]})
    cols, fl = pd.DataFrame(rows), pd.DataFrame(flags)
    cols.to_csv(out / "tag_comparison_columns.csv", index=False)
    fl.to_csv(out / "tag_comparison_flags.csv", index=False)
    print(f"{cols.table.nunique()} tables, {len(cols)} numeric columns compared; skipped {len(skipped)}: {skipped[:6]}")
    print(f"flags: {fl.kind.value_counts().to_dict() if len(fl) else 'none'}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
