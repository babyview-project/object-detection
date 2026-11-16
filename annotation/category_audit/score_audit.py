#!/usr/bin/env python3
"""Category audit, step 3: score the RA exports and apply the exclusion rule.

Reads out/private/items.csv, assignment.csv and every export in out/private/responses/*.csv (the page's
"Export backup / answers" files; the latest submitted answer per rater, item and step wins).

Per item: name_ok = not marked in step 1 (shared items: mean over raters); depicted = marked in step 2.
Per category x dataset (aggregate, results/audit_by_category.csv):
  name_precision      share of Qwen's boxes that are the named category (Wilson 95% CI)
  rater_depicted      share depicted among boxes that are the category (rater, step 2)
  qwen_depicted       the same share by Qwen's format label, on the same boxes
  format_agreement    share of those boxes where Qwen's real / depicted call matches the rater's
Agreement on shared items (results/audit_agreement.csv): percent agreement and Cohen's kappa per step.

Exclusion rule (fixed before rating, 2026-11-15): drop a category if its name precision is below RULE in BabyView or
in VEDB -> results/excluded_categories.txt. The analyses are then rerun with and without them (sensitivity check).

  /data/j7yang/anaconda3/bin/python score_audit.py [--rule 0.70]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
RES = HERE / "results"
DEPICTED = {"toy", "photo", "drawing", "picture", "screen"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def answers(folder: Path) -> pd.DataFrame:
    fs = sorted(folder.glob("*.csv"))
    if not fs:
        raise SystemExit(f"no exports in {folder}")
    r = pd.concat([pd.read_csv(f, dtype={"item_id": str}) for f in fs])
    r = r[r.study == "category-audit"]
    r = r[r.step_submitted.notna() & (r.step_submitted.astype(str) != "")]
    r = r.sort_values("exported_at").drop_duplicates(["rater", "item_id", "phase"], keep="last")
    return r.pivot_table(index=["rater", "item_id"], columns="phase", values="marked", aggfunc="first").reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", type=float, default=0.70)
    ap.add_argument("--responses", type=Path, default=OUT / "responses")
    ap.add_argument("--results", type=Path, default=RES)
    a = ap.parse_args()
    res = a.results; res.mkdir(parents=True, exist_ok=True)
    it = pd.read_csv(OUT / "items.csv", dtype={"item_id": str})
    sh = pd.read_csv(OUT / "assignment.csv", dtype={"item_id": str}).set_index("item_id").shared
    A = answers(a.responses).rename(columns={"crop": "not_it", "frame": "depicted"})
    # agreement on shared items, both steps
    ag = []
    S = A[A.item_id.map(sh).fillna(False).astype(bool)]
    raters = sorted(S.rater.unique())
    if len(raters) >= 2:
        w = S.pivot_table(index="item_id", columns="rater", values=["not_it", "depicted"])
        for step in ("not_it", "depicted"):
            x = w[step][raters[:2]].dropna()
            po, k = kappa(x[raters[0]], x[raters[1]])
            ag.append(dict(step=step, raters=" vs ".join(raters[:2]), items=len(x), agreement=po, kappa=k))
    pd.DataFrame(ag).to_csv(res / "audit_agreement.csv", index=False)
    # per item: mean over raters
    per = A.groupby("item_id")[["not_it", "depicted"]].mean().reset_index().merge(it, on="item_id")
    per["name_ok"] = 1 - per.not_it
    per["qwen_dep"] = per.qwen_format.isin(DEPICTED).astype(float)
    rows = []
    for (cat, ds), g in per.groupby(["category", "dataset"]):
        ok = g[g.name_ok >= .5]
        k = g.name_ok.sum(); n = len(g); lo, hi = wilson(k, n)
        rows.append(dict(category=cat, dataset=ds, items=n, name_precision=k / n, ci_low=lo, ci_high=hi,
                         rater_depicted=ok.depicted.mean() if len(ok) else np.nan,
                         qwen_depicted=ok.qwen_dep.mean() if len(ok) else np.nan,
                         format_agreement=((ok.depicted >= .5) == (ok.qwen_dep >= .5)).mean() if len(ok) else np.nan))
    T = pd.DataFrame(rows); T.to_csv(res / "audit_by_category.csv", index=False)
    w = T.pivot(index="category", columns="dataset", values="name_precision")
    bad = sorted(w.index[(w < a.rule).any(axis=1)])
    (res / "excluded_categories.txt").write_text("".join(f"{c}\n" for c in bad))
    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False)); print(pd.DataFrame(ag).round(3).to_string(index=False))
    print(f"\nrule: name precision < {a.rule} in either dataset -> exclude {len(bad)}: {', '.join(bad)}")


if __name__ == "__main__":
    main()
