#!/usr/bin/env python3
"""Merge the RAs' exported answers (web task) into human labels, and report agreement.

Input: every objcheck_<rater>_<bundle>_<time>.csv in --responses (default
out/private/inhouse/responses/). Each export holds all of a rater's answers so far, so
only the newest file per rater is used. Only submitted steps count.

Outputs (out/private/, git-ignored)
  human_labels.csv     item_id, human_crop, human_context (1 = the named object, 0 = not),
                       n_raters_crop, n_raters_context; shared items by majority, ties left
                       empty. Read by compare_clip_gemini.py (results/vs_human.csv).
  inhouse/agreement.json   per step: pairwise Cohen's kappa and Fleiss' kappa on the
                       shared items, percent agreement, and progress per rater.

Usage
  python merge_inhouse_responses.py [--responses DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
INH = OUT / "inhouse"


def cohen(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fleiss(m):
    """m: items x raters, values 0/1, complete rows."""
    n = m.shape[1]
    ones = m.sum(1)
    counts = np.stack([n - ones, ones], 1)
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", type=Path, default=INH / "responses")
    args = ap.parse_args()
    files = sorted(args.responses.glob("objcheck_*.csv"))
    if not files:
        raise SystemExit(f"no objcheck_*.csv in {args.responses}")
    newest = {}
    for f in files:
        d = pd.read_csv(f, dtype=str)
        r = d.rater.iloc[0]
        t = d.exported_at.iloc[0]
        if r not in newest or t > newest[r][0]:
            newest[r] = (t, f, d)
    a = pd.read_csv(INH / "assignments.csv")
    d = pd.concat([v[2] for v in newest.values()], ignore_index=True)
    d = d[d.step_submitted.notna() & (d.step_submitted != "")]
    d["valid"] = 1 - d.marked_not_object.astype(int)
    unknown = set(d.item_id) - set(a.item_id)
    assert not unknown, f"{len(unknown)} items not in assignments.csv (wrong sample?)"

    progress = {}
    for r, (t, f, _) in newest.items():
        mine = a[a.rater == r]
        got = d[d.rater == r]
        progress[r] = {"file": f.name, "exported_at": t, "assigned": int(len(mine)),
                       "crop_done": int((got.phase == "crop").sum()), "frame_done": int((got.phase == "frame").sum())}

    shared = set(a[a.shared].item_id)
    labels, agree = {}, {}
    for phase, col in (("crop", "human_crop"), ("frame", "human_context")):
        w = d[d.phase == phase].pivot_table(index="item_id", columns="rater", values="valid", aggfunc="first")
        n = w.notna().sum(1)
        mean = w.mean(1)
        lab = pd.Series(np.where(mean > 0.5, 1.0, np.where(mean < 0.5, 0.0, np.nan)), index=w.index)
        labels[col], labels[f"n_raters_{'crop' if phase == 'crop' else 'context'}"] = lab, n
        s = w[w.index.isin(shared)]
        pairs = {}
        for r1, r2 in itertools.combinations(s.columns, 2):
            both = s[[r1, r2]].dropna()
            if len(both) >= 20:
                pairs[f"{r1}-{r2}"] = {"n": int(len(both)), "kappa": round(cohen(both[r1], both[r2]), 3),
                                       "agreement": round(float((both[r1] == both[r2]).mean()), 3)}
        full = s.dropna()
        agree[phase] = {"pairwise": pairs,
                        "fleiss_kappa": round(fleiss(full.to_numpy()), 3) if len(full) >= 20 and full.shape[1] > 2 else None,
                        "n_shared_all_raters": int(len(full)),
                        "ties_left_empty": int(lab[n > 1].isna().sum())}
    out = pd.DataFrame(labels)
    out.index.name = "item_id"
    out.to_csv(OUT / "human_labels.csv")
    rep = {"progress": progress, "agreement": agree}
    (INH / "agreement.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"wrote {OUT / 'human_labels.csv'} ({out.human_crop.notna().sum()} crop, "
          f"{out.human_context.notna().sum()} context labels)")


if __name__ == "__main__":
    main()
