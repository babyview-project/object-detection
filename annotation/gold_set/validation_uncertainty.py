#!/usr/bin/env python3
"""Per-child values and child-level bootstrap SEs for the validation scorecard (manuscript Fig. 1, a01_scorecard).

Reads out/private/ra_scores/boxes.csv (score_ra_qwen_t1.py). The pooled values are those of summary.csv / interrater.csv
(asserted). Uncertainty: 10,000 resamples of the 17 children (cluster bootstrap; every frame of a drawn child comes along),
pooled measure recomputed on each; SE = SD of the resamples, 95% percentile CI. Per child: the same measures on that child's
frames (both raters' reviews pooled, as in summary.csv).

Outputs (private: per child): out/private/ra_scores/scorecard_by_child.csv, scorecard_uncertainty.csv
  /data/j7yang/anaconda3/bin/python validation_uncertainty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_ra_qwen_t1 as S  # noqa: E402

N_BOOT = 10000


def frame_table(B: pd.DataFrame) -> pd.DataFrame:
    """One row per (rater, frame): counts the pooled measures are ratios of."""
    g = B.assign(q=B.outcome != "added", m=B.outcome == "matched", a=B.outcome == "added",
                 nk=(B.outcome == "matched") * B.name_kept_norm.fillna(0), fk=(B.outcome == "matched") * B.format_kept.fillna(0))
    return g.groupby(["rater", "frame_uid", "subject_id"])[["q", "m", "a", "nk", "fk"]].sum().reset_index()


def measures(t: pd.DataFrame) -> dict:
    q, m, a, nk, fk = (t[c].sum() for c in ("q", "m", "a", "nk", "fk"))
    p, r = m / q, m / (m + a)
    return {"precision": p, "recall": r, "name_kept": nk / m, "format_kept": fk / m, "f1": 2 * p * r / (p + r)}


def interrater(B: pd.DataFrame) -> pd.DataFrame:
    """Per shared frame: boxes both raters drew, Emma only, Mira only (as score_ra_qwen_t1.py)."""
    nm = {s: S.norm(s) for s in set(B.q_name.dropna()) | set(B.ra_name.dropna())}
    sh = B[B.shared == True]
    both = set.intersection(*[set(g.frame_uid) for _, g in sh.groupby("rater")])
    final = sh[sh.outcome != "deleted"]; cols = ["ra_x0", "ra_y0", "ra_x1", "ra_y1"]; out = []
    for uid in sorted(both):
        a, b = (final[(final.frame_uid == uid) & (final.rater == r)].rename(columns=dict(zip(cols, ["x0", "y0", "x1", "y1"])))
                .assign(name=lambda x: x.ra_name, source="manual").reset_index(drop=True) for r in ("Emma", "Mira"))
        k = {"both": 0, "a": 0, "b": 0}
        for i, j, _ in S.match(a, b, exact=False):
            k["both" if i is not None and j is not None else ("a" if j is None else "b")] += 1
        out.append({"frame_uid": uid, "subject_id": sh[sh.frame_uid == uid].subject_id.iloc[0], **k})
    return pd.DataFrame(out)


def boot(tab, fn, rng):
    kids = tab.subject_id.unique(); rows = {k: tab.index[tab.subject_id == k].values for k in kids}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(kids, len(kids))
        vals.append(fn(tab.loc[np.concatenate([rows[k] for k in pick])]))
    return pd.DataFrame(vals)


def main():
    B = pd.read_csv(S.OUT / "boxes.csv")
    bp = B.q_name.isin(S.BODY_PARTS) | B.ra_name.isin(S.BODY_PARTS)
    sets = {"all": frame_table(B), "without body parts": frame_table(B[~bp])}
    summ = pd.read_csv(S.OUT / "summary.csv").set_index("set")
    rng = np.random.default_rng(0); rows, kids = [], []
    for name, t in sets.items():
        obs = measures(t)
        for k_ in ("precision", "recall"):
            assert abs(obs[k_] - summ.loc[name, k_]) < 1e-9, (name, k_)
        bs = boot(t, measures, rng)
        for k_, v in obs.items():
            rows.append({"set": name, "measure": k_, "value": v, "se": bs[k_].std(ddof=1),
                         "ci_low": bs[k_].quantile(.025), "ci_high": bs[k_].quantile(.975), "children": t.subject_id.nunique()})
        for kid, g in t.groupby("subject_id"):
            kids.append({"set": name, "subject_id": kid, "frames": g.frame_uid.nunique(), **measures(g)})
    I = interrater(B)
    f1 = lambda t: {"interrater_f1": 2 * t.both.sum() / (2 * t.both.sum() + t.a.sum() + t.b.sum())}
    obs = f1(I)["interrater_f1"]
    ir = pd.read_csv(S.OUT / "interrater.csv").iloc[0]
    assert abs(obs - ir.box_agreement_f1) < 1e-9 and len(I) == ir.shared_frames_both_reviewed
    bs = boot(I, f1, rng)
    rows.append({"set": "shared frames", "measure": "interrater_f1", "value": obs, "se": bs.interrater_f1.std(ddof=1),
                 "ci_low": bs.interrater_f1.quantile(.025), "ci_high": bs.interrater_f1.quantile(.975),
                 "children": I.subject_id.nunique()})
    for kid, g in I.groupby("subject_id"):
        kids.append({"set": "shared frames", "subject_id": kid, "frames": len(g), **f1(g)})
    U = pd.DataFrame(rows); U.to_csv(S.OUT / "scorecard_uncertainty.csv", index=False)
    pd.DataFrame(kids).to_csv(S.OUT / "scorecard_by_child.csv", index=False)
    print(U.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
