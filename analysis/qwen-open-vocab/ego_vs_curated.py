#!/usr/bin/env python3
"""Egocentric vs curated domain clustering: every egocentric dataset (BabyView, VEDB) minus every curated one (THINGS,
ImageNet, LVIS) on the same 1,000 within-domain category-bootstrap draws as crossdataset_analysis.ipynb section 4
(core categories, renamed labels), for raw delta, delta / SD and AUC (AUC point estimate only). Two-sided bootstrap p
(floor 2/1000), Benjamini-Hochberg across the six comparisons within each model and measure. Also the mean of the two
egocentric datasets minus the mean of the three curated ones.

Output: results/crossdataset_renamed_core/ego_vs_curated.csv (overall), ego_vs_curated_by_domain.csv (per CDI domain:
delta / SD, BH across domains for the pooled contrast and across domains x pairs for the six pairwise ones)
  QOV_LABELS=renamed QOV_CATS=core python ego_vs_curated.py
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy import stats

import variant as V
from pathlib import Path

HERE = Path(__file__).resolve().parent
XD = V.out_dir(HERE / "out" / "crossdataset")
RES = V.out_dir(HERE / "results" / "crossdataset"); RES = RES.with_name(RES.name + "_core"); RES.mkdir(parents=True, exist_ok=True)
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
EGO, CUR = ["bv", "vedb"], ["things", "imagenet", "lvis"]


def zrows(X): return (X - X.mean(0)) / (X.std(0) + 1e-10)
def rdm(df, cs): return squareform(pdist(zrows(df.loc[cs].values.astype(float)), metric="cosine"))


def draws(cs, n=1000, seed=2):
    rng = np.random.default_rng(seed); sem = np.array([DOM.get(c, "other") for c in cs])
    by = {d: np.flatnonzero(sem == d) for d in sorted(set(sem))}
    return [np.concatenate([rng.choice(v, len(v)) for v in by.values()]) for _ in range(n)]


def cluster(D, cs, idx=None):
    sem = np.array([DOM.get(c, "other") for c in cs]); ix = np.arange(len(cs)) if idx is None else idx
    i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]; ok = a != b
    same = sem[a][ok] == sem[b][ok]; d = D[a[ok], b[ok]]; w, bt = d[same], d[~same]
    auc = stats.mannwhitneyu(bt, w).statistic / (len(bt) * len(w)) if idx is None else np.nan
    return bt.mean() - w.mean(), (bt.mean() - w.mean()) / d.std(), auc


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


rows = []
for m in ("dinov3", "clip"):
    z = np.load(XD / f"tables_{m}.npz", allow_pickle=True)
    T = {k: pd.DataFrame(z[f"{k}__means"], index=list(z[f"{k}__categories"])) for k in EGO + CUR}
    CORE = sorted(set.intersection(*[set(T[k].index) for k in EGO + CUR]))
    Ds = {k: rdm(T[k], CORE) for k in T}
    pt = {k: cluster(D, CORE) for k, D in Ds.items()}
    B = draws(CORE)
    boot = {k: np.array([cluster(D, CORE, ix)[:2] for ix in B]) for k, D in Ds.items()}
    pairs = [(e, c) for e in EGO for c in CUR] + [("ego", "curated")]
    for k_, meas in enumerate(("delta", "delta_sd")):
        for e, c in pairs:
            if e == "ego":
                x = np.mean([boot[k][:, k_] for k in EGO], 0) - np.mean([boot[k][:, k_] for k in CUR], 0)
                diff = np.mean([pt[k][k_] for k in EGO]) - np.mean([pt[k][k_] for k in CUR])
                auc_d = np.mean([pt[k][2] for k in EGO]) - np.mean([pt[k][2] for k in CUR])
            else:
                x = boot[e][:, k_] - boot[c][:, k_]; diff = pt[e][k_] - pt[c][k_]; auc_d = pt[e][2] - pt[c][2]
            rows.append({"model": m, "measure": meas, "pair": f"{e} - {c}", "n_categories": len(CORE), "diff": diff,
                         "ci_low": np.percentile(x, 2.5), "ci_high": np.percentile(x, 97.5),
                         "p_boot": max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / len(x)), "auc_diff": auc_d})
r = pd.DataFrame(rows)
pair6 = r.pair != "ego - curated"
r["q_bh"] = np.nan
r.loc[pair6, "q_bh"] = r[pair6].groupby(["model", "measure"]).p_boot.transform(bh)
r.loc[~pair6, "q_bh"] = r.loc[~pair6, "p_boot"]
r.to_csv(RES / "ego_vs_curated.csv", index=False)
pd.set_option("display.width", 200); print(r.round(3).to_string())


# ---------------------------------------------------------------- per domain (as notebook section 4b)
import fig_style as FS
ORDER = [d for d in FS.DOMAIN_ORDER if d not in ("people", "other")]


def domain_deltas(D, sem, doms, orig=None):
    n = len(D); valid = np.triu(np.ones((n, n), bool), 1)
    if orig is not None: valid &= orig[:, None] != orig[None, :]
    out = []
    for d in doms:
        a = sem == d; w = valid & a[:, None] & a[None, :]; b = a[:, None] & ~a[None, :]
        out.append(D[b].mean() - D[w].mean() if w.any() and b.any() else np.nan)
    return np.array(out), D[valid | valid.T].std()


rows = []
for m in ("dinov3", "clip"):
    z = np.load(XD / f"tables_{m}.npz", allow_pickle=True)
    T = {k: pd.DataFrame(z[f"{k}__means"], index=list(z[f"{k}__categories"])) for k in EGO + CUR}
    CORE = sorted(set.intersection(*[set(T[k].index) for k in EGO + CUR]))
    sem0 = np.array([DOM.get(c, "other") for c in CORE]); vc = pd.Series(sem0).value_counts()
    DOMS = [d for d in ORDER if vc.get(d, 0) >= 3]
    B = draws(CORE, 1000, seed=3)
    PT, BSD = {}, {}
    for k in EGO + CUR:
        D = rdm(T[k], CORE); dl, sd = domain_deltas(D, sem0, DOMS); PT[k] = dl / sd
        bo = [domain_deltas(D[np.ix_(ix, ix)], sem0[ix], DOMS, ix) for ix in B]
        BSD[k] = np.array([b[0] for b in bo]) / np.array([b[1] for b in bo])[:, None]
    for i, dm in enumerate(DOMS):
        for e, c in [(e, c) for e in EGO for c in CUR] + [("ego", "curated")]:
            if e == "ego":
                x = np.mean([BSD[k][:, i] for k in EGO], 0) - np.mean([BSD[k][:, i] for k in CUR], 0)
                diff = np.mean([PT[k][i] for k in EGO]) - np.mean([PT[k][i] for k in CUR])
            else:
                x = BSD[e][:, i] - BSD[c][:, i]; diff = PT[e][i] - PT[c][i]
            x = x[~np.isnan(x)]
            rows.append({"model": m, "domain": dm, "n_categories": int(vc[dm]), "pair": f"{e} - {c}", "diff": diff,
                         "ci_low": np.percentile(x, 2.5), "ci_high": np.percentile(x, 97.5),
                         "p_boot": max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / len(x))})
r2 = pd.DataFrame(rows)
pooled = r2.pair == "ego - curated"
r2["q_bh"] = np.nan
r2.loc[pooled, "q_bh"] = r2[pooled].groupby("model").p_boot.transform(bh)          # across domains
r2.loc[~pooled, "q_bh"] = r2[~pooled].groupby("model").p_boot.transform(bh)        # across domains x six pairs
r2.to_csv(RES / "ego_vs_curated_by_domain.csv", index=False)
print(r2[pooled].round(3).to_string())
