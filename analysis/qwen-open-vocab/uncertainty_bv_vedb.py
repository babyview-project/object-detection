#!/usr/bin/env python3
"""Standard errors for the BabyView vs VEDB figures (Figs. 2E, 8A, 8C), resampling children and VEDB sessions.

The figures show ±1 SE. Point estimates are the ones bv_vedb_analysis.ipynb reports; this script adds the SEs.

  scene statistics   per unit (child or session) mean, then mean over units; SE = SD over units / sqrt(n units)
                     (per-unit values written to out/private/ for the dot plots)
  frequency ratio    log2(BabyView / VEDB frame prevalence) per category; SE = SD of 1,000 resamples of children and
                     sessions (each dataset resampled separately; same draws and seed as the notebook, so the 95% CIs match)
  JS divergence      split-half estimate as in the notebook (mean over random halves). SE: 200 resamples of the children
  1 - RDM rho        of the bin and of the VEDB sessions; on each resample the split-half estimate is recomputed (10 random
                     splits for JS, 5 for RDM; a resampled unit and its copies always go to the same half); SE = SD over
                     resamples. Replaces the notebook's band, which was the spread of single split-half draws, not an SE.
  MMD excess         median over categories, as in the notebook; SE = SD of the notebook's 1,000 category-bootstrap medians

  QOV_LABELS=renamed QOV_CATS=core [QOV_DOMAINS=things27] python uncertainty_bv_vedb.py
Outputs (results/bv_vedb_renamed[_things27]_core/): scene_statistics_se.csv, category_frequency_ratio_se.csv,
  category_frequency_ratio_home_se.csv, js_divergence_se.csv, rdm_split_half_se.csv, appearance_mmd_by_age_se.csv;
  out/private/bv_vedb_renamed[_things27]/scene_unit_means.csv (per child / session; private).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import variant as V

HERE = Path(__file__).resolve().parent
D = V.out_dir(HERE / "out" / "private" / "bv_vedb")
CORE_MODE = os.environ.get("QOV_CATS", "tier1")
SFX = "_core" if CORE_MODE == "core" else ""
RES = V.out_dir(HERE / "results" / "bv_vedb"); RES = RES.with_name(RES.name + SFX); RES.mkdir(parents=True, exist_ok=True)
_data = HERE.parents[1] / "data"
CORE = {l.strip() for l in open(_data / "included_categories_qwen_core.txt") if l.strip()}
CORE_REAL = {l.strip() for l in open(_data / "included_categories_qwen_core_real.txt") if l.strip()}
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
BINS = ["<12 mo", "12-18 mo", "18-24 mo", "24+ mo"]
GROUPS = BINS + ["all ages"]


def in_core(c, real=False):
    return CORE_MODE != "core" or c in (CORE_REAL if real else CORE)


fr = pd.read_csv(D / "frames.csv.gz", dtype={"unit": str})
pres = pd.read_csv(D / "presence.csv.gz", dtype={"unit": str})
fr = fr[(fr.dataset == "vedb") | fr.age_bin.isin(BINS)]
pres = pres[(pres.dataset == "vedb") | pres.age_bin.isin(BINS)]
fr["dep_share"] = fr.n_depicted / fr.n_obj.clip(lower=1)


# ---- scene statistics -------------------------------------------------------------------------------------------------
M1 = ["n_obj", "n_distinct", "n_person", "n_hand", "n_face", "dep_share", "n_screen", "median_area", "mean_cy"]
rows, units = [], []
for scope, f0 in (("all frames", fr), ("home-scene frames", fr[fr.home])):
    for g in GROUPS + ["VEDB"]:
        f = f0[f0.dataset == "vedb"] if g == "VEDB" else f0[(f0.dataset == "bv") & ((f0.age_bin == g) | (g == "all ages"))]
        um = f.groupby("unit")[M1].mean()
        units.append(um.reset_index().assign(scope=scope, group=g))
        for k in M1:
            x = um[k].dropna()
            rows.append(dict(scope=scope, group=g, measure=k, mean=x.mean(), sd=x.std(ddof=1), se=x.std(ddof=1) / np.sqrt(len(x)),
                             units=len(x)))
pd.DataFrame(rows).to_csv(RES / "scene_statistics_se.csv", index=False)
pd.concat(units).to_csv(D / "scene_unit_means.csv", index=False)
print("scene statistics done")


# ---- category frequency ratio (same draws as bv_vedb_analysis section 2) ------------------------------------------------
def prevalence(ds, group="all ages", home=None):
    p = pres[pres.dataset == ds]; f = fr[fr.dataset == ds]
    if ds == "bv" and group != "all ages":
        p, f = p[p.age_bin == group], f[f.age_bin == group]
    if home is not None:
        p, f = p[p.home == home], f[f.home == home]
    cnt = p.groupby(["unit", "category"]).frames.sum().unstack(fill_value=0)
    tot = f.groupby("unit").size()
    return cnt.reindex(tot.index, fill_value=0), tot


def ratio_table(home=None, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(1)
    (cb, tb), (cv, tv) = prevalence("bv", home=home), prevalence("vedb", home=home)
    cats = sorted(set(cb.columns) & set(cv.columns))
    cats = [c for c in cats if cb[c].sum() >= 50 and cv[c].sum() >= 50]
    cb, cv = cb[cats], cv[cats]
    pb, pv = cb.sum() / tb.sum(), cv.sum() / tv.sum()
    PB, PV = [], []
    for _ in range(n_boot):
        ib, iv = rng.integers(0, len(tb), len(tb)), rng.integers(0, len(tv), len(tv))
        PB.append(cb.values[ib].sum(0) / tb.values[ib].sum()); PV.append(cv.values[iv].sum(0) / tv.values[iv].sum())
    with np.errstate(divide="ignore"):
        B = np.log2(np.array(PB) / np.array(PV))
    Bf = np.where(np.isfinite(B), B, np.nan)
    return pd.DataFrame({"bv_prev": pb, "vedb_prev": pv, "log2_ratio": np.log2(pb / pv), "se": np.nanstd(Bf, 0, ddof=1),
                         "lo": np.percentile(B, 2.5, 0), "hi": np.percentile(B, 97.5, 0),
                         "domain": [DOM.get(c, "other") for c in cats]}, index=cats).sort_values("log2_ratio")


ratio_table().to_csv(RES / "category_frequency_ratio_se.csv")
ratio_table(home=True).to_csv(RES / "category_frequency_ratio_home_se.csv")
print("frequency ratio done")


# ---- cluster bootstrap of the split-half divergences -----------------------------------------------------------------
def boot_units(u, rng):
    """Resample units with replacement; return the distinct units drawn and how often each was drawn."""
    u = np.asarray(u); d = rng.choice(len(u), len(u))
    k, w = np.unique(d, return_counts=True)
    return u[k], w


def split(u, w, rng):
    """Random halves of the distinct units (copies of a unit stay together); returns (units, weights) per half."""
    o = rng.permutation(len(u)); h = len(u) // 2
    return (u[o[:h]], w[o[:h]]), (u[o[h:2 * h]], w[o[h:2 * h]])


def js(p, q):
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0); b = np.where(q > 0, q * np.log2(q / m), 0)
    return .5 * a.sum() + .5 * b.sum()


def counts(ds, group, home, level):
    c, _ = prevalence(ds, group, home)
    if level == "domain":
        c = c.T.groupby(lambda x: DOM.get(x, "other")).sum().T
    return c


def js_boot(group, home, level, n_boot=200, n_split=10, rng=None):
    cb, cv = counts("bv", group, home, level), counts("vedb", "all ages", home, level)
    cols = sorted(set(cb.columns) | set(cv.columns))
    cb, cv = cb.reindex(columns=cols, fill_value=0), cv.reindex(columns=cols, fill_value=0)
    Cb, Cv = cb.values.astype(float), cv.values.astype(float)
    nb, nv = len(cb), len(cv)
    out = []
    for _ in range(n_boot):
        ub, wb = boot_units(np.arange(nb), rng); uv, wv = boot_units(np.arange(nv), rng)
        r = []
        for _ in range(n_split):
            (b1, w1), (b2, w2) = split(ub, wb, rng); (v1, x1), _v2 = split(uv, wv, rng)
            X = [w1 @ Cb[b1], w2 @ Cb[b2], x1 @ Cv[v1]]
            N = int(min(x.sum() for x in X))
            X = [rng.multivariate_hypergeometric(x.astype(np.int64), N) for x in X]
            r.append((js(X[0], X[2]), js(X[0], X[1])))
        r = np.array(r).mean(0)
        out.append((r[0], r[1], r[0] - r[1]))
    return np.array(out)


rng = np.random.default_rng(12)
rows = []
for home in (None, True):
    for level in ("category", "domain"):
        for g in GROUPS:
            o = js_boot(g, home, level, rng=rng)
            for k, name in enumerate(("BabyView vs VEDB", "within BabyView", "excess over within BabyView")):
                rows.append(dict(scope="home-scene frames" if home else "all frames", level=level, group=g, comparison=name,
                                 se=o[:, k].std(ddof=1), boot_mean=o[:, k].mean(), n_boot=len(o)))
        print("JS", home, level, "done")
J = pd.read_csv(RES / "js_divergence.csv").merge(pd.DataFrame(rows), on=["scope", "level", "group", "comparison"], how="left")
J.to_csv(RES / "js_divergence_se.csv", index=False)


def l2(X):
    X = X.astype(np.float32); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def rdm_boot(m="dinov3", real=False, n_boot=200, n_split=5, min_n=60, rng=None):
    s = np.load(D / f"sample_{m}.npz", allow_pickle=True)
    ds, ab, cat, un, fm = s["dataset"], s["age_bin"], s["category"], s["unit"], s["format"]
    ok = (fm == "real") if real else np.ones(len(ds), bool)
    cnt = pd.DataFrame({"g": np.where(ds == "vedb", "VEDB", ab), "c": cat})[ok].value_counts().unstack(0).fillna(0)
    cats = sorted(c for c in cnt.index[(cnt[BINS + ["VEDB"]] >= min_n).all(1)] if in_core(c, real))
    ci = {c: i for i, c in enumerate(cats)}
    sel = ok & np.isin(cat, cats)
    E = l2(s["emb"][sel]); ds, ab, cat, un = ds[sel], ab[sel], cat[sel], un[sel]
    k = np.array([ci[c] for c in cat])
    key = np.where(ds == "vedb", "V|" + un, "B|" + un + "|" + ab)            # sums per child x age bin, or per session
    keys, kidx = np.unique(key, return_inverse=True)
    S = np.zeros((len(keys), len(cats), E.shape[1]), np.float32); N = np.zeros((len(keys), len(cats)))
    np.add.at(S, (kidx, k), E); np.add.at(N, (kidx, k), 1)
    del E
    zr = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-10)
    rd = lambda X: squareform(pdist(zr(X), "cosine"))[np.triu_indices(len(X), 1)]

    S2 = S.reshape(len(keys), -1); del S
    vedb = {u.split("|")[1]: [i] for i, u in enumerate(keys) if u.startswith("V|")}
    rows = []
    for g in GROUPS:
        bv = {}
        for i, u in enumerate(keys):
            if u.startswith("B|") and (g == "all ages" or u.split("|")[2] == g):
                bv.setdefault(u.split("|")[1], []).append(i)

        def means(units, w, table):
            wv = np.zeros(len(keys), np.float32)
            for u, wt in zip(units, w):
                wv[table[u]] += wt
            X = (wv @ S2).reshape(len(cats), -1); n = wv @ N
            return X / np.maximum(n, 1)[:, None], n
        out = []
        for _ in range(n_boot):
            ub, wb = boot_units(sorted(bv), rng); uv, wv = boot_units(sorted(vedb), rng)
            r = []
            for _ in range(n_split):
                (b1, w1), (b2, w2) = split(ub, wb, rng); (v1, x1), (v2, x2) = split(uv, wv, rng)
                M = [means(b1, w1, bv), means(b2, w2, bv), means(v1, x1, vedb), means(v2, x2, vedb)]
                keep = np.all([n >= 5 for _, n in M], axis=0)
                if keep.sum() < 30:
                    continue
                R = [rd(X[keep]) for X, _ in M]
                r.append((stats.spearmanr(R[0], R[2]).statistic, stats.spearmanr(R[0], R[1]).statistic))
            if r:
                out.append(np.mean(r, 0))
        out = np.array(out)
        for j, name in enumerate(("BabyView vs VEDB", "within BabyView")):
            rows.append(dict(model=m, real=real, group=g, comparison=name, se=out[:, j].std(ddof=1), boot_mean=out[:, j].mean(),
                             n_boot=len(out)))
        print("RDM", g, len(out), "resamples")
    return pd.DataFrame(rows)


RR = pd.read_csv(RES / "rdm_split_half.csv")
_cdi = HERE / "results" / f"bv_vedb_renamed{SFX}" / "rdm_split_half_se.csv"     # the RDMs do not depend on the domain scheme
if V.DOMAINS != "cdi" and V.LABELS == "renamed" and _cdi.exists():
    RS = pd.read_csv(_cdi)[["model", "real", "group", "comparison", "se", "boot_mean", "n_boot"]]
else:
    RS = rdm_boot(rng=np.random.default_rng(14))
RR.merge(RS, on=["model", "real", "group", "comparison"], how="left").to_csv(RES / "rdm_split_half_se.csv", index=False)

# ---- MMD: SD of the notebook's category-bootstrap medians (same seeds) -----------------------------------------------
A = pd.read_csv(RES / "appearance_mmd.csv")
rows = []
for (m, r), a in A.groupby(["model", "real"]):
    every = set.intersection(*[set(a[a.group == g].category) for g in GROUPS])
    for g in GROUPS:
        x = a[(a.group == g) & a.category.isin(every)].set_index("category").excess
        b = [np.median(np.random.default_rng(i).choice(x.values, len(x))) for i in range(1000)]
        rows.append(dict(model=m, real=r, group=g, median_excess=x.median(), se=np.std(b, ddof=1),
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), categories=len(x)))
pd.DataFrame(rows).to_csv(RES / "appearance_mmd_by_age_se.csv", index=False)
print("done")
