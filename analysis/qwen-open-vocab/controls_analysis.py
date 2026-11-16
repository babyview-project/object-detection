#!/usr/bin/env python3
"""Context, geometry and crop-quality controls on the Qwen open Tier 1 categories (the comparison-datasets notebook
11-16 controls, rerun for the open-vocabulary main analysis).

Sections (each writes results/controls[_renamed]/<section>_*.csv):
  cooc     co-occurrence control (nb 12): same-domain coefficient with Jaccard + shared-context covariates
  geom     crop geometry BabyView vs VEDB (nb 15): AUC within levels, geometry matching, noise ceilings, equivalence
  size     short-side floors on BabyView and VEDB (nb 16 analysis 1); LVIS / THINGS / ImageNet unchanged
  count    K exemplars per category in every dataset (nb 16 analysis 2): BabyView, VEDB, THINGS
  perm     BabyView-THINGS RDM correlation: rows-only label shuffle vs rows-and-columns (Mantel) null
  things   BabyView vs THINGS (WordNet mapping), crossed children x images x categories bootstrap; per child
  prev     category frequency agreement, BabyView vs VEDB (frame prevalence), ImageNet and LVIS (exemplar counts)

Conventions as crossdataset_analysis.ipynb: category vector = mean of L2-normalized box vectors; every dataset z-scored
across the categories compared; cosine RDMs; AUC = P(between-domain distance > within-domain distance) over category
pairs (primary, scale-free), raw delta and delta / SD beside it; CIs and differences from a within-domain category
bootstrap with shared draws (self-pairs dropped). Domains: categories.csv (Qwen hierarchy + hand overrides).
Inputs: out/private/open*/ (build_open_means.py), out/crossdataset*/tables_*.npz (build_crossdataset_tables.py),
out/private/controls*/ (build_controls_data.py), out/private/bv_vedb*/sample_*.npz (build_bv_vedb_data.py).

  QOV_LABELS=renamed nice -n 19 python controls_analysis.py [--sections cooc geom ...] [--quick]

QOV_CATS=core: every embedding comparison uses the cross-dataset core (data/included_categories_qwen_core.txt: the 183
categories present in all five datasets) or, for real-object comparisons, its 164 categories with real-object
exemplars in BabyView and VEDB (included_categories_qwen_core_real.txt); sections with extra requirements (a geometry cell) use the core categories that meet them. All-formats and real-only datasets are compared
separately so all-formats comparisons keep the full core. Frequency agreement (prev) is not restricted. Results go to
results/controls[_renamed]_core/.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import variant as V

HERE = Path(__file__).resolve().parent
OPEN, OPEN_VEDB = V.out_dir(HERE / "out" / "private" / "open"), V.out_dir(HERE / "out" / "private" / "open_vedb")
XD = V.out_dir(HERE / "out" / "crossdataset")
CTRL = V.out_dir(HERE / "out" / "private" / "controls")
SAMPLE = V.out_dir(HERE / "out" / "private" / "bv_vedb")
THINGS_IMG = Path("/data2/dataset/babyview/related_data/things/embeddings")
CORE_MODE = os.environ.get("QOV_CATS", "tier1")
RES = V.out_dir(HERE / "results" / "controls")
if CORE_MODE == "core":
    RES = RES.with_name(RES.name + "_core")
    _data = HERE.parents[1] / "data"
    CORE = {l.strip() for l in open(_data / "included_categories_qwen_core.txt") if l.strip()}
    CORE_REAL = {l.strip() for l in open(_data / "included_categories_qwen_core_real.txt") if l.strip()}
RES.mkdir(parents=True, exist_ok=True)


def restrict(cs, real=False):
    """Category list -> the core (core_real for real-object comparisons) when QOV_CATS=core; unchanged otherwise."""
    if CORE_MODE != "core":
        return list(cs)
    keep = CORE_REAL if real else CORE
    return [c for c in cs if c in keep]


def restrict_idx(ci, real=False):
    keep = set(restrict([CATLIST[i] for i in ci], real))
    return np.array([i for i in ci if CATLIST[i] in keep], dtype=int)


def is_real(dss):
    return any(str(d).endswith("_real") for d in dss)
MODELS = ("clip", "dinov3")
CATS = pd.read_csv(OPEN / "categories.csv", index_col=0)
DOM = CATS.domain.to_dict()
CATLIST = CATS.index.tolist()
N_BOOT = 1000
N_BOOT_MATCH = 10000    # geometry matching (Response Figure R1e): its tests are drawn, so p can resolve below .001


# ---------------------------------------------------------------- shared helpers
def zrows(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-10)


def cosd(A, B=None):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = A if B is None else B / np.linalg.norm(B, axis=1, keepdims=True)
    return 1 - A @ B.T


def sem_of(cs):
    return np.array([DOM.get(c, "other") for c in cs])


def pairs(n, idx=None):
    idx = np.arange(n) if idx is None else idx
    i, j = np.triu_indices(len(idx), 1)
    a, b = idx[i], idx[j]
    ok = a != b
    return a[ok], b[ok]


def measures(D, sem, idx=None):
    """(AUC, raw delta, delta / SD) for RDM D; idx = bootstrap draw of category indices."""
    a, b = pairs(len(sem), idx)
    d = D[a, b]; same = sem[a] == sem[b]
    w, bt = d[same], d[~same]
    r = stats.rankdata(np.concatenate([w, bt]))
    auc = (r[len(w):].sum() - len(bt) * (len(bt) + 1) / 2) / (len(w) * len(bt))
    dl = bt.mean() - w.mean()
    return auc, dl, dl / d.std()


def draws(sem, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    by = [np.flatnonzero(sem == d) for d in sorted(set(sem))]
    return [np.concatenate([rng.choice(v, len(v)) for v in by]) for _ in range(n)]


def ci_p(x):
    """95% percentile CI and two-sided bootstrap p, floored at the resolution of the draws (2 / n)."""
    x = np.asarray(x)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return lo, hi, float(min(1, max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / len(x))))


def compare(R: dict, cs, tag: dict, n_boot=N_BOOT, seed=0, return_boot=False):
    """AUC / delta / delta-SD per dataset with bootstrap CIs and SEs, and every pairwise difference. R: name -> RDM on cs."""
    sem = sem_of(cs)
    names = list(R)
    obs = {k: measures(R[k], sem) for k in names}
    B = np.array([[measures(R[k], sem, ix) for k in names] for ix in draws(sem, n_boot, seed)])  # draw x ds x 3
    rows, diffs = [], []
    for i, k in enumerate(names):
        lo, hi = np.percentile(B[:, i, 0], [2.5, 97.5])
        rows.append({**tag, "dataset": k, "n_categories": len(cs), "auc": obs[k][0], "auc_ci_lo": lo, "auc_ci_hi": hi,
                     "delta": obs[k][1], "delta_sd": obs[k][2], "auc_se": B[:, i, 0].std(ddof=1),
                     "delta_se": B[:, i, 1].std(ddof=1), "delta_sd_se": B[:, i, 2].std(ddof=1)})
    for i, j in combinations(range(len(names)), 2):
        for q, metric in enumerate(("auc", "delta", "delta_sd")):
            lo, hi, p = ci_p(B[:, i, q] - B[:, j, q])
            diffs.append({**tag, "pair": f"{names[i]} - {names[j]}", "metric": metric, "n_categories": len(cs),
                          "diff": obs[names[i]][q] - obs[names[j]][q], "ci_lo": lo, "ci_hi": hi, "p": p})
    return (rows, diffs, B, obs) if return_boot else (rows, diffs)


def tables(m):
    z = np.load(XD / f"tables_{m}.npz", allow_pickle=True)
    out = {}
    for k in sorted({f.split("__")[0] for f in z.files}):
        out[k] = pd.DataFrame(z[f"{k}__means"], index=list(z[f"{k}__categories"]))
        out[k].attrs["n"] = pd.Series(z[f"{k}__counts"], index=out[k].index)
    return out


def unit_sums(m, ds, part="all"):
    """Per-unit sums: (units, S [unit x cat x dim], N [unit x cat]) for BabyView children / VEDB sessions."""
    z = np.load((OPEN if ds == "bv" else OPEN_VEDB) / f"sums_{m}.npz", allow_pickle=True)
    cats = list(z["categories"])
    order = [cats.index(c) for c in CATLIST if c in cats]
    assert len(order) == len(CATLIST) or ds == "vedb"
    S = np.zeros((len(z["subjects"]), len(CATLIST), z["sum_all"].shape[2]), np.float64)
    N = np.zeros((len(z["subjects"]), len(CATLIST)))
    tgt = [CATLIST.index(c) for c in np.array(cats)[order]]
    S[:, tgt] = z[f"sum_{part}"][:, order]; N[:, tgt] = z[f"n_{part}"][:, order]
    return np.array(z["subjects"]), S, N


def things_images(m):
    """THINGS per-image vectors (L2-normalized) with their Tier 1 category (mapping/map_things.csv)."""
    z = np.load(THINGS_IMG / f"things_images_{m}.npz", allow_pickle=True)
    mp = pd.read_csv(HERE / "mapping" / "map_things.csv").set_index("id").category.to_dict()
    cat = np.array([mp.get(str(c), "") for c in z["concept"]])
    keep = np.isin(cat, CATLIST)
    X = z["emb"][keep].astype(np.float64)
    return np.array([CATLIST.index(c) for c in cat[keep]]), X / np.linalg.norm(X, axis=1, keepdims=True)


def keyed(path, m):
    z = np.load(path, allow_pickle=True)
    return dict(unit=z["unit"], cat=z["cat"], real=z["real"], code=z["code"], n=z["n"].astype(float), X=z[f"sum_{m}"].astype(np.float64))


def means_from(d, sel, n_cats=len(CATLIST)):
    X = np.zeros((n_cats, d["X"].shape[1])); n = np.zeros(n_cats)
    np.add.at(X, d["cat"][sel], d["X"][sel]); np.add.at(n, d["cat"][sel], d["n"][sel])
    return X / np.maximum(n, 1)[:, None], n


def save(rows, name):
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- cooc (nb 12)
def rank_z(x):
    r = stats.rankdata(x); return (r - r.mean()) / r.std()


def ols_coef(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def covariates(ds, cs):
    z = np.load(CTRL / f"cooc_{ds}.npz", allow_pickle=True)
    C = z["C"].astype(float); N = float(z["n_frames"]); nf = np.diag(C).copy()
    idx = np.array([CATLIST.index(c) for c in cs])
    Cs = C[np.ix_(idx, idx)]; n = nf[idx]
    jac = Cs / (n[:, None] + n[None, :] - Cs + 1e-12)
    with np.errstate(divide="ignore"):
        ppmi = np.maximum(np.log((C * N + 1e-12) / (np.outer(nf, nf) + 1e-12)), 0)
    np.fill_diagonal(ppmi, 0)
    P = ppmi[idx]                                      # rows: compared categories, columns: all Tier 1
    # cosine of the two profiles with the pair's own two columns left out (notebook 12):
    # sum_{k != a, b} P[a, k] P[b, k] = P[a] . P[b] - P[a, a] P[b, a] - P[a, b] P[b, b]
    own = P[np.arange(len(idx)), idx]                  # P[a, a] (0: diagonal zeroed)
    Pab = P[:, idx]                                    # P[a, b]
    sq = (P ** 2).sum(1)
    dot = P @ P.T - own[:, None] * Pab.T - Pab * own[None, :]
    na = sq[:, None] - own[:, None] ** 2 - Pab ** 2
    nb = sq[None, :] - own[None, :] ** 2 - Pab.T ** 2
    prof = dot / np.sqrt(np.maximum(na, 1e-24) * np.maximum(nb, 1e-24))
    return jac, prof


def sec_cooc(q):
    n_perm = 200 if q else 1000
    rows, diffs = [], []
    for m in MODELS:
        T = tables(m)
        groups = {"bv_things": ["bv", "bv_real", "things"], "bv_vedb_things": ["bv", "bv_real", "vedb", "things"],
                  "bv_lvis": ["bv", "bv_real", "lvis"], "bv_imagenet": ["bv", "bv_real", "imagenet"]}
        if CORE_MODE == "core":                       # all formats and real objects compared separately
            groups = {"bv_things": ["bv", "things"], "bv_things_real": ["bv_real", "things"],
                      "bv_vedb_things": ["bv", "vedb", "things"], "bv_vedb_things_real": ["bv_real", "vedb_real", "things"],
                      "bv_lvis": ["bv", "lvis"], "bv_lvis_real": ["bv_real", "lvis"],
                      "bv_imagenet": ["bv", "imagenet"], "bv_imagenet_real": ["bv_real", "imagenet"]}
        for name, dss in groups.items():
            cs = restrict(sorted(set.intersection(*[set(T[d].index) for d in dss])), is_real(dss))
            sem = sem_of(cs); a, b = pairs(len(cs)); same = (sem[a] == sem[b]).astype(float)
            for cov_ds in (["bv", "vedb"] if any(d.startswith("vedb") for d in dss) else ["bv"]):
                jac, prof = covariates(cov_ds, cs)
                xj, xp = rank_z(jac[a, b]), rank_z(prof[a, b])
                R = {d: cosd(zrows(T[d].loc[cs].values.astype(float))) for d in dss}
                coef = {}
                for d in dss:
                    y = rank_z(R[d][a, b]); xs = (same - same.mean()) / same.std()
                    alone, ctrl = -ols_coef(y, [xs])[0], -ols_coef(y, [xs, xj, xp])[0]
                    rng = np.random.default_rng(12)
                    null = []
                    for _ in range(n_perm):
                        sp = rng.permutation(sem); s_ = (sp[a] == sp[b]).astype(float); s_ = (s_ - s_.mean()) / s_.std()
                        null.append(-ols_coef(y, [s_, xj, xp])[0])
                    coef[d] = ctrl
                    rows.append({"model": m, "comparison": name, "covariates_from": cov_ds, "dataset": d, "n_categories": len(cs),
                                 "coef_alone": alone, "coef_controlled": ctrl, "retained": ctrl / alone,
                                 "p_perm": (1 + np.sum(np.array(null) >= ctrl)) / (1 + n_perm),
                                 "auc": measures(R[d], sem)[0]})
                # differences between datasets under the same controls: within-domain category bootstrap
                B = []
                for ix in draws(sem, 300 if q else N_BOOT, seed=5):
                    a2, b2 = pairs(len(cs), ix); s2 = (sem[a2] == sem[b2]).astype(float)
                    if s2.std() == 0: continue
                    xs2 = (s2 - s2.mean()) / s2.std(); xj2, xp2 = rank_z(jac[a2, b2]), rank_z(prof[a2, b2])
                    B.append([-ols_coef(rank_z(R[d][a2, b2]), [xs2, xj2, xp2])[0] for d in dss])
                B = np.array(B)
                for i, j in combinations(range(len(dss)), 2):
                    lo, hi, p = ci_p(B[:, i] - B[:, j])
                    diffs.append({"model": m, "comparison": name, "covariates_from": cov_ds, "pair": f"{dss[i]} - {dss[j]}",
                                  "diff_controlled": coef[dss[i]] - coef[dss[j]], "ci_lo": lo, "ci_hi": hi, "p": p})
                print(f"cooc {m} {name} ({cov_ds} co-occurrence): {len(cs)} categories", flush=True)
    r = save(rows, "cooc_coefficients"); d = save(diffs, "cooc_differences")
    print(r.round(3).to_string()); print(d.round(3).to_string())


# ---------------------------------------------------------------- geom (nb 15)
def sec_geom(q):
    meta = json.load(open(CTRL / "meta.json"))
    MIN_N, MIN_LEVEL, MIN_CELL, MIN_MATCHED, EQ = 20, 10, 3, 20, 0.04
    n_boot = 300 if q else N_BOOT
    rows, diffs, desc = [], [], []
    for m in MODELS:
        G = {ds: keyed(CTRL / f"geo_{ds}.npz", m) for ds in ("bv", "vedb")}
        for part in ("all", "real"):
            sel = {ds: (np.ones(len(g["n"]), bool) if part == "all" else g["real"] == 1) for ds, g in G.items()}
            # cell x category sums per dataset
            CS, CN = {}, {}
            for ds, g in G.items():
                s = sel[ds]
                X = np.zeros((36, len(CATLIST), g["X"].shape[1])); n = np.zeros((36, len(CATLIST)))
                np.add.at(X, (g["code"][s], g["cat"][s]), g["X"][s]); np.add.at(n, (g["code"][s], g["cat"][s]), g["n"][s])
                CS[ds], CN[ds] = X, n
            tot = {ds: CN[ds].sum(0) for ds in CN}
            ci = restrict_idx(np.flatnonzero((tot["bv"] >= MIN_N) & (tot["vedb"] >= MIN_N)), part == "real"); cs = [CATLIST[i] for i in ci]
            # 1. describe: share of boxes per size sextile / vertical third / truncated, per dataset and domain
            size = np.arange(36) // 6; vpos = (np.arange(36) // 2) % 3; trunc = np.arange(36) % 2
            for ds in CN:
                for dom in sorted(set(sem_of(cs))):
                    cc = ci[sem_of(cs) == dom]; w = CN[ds][:, cc].sum(1); w = w / w.sum()
                    desc.append({"model": m, "part": part, "dataset": ds, "domain": dom, "n_categories": len(cc),
                                 "mean_size_sextile": (w * size).sum(), "share_bottom_third": w[vpos == 2].sum(),
                                 "share_top_third": w[vpos == 0].sum(), "share_truncated": w[trunc == 1].sum()})
            # 2. AUC within levels
            levels = {"size small (sextiles 1-2)": size < 2, "size middle (3-4)": (size >= 2) & (size < 4),
                      "size large (5-6)": size >= 4, "top third": vpos == 0, "middle third": vpos == 1,
                      "bottom third": vpos == 2, "not truncated": trunc == 0, "truncated": trunc == 1}
            for L, cm in levels.items():
                M = {ds: (CS[ds][cm].sum(0), CN[ds][cm].sum(0)) for ds in CS}
                cl = [i for i in ci if all(M[ds][1][i] >= MIN_LEVEL for ds in M)]
                if len(cl) < 30: continue
                R = {ds: cosd(zrows(M[ds][0][cl] / M[ds][1][cl][:, None])) for ds in M}
                r, dd = compare(R, [CATLIST[i] for i in cl], {"model": m, "part": part, "analysis": "level", "level": L}, n_boot=n_boot)
                rows += r; diffs += dd
            # 3. geometry matching: common cell weights per category
            Mm, okm = {ds: np.zeros((len(ci), CS[ds].shape[2])) for ds in CS}, np.ones(len(ci), bool)
            for k, i in enumerate(ci):
                both = (CN["bv"][:, i] >= MIN_CELL) & (CN["vedb"][:, i] >= MIN_CELL)
                if CN["bv"][both, i].sum() < MIN_MATCHED or CN["vedb"][both, i].sum() < MIN_MATCHED:
                    okm[k] = False; continue
                w = (CN["bv"][both, i] / CN["bv"][both, i].sum() + CN["vedb"][both, i] / CN["vedb"][both, i].sum()) / 2
                for ds in CS:
                    Mm[ds][k] = (w[:, None] * CS[ds][both, i] / CN[ds][both, i][:, None]).sum(0)
            cm_ = [cs[k] for k in np.flatnonzero(okm)]
            BM = {}                                   # same seed and categories -> the two conditions share their draws
            for cond in ("unmatched", "matched"):
                if cond == "unmatched":
                    R = {ds: cosd(zrows(CS[ds][:, ci[okm]].sum(0) / CN[ds][:, ci[okm]].sum(0)[:, None])) for ds in CS}
                else:
                    R = {ds: cosd(zrows(Mm[ds][okm])) for ds in CS}
                r, dd, BM[cond], obs = compare(R, cm_, {"model": m, "part": part, "analysis": "matching", "level": cond},
                                               n_boot=N_BOOT_MATCH if not q else n_boot, return_boot=True)
                BM[cond + "_obs"] = obs
                a, b = pairs(len(cm_))
                rho = stats.spearmanr(R["bv"][a, b], R["vedb"][a, b]).statistic
                for x in r: x["bv_vedb_rho"] = rho
                rows += r; diffs += dd
            # does matching change clustering? matched - unmatched per dataset, and the change in the BabyView - VEDB gap
            names = list(BM["matched_obs"])
            for qi, metric in enumerate(("auc", "delta", "delta_sd")):
                ch = {k: BM["matched"][:, i, qi] - BM["unmatched"][:, i, qi] for i, k in enumerate(names)}
                och = {k: BM["matched_obs"][k][qi] - BM["unmatched_obs"][k][qi] for k in names}
                for lab_, x, o in [(f"{k}: matched - unmatched", ch[k], och[k]) for k in names] + \
                                  [("(bv - vedb): matched - unmatched", ch["bv"] - ch["vedb"], och["bv"] - och["vedb"])]:
                    lo, hi, p = ci_p(x)
                    diffs.append({"model": m, "part": part, "analysis": "matching", "level": "matched - unmatched", "pair": lab_,
                                  "metric": metric, "n_categories": len(cm_), "diff": o, "ci_lo": lo, "ci_hi": hi, "p": p})
            print(f"geom {m} {part}: {len(ci)} categories, {len(cm_)} matched", flush=True)
        # 4. noise ceilings and 5. equivalence (all formats and real), on per-unit sums
        for part in ("all", "real"):
            U = {ds: unit_sums(m, ds, part) for ds in ("bv", "vedb")}
            N0 = {ds: U[ds][2].sum(0) for ds in U}
            ci = restrict_idx(np.flatnonzero((N0["bv"] >= MIN_N) & (N0["vedb"] >= MIN_N)), part == "real"); cs = [CATLIST[i] for i in ci]; sem = sem_of(cs)
            a, b = pairs(len(cs))
            full = {ds: cosd(zrows(U[ds][1].sum(0)[ci] / N0[ds][ci][:, None])) for ds in U}
            rel = {}
            for ds in U:
                _, S, N = U[ds]; rs = []
                for k in range(10 if q else 20):
                    rng = np.random.default_rng(1000 + k); inA = np.zeros(len(S), bool); inA[rng.permutation(len(S))[: len(S) // 2]] = True
                    hA = S[inA].sum(0)[ci] / np.maximum(N[inA].sum(0)[ci], 1)[:, None]; hB = S[~inA].sum(0)[ci] / np.maximum(N[~inA].sum(0)[ci], 1)[:, None]
                    okh = (N[inA].sum(0)[ci] >= 5) & (N[~inA].sum(0)[ci] >= 5)
                    ah, bh = pairs(okh.sum())
                    RA, RB = cosd(zrows(hA[okh])), cosd(zrows(hB[okh]))
                    r_ = stats.spearmanr(RA[ah, bh], RB[ah, bh]).statistic; rs.append(2 * r_ / (1 + r_))
                rel[ds] = float(np.mean(rs))
            rho = stats.spearmanr(full["bv"][a, b], full["vedb"][a, b]).statistic
            rows.append({"model": m, "part": part, "analysis": "noise_ceiling", "n_categories": len(cs), "bv_vedb_rho": rho,
                         "rel_bv": rel["bv"], "rel_vedb": rel["vedb"], "ceiling": np.sqrt(rel["bv"] * rel["vedb"]),
                         "rho_over_ceiling": rho / np.sqrt(rel["bv"] * rel["vedb"])})
            # equivalence: crossed unit x category bootstrap of AUC(BV) - AUC(VEDB)
            rng = np.random.default_rng(15); dB = []
            for ix in draws(sem, n_boot // 2 if q else N_BOOT, seed=15):
                v = []
                for ds in U:
                    _, S, N = U[ds]; uu = rng.integers(0, len(S), len(S))
                    Sx, Nx = S[uu].sum(0)[ci], N[uu].sum(0)[ci]
                    v.append(measures(cosd(zrows(Sx / np.maximum(Nx, 1)[:, None])), sem, ix)[0])
                dB.append(v[0] - v[1])
            dB = np.array(dB); obs = measures(full["bv"], sem)[0] - measures(full["vedb"], sem)[0]
            lo90, hi90 = np.percentile(dB, [5, 95]); lo95, hi95 = np.percentile(dB, [2.5, 97.5])
            rows.append({"model": m, "part": part, "analysis": "equivalence", "n_categories": len(cs), "diff_auc_bv_minus_vedb": obs,
                         "ci90_lo": lo90, "ci90_hi": hi90, "ci95_lo": lo95, "ci95_hi": hi95, "bound": EQ,
                         "equivalent": bool(lo90 > -EQ and hi90 < EQ)})
            print(f"geom {m} {part}: noise ceiling + equivalence on {len(cs)} categories", flush=True)
    r = save(rows, "geom_summary"); save(diffs, "geom_differences"); save(desc, "geom_describe")
    print(r[r.analysis.isin(["noise_ceiling", "equivalence"])].dropna(axis=1, how="all").round(3).to_string())
    print(r[r.analysis == "matching"][["model", "part", "level", "dataset", "n_categories", "auc", "bv_vedb_rho"]].round(3).to_string())


# ---------------------------------------------------------------- size (nb 16, analysis 1)
def sec_size(q):
    MIN_N = 10
    rows, diffs = [], []
    for m in MODELS:
        T = tables(m)
        S = {ds: keyed(CTRL / f"side_{ds}.npz", m) for ds in ("bv", "vedb")}
        for comp in (["things"], ["lvis"], ["imagenet"]):
            for floor, lo in (("none", 0), ("ge32", 1), ("ge64", 2)):
                M = {}
                for ds, d in S.items():
                    for part in ("all", "real"):
                        sel = (d["code"] >= lo) & ((d["real"] == 1) if part == "real" else True)
                        M[ds if part == "all" else f"{ds}_real"] = means_from(d, sel)
                key_sets = [list(M)] if CORE_MODE != "core" else [["bv", "vedb"], ["bv_real", "vedb_real"]]
                for keys in key_sets:
                    cs = restrict([c for i, c in enumerate(CATLIST) if all(M[k][1][i] >= MIN_N for k in keys) and c in T[comp[0]].index],
                                  is_real(keys))
                    ci = [CATLIST.index(c) for c in cs]
                    R = {k: cosd(zrows(M[k][0][ci])) for k in keys}
                    R[comp[0]] = cosd(zrows(T[comp[0]].loc[cs].values.astype(float)))
                    r, dd = compare(R, cs, {"model": m, "versus": comp[0], "floor": floor,
                                            "part": "real" if is_real(keys) else "all"}, n_boot=300 if q else N_BOOT)
                    rows += r; diffs += dd
            print(f"size {m} vs {comp[0]} done", flush=True)
    r = save(rows, "size_summary"); save(diffs, "size_differences")
    print(r.pivot_table(index=["model", "versus", "floor", "n_categories"], columns="dataset", values="auc").round(3).to_string())


# ---------------------------------------------------------------- count (nb 16, analysis 2)
def sec_count(q):
    K, N_DRAW = 12, (100 if q else 500)
    rows, diffs = [], []
    for m in MODELS:
        z = np.load(SAMPLE / f"sample_{m}.npz", allow_pickle=True)
        X = z["emb"].astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
        ex = {}
        for ds in ("bv", "vedb"):
            s = z["dataset"] == ds
            cat = np.array([CATLIST.index(c) if c in DOM else -1 for c in z["category"][s]])
            ex[ds] = (cat[cat >= 0], X[s][cat >= 0])
        ex["things"] = things_images(m)
        cnt = {ds: np.bincount(c, minlength=len(CATLIST)) for ds, (c, _) in ex.items()}
        ci = restrict_idx(np.flatnonzero(np.logical_and.reduce([cnt[ds] >= K for ds in cnt]))); cs = [CATLIST[i] for i in ci]; sem = sem_of(cs)
        byc = {ds: {i: np.flatnonzero(ex[ds][0] == i) for i in ci} for ds in ex}
        rng = np.random.default_rng(16); A, Bd = [], []
        bdraws = draws(sem, N_DRAW, seed=16)
        for t in range(N_DRAW):
            R = {}
            for ds in ex:
                Mx = np.array([ex[ds][1][rng.choice(byc[ds][i], K, replace=False)].mean(0) for i in ci])
                R[ds] = cosd(zrows(Mx))
            A.append([measures(R[ds], sem)[0] for ds in ex])
            Bd.append([measures(R[ds], sem, bdraws[t])[0] for ds in ex])
        A, Bd = np.array(A), np.array(Bd)
        names = list(ex)
        for i, ds in enumerate(names):
            rows.append({"model": m, "K": K, "dataset": ds, "n_categories": len(cs), "auc_mean": A[:, i].mean(),
                         "draw_lo": np.percentile(A[:, i], 2.5), "draw_hi": np.percentile(A[:, i], 97.5)})
        for i, j in combinations(range(len(names)), 2):
            lo, hi, p = ci_p(Bd[:, i] - Bd[:, j])
            diffs.append({"model": m, "K": K, "pair": f"{names[i]} - {names[j]}", "diff_auc": (A[:, i] - A[:, j]).mean(),
                          "ci_lo": lo, "ci_hi": hi, "p": p})
        print(f"count {m}: {len(cs)} categories", flush=True)
    print(save(rows, "count_summary").round(3).to_string()); print(save(diffs, "count_differences").round(3).to_string())


# ---------------------------------------------------------------- perm
def sec_perm(q):
    n_perm = 1000 if q else 5000
    rows = []
    for m in MODELS:
        T = tables(m)
        for other in ("things", "vedb", "lvis", "imagenet"):
            cs = restrict(sorted(set(T["bv"].index) & set(T[other].index)))
            A, Bm = cosd(zrows(T["bv"].loc[cs].values.astype(float))), cosd(zrows(T[other].loc[cs].values.astype(float)))
            a, b = pairs(len(cs)); obs = stats.spearmanr(A[a, b], Bm[a, b]).statistic
            rng = np.random.default_rng(3); rows_only, both = [], []
            rB = stats.rankdata(Bm[a, b])
            for _ in range(n_perm):
                p = rng.permutation(len(cs))
                both.append(stats.spearmanr(A[a, b], Bm[p][:, p][a, b]).statistic)
                rows_only.append(stats.spearmanr(A[a, b], Bm[p][a, b]).statistic)   # rows shuffled, columns not
            for nm, nl in (("rows only", rows_only), ("rows and columns (Mantel)", both)):
                nl = np.array(nl)
                rows.append({"model": m, "versus": other, "n_categories": len(cs), "rho": obs, "null": nm,
                             "null_mean": nl.mean(), "null_sd": nl.std(), "null_95": np.percentile(nl, 95),
                             "p": (1 + (nl >= obs).sum()) / (1 + n_perm)})
        print(f"perm {m} done", flush=True)
    print(save(rows, "perm_rdm_null").round(4).to_string())


# ---------------------------------------------------------------- prev (comparison-datasets nb 10)
def sec_prev(q):
    """Category frequency agreement: BabyView / VEDB frame prevalence vs ImageNet images and LVIS boxes per category."""
    def frame_prev(root):
        fbc = pd.read_csv(root / "frames_by_child.csv")
        tot = pd.read_csv(root / "frames_total.csv").frames.sum()
        return fbc.groupby("cat").frames_all.sum() / tot
    T = tables("clip")
    P = {"bv": frame_prev(OPEN), "vedb": frame_prev(OPEN_VEDB),
         "imagenet": T["imagenet"].attrs["n"].astype(float), "lvis": T["lvis"].attrs["n"].astype(float)}
    rows = []
    rng = np.random.default_rng(10)
    for other in ("vedb", "imagenet", "lvis"):
        cs = sorted(set(P["bv"][P["bv"] > 0].index) & set(P[other][P[other] > 0].index))
        x, y = P["bv"].loc[cs].to_numpy(), P[other].loc[cs].to_numpy()
        rho, p = stats.spearmanr(x, y)
        bs = [stats.spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(cs), len(cs)) for _ in range(2000))]
        rows.append({"versus": other, "n_categories": len(cs), "spearman": rho, "p": p,
                     "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                     "measure": "frames with the category / all frames" if other == "vedb" else "exemplars per category"})
    print(save(rows, "prevalence_spearman").round(3).to_string())


# ---------------------------------------------------------------- things (BabyView vs THINGS, crossed bootstrap)
def sec_things(q):
    """BabyView - THINGS domain clustering on the WordNet-mapped categories, with a crossed bootstrap that resamples
    children (BabyView), images within each category (THINGS) and categories within domains, so all three sources of
    sampling error enter the interval; plus each child's own AUC against THINGS on the child's categories."""
    n_boot = 200 if q else N_BOOT
    rows, kids = [], []
    for m in MODELS:
        tc, tX = things_images(m)
        by_cat = {i: np.flatnonzero(tc == i) for i in np.unique(tc)}
        for part in ("all", "real"):
            _, S, N = unit_sums(m, "bv", part)
            Nt = np.bincount(tc, minlength=len(CATLIST))
            ci = restrict_idx(np.flatnonzero((N.sum(0) >= 30) & (Nt > 0)), part == "real"); cs = [CATLIST[i] for i in ci]; sem = sem_of(cs)
            Mt = np.array([tX[by_cat[i]].mean(0) for i in ci])
            Rb = cosd(zrows(S.sum(0)[ci] / N.sum(0)[ci][:, None])); Rt = cosd(zrows(Mt))
            obs = [measures(Rb, sem), measures(Rt, sem)]
            rng = np.random.default_rng(21); B = []
            for ix in draws(sem, n_boot, seed=21):
                uu = rng.integers(0, len(S), len(S))
                Sx, Nx = S[uu].sum(0)[ci], N[uu].sum(0)[ci]
                Mtx = np.array([tX[rng.choice(by_cat[i], len(by_cat[i]))].mean(0) for i in ci])
                B.append([measures(cosd(zrows(Sx / np.maximum(Nx, 1)[:, None])), sem, ix),
                          measures(cosd(zrows(Mtx)), sem, ix)])
            B = np.array(B)                                    # draw x (bv, things) x (auc, delta, delta_sd)
            for q_, metric in enumerate(("auc", "delta", "delta_sd")):
                lo, hi, p = ci_p(B[:, 0, q_] - B[:, 1, q_])
                rows.append({"model": m, "part": part, "metric": metric, "n_categories": len(cs), "bv": obs[0][q_],
                             "things": obs[1][q_], "diff": obs[0][q_] - obs[1][q_], "ci_lo": lo, "ci_hi": hi, "p": p,
                             "bootstrap": "children x THINGS images x categories"})
            if part == "all":                                  # each child on its own categories (>= 20 boxes)
                units, S1, N1 = unit_sums(m, "bv", "all")
                ftot = pd.read_csv(OPEN / "frames_total.csv").set_index("subject").frames
                for k, u in enumerate(units):
                    if ftot.get(u, 0) < 10_000: continue           # children analysed individually (open notebook 4b)
                    cc = restrict_idx(np.flatnonzero((N1[k] >= 20) & (Nt > 0))); s_ = sem_of([CATLIST[i] for i in cc])
                    a_c = measures(cosd(zrows(S1[k][cc] / N1[k][cc][:, None])), s_)
                    a_t = measures(cosd(zrows(np.array([tX[by_cat[i]].mean(0) for i in cc]))), s_)
                    kids.append({"model": m, "subject": u, "boxes": int(N1[k].sum()), "n_categories": len(cc),
                                 "auc_child": a_c[0], "auc_things": a_t[0], "delta_child": a_c[1], "delta_things": a_t[1]})
            print(f"things {m} {part}: {len(cs)} categories", flush=True)
    r = save(rows, "things_crossed_bootstrap"); k = save(kids, "things_per_child")
    print(r.round(4).to_string())
    k["auc_lower_than_things"] = k.auc_child < k.auc_things; k["delta_higher_than_things"] = k.delta_child > k.delta_things
    print(k.groupby("model")[["auc_lower_than_things", "delta_higher_than_things"]].agg(["sum", "size"]))


SECTIONS = {"cooc": sec_cooc, "geom": sec_geom, "size": sec_size, "count": sec_count, "perm": sec_perm, "prev": sec_prev,
            "things": sec_things}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    ap.add_argument("--quick", action="store_true", help="fewer splits / draws, for checking")
    a = ap.parse_args()
    print("labels:", V.LABELS, "->", RES, flush=True)
    for s in a.sections:
        SECTIONS[s](a.quick)
