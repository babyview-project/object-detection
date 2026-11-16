#!/usr/bin/env python3
"""Write crossdataset_analysis.ipynb: comparison-datasets notebooks 07-10 on the Tier 1 open categories.

Datasets: BabyView (Qwen boxes; all formats and real only), VEDB (Qwen boxes, same settings; when present), THINGS,
LVIS (human boxes), ImageNet (whole images). Tables from build_crossdataset_tables.py; mapping from
build_crossdataset_mapping.py (provisional: Qwen-chosen WordNet senses, RA review pending). Methods as
comparison-datasets: every dataset z-scored within the categories compared, cosine RDMs, Spearman on the upper
triangle with a category bootstrap (shared draws), domain clustering as raw delta (between - within), delta / SD and AUC,
pair differences with bootstrap CIs. Format prevalence: Qwen's labels for BabyView and VEDB, the BabyView-trained
format probe (format_classifier.py) elsewhere.

  python build_crossdataset_nb.py
  jupyter nbconvert --to notebook --execute crossdataset_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \\
      --ExecutePreprocessor.timeout=7200 --output-dir out/private --output crossdataset_analysis_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# BabyView vs VEDB vs THINGS vs LVIS vs ImageNet on Qwen's open categories (Tier 1)

Replicates comparison-datasets notebooks 07-10 (pairwise RDM correlations, RDMs clustered within domain, dataset x
dataset correlation matrix, domain cluster strength) on the 828 Tier 1 categories instead of valid129, plus visual-format
prevalence across datasets. Each dataset's categories come from the cross-dataset mapping (WordNet senses chosen by Qwen
or hand-checked; **provisional until RA review**). Every comparison uses only the categories the datasets compared share,
each dataset z-scored within them. Datasets: `bv` (all formats), `bv_real`, `vedb`, `vedb_real` (when present),
`things`, `lvis`, `imagenet`."""),
code("""import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.spatial.distance import pdist, squareform
HERE = Path.cwd()
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
from make_subject_figures import COLORS, ORDER, cluster_within
import fig_style as FS                      # manuscript style: Lato, bold labels / legends, dataset colours
FS.apply()
import variant as V                      # QOV_LABELS=renamed (environment): renamed-label tables and parallel output folders
print("labels:", V.LABELS)
XD = V.out_dir(HERE / "out" / "crossdataset")
import os
CORE_MODE = os.environ.get("QOV_CATS", "tier1")     # core: every comparison on the 183-category core (164 for real objects)
SFX = "_core" if CORE_MODE == "core" else ""
FIG = V.out_dir(HERE / "figures" / "crossdataset"); FIG = FIG.with_name(FIG.name + SFX); FIG.mkdir(parents=True, exist_ok=True)
RES = V.out_dir(HERE / "results" / "crossdataset"); RES = RES.with_name(RES.name + SFX); RES.mkdir(parents=True, exist_ok=True)
print("categories:", CORE_MODE)
_data = HERE.parents[1] / "data"
CORE_REAL_LIST = [l.strip() for l in open(_data / "included_categories_qwen_core_real.txt") if l.strip()]
cats = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0)
DOM = cats.domain.to_dict()
LABEL = {"bv": "BabyView", "bv_real": "BabyView (real)", "vedb": "VEDB", "vedb_real": "VEDB (real)", "things": "THINGS",
         "lvis": "LVIS", "imagenet": "ImageNet"}
T = {}
for m in ("clip", "dinov3"):
    z = np.load(XD / f"tables_{m}.npz", allow_pickle=True)
    for k in sorted({f.split("__")[0] for f in z.files}):
        T[m, k] = pd.DataFrame(z[f"{k}__means"], index=list(z[f"{k}__categories"]))
DS = [d for d in LABEL if ("clip", d) in T]
print("datasets:", {d: len(T["clip", d]) for d in DS})

def zrows(X): return (X - X.mean(0)) / (X.std(0) + 1e-10)
def rdm(df, cs): return squareform(pdist(zrows(df.loc[cs].values.astype(float)), metric="cosine"))
def upper(D): return D[np.triu_indices(len(D), 1)]
def shared(*ds, m="clip"):
    s = set(T[m, ds[0]].index)
    for d in ds[1:]: s &= set(T[m, d].index)
    return sorted(s)
def draws(cs, n=1000, seed=0):
    rng = np.random.default_rng(seed); sem = np.array([DOM.get(c, "other") for c in cs])
    by = {d: np.flatnonzero(sem == d) for d in sorted(set(sem))}       # sorted: draws do not depend on the hash seed
    return [np.concatenate([rng.choice(v, len(v)) for v in by.values()]) for _ in range(n)]
def cluster(D, cs, idx=None):
    sem = np.array([DOM.get(c, "other") for c in cs]); ix = np.arange(len(cs)) if idx is None else idx
    i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]; ok = a != b
    same = sem[a][ok] == sem[b][ok]; d = D[a[ok], b[ok]]
    w, bt = d[same], d[~same]
    auc = stats.mannwhitneyu(bt, w).statistic / (len(bt) * len(w)) if idx is None else np.nan
    return bt.mean() - w.mean(), (bt.mean() - w.mean()) / d.std(), auc"""),

md("## 1. Coverage: categories per dataset and shared sets"),
code("""cov = pd.DataFrame({LABEL[d]: [len(T["clip", d])] for d in DS}, index=["Tier 1 categories with exemplars"])
display(cov)
core = [d for d in ("bv", "vedb", "things", "lvis", "imagenet") if d in DS]
CORE = shared(*core)
print(f"core set (in every one of {', '.join(LABEL[d] for d in core)}): {len(CORE)} categories")
print(pd.Series([DOM.get(c) for c in CORE]).value_counts().to_dict())
M = pd.DataFrame(index=[LABEL[d] for d in DS], columns=[LABEL[d] for d in DS], dtype=float)
for a in DS:
    for b in DS:
        M.loc[LABEL[a], LABEL[b]] = len(shared(a, b))
display(M.astype(int).style.set_caption("shared categories per dataset pair"))"""),

md("""## 2. Pairwise RDM correlations (notebook 07/08)
Each pair on the categories it shares; 95% CI from a category bootstrap within domains."""),
code("""rows = []
for m in ("clip", "dinov3"):
    for a, b in combinations(DS, 2):
        if CORE_MODE != "core":
            cs = shared(a, b, m=m)
        else:                                            # the 164-category real list whenever a real-object table is involved
            lst = CORE_REAL_LIST if (a.endswith("_real") or b.endswith("_real")) else CORE
            cs = [c for c in lst if c in set(T[m, a].index) and c in set(T[m, b].index)]
        if len(cs) < 20: continue
        Da, Db = rdm(T[m, a], cs), rdm(T[m, b], cs)
        rho = stats.spearmanr(upper(Da), upper(Db)).statistic
        bs = []
        for ix in draws(cs, 300, seed=1):
            i, j = np.triu_indices(len(ix), 1); ok = ix[i] != ix[j]
            bs.append(stats.spearmanr(Da[ix[i][ok], ix[j][ok]], Db[ix[i][ok], ix[j][ok]]).statistic)
        rows.append({"model": m, "a": a, "b": b, "n_categories": len(cs), "spearman": rho,
                     "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5), "se": np.std(bs, ddof=1)})
pr = pd.DataFrame(rows); pr.to_csv(RES / "pairwise_rdm_correlations.csv", index=False)
for m in ("clip", "dinov3"):
    P = pr[pr.model == m]; mat = pd.DataFrame(np.eye(len(DS)), index=DS, columns=DS)
    for r in P.itertuples(): mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.spearman
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(DS)), [LABEL[d] for d in DS], rotation=40, ha="right"); ax.set_yticks(range(len(DS)), [LABEL[d] for d in DS])
    for i in range(len(DS)):
        for j in range(len(DS)):
            if i != j: ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if mat.values[i, j] > .6 else FS.INK)
    fig.colorbar(im, ax=ax, shrink=.8, label="Spearman between RDMs" + (" (core categories)" if CORE_MODE == "core" else " (shared categories)"))
    ax.set_title(f"Dataset x dataset RDM agreement, {m.upper() if m == 'clip' else 'DINOv3'}", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"rdm_agreement_matrix_{m}.png", dpi=200, bbox_inches="tight"); plt.show()
display(pr.round(3))"""),

md("""## 3. RDMs on the core set, clustered within domain (notebook 07)
Panel titles: Spearman between each dataset's RDM and BabyView's on the same core categories (full matrix in
results/crossdataset/core_rdm_correlations_<model>.csv)."""),
code("""for m in ("clip", "dinov3"):
    order = sorted(range(len(CORE)), key=lambda k: (ORDER.index(DOM.get(CORE[k], "other")) if DOM.get(CORE[k]) in ORDER else 99, CORE[k]))
    cs = [CORE[k] for k in order]
    idx = cluster_within(cs, rdm(T[m, "bv"], cs), DOM); cs = [cs[k] for k in idx]
    R = {d: rdm(T[m, d], cs) for d in core}
    cm = pd.DataFrame([[stats.spearmanr(upper(R[a]), upper(R[b])).statistic for b in core] for a in core],
                      index=[LABEL[d] for d in core], columns=[LABEL[d] for d in core])
    cm.to_csv(RES / f"core_rdm_correlations_{m}.csv")
    display(cm.round(3).style.set_caption(f"Spearman between RDMs on the {len(cs)} core categories, {m}"))
    fig, axs = plt.subplots(1, len(core), figsize=(4.2 * len(core), 4.7))
    for ax, d in zip(np.atleast_1d(axs), core):
        D = R[d]
        ax.imshow(D, cmap="viridis", vmin=0, vmax=2, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ds = [DOM.get(c, "other") for c in cs]
        for k in range(1, len(ds)):
            if ds[k] != ds[k - 1]: ax.axhline(k - .5, color="white", lw=.6); ax.axvline(k - .5, color="white", lw=.6)
        if d == core[0]:                                # domain labels in the manuscript colours
            st = [0] + [k for k in range(1, len(ds)) if ds[k] != ds[k - 1]]; en = st[1:] + [len(ds)]
            ax.set_yticks([(a + b - 1) / 2 for a, b in zip(st, en)], [FS.DOMAIN_LABEL.get(ds[a], ds[a]) for a in st], fontsize=8)
            ax.tick_params(axis="y", length=0); FS.color_domain_ticks(ax.get_yticklabels(), [ds[a] for a in st])
        ax.set_title(LABEL[d] + ("" if d == "bv" else f"\\nSpearman with BabyView = {cm.loc[LABEL[d], LABEL['bv']]:.2f}"), fontsize=11)
    fig.suptitle(f"{len(cs)} core categories, {m.upper() if m == 'clip' else 'DINOv3'} (order: BabyView, clustered within domain)", x=.01, ha="left", y=.99)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(FIG / f"core_rdms_{m}.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 4. Domain cluster strength (notebook 09)
Per dataset on the core set: raw delta (between - within cosine distance), delta / SD (SD of all pair distances) and
AUC; 95% CIs from shared category-bootstrap draws; BabyView minus each dataset."""),
code("""rows, diffs = [], []
for m in ("clip", "dinov3"):
    Ds = {d: rdm(T[m, d], CORE) for d in DS if set(CORE) <= set(T[m, d].index)}
    B = draws(CORE, 10000, seed=2)
    boot = {d: np.array([cluster(D, CORE, ix)[:2] for ix in B]) for d, D in Ds.items()}
    for d, D in Ds.items():
        dl, sd, auc = cluster(D, CORE)
        rows.append({"model": m, "dataset": d, "n": len(CORE), "delta": dl, "delta_ci": np.percentile(boot[d][:, 0], [2.5, 97.5]).round(3).tolist(),
                     "delta_sd": sd, "delta_sd_ci": np.percentile(boot[d][:, 1], [2.5, 97.5]).round(3).tolist(), "auc": auc,
                     "delta_se": boot[d][:, 0].std(ddof=1), "delta_sd_se": boot[d][:, 1].std(ddof=1)})
    for d in Ds:
        if d == "bv": continue
        for k_, meas in enumerate(("delta", "delta_sd")):
            x = boot["bv"][:, k_] - boot[d][:, k_]
            diffs.append({"model": m, "measure": meas, "pair": f"bv - {d}", "diff": cluster(Ds["bv"], CORE)[k_] - cluster(Ds[d], CORE)[k_],
                          "ci_low": np.percentile(x, 2.5), "ci_high": np.percentile(x, 97.5),
                          "p_boot": max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / len(x))})   # floor: resolution of the draws
def bh(p):                                              # Benjamini-Hochberg q-values
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
cs_ = pd.DataFrame(rows); cs_.to_csv(RES / "cluster_strength_core.csv", index=False); display(cs_.round(3))
df_ = pd.DataFrame(diffs); df_["q_bh"] = df_.groupby(["model", "measure"]).p_boot.transform(bh)   # across the four comparisons
df_.to_csv(RES / "cluster_strength_differences_core.csv", index=False); display(df_.round(3))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, m in zip(axs, ("clip", "dinov3")):
    s = cs_[cs_.model == m]
    lo = [v[0] for v in s.delta_sd_ci]; hi = [v[1] for v in s.delta_sd_ci]
    for k, (d, v, l_, h_) in enumerate(zip(s.dataset, s.delta_sd, lo, hi)):
        ax.errorbar([k], [v], yerr=[[v - l_], [h_ - v]], fmt="o", color=FS.DS_COLOR.get(d, FS.BASELINE), ms=7, mec="white", capsize=3)
        if d != "bv":                                   # differs from BabyView (Δ / SD, BH across the four comparisons)
            q = df_[(df_.model == m) & (df_.measure == "delta_sd") & (df_.pair == f"bv - {d}")].q_bh.iloc[0]
            ax.text(k, h_ + .02, FS.stars(q), ha="center", va="bottom", fontsize=9, fontweight="bold" if q < .05 else "normal")
    ax.set_xticks(range(len(s)), [LABEL[d] for d in s.dataset], rotation=30, ha="right")
    ax.set_title(m.upper() if m == "clip" else "DINOv3", loc="left", fontsize=11); ax.axhline(0, color="#8a8984", lw=1, ls="--")
axs[0].set_ylabel("domain clustering (Δ / SD)")
fig.suptitle(f"Domain cluster strength on {len(CORE)} core categories (95% CIs; stars: differs from BabyView, BH: * q < .05, ** < .01, *** < .001)",
             x=.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "cluster_strength_core.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 4b. Cluster strength by semantic domain (notebook 09)
Per domain on the core set: mean distance from the domain's categories to every other category minus the mean distance
among the domain's own categories (delta; > 0 = the domain forms a cluster), and delta / SD of all pair distances. 95% CIs
from the shared within-domain category-bootstrap draws (pairs of a category with its own copy left out). Domains with at
least 3 core categories. Significance: BabyView minus each dataset on the same draws, two-sided bootstrap p (floor 0.002
with 10,000 draws), Benjamini-Hochberg across domains x datasets within each model and measure; stars on the figure (raw
delta), both measures in results/crossdataset/cluster_strength_by_domain_differences_core.csv."""),
code("""def domain_deltas(D, sem, doms, orig=None):
    n = len(D); valid = np.triu(np.ones((n, n), bool), 1)
    if orig is not None: valid &= orig[:, None] != orig[None, :]
    out = []
    for d in doms:
        a = sem == d; w = valid & a[:, None] & a[None, :]; b = a[:, None] & ~a[None, :]
        out.append(D[b].mean() - D[w].mean() if w.any() and b.any() else np.nan)
    return np.array(out), D[valid | valid.T].std()
sem0 = np.array([DOM.get(c, "other") for c in CORE])
vc = pd.Series(sem0).value_counts()
DOMS = [d for d in ORDER if vc.get(d, 0) >= 3] + sorted(d for d in vc.index if d not in ORDER and vc[d] >= 3)
B = draws(CORE, 10000, seed=3)
rows, BS, BSD, PT = [], {}, {}, {}
for m in ("clip", "dinov3"):
    for d in core:
        D = rdm(T[m, d], CORE)
        dl, sd = domain_deltas(D, sem0, DOMS)
        bo = [domain_deltas(D[np.ix_(ix, ix)], sem0[ix], DOMS, ix) for ix in B]
        bs = np.array([b[0] for b in bo]); BS[m, d] = bs; BSD[m, d] = bs / np.array([b[1] for b in bo])[:, None]; PT[m, d] = (dl, dl / sd)
        for k, dm in enumerate(DOMS):
            rows.append({"model": m, "dataset": d, "domain": dm, "n_categories": int(vc[dm]), "delta": dl[k], "delta_sd": dl[k] / sd,
                         "ci_low": np.nanpercentile(bs[:, k], 2.5), "ci_high": np.nanpercentile(bs[:, k], 97.5),
                         "ci_low_sd": np.nanpercentile(BSD[m, d][:, k], 2.5), "ci_high_sd": np.nanpercentile(BSD[m, d][:, k], 97.5),
                         "se": np.nanstd(bs[:, k], ddof=1), "se_sd": np.nanstd(BSD[m, d][:, k], ddof=1)})
dd = pd.DataFrame(rows); dd.to_csv(RES / "cluster_strength_by_domain_core.csv", index=False)
def bh(p):                                              # Benjamini-Hochberg q-values
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
sig = []
for m in ("clip", "dinov3"):
    for d in core:
        if d == "bv": continue
        for meas, A, k_ in (("delta", BS, 0), ("delta_sd", BSD, 1)):
            x = A[m, "bv"] - A[m, d]
            for k, dm in enumerate(DOMS):
                xk = x[:, k][~np.isnan(x[:, k])]
                p_ = max(2 * min((xk <= 0).mean(), (xk >= 0).mean()), 2 / len(xk))   # floor: resolution of the draws
                sig.append({"model": m, "measure": meas, "domain": dm, "pair": f"bv - {d}", "dataset": d,
                            "diff": PT[m, "bv"][k_][k] - PT[m, d][k_][k], "ci_low": np.percentile(xk, 2.5),
                            "ci_high": np.percentile(xk, 97.5), "se": xk.std(ddof=1), "p_boot": p_})
sig = pd.DataFrame(sig)
sig["q_bh"] = sig.groupby(["model", "measure"]).p_boot.transform(bh)    # across domains x datasets, per model and measure
sig["stars"] = [FS.stars(q, "") for q in sig.q_bh]
sig.to_csv(RES / "cluster_strength_by_domain_differences_core.csv", index=False)
for m in ("clip", "dinov3"):
    for meas in ("delta", "delta_sd"):
        t = sig[(sig.model == m) & (sig.measure == meas)]
        display(t.assign(cell=[f"{a:+.3f} {s_}" for a, s_ in zip(t["diff"], t.stars)]).pivot(index="domain", columns="pair", values="cell")
                .reindex(DOMS).style.set_caption(f"BabyView minus dataset, {meas}, {m} (BH q: * < .05, ** < .01)"))
for m in ("clip", "dinov3"):
    display(dd[dd.model == m].pivot(index="domain", columns="dataset", values="delta").reindex(DOMS)[core]
            .rename(columns=LABEL).round(3).style.set_caption(f"delta by domain, {m}"))
DCOL = dict(FS.DS_COLOR)
fig, axs = plt.subplots(2, 1, figsize=(max(9, 1.1 * len(DOMS)), 7.6), sharex=True)
w = .8 / len(core)
for ax, m in zip(axs, ("clip", "dinov3")):
    for j, d in enumerate(core):
        s = dd[(dd.model == m) & (dd.dataset == d)].set_index("domain").reindex(DOMS)
        x = np.arange(len(DOMS)) - .4 + w * (j + .5)
        ax.errorbar(x, s.delta_sd, yerr=[s.delta_sd - s.ci_low_sd, s.ci_high_sd - s.delta_sd], fmt="o", color=DCOL.get(d, FS.BASELINE),
                    ms=6, mec="white", capsize=2, lw=1.2, label=LABEL[d])
        if d != "bv":                                   # stars: differs from BabyView (BH q)
            st = sig[(sig.model == m) & (sig.measure == "delta_sd") & (sig.dataset == d)].set_index("domain").reindex(DOMS).stars
            for xi, hi, t_ in zip(x, s.ci_high_sd, st):
                if t_: ax.text(xi, hi + .04, t_, ha="center", va="bottom", fontsize=9, color=FS.INK)
    ax.axhline(0, color="#8a8984", lw=1, ls="--")
    ax.set_title(m.upper() if m == "clip" else "DINOv3", loc="left", fontsize=11); ax.set_ylabel("Δ / SD")
axs[0].legend(ncol=len(core), frameon=False, loc="upper right", fontsize=9)
axs[1].set_xticks(range(len(DOMS)), [f"{FS.DOMAIN_LABEL.get(d, d)}\\n({vc[d]})" for d in DOMS], rotation=0, fontsize=9)
FS.color_domain_ticks(axs[1].get_xticklabels(), DOMS)
for ax in axs: ax.set_ylim(top=ax.get_ylim()[1] + .3)
fig.suptitle(f"Domain cluster strength (Δ / SD) by semantic domain, {len(CORE)} core categories (n per domain in brackets)\\n"
             "stars: differs from BabyView, bootstrap test, BH-corrected across domains x datasets (* q < .05, ** < .01, *** < .001)",
             x=.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "cluster_strength_by_domain_core.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 4c. Real objects only: does the domain clustering survive without toys and pictures?
Section 4b again with BabyView and VEDB restricted to boxes Qwen labelled `real` (THINGS, LVIS and ImageNet as before),
on the categories all five share. Also BabyView all formats vs BabyView real on the same categories: the part of each
domain's clustering that comes from depictions. Same bootstrap (10,000 within-domain draws), BH across domains x
comparisons within each model and measure."""),
code("""CORE_R = shared("bv_real", "vedb_real", "things", "lvis", "imagenet")
CORE_R = [c for c in CORE_R if c in set(T["clip", "bv"].index)]
semR = np.array([DOM.get(c, "other") for c in CORE_R]); vcR = pd.Series(semR).value_counts()
DOMS_R = [d for d in ORDER if vcR.get(d, 0) >= 3]
SETS = ["bv", "bv_real", "vedb_real", "things", "lvis", "imagenet"]
BR = draws(CORE_R, 10000, seed=5)
rowsR, BSR, BSDR, PTR = [], {}, {}, {}
for m in ("clip", "dinov3"):
    for d in SETS:
        Dm = rdm(T[m, d], CORE_R)
        dl, sd = domain_deltas(Dm, semR, DOMS_R)
        bo = [domain_deltas(Dm[np.ix_(ix, ix)], semR[ix], DOMS_R, ix) for ix in BR]
        bs = np.array([b[0] for b in bo]); BSR[m, d] = bs; BSDR[m, d] = bs / np.array([b[1] for b in bo])[:, None]; PTR[m, d] = (dl, dl / sd)
        dlt, sdn, auc = cluster(Dm, CORE_R)
        rowsR.append({"model": m, "dataset": d, "domain": "ALL", "delta": dlt, "delta_sd": sdn, "auc": auc})
        for k, dm in enumerate(DOMS_R):
            rowsR.append({"model": m, "dataset": d, "domain": dm, "n_categories": int(vcR[dm]), "delta": dl[k], "delta_sd": dl[k] / sd,
                          "ci_low": np.nanpercentile(bs[:, k], 2.5), "ci_high": np.nanpercentile(bs[:, k], 97.5),
                          "ci_low_sd": np.nanpercentile(BSDR[m, d][:, k], 2.5), "ci_high_sd": np.nanpercentile(BSDR[m, d][:, k], 97.5),
                          "se": np.nanstd(bs[:, k], ddof=1), "se_sd": np.nanstd(BSDR[m, d][:, k], ddof=1)})
DR = pd.DataFrame(rowsR); DR.to_csv(RES / "cluster_strength_by_domain_real.csv", index=False)
print(f"{len(CORE_R)} categories shared by all five (BabyView and VEDB real only)")
for m in ("clip", "dinov3"):
    display(DR[DR.model == m].pivot(index="domain", columns="dataset", values="delta").reindex(DOMS_R + ["ALL"])[SETS]
            .rename(columns=LABEL).round(3).style.set_caption(f"raw delta, {m} (ALL = all domains pooled)"))
    display(DR[(DR.model == m) & (DR.domain == "ALL")].set_index("dataset")[["delta", "delta_sd", "auc"]].loc[SETS].round(3)
            .style.set_caption(f"pooled measures, {m}"))
sigR = []
for m in ("clip", "dinov3"):
    for a, b in [("bv_real", d) for d in ("vedb_real", "things", "lvis", "imagenet")] + [("bv", "bv_real")]:
        for meas, A, k_ in (("delta", BSR, 0), ("delta_sd", BSDR, 1)):
            x = A[m, a] - A[m, b]
            for k, dm in enumerate(DOMS_R):
                xk = x[:, k][~np.isnan(x[:, k])]
                sigR.append({"model": m, "measure": meas, "domain": dm, "pair": f"{a} - {b}", "diff": PTR[m, a][k_][k] - PTR[m, b][k_][k],
                             "ci_low": np.percentile(xk, 2.5), "ci_high": np.percentile(xk, 97.5), "se": xk.std(ddof=1),
                             "p_boot": max(2 * min((xk <= 0).mean(), (xk >= 0).mean()), 2 / len(xk))})
sigR = pd.DataFrame(sigR); sigR["q_bh"] = sigR.groupby(["model", "measure"]).p_boot.transform(bh)
sigR["stars"] = [FS.stars(q, "") for q in sigR.q_bh]
sigR.to_csv(RES / "cluster_strength_by_domain_real_differences.csv", index=False)
for m in ("clip", "dinov3"):
    for meas in ("delta", "delta_sd"):
        t = sigR[(sigR.model == m) & (sigR.measure == meas)]
        display(t.assign(cell=[f"{a:+.3f} {s_}" for a, s_ in zip(t["diff"], t.stars)]).pivot(index="domain", columns="pair", values="cell")
                .reindex(DOMS_R).style.set_caption(f"{meas}, {m} (BH q: * < .05, ** < .01); 'bv - bv_real' = depiction contribution"))
fig, axs = plt.subplots(2, 1, figsize=(max(9, 1.2 * len(DOMS_R)), 7.6), sharex=True)
show = ["bv", "bv_real", "vedb_real", "things", "lvis", "imagenet"]
col = dict(FS.DS_COLOR)
w = .8 / len(show)
for ax, m in zip(axs, ("clip", "dinov3")):
    for j, d in enumerate(show):
        s_ = DR[(DR.model == m) & (DR.dataset == d) & (DR.domain != "ALL")].set_index("domain").reindex(DOMS_R)
        x = np.arange(len(DOMS_R)) - .4 + w * (j + .5)
        ax.errorbar(x, s_.delta_sd, yerr=[s_.delta_sd - s_.ci_low_sd, s_.ci_high_sd - s_.delta_sd], fmt="o", color=col[d], ms=6,
                    mfc="white" if d == "bv" else col[d], mec=col[d], capsize=2, lw=1.2, label=LABEL[d])
        if d != "bv_real":                              # differs from BabyView real (Δ / SD, BH); open circle: BabyView all vs real
            pair = "bv - bv_real" if d == "bv" else f"bv_real - {d}"
            st = sigR[(sigR.model == m) & (sigR.measure == "delta_sd") & (sigR.pair == pair)].set_index("domain").reindex(DOMS_R).stars
            for xi, hi, t_ in zip(x, s_.ci_high_sd, st):
                if t_: ax.text(xi, hi + .04, t_, ha="center", va="bottom", fontsize=8, color=FS.INK)
    ax.axhline(0, color="#8a8984", lw=1, ls="--")
    ax.set_title(m.upper() if m == "clip" else "DINOv3", loc="left", fontsize=11); ax.set_ylabel("Δ / SD")
axs[0].legend(ncol=len(show), frameon=False, loc="upper right", fontsize=8)
axs[1].set_xticks(range(len(DOMS_R)), [f"{FS.DOMAIN_LABEL.get(d, d)}\\n({vcR[d]})" for d in DOMS_R], fontsize=9)
FS.color_domain_ticks(axs[1].get_xticklabels(), DOMS_R)
for ax in axs: ax.set_ylim(top=ax.get_ylim()[1] + .3)
fig.suptitle(f"Domain clustering (Δ / SD) with real objects only, {len(CORE_R)} categories (open circles: BabyView all formats)\\n"
             "stars: differs from BabyView real (open circles: all formats vs real), BH across domains x comparisons (* q < .05, ** < .01, *** < .001)",
             x=.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "cluster_strength_by_domain_real.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 5. Visual formats across datasets
Share of exemplars that are toys, pictures or screens. BabyView and VEDB: Qwen's own format labels (boxes); THINGS, LVIS,
ImageNet: the format probe (format_classifier.py, 88.5% held-out agreement with Qwen on BabyView). `excl.` leaves out
categories that are depictions or toys by definition (picture, photo, drawing, painting, poster, screen, toy, block, ...)."""),
code("""INTRINSIC = {"picture", "photo", "drawing", "painting", "poster", "sticker", "screen", "television", "toy", "block", "puzzle",
             "puzzle piece", "doll", "teddy bear", "lego", "lego block", "stuffed animal", "figurine", "cartoon", "map", "calendar",
             "book", "picture book", "card", "tablet", "phone", "laptop", "computer monitor", "monitor"}
FMT = pd.read_csv(XD / "format_by_category.csv", index_col=[0, 1]) if (XD / "format_by_category.csv").exists() else None
if FMT is None:
    print("format_by_category.csv not built yet (build_format_tables.py after the LVIS / ImageNet probe runs)")
else:
    rows = []
    for d, g in FMT.groupby(level=0):
        g = g.droplevel(0)
        for name, gg in (("all", g), ("excl.", g[~g.index.isin(INTRINSIC)])):
            n = gg.n.sum()
            rows.append({"dataset": d, "categories": name, "exemplars": int(n),
                         **{f: (gg[f] * gg.n).sum() / n for f in ("real", "toy", "picture", "screen")}})
    fp = pd.DataFrame(rows); fp.to_csv(RES / "format_prevalence.csv", index=False); display(fp.round(3))
    dom = []
    for d, g in FMT.groupby(level=0):
        g = g.droplevel(0); g = g[~g.index.isin(INTRINSIC)].assign(domain=lambda x: [DOM.get(c, "other") for c in x.index])
        for dm, gg in g.groupby("domain"):
            dom.append({"dataset": d, "domain": dm, "depicted": 1 - (gg.real * gg.n).sum() / gg.n.sum(), "n": int(gg.n.sum())})
    dom = pd.DataFrame(dom).pivot(index="domain", columns="dataset", values="depicted")
    display(dom.round(3).style.set_caption("depicted share by domain (intrinsic depictions left out)"))
    dom.to_csv(RES / "format_depicted_by_domain.csv")"""),

md("""## 6. Categories rich in depictions in BabyView but not elsewhere
Depicted share (toy + picture + screen) per category in BabyView vs the mean of the other datasets that have the
category (>= 30 exemplars each), and BabyView vs each dataset on its own (depiction_gap_bv_vs_<dataset>.csv).
Montages: make_crossdataset_montages.py [--vs <dataset>]."""),
code("""if FMT is not None:
    dep = (1 - FMT.real).unstack(level=0); n = FMT.n.unstack(level=0)
    dep = dep.where(n >= 30)
    others = [d for d in dep.columns if d not in ("bv", "bv_real", "vedb_real")]
    cand = pd.DataFrame({"bv_depicted": dep["bv"], "others_depicted": dep[others].mean(axis=1), "n_other_datasets": dep[others].notna().sum(axis=1)})
    cand = cand[(cand.n_other_datasets >= 1) & ~cand.index.isin(INTRINSIC)].dropna()
    cand["gap"] = cand.bv_depicted - cand.others_depicted
    cand["domain"] = [DOM.get(c) for c in cand.index]
    cand = cand.sort_values("gap", ascending=False); cand.to_csv(RES / "depiction_gap_categories.csv")
    display(cand.head(40).round(3))
    for d in others:                                     # BabyView vs each dataset on its own
        g = pd.DataFrame({"bv_depicted": dep["bv"], f"{d}_depicted": dep[d], "bv_n": n["bv"], f"{d}_n": n[d]}).dropna()
        g = g[~g.index.isin(INTRINSIC)].assign(gap=lambda x: x.bv_depicted - x[f"{d}_depicted"],
                                               domain=lambda x: [DOM.get(c) for c in x.index])
        g.sort_values("gap", ascending=False).to_csv(RES / f"depiction_gap_bv_vs_{d}.csv")
        print(f"BabyView vs {d}: {len(g)} shared categories; top 15:", ", ".join(g.sort_values("gap", ascending=False).index[:15]))"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "crossdataset_analysis.ipynb")
print("wrote", HERE / "crossdataset_analysis.ipynb")
