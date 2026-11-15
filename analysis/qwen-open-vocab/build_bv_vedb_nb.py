#!/usr/bin/env python3
"""Write bv_vedb_analysis.ipynb: how the child's view (BabyView) differs from an adult's (VEDB), and whether the
difference shrinks with age.

Inputs: build_bv_vedb_data.py (out/private/bv_vedb/), Qwen boxes of both datasets, same model, prompt, temperature and
pixel count. Unit: BabyView child, VEDB session. BabyView age bins: <12, 12-18, 18-24, 24+ months.
Sections: scene statistics; category frequencies (BabyView / VEDB ratios); divergence of what is seen (Jensen-Shannon
divergence of the category and domain distributions), of how it looks (within-category MMD of embeddings) and of
between-category structure (RDM correlation); age trends. Every divergence uses one split-half design: on each draw the
children of a bin and the VEDB sessions are split into random halves; BabyView half vs VEDB half is the comparison,
half vs half within each dataset the baseline, with equal sample sizes on every side. Home-scene frames (frames with a
bed, couch, crib, high chair, fridge, stove, ...) give a content-matched check, since VEDB sessions span many settings.

  python build_bv_vedb_nb.py
  jupyter nbconvert --to notebook --execute bv_vedb_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \\
      --ExecutePreprocessor.timeout=14400 --output-dir out/private --output bv_vedb_analysis_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# BabyView vs VEDB: the child's view and the adult's

Both datasets went through the same Qwen3.8-27B pipeline (open labels, formats, boxes; same prompt, temperature and pixel
count) and the same CLIP / DINOv3 embeddings. Units are BabyView children and VEDB sessions; BabyView is split into four
age bins. **Divergences** (sections 4-6) all use the same split-half design, so every number has a matched baseline:
on each draw, the children of an age bin and the VEDB sessions are each split into random halves;
*BabyView half vs VEDB half* is the comparison, *half vs half within each dataset* the baseline, with equal sample sizes
throughout. A divergence that is no larger than the within-BabyView baseline means the child-adult difference is no
bigger than the difference between groups of families.

Caveat: VEDB sessions were recorded in many settings (homes, shops, campuses, outdoors); BabyView mostly at home.
*Home-scene* frames (with a bed, couch, crib, high chair, fridge, stove, ...) give a content-matched comparison."""),
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
HERE = Path.cwd()
D = HERE / "out" / "private" / "bv_vedb"
sys.path.insert(0, str(HERE)); import variant as V   # QOV_LABELS=renamed (environment): parallel folders
print("labels:", V.LABELS)
D = V.out_dir(D)
import os
CORE_MODE = os.environ.get("QOV_CATS", "tier1")   # core: appearance and RDM sections on the 183 / 164 core; frequencies on Tier 1
SFX = "_core" if CORE_MODE == "core" else ""
FIG = V.out_dir(HERE / "figures" / "bv_vedb"); FIG = FIG.with_name(FIG.name + SFX); FIG.mkdir(parents=True, exist_ok=True)
RES = V.out_dir(HERE / "results" / "bv_vedb"); RES = RES.with_name(RES.name + SFX); RES.mkdir(parents=True, exist_ok=True)
_data = HERE.parents[1] / "data"
CORE = {l.strip() for l in open(_data / "included_categories_qwen_core.txt") if l.strip()}
CORE_REAL = {l.strip() for l in open(_data / "included_categories_qwen_core_real.txt") if l.strip()}
def in_core(c, real=False):
    return CORE_MODE != "core" or c in (CORE_REAL if real else CORE)
print("categories:", CORE_MODE)
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
BINS = ["<12 mo", "12-18 mo", "18-24 mo", "24+ mo"]
GROUPS = BINS + ["all ages"]
import fig_style as FS                      # manuscript style: Lato, bold labels / legends, dataset colours
FS.apply()
C_BV, C_VEDB, C_BASE = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"], FS.BASELINE
fr = pd.read_csv(D / "frames.csv.gz", dtype={"unit": str})
pres = pd.read_csv(D / "presence.csv.gz", dtype={"unit": str})
geo = pd.read_csv(D / "geometry.csv.gz")
S = {m: dict(np.load(D / f"sample_{m}.npz", allow_pickle=True)) for m in ("clip", "dinov3")}
fr = fr[(fr.dataset == "vedb") | fr.age_bin.isin(BINS)]
pres = pres[(pres.dataset == "vedb") | pres.age_bin.isin(BINS)]
rng0 = np.random.default_rng(0)
def units(ds, group):
    f = fr[fr.dataset == ds]
    if ds == "bv" and group != "all ages": f = f[f.age_bin == group]
    return sorted(f.unit.unique())
def halves(u, rng):
    u = np.array(u); rng.shuffle(u); h = len(u) // 2
    return list(u[:h]), list(u[h:2 * h])
summary = []          # (measure, group, comparison, value, lo, hi)
print({ds: (fr[fr.dataset == ds].unit.nunique(), len(fr[fr.dataset == ds])) for ds in ("bv", "vedb")})
display(fr.groupby(["dataset", "age_bin"]).agg(units=("unit", "nunique"), frames=("unit", "size"), home_share=("home", "mean")).round(3))"""),

md("""## 1. Scene statistics
Per frame: objects, distinct categories, people, hands, faces (face / eye / mouth / nose), objects shown as toys, pictures
or screens, median object size (box area / frame area; larger = closer or bigger) and mean vertical position (0 top,
1 bottom). Averaged per child or session first, then across units; 95% CIs from resampling units."""),
code("""M1 = {"n_obj": "objects / frame", "n_distinct": "distinct categories / frame", "n_person": "people / frame",
      "n_hand": "hands / frame", "n_face": "face parts / frame", "dep_share": "share of objects depicted",
      "n_screen": "screen objects / frame", "median_area": "median object size (frame share)", "mean_cy": "mean object height (0 top, 1 bottom)"}
fr["dep_share"] = fr.n_depicted / fr.n_obj.clip(lower=1)
def unit_means(f):
    return f.groupby("unit")[list(M1)].mean()
def boot_ci(x, n=2000, rng=rng0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
rows = []
for scope, f0 in (("all frames", fr), ("home-scene frames", fr[fr.home])):
    for g in GROUPS + ["VEDB"]:
        f = f0[f0.dataset == "vedb"] if g == "VEDB" else f0[(f0.dataset == "bv") & ((f0.age_bin == g) | (g == "all ages"))]
        um = unit_means(f)
        for k in M1:
            v, lo, hi = boot_ci(um[k])
            rows.append(dict(scope=scope, group=g, measure=k, mean=v, lo=lo, hi=hi, units=len(um)))
sc = pd.DataFrame(rows); sc.to_csv(RES / "scene_statistics.csv", index=False)
display(sc[sc.scope == "all frames"].pivot(index="measure", columns="group", values="mean")[GROUPS + ["VEDB"]].round(3))
fig, axs = plt.subplots(3, 3, figsize=(12, 9)); axs = axs.ravel()
for ax, k in zip(axs, M1):
    for j, (scope, mk) in enumerate((("all frames", "o"), ("home-scene frames", "s"))):
        s = sc[(sc.scope == scope) & (sc.measure == k)].set_index("group")
        x = np.arange(len(BINS)) + (j - .5) * .18
        b = s.loc[BINS]
        ax.errorbar(x, b["mean"], yerr=[b["mean"] - b.lo, b.hi - b["mean"]], fmt=mk + "-", color=C_BV, ms=5,
                    mfc=C_BV if j == 0 else "white", capsize=2, lw=1.2, label=f"BabyView, {scope}")
        v = s.loc["VEDB"]
        ax.axhspan(v.lo, v.hi, color=C_VEDB, alpha=.12 if j == 0 else .06)
        ax.axhline(v["mean"], color=C_VEDB, lw=1.4, ls="-" if j == 0 else "--", label=f"VEDB, {scope}")
    ax.set_xticks(range(len(BINS)), BINS, fontsize=8); ax.set_title(M1[k], fontsize=10, loc="left")
axs[0].legend(fontsize=7, frameon=False)
fig.suptitle("Scene statistics: BabyView by age (blue) vs VEDB adults (orange band = 95% CI)", x=.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "scene_statistics.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 1b. Scene statistics within children: change with age, or differences between children?
The age bins in section 1 hold different children (15-29 per bin), so a trend across bins can come from which children
are in a bin. Here each measure is averaged per child and month of age (child-months with >= 200 frames) and age is split
into its within-child part (month minus the child's mean month) and its between-child part (the child's mean month).
Two estimates of the within-child slope:
- **mixed model**: random intercept and random age slope per child (statsmodels MixedLM; if it does not converge, random
  intercept only, flagged in `model`);
- **child fixed effects**: OLS of the measure on age with one intercept per child, standard errors clustered by child
  (always estimable; same within-child slope).
Slopes per year of age; BH-FDR across measures. Also the share of children (>= 3 child-months) whose own linear trend has
the same sign. Figure: thin lines = individual children (child-month means), thick = fixed-effects fit, orange = VEDB."""),
code("""import warnings
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
fb = fr[fr.dataset == "bv"].copy()
fb["month"] = np.floor(fb.age_months)
MW = ["n_obj", "n_distinct", "n_person", "n_hand", "n_face", "n_screen", "dep_share", "median_area", "mean_cy"]
cmo = fb.groupby(["unit", "month"]).agg(frames=("n_obj", "size"), **{m: (m, "mean") for m in MW}).reset_index()
cmo = cmo[cmo.frames >= 200].copy()
mu = cmo.groupby("unit").month.transform("mean")
cmo["age_within"] = cmo.month - mu; cmo["age_between"] = mu - cmo.month.mean()
print(f"{cmo.unit.nunique()} children, {len(cmo)} child-months; children with >= 3 child-months: {(cmo.groupby('unit').size() >= 3).sum()}")
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
rows = []
for m in MW:
    sd_m = cmo[m].std()
    d = cmo.assign(y=(cmo[m] - cmo[m].mean()) / sd_m, mw=cmo.age_within / 12, mb=cmo.age_between / 12)   # z-scored y, age in years
    fit, model = None, "random intercept + slope"
    for re in ("~age_within", None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                f = smf.mixedlm("y ~ mw + mb", d, groups=d.unit, re_formula="~mw" if re else None).fit(method=["lbfgs", "powell"], reml=True)
                ok = f.converged and not any(issubclass(x.category, ConvergenceWarning) for x in w)
            except Exception:
                ok = False
        if ok:
            fit = f; break
        model = "random intercept only"
    fe = smf.ols("y ~ month + C(unit)", d.assign(y=cmo[m])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit)[0]})
    sl = [np.polyfit(x.month, x[m], 1)[0] for _, x in cmo.groupby("unit") if len(x) >= 3]
    rows.append(dict(measure=m, label=M1.get(m, m),
                     mixed_within_per_year=fit.params.mw * sd_m if fit is not None else np.nan,
                     mixed_ci_lo=fit.conf_int().loc["mw", 0] * sd_m if fit is not None else np.nan,
                     mixed_ci_hi=fit.conf_int().loc["mw", 1] * sd_m if fit is not None else np.nan,
                     p_mixed=fit.pvalues.mw if fit is not None else np.nan, model=model if fit is not None else "did not converge",
                     between_per_year=fit.params.mb * sd_m if fit is not None else np.nan,
                     p_between=fit.pvalues.mb if fit is not None else np.nan,
                     fe_within_per_year=fe.params.month * 12, fe_ci_lo=fe.conf_int().loc["month", 0] * 12,
                     fe_ci_hi=fe.conf_int().loc["month", 1] * 12, p_fe=fe.pvalues.month,
                     mean=cmo[m].mean(), pct_change_per_year=fe.params.month * 12 / cmo[m].mean(),
                     children_same_sign=np.mean(np.sign(sl) == np.sign(fe.params.month)), children=len(sl)))
WC = pd.DataFrame(rows)
WC["q_mixed"] = bh(WC.p_mixed.fillna(1)); WC["q_fe"] = bh(WC.p_fe); WC["q_between"] = bh(WC.p_between.fillna(1))
WC.to_csv(RES / "scene_statistics_within_child.csv", index=False)
display(WC[["label", "mixed_within_per_year", "q_mixed", "model", "fe_within_per_year", "fe_ci_lo", "fe_ci_hi", "q_fe",
            "pct_change_per_year", "between_per_year", "q_between", "children_same_sign"]].round(4))
vedb_mean = fr[fr.dataset == "vedb"].groupby("unit")[MW].mean().mean()
fig, axs = plt.subplots(3, 3, figsize=(12, 9.5)); axs = axs.ravel()
for ax, m in zip(axs, MW):
    for u, x in cmo.groupby("unit"):
        if len(x) >= 3: ax.plot(x.month, x[m], "-", color=C_BV, alpha=.18, lw=.8)
    r = WC.set_index("measure").loc[m]
    xs = np.linspace(cmo.month.min(), cmo.month.max(), 50)
    ax.plot(xs, cmo[m].mean() + r.fe_within_per_year / 12 * (xs - cmo.month.mean()), color=C_BV, lw=2.4)
    ax.axhline(vedb_mean[m], color=C_VEDB, lw=1.5)
    star = "**" if r.q_fe < .01 else "*" if r.q_fe < .05 else "n.s."
    ax.set_title(f"{M1.get(m, m)}\\n{r.fe_within_per_year:+.3g}/yr ({r.pct_change_per_year:+.0%}), {star}; {r.children_same_sign:.0%} of children",
                 loc="left", fontsize=9)
    ax.set_xlabel("age (months)", fontsize=8)
fig.suptitle("Within-child change with age (thin: each child; thick: child fixed-effects slope; orange: VEDB adults)", x=.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "scene_statistics_within_child.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 2. Category frequencies
Frame prevalence per Tier 1 category (share of frames containing it), BabyView vs VEDB, as log2 ratio with 95% CIs from
resampling children and sessions. Categories with >= 50 frames in each dataset. Domain level: summed prevalence of the
domain's categories."""),
code("""def prevalence(ds, group="all ages", home=None):
    p = pres[pres.dataset == ds]; f = fr[fr.dataset == ds]
    if ds == "bv" and group != "all ages": p, f = p[p.age_bin == group], f[f.age_bin == group]
    if home is not None: p, f = p[p.home == home], f[f.home == home]
    cnt = p.groupby(["unit", "category"]).frames.sum().unstack(fill_value=0)
    tot = f.groupby("unit").size()
    return cnt.reindex(tot.index, fill_value=0), tot
def ratio_table(home=None, n_boot=1000, rng=np.random.default_rng(1)):
    (cb, tb), (cv, tv) = prevalence("bv", home=home), prevalence("vedb", home=home)
    cats = sorted(set(cb.columns) & set(cv.columns))
    cats = [c for c in cats if cb[c].sum() >= 50 and cv[c].sum() >= 50]
    cb, cv = cb[cats], cv[cats]
    pb, pv = cb.sum() / tb.sum(), cv.sum() / tv.sum()
    lr = np.log2(pb / pv)
    PB, PV = [], []
    for _ in range(n_boot):
        ib, iv = rng.integers(0, len(tb), len(tb)), rng.integers(0, len(tv), len(tv))
        PB.append(cb.values[ib].sum(0) / tb.values[ib].sum()); PV.append(cv.values[iv].sum(0) / tv.values[iv].sum())
    PB, PV = np.array(PB), np.array(PV)
    with np.errstate(divide="ignore"):
        B = np.log2(PB / PV)
    t = pd.DataFrame({"bv_prev": pb, "vedb_prev": pv, "log2_ratio": lr, "lo": np.percentile(B, 2.5, 0), "hi": np.percentile(B, 97.5, 0),
                      "domain": [DOM.get(c, "other") for c in cats]}, index=cats)
    dm = {}
    for d, g in t.groupby("domain"):
        cols = [cats.index(c) for c in g.index]
        bd = np.log2(PB[:, cols].sum(1) / PV[:, cols].sum(1))
        dm[d] = dict(bv_prev=pb[g.index].sum(), vedb_prev=pv[g.index].sum(), log2_ratio=np.log2(pb[g.index].sum() / pv[g.index].sum()),
                     lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), n=len(g))
    return t.sort_values("log2_ratio"), pd.DataFrame(dm).T
rt, rd = ratio_table(); rt.to_csv(RES / "category_frequency_ratio.csv")
rth, rdh = ratio_table(home=True); rth.to_csv(RES / "category_frequency_ratio_home.csv")
print(f"{len(rt)} categories; more frequent for children (log2 ratio > 1): {(rt.lo > 1).sum()}, for adults (< -1): {(rt.hi < -1).sum()}")
display(rt.tail(25).round(3).style.set_caption("most over-represented in BabyView"))
display(rt.head(25).round(3).style.set_caption("most over-represented in VEDB"))
rd = rd.sort_values("log2_ratio"); rd.to_csv(RES / "domain_frequency_ratio.csv"); display(rd.round(3))
fig, axs = plt.subplots(1, 2, figsize=(12, 7))
for ax, (t, title) in zip(axs, ((rt, "all frames"), (rth, "home-scene frames"))):
    s = pd.concat([t.head(15), t.tail(15)])
    y = np.arange(len(s))
    ax.errorbar(s.log2_ratio, y, xerr=[s.log2_ratio - s.lo, s.hi - s.log2_ratio], fmt="o", ms=4, capsize=2,
                color="#3d3c39", ecolor="#b9b8b3")
    ax.scatter(s.log2_ratio, y, c=[C_BV if v > 0 else C_VEDB for v in s.log2_ratio], s=22, zorder=3)
    ax.set_yticks(y, s.index, fontsize=8); ax.axvline(0, color=C_BASE, lw=1, ls="--")
    ax.set_xlabel("log2 frame prevalence, BabyView / VEDB"); ax.set_title(title, loc="left", fontsize=10)
fig.suptitle("Categories most over-represented for children (blue) and adults (orange)", x=.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "category_frequency_ratio.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 3. Divergence of what is seen
Jensen-Shannon divergence (base 2, 0 = identical, 1 = disjoint) between category distributions (frames containing each
Tier 1 category, normalized), and between domain distributions. Split-half design (200 draws): BabyView half vs VEDB half,
and the within-BabyView and within-VEDB baselines; every side subsampled to the same number of category-frame events.
Per-category contributions to the pooled JS divergence say which categories drive it."""),
code("""def js(p, q):
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0); b = np.where(q > 0, q * np.log2(q / m), 0)
    return .5 * a.sum() + .5 * b.sum(), .5 * a + .5 * b
def counts(ds, group, home, level):
    c, _ = prevalence(ds, group, home)
    if level == "domain":
        c = c.T.groupby(lambda x: DOM.get(x, "other")).sum().T
    return c
def js_split(group, home=None, level="category", n_draw=200, rng=np.random.default_rng(2)):
    cb, cv = counts("bv", group, home, level), counts("vedb", "all ages", home, level)
    cols = sorted(set(cb.columns) | set(cv.columns)); cb, cv = cb.reindex(columns=cols, fill_value=0), cv.reindex(columns=cols, fill_value=0)
    out = []
    for _ in range(n_draw):
        b1, b2 = halves(cb.index, rng); v1, v2 = halves(cv.index, rng)
        X = [cb.loc[b1].values.sum(0), cb.loc[b2].values.sum(0), cv.loc[v1].values.sum(0), cv.loc[v2].values.sum(0)]
        N = int(min(x.sum() for x in X))
        X = [rng.multivariate_hypergeometric(x.astype(np.int64), N) for x in X]
        out.append((js(X[0], X[2])[0], js(X[0], X[1])[0], js(X[2], X[3])[0]))
    return np.array(out)
rows = []
for home in (None, True):
    for level in ("category", "domain"):
        for g in GROUPS:
            o = js_split(g, home, level)
            for k, name in enumerate(("BabyView vs VEDB", "within BabyView", "within VEDB")):
                rows.append(dict(scope="home-scene frames" if home else "all frames", level=level, group=g, comparison=name,
                                 js=o[:, k].mean(), lo=np.percentile(o[:, k], 2.5), hi=np.percentile(o[:, k], 97.5)))
            d = o[:, 0] - o[:, 1]
            rows.append(dict(scope="home-scene frames" if home else "all frames", level=level, group=g, comparison="excess over within BabyView",
                             js=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
J = pd.DataFrame(rows); J.to_csv(RES / "js_divergence.csv", index=False)
display(J[J.level == "category"].pivot_table(index=["scope", "comparison"], columns="group", values="js")[GROUPS].round(3))
display(J[J.level == "domain"].pivot_table(index=["scope", "comparison"], columns="group", values="js")[GROUPS].round(3))
cb, cv = counts("bv", "all ages", None, "category").sum(), counts("vedb", "all ages", None, "category").sum()
cols = sorted(set(cb.index) | set(cv.index)); tot, contrib = js(cb.reindex(cols, fill_value=0).values, cv.reindex(cols, fill_value=0).values)
ct = pd.DataFrame({"contribution": contrib, "share_of_js": contrib / tot, "bv_share": (cb / cb.sum()).reindex(cols, fill_value=0).values,
                   "vedb_share": (cv / cv.sum()).reindex(cols, fill_value=0).values, "domain": [DOM.get(c, "other") for c in cols]}, index=cols)
ct = ct.sort_values("contribution", ascending=False); ct.to_csv(RES / "js_contributions.csv")
print(f"pooled JS divergence {tot:.3f}; top 20 categories carry {ct.share_of_js.head(20).sum():.0%}")
display(ct.head(20).round(4)); display(ct.groupby("domain").share_of_js.sum().sort_values(ascending=False).round(3))"""),

md("""## 4. Divergence of how things look (within category)
For each Tier 1 category, the maximum mean discrepancy (MMD², RBF kernel on L2-normalized embeddings, bandwidth = median
pairwise distance) between BabyView and VEDB exemplars, against the within-dataset baselines, in the same split-half design
(halves by child / session; 40 exemplars per half; 30 draws). Reported per category as the excess over the larger
within-dataset baseline, and per age bin as the median over the categories every bin has. `real` repeats it with real
objects only, to separate the effect of toys and pictures."""),
code("""NH = 40
def l2(X): X = X.astype(np.float32); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
def mmd2(X, Y, s):
    k = lambda A, B: np.exp(-cdist(A, B, "sqeuclidean") / (2 * s * s))
    return k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean()
def pick(idx, unit, us, n, rng):
    j = idx[np.isin(unit[idx], us)]
    return rng.choice(j, n, replace=False) if len(j) >= n else None
def appearance(m, real=False, n_draw=30, rng=np.random.default_rng(3)):
    s = S[m]; E = l2(s["emb"]); ds, ab, cat, un, fm = s["dataset"], s["age_bin"], s["category"], s["unit"], s["format"]
    ok = (fm == "real") if real else np.ones(len(E), bool)
    rows = []
    for c in sorted(x for x in set(cat) if in_core(x, real)):
        iv = np.flatnonzero((cat == c) & (ds == "vedb") & ok)
        if len(iv) < 2 * NH: continue
        for g in GROUPS:
            ib = np.flatnonzero((cat == c) & (ds == "bv") & ok & ((ab == g) | (g == "all ages")))
            if len(ib) < 2 * NH: continue
            ub, uv = sorted(set(un[ib])), sorted(set(un[iv]))
            if len(ub) < 4 or len(uv) < 4: continue
            vals = []
            for _ in range(n_draw):
                b1, b2 = halves(ub, rng); v1, v2 = halves(uv, rng)
                P = [pick(ib, un, b1, NH, rng), pick(ib, un, b2, NH, rng), pick(iv, un, v1, NH, rng), pick(iv, un, v2, NH, rng)]
                if any(p is None for p in P): continue
                X = [E[p] for p in P]
                sig = np.median(pdist(np.vstack(X)))
                vals.append((mmd2(X[0], X[2], sig), mmd2(X[0], X[1], sig), mmd2(X[2], X[3], sig)))
            if len(vals) >= 10:
                v = np.array(vals).mean(0)
                rows.append(dict(model=m, real=real, category=c, group=g, between=v[0], within_bv=v[1], within_vedb=v[2],
                                 excess=v[0] - max(v[1], v[2]), domain=DOM.get(c, "other")))
    return pd.DataFrame(rows)
A = pd.concat([appearance(m, real) for m in ("clip", "dinov3") for real in (False, True)], ignore_index=True)
A.to_csv(RES / "appearance_mmd.csv", index=False)
def by_group(a):
    every = set.intersection(*[set(a[a.group == g].category) for g in GROUPS]) if len(a) else set()
    out = []
    for g in GROUPS:
        x = a[(a.group == g) & a.category.isin(every)].set_index("category").excess
        b = [np.median(np.random.default_rng(i).choice(x.values, len(x))) for i in range(1000)]
        out.append(dict(group=g, median_excess=x.median(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                        share_above_baseline=(x > 0).mean(), categories=len(x)))
    return pd.DataFrame(out)
AG = pd.concat([by_group(A[(A.model == m) & (A.real == r)]).assign(model=m, real=r) for m in ("clip", "dinov3") for r in (False, True)])
AG.to_csv(RES / "appearance_mmd_by_age.csv", index=False); display(AG.round(4))
for m in ("clip", "dinov3"):
    a = A[(A.model == m) & ~A.real & (A.group == "all ages")].sort_values("excess", ascending=False)
    display(a.head(20)[["category", "domain", "between", "within_bv", "within_vedb", "excess"]].round(4)
            .style.set_caption(f"{m}: categories that look most different (children vs adults)"))
    display(a.groupby("domain").excess.agg(["median", "size"]).sort_values("median", ascending=False).round(4))"""),

md("""## 5. Divergence of between-category structure
Per age bin, category means from the embedding sample (categories with >= 60 exemplars in every bin and in VEDB),
z-scored within the set, cosine RDMs. BabyView half vs VEDB half Spearman, with within-dataset half vs half as the
reliability (split by child / session; 100 draws); 1 - rho is the divergence."""),
code("""def rdm_split(m, real=False, n_draw=100, rng=np.random.default_rng(4), min_n=60):
    s = S[m]; E = l2(s["emb"]); ds, ab, cat, un, fm = s["dataset"], s["age_bin"], s["category"], s["unit"], s["format"]
    ok = (fm == "real") if real else np.ones(len(E), bool)
    cnt = pd.DataFrame({"g": np.where(ds == "vedb", "VEDB", ab), "c": cat})[ok].value_counts().unstack(0).fillna(0)
    cats = sorted(c for c in cnt.index[(cnt[BINS + ["VEDB"]] >= min_n).all(1)] if in_core(c, real))
    ci = {c: i for i, c in enumerate(cats)}
    zr = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-10)
    def means(sel):
        X = np.zeros((len(cats), E.shape[1])); n = np.zeros(len(cats))
        k = np.array([ci.get(c, -1) for c in cat[sel]]); v = k >= 0
        np.add.at(X, k[v], E[sel][v]); np.add.at(n, k[v], 1)
        return X / np.maximum(n, 1)[:, None], n
    rd = lambda X: squareform(pdist(zr(X), "cosine"))[np.triu_indices(len(X), 1)]
    rows = []
    uv = sorted(set(un[ds == "vedb"]))
    for g in GROUPS:
        ub = sorted(set(un[(ds == "bv") & ((ab == g) | (g == "all ages"))]))
        r = []
        for _ in range(n_draw):
            b1, b2 = halves(ub, rng); v1, v2 = halves(uv, rng)
            M = [means(ok & (ds == "bv") & ((ab == g) | (g == "all ages")) & np.isin(un, h)) for h in (b1, b2)] + \\
                [means(ok & (ds == "vedb") & np.isin(un, h)) for h in (v1, v2)]
            keep = np.all([n >= 5 for _, n in M], axis=0)        # categories with >= 5 exemplars in all four halves
            if keep.sum() < 30: continue
            R = [rd(X[keep]) for X, _ in M]
            r.append((stats.spearmanr(R[0], R[2]).statistic, stats.spearmanr(R[0], R[1]).statistic, stats.spearmanr(R[2], R[3]).statistic))
        r = np.array(r).reshape(-1, 3)
        if not len(r): continue
        for k, name in enumerate(("BabyView vs VEDB", "within BabyView", "within VEDB")):
            rows.append(dict(model=m, real=real, group=g, comparison=name, rho=r[:, k].mean(), lo=np.percentile(r[:, k], 2.5),
                             hi=np.percentile(r[:, k], 97.5), categories=len(cats), draws=len(r)))
        if not len(r):
            print(f"{m} {g}: no usable draws")
    return pd.DataFrame(rows)
RR = pd.concat([rdm_split(m, real) for m in ("clip", "dinov3") for real in (False, True)], ignore_index=True)
RR.to_csv(RES / "rdm_split_half.csv", index=False)
display(RR.pivot_table(index=["model", "real", "comparison"], columns="group", values="rho")[GROUPS].round(3))"""),

md("""## 6. Age trends
One panel per divergence, BabyView age bins on x: what is seen (JS divergence, category and domain), how often objects are
depictions, how things look (median MMD excess), and between-category structure (1 - rho). Orange: BabyView vs VEDB;
grey: the within-BabyView baseline (and within-VEDB where it applies). A trend in the orange line that the grey line does
not share is a change with age in how far children's experience is from adults'."""),
code("""fig, axs = plt.subplots(2, 3, figsize=(13, 7.5)); axs = axs.ravel(); x = np.arange(len(BINS))
def band(ax, t, col, ycol, label, ls="-"):
    t = t.set_index("group").loc[BINS]
    ax.plot(x, t[ycol], ls, color=col, marker="o", ms=5, lw=1.5, label=label)
    ax.fill_between(x, t.lo, t.hi, color=col, alpha=.15, lw=0)
for ax, level in zip(axs[:2], ("category", "domain")):
    j = J[(J.level == level) & (J.scope == "all frames")]
    band(ax, j[j.comparison == "BabyView vs VEDB"], C_VEDB, "js", "BabyView vs VEDB")
    band(ax, j[j.comparison == "within BabyView"], C_BASE, "js", "within BabyView (baseline)")
    jh = J[(J.level == level) & (J.scope == "home-scene frames") & (J.comparison == "BabyView vs VEDB")]
    band(ax, jh, C_VEDB, "js", "BabyView vs VEDB, home scenes", ls="--")
    ax.set_title(f"what is seen: JS divergence ({level})", loc="left", fontsize=10)
ax = axs[2]
s = sc[(sc.scope == "all frames") & (sc.measure == "dep_share")].set_index("group")
ax.errorbar(x, s.loc[BINS, "mean"], yerr=[s.loc[BINS, "mean"] - s.loc[BINS, "lo"], s.loc[BINS, "hi"] - s.loc[BINS, "mean"]],
            fmt="o-", color=C_BV, capsize=2, label="BabyView")
ax.axhline(s.loc["VEDB", "mean"], color=C_VEDB, lw=1.5, label="VEDB")
ax.set_title("share of objects depicted (toy / picture / screen)", loc="left", fontsize=10)
for ax, m in zip(axs[3:5], ("clip", "dinov3")):
    for r, ls in ((False, "-"), (True, "--")):
        band(ax, AG[(AG.model == m) & (AG.real == r)].rename(columns={"median_excess": "y"}), C_VEDB, "y",
             "all formats" if not r else "real objects only", ls)
    ax.axhline(0, color=C_BASE, lw=1, ls=":")
    ax.set_title(f"how things look: MMD excess over baseline ({m.upper() if m == 'clip' else 'DINOv3'})", loc="left", fontsize=10)
ax = axs[5]
for m, mk in (("clip", "o"), ("dinov3", "s")):
    t = RR[(RR.model == m) & ~RR.real]
    for comp, col in (("BabyView vs VEDB", C_VEDB), ("within BabyView", C_BASE)):
        tt = t[t.comparison == comp].set_index("group").loc[BINS]
        ax.plot(x, 1 - tt.rho, "-", marker=mk, color=col, ms=5, lw=1.4, label=f"{comp}, {m}")
ax.set_title("between-category structure: 1 - RDM rho", loc="left", fontsize=10)
for ax in axs:
    ax.set_xticks(x, BINS, fontsize=8); ax.legend(fontsize=7, frameon=False)
fig.suptitle("Does children's visual experience approach adults' with age?", x=.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "age_trends.png", dpi=200, bbox_inches="tight"); plt.show()
def trend(t, ycol):
    t = t.set_index("group").loc[BINS]
    return stats.spearmanr(range(len(BINS)), t[ycol]).statistic
print("Spearman of each divergence with age bin (4 bins; descriptive):")
print(" JS category, BabyView vs VEDB:", round(trend(J[(J.level == 'category') & (J.scope == 'all frames') & (J.comparison == 'BabyView vs VEDB')], 'js'), 2))
print(" JS category, excess over within-BabyView:", round(trend(J[(J.level == 'category') & (J.scope == 'all frames') & (J.comparison == 'excess over within BabyView')], 'js'), 2))
for m in ("clip", "dinov3"):
    print(f" MMD excess {m}:", round(trend(AG[(AG.model == m) & ~AG.real], 'median_excess'), 2),
          f" 1-rho {m}:", round(trend(RR[(RR.model == m) & ~RR.real & (RR.comparison == 'BabyView vs VEDB')].assign(d=lambda t: 1 - t.rho), 'd'), 2))"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "bv_vedb_analysis.ipynb")
print("wrote", HERE / "bv_vedb_analysis.ipynb")
