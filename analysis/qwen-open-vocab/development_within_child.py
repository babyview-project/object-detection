#!/usr/bin/env python3
"""Within-child robustness checks for the development results (manuscript: "The layout of the child's view changes with
age, while its category structure is stable"; Fig. 8C).

The age-bin comparisons of Fig. 8C pool different children in each bin (15-29 per bin), so a trend or its absence could
come from who was recorded at which age. Here every measure is computed per child x age window (3-month windows of age,
>= MIN_FRAMES frames) and modeled like the scene statistics: y ~ age_within + age_between, random intercept and
age_within slope per child (statsmodels MixedLM, REML; random intercept only if the slope model fails). Checks: the
same model with log(objects) as a covariate, and a child fixed-effects OLS slope with child-clustered SEs.

Measures (DINOv3, core categories, VEDB = adults' reference; "excess" = vs VEDB minus vs the other BabyView children,
leave-one-child-out, on the same categories, so category composition and sampling noise cancel):
  rdm_rho_vedb        Spearman between the child-window RDM and VEDB's RDM (categories with >= MIN_OBJ objects)
  rdm_rho_excess      rho with the other children's RDM minus rho with VEDB's (how much less adult-like than child-typical)
  delta_sd            domain clustering of the child-window RDM (between - within domain distance / SD)
  mmd_excess          linear-kernel unbiased MMD^2 to VEDB minus to the other children, mean over categories
  js_vedb / js_excess Jensen-Shannon divergence of the core-category distribution (frame presence) to VEDB's,
                      on SUB_FRAMES random frames per window (equal sample size), all frames and home-scene frames
Object size per category: log box area per (child, month, category) median, same mixed model per category (real
objects too); pooled within-category slope (child x category fixed effects); and the overall median-size decline
split into within-category change vs change in which categories are seen.

  QOV_LABELS=renamed python development_within_child.py
Outputs: results/bv_vedb_renamed/dev_within_child_{measures,slopes,size_by_category,size_summary}.csv,
         figures/bv_vedb_renamed/dev_within_child.png, dev_size_by_category.png
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial.distance import pdist, jensenshannon

import variant as V

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
D = V.out_dir(HERE / "out" / "private" / "dev_within_child")
RES = V.out_dir(HERE / "results" / "bv_vedb"); FIG = V.out_dir(HERE / "figures" / "bv_vedb")
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
WIN, MIN_FRAMES, SUB_FRAMES, MIN_OBJ, MIN_CATS = 3, 1000, 1000, 15, 40
import fig_style as FS
FS.apply()
C_BV = FS.DS_COLOR["bv"]


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def zrows(X): return (X - X.mean(0)) / (X.std(0) + 1e-10)
def rdm(M): return pdist(zrows(M), "cosine")


def delta_sd(d, sem):
    n = len(sem); i, j = np.triu_indices(n, 1); same = sem[i] == sem[j]
    return (d[~same].mean() - d[same].mean()) / d.std()


def u2(S, n):
    """Unbiased ||mean||^2 from the sum of n unit vectors."""
    return ((S * S).sum(-1) - n) / np.maximum(n * (n - 1), 1)


def mixed(df, y, extra=""):
    """y ~ aw + ab (+ extra), random intercept + aw slope per child; returns slope/yr, CI, p, model used."""
    d = df.dropna(subset=[y]).copy()
    sd = d[y].std(); d["_y"] = (d[y] - d[y].mean()) / sd
    f = "_y ~ aw + ab" + extra
    for re in ("~aw", None):
        try:
            m = smf.mixedlm(f, d, groups=d.unit, re_formula=re).fit(reml=True, method=["lbfgs", "powell"])
            if not m.converged and re is not None:
                continue
            b, se = m.params["aw"], m.bse["aw"]
            return dict(slope=b * sd, lo=(b - 1.96 * se) * sd, hi=(b + 1.96 * se) * sd, p=m.pvalues["aw"],
                        between=m.params["ab"] * sd, p_between=m.pvalues["ab"], re="slope" if re else "intercept",
                        n=len(d), children=d.unit.nunique())
        except Exception:
            continue
    return dict(slope=np.nan, n=len(d), children=d.unit.nunique(), re="failed")


def child_fe(df, y):
    """Within-child OLS slope (child-demeaned), SE clustered by child."""
    d = df.dropna(subset=[y])
    d = d[d.groupby("unit").unit.transform("size") >= 2]
    yy = d[y] - d.groupby("unit")[y].transform("mean"); x = d.age - d.groupby("unit").age.transform("mean")
    b = (x * yy).sum() / (x * x).sum(); e = yy - b * x
    g = (x * e).groupby(d.unit).sum(); G = len(g)
    se = np.sqrt((g ** 2).sum() * G / (G - 1)) / (x * x).sum()
    return b, se, 2 * stats.t.sf(abs(b / se), G - 1)


def add_age(df):
    mu = df.groupby("unit").age.transform("mean")
    df["aw"] = (df.age - mu) / 12; df["ab"] = (mu - df.age.mean()) / 12
    return df


# ---------------------------------------------------------------------------------------------------------- measures
def window_measures():
    z = np.load(D / "sums_dinov3.npz")
    keys, cats = z["keys"], list(z["categories"])
    unit = np.array([k.split("|")[0] for k in keys]); month = np.array([int(k.split("|")[1]) for k in keys])
    win = month // WIN
    sem = np.array([DOM.get(c, "other") for c in cats])
    fr = pd.read_parquet(D / "frames.parquet"); bx = pd.read_parquet(D / "boxes.parquet")
    fr["win"] = fr.month // WIN; bx["win"] = bx.month // WIN
    # VEDB reference: means and frame-presence distribution on the same core categories
    zv = np.load(V.out_dir(HERE / "out" / "private" / "open_vedb") / "sums_dinov3.npz")
    vi = [list(zv["categories"]).index(c) for c in cats]
    SV, NV = zv["sum_all"][:, vi].sum(0).astype(np.float64), zv["n_all"][:, vi].sum(0)
    pv = pd.read_csv(V.out_dir(HERE / "out" / "private" / "bv_vedb") / "presence.csv.gz")
    pv = pv[pv.dataset == "vedb"]
    pres_v = {h: pv[pv.home | (h == "all")].groupby("category").frames.sum().reindex(cats).fillna(0).to_numpy()
              for h in ("all", "home")}
    S_all = z["sum_all"].astype(np.float64); N_all = z["n_all"]
    S_tot, N_tot = S_all.sum(0), N_all.sum(0)
    rng = np.random.default_rng(0)
    cidx = {c: i for i, c in enumerate(cats)}
    bx["ci"] = bx.category.map(cidx)
    rows = []
    for (u, w), f in fr.groupby(["unit", "win"]):
        if len(f) < MIN_FRAMES:
            continue
        sel = (unit == u) & (win == w)
        S, N = S_all[sel].sum(0), N_all[sel].sum(0)
        So = S_tot - S_all[unit == u].sum(0); No = N_tot - N_all[unit == u].sum(0)   # other children
        ok = (N >= MIN_OBJ) & (NV >= MIN_OBJ) & (No >= MIN_OBJ)
        r = dict(unit=u, win=w, age=f.age_months.mean(), frames=len(f), objects=int(N.sum()), n_cats=int(ok.sum()))
        if ok.sum() >= MIN_CATS:
            dc, dv, do = (rdm(X[ok] / n[ok, None]) for X, n in ((S, N), (SV, NV), (So, No)))
            r["rdm_rho_vedb"] = stats.spearmanr(dc, dv).statistic
            r["rdm_rho_others"] = stats.spearmanr(dc, do).statistic
            r["rdm_rho_excess"] = r["rdm_rho_others"] - r["rdm_rho_vedb"]
            r["delta_sd"] = delta_sd(dc, sem[ok])
        m = (N >= MIN_OBJ) & (NV >= 2) & (No >= 2)
        if m.sum() >= 10:
            mc, mv, mo = S[m] / N[m, None], SV[m] / NV[m, None], So[m] / No[m, None]
            mmd_v = u2(S[m], N[m]) + u2(SV[m], NV[m]) - 2 * (mc * mv).sum(1)
            mmd_o = u2(S[m], N[m]) + u2(So[m], No[m]) - 2 * (mc * mo).sum(1)
            r["mmd_vedb"], r["mmd_excess"] = mmd_v.mean(), (mmd_v - mmd_o).mean()
        # category distribution on SUB_FRAMES random frames (equal sample size for every window)
        b = bx[(bx.unit == u) & (bx.win == w)].drop_duplicates(["video_id", "frame_number", "ci"])
        fk = f.video_id + "/" + f.frame_number.astype(str); bk = b.video_id + "/" + b.frame_number.astype(str)
        for h in ("all", "home"):
            pool = fk[f.home].to_numpy() if h == "home" else fk.to_numpy()
            if len(pool) < (SUB_FRAMES if h == "all" else 300):
                continue
            n_sub = SUB_FRAMES if h == "all" else 300
            js = []
            for _ in range(20):
                s = set(rng.choice(pool, n_sub, replace=False))
                cnt = np.bincount(b.ci[bk.isin(s).to_numpy()], minlength=len(cats))
                js.append(jensenshannon(cnt, pres_v[h], base=2) ** 2)
            r[f"js_vedb_{h}"] = np.mean(js)
        rows.append(r)
    return add_age(pd.DataFrame(rows))


MEAS = {"rdm_rho_vedb": "RDM ρ with adults (VEDB)", "rdm_rho_others": "RDM ρ with other children",
        "rdm_rho_excess": "RDM ρ: other children − adults", "delta_sd": "domain clustering (Δ/SD)",
        "mmd_excess": "appearance: MMD² to adults − to other children", "js_vedb_all": "category distribution: JS to adults",
        "js_vedb_home": "category distribution, home scenes: JS to adults"}


# ------------------------------------------------------------------------------------------------------- object size
def size_by_category():
    bx = pd.read_parquet(D / "boxes.parquet")
    bx = bx[bx.area > 0].assign(la=lambda d: np.log(d.area), dom=lambda d: d.category.map(DOM).fillna("other"))
    out, summ = [], []
    for fmt in ("all", "real"):
        b = bx if fmt == "all" else bx[bx.format == "real"]
        cm = b.groupby(["unit", "month", "category"]).agg(la=("la", "median"), n=("la", "size"), age=("age_months", "mean")).reset_index()
        cm = cm[cm.n >= 3]
        rows = []
        for c, g in cm.groupby("category"):
            g = g[g.groupby("unit").unit.transform("size") >= 2]
            if g.unit.nunique() < 5 or len(g) < 20:
                continue
            r = mixed(add_age(g.copy()), "la")
            b_, se_, p_ = child_fe(g, "la")
            rows.append(dict(format=fmt, category=c, domain=DOM.get(c, "other"), child_months=len(g), children=g.unit.nunique(),
                             boxes=int(g.n.sum()), pct_per_year=np.expm1(r["slope"]), lo=np.expm1(r.get("lo", np.nan)),
                             hi=np.expm1(r.get("hi", np.nan)), p=r.get("p", np.nan), re=r["re"],
                             fe_pct_per_year=np.expm1(b_ * 12), fe_p=p_))
        t = pd.DataFrame(rows); t["q"] = bh(t.p.fillna(1)); out.append(t)
        # pooled within-category slope: child x category means removed, random age slope per child
        g = cm.copy(); g["cc"] = g.unit + "|" + g.category
        g = g[g.groupby("cc").cc.transform("size") >= 2]
        g["y"] = g.la - g.groupby("cc").la.transform("mean"); g["x"] = (g.age - g.groupby("cc").age.transform("mean")) / 12
        bfe = (g.x * g.y).sum() / (g.x ** 2).sum(); e = g.y - bfe * g.x; sc = (g.x * e).groupby(g.unit).sum(); G = len(sc)
        sefe = np.sqrt((sc ** 2).sum() * G / (G - 1)) / (g.x ** 2).sum()
        m = smf.mixedlm("y ~ x", g, groups=g.unit, re_formula="~x").fit(reml=True, method=["lbfgs", "powell"])
        # overall trend with and without changes in category mix: per child-month mean log area, raw vs reweighted
        # to the overall category mix (mean over categories of the category's mean, weights = overall box share)
        wts = b.category.value_counts(normalize=True)
        raw = b.groupby(["unit", "month"]).agg(y=("la", "median"), age=("age_months", "mean"), n=("la", "size")).reset_index()
        cmw = cm.assign(w=cm.category.map(wts))
        fix = cmw.groupby(["unit", "month"]).apply(lambda d: np.average(d.la, weights=d.w)).rename("y_fix").reset_index()
        raw = add_age(raw.merge(fix, on=["unit", "month"])); raw = raw[raw.n >= 200]
        a, f_ = mixed(raw, "y"), mixed(raw, "y_fix")
        summ.append(dict(format=fmt, categories=len(t), share_decreasing=(t.pct_per_year < 0).mean(),
                         n_decr_q05=int(((t.q < .05) & (t.pct_per_year < 0)).sum()),
                         n_incr_q05=int(((t.q < .05) & (t.pct_per_year > 0)).sum()),
                         sign_test_p=stats.binomtest(int((t.pct_per_year < 0).sum()), len(t)).pvalue,
                         median_cat_pct_per_year=t.pct_per_year.median(),
                         pooled_within_cat_pct=np.expm1(bfe), pooled_lo=np.expm1(bfe - 1.96 * sefe),
                         pooled_hi=np.expm1(bfe + 1.96 * sefe), pooled_p=2 * stats.t.sf(abs(bfe / sefe), G - 1),
                         pooled_mixed_pct=np.expm1(m.params["x"]), pooled_mixed_p=m.pvalues["x"], pooled_mixed_converged=m.converged,
                         overall_median_pct=np.expm1(a["slope"]), overall_p=a["p"],
                         fixed_mix_pct=np.expm1(f_["slope"]), fixed_mix_p=f_["p"]))
    U = pd.DataFrame(summ)
    fr = pd.read_csv(V.out_dir(HERE / "out" / "private" / "bv_vedb") / "frames.csv.gz",
                     usecols=["dataset", "unit", "age_months", "median_area"])
    fr = fr[fr.dataset == "bv"].assign(month=lambda d: np.floor(d.age_months))
    fm = bx.groupby(["unit", "month", "video_id", "frame_number"]).area.median().rename("median_area").reset_index()
    for name, f in (("frame median, all boxes (manuscript measure)", fr), ("frame median, core-category boxes", fm)):
        c = f.groupby(["unit", "month"]).agg(y=("median_area", "mean"), age=("month", "mean"), n=("median_area", "size")).reset_index()
        c = add_age(c[c.n >= 200]); r = mixed(c, "y")
        U.loc[len(U), ["format", "overall_median_pct", "overall_p"]] = [name, r["slope"] / c.y.mean(), r["p"]]
    return pd.concat(out, ignore_index=True), U


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    W = window_measures()
    W.to_csv(RES / "dev_within_child_measures.csv", index=False)
    print(f"{W.unit.nunique()} children, {len(W)} child x {WIN}-month windows "
          f"({W.groupby('unit').size().ge(2).sum()} children with >= 2 windows)")
    rows = []
    for m, lab in MEAS.items():
        a = mixed(W, m); c = mixed(W, m, " + np.log(objects)"); b, se, p = child_fe(W, m)
        rows.append(dict(measure=m, label=lab, mean=W[m].mean(), **{k: a.get(k) for k in ("slope", "lo", "hi", "p", "between", "p_between", "re", "n", "children")},
                         slope_logobj=c.get("slope"), p_logobj=c.get("p"), fe_slope=b * 12, fe_p=p))
    S = pd.DataFrame(rows); S["q"] = bh(S.p.fillna(1))
    S.to_csv(RES / "dev_within_child_slopes.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.drop(columns=["measure"]).round(4).to_string(index=False))

    T, U = size_by_category()
    T.to_csv(RES / "dev_size_by_category.csv", index=False); U.to_csv(RES / "dev_size_summary.csv", index=False)
    print(U.round(4).T.to_string())
    print(T[T.format == "all"].groupby("domain").agg(categories=("category", "size"), median_pct=("pct_per_year", "median"),
                                                     share_decr=("pct_per_year", lambda s: (s < 0).mean())).round(3))

    # figure: per-measure child trajectories + mixed slope; size slopes per category
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.5)); axs = axs.ravel()
    for ax, (m, lab) in zip(axs, MEAS.items()):
        r = S.set_index("measure").loc[m]
        for u, x in W.dropna(subset=[m]).groupby("unit"):
            ax.plot(x.age, x[m], "-o", color=C_BV, alpha=.35, ms=2.5, lw=.8)
        xs = np.linspace(W.age.min(), W.age.max(), 2)
        ax.plot(xs, W[m].mean() + r.slope / 12 * (xs - W.age.mean()), color="#0d366b", lw=2.5)
        ax.set_title(f"{lab}\n{r.slope:+.3f}/yr [{r.lo:+.3f}, {r.hi:+.3f}], q = {r.q:.2g}", loc="left", fontsize=9)
        ax.set_xlabel("age (months)", fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax = axs[-1]; t = T[T.format == "all"].sort_values("pct_per_year")
    ax.scatter(np.arange(len(t)), 100 * t.pct_per_year, s=8, c=np.where(t.q < .05, "#0d366b", "#9bb7d4"))
    ax.axhline(0, color="k", lw=.6)
    ax.set_title(f"object size, within-child change per category\n{(t.pct_per_year < 0).mean():.0%} of {len(t)} categories shrink "
                 f"(dark: q < .05)", loc="left", fontsize=9)
    ax.set_xlabel("categories (sorted)", fontsize=8); ax.set_ylabel("% change in box area per year", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Within-child development checks: child x {WIN}-month windows (lines: one child); thick: mixed-model "
                 "within-child slope (random intercept + age slope per child)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(FIG / "dev_within_child.png", dpi=200, bbox_inches="tight")
    print("saved", FIG / "dev_within_child.png")


if __name__ == "__main__":
    main()
