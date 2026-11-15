#!/usr/bin/env python3
"""Within-child scene statistics (bv_vedb_analysis.ipynb section 1b) with child-month data density taken into account.

Section 1b fits child-month means (>= 200 frames) unweighted, although child-months hold 200-15,000 frames from 2-115
videos. Here the same within / between-child age decomposition (y ~ age_within + age_between, random intercept and
age_within slope per child, REML) is fitted three ways:

  unweighted     residual variance s2                         (= section 1b's mixed model)
  frame weights  residual variance s2 / frames                (lmer(weights = frames)); lets the densest months dominate
  meta-analytic  residual variance tau2 + se2                 se2 = sampling variance of the child-month mean, clustered
                                                              by video (frames within a video are not independent);
                                                              precise months count more, but only up to tau2, the real
                                                              month-to-month variation

Report the frame-weighted fit. Frame counts are unrelated to the measures (|rank r| <= .15), but the video-clustered
standard errors are not: for rare, bursty measures a month's SE tracks its own mean (rank r = .97 screens, .86 faces,
.61 people), so the meta-analytic fit down-weights exactly the months with many screens / faces and biases those slopes
toward 0. It is kept as a check for the measures where that coupling is weak (objects, categories, height).
Wald 95% CIs and p; BH-FDR across measures within each fit.
Figure: one panel per measure; points = child-month means, area by frames; thin lines = each child's own frame-weighted
linear trend over the ages it was recorded (children with >= 3 child-months); thick line = the frame-weighted
within-child slope.

  python scene_within_child_density.py                      # original labels
  QOV_LABELS=renamed python scene_within_child_density.py   # renamed labels
Outputs: results/bv_vedb[_renamed]/scene_statistics_within_child_density.csv,
         figures/bv_vedb[_renamed]/scene_statistics_within_child_density.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import variant as V

HERE = Path(__file__).resolve().parent
D = V.out_dir(HERE / "out" / "private" / "bv_vedb")
FIG = V.out_dir(HERE / "figures" / "bv_vedb"); FIG.mkdir(parents=True, exist_ok=True)
RES = V.out_dir(HERE / "results" / "bv_vedb"); RES.mkdir(parents=True, exist_ok=True)
import fig_style as FS                      # manuscript style: Lato, bold labels / legends, dataset colours
FS.apply()
C_BV, C_VEDB = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"]
M1 = {"n_obj": "objects / frame", "n_distinct": "distinct categories / frame", "n_person": "people / frame",
      "n_hand": "hands / frame", "n_face": "face parts / frame", "n_screen": "objects shown on screens / frame",
      "dep_share": "share of objects depicted", "median_area": "median object size (frame share)",
      "mean_cy": "mean object height (0 top, 1 bottom)"}
MW = list(M1)
FITS = ("unweighted", "frame weights", "meta-analytic")


def child_months(fb):
    """Child-month means, frame counts and video-clustered sampling variance of each mean, per measure."""
    fb = fb.copy()
    fb["month"] = np.floor(fb.age_months)
    rows = []
    for (u, mo), f in fb.groupby(["unit", "month"]):
        if len(f) < 200:
            continue
        r = dict(unit=u, month=mo, frames=len(f), videos=f.video_id.nunique())
        for m in MW:
            g = f[["video_id", m]].dropna()
            n = len(g)
            mean = g[m].mean()
            # Cluster-robust (by video) variance of the frame mean: sum_v (sum of residuals in v)^2 / n^2 * V / (V - 1).
            s = (g[m] - mean).groupby(g.video_id).sum()
            k = len(s)
            r[m], r[m + "__se2"], r[m + "__n"] = mean, (s ** 2).sum() / n ** 2 * k / (k - 1) if k > 1 else np.nan, n
        rows.append(r)
    cmo = pd.DataFrame(rows)
    mu = cmo.groupby("unit").month.transform("mean")
    cmo["mw"] = (cmo.month - mu) / 12                         # years
    cmo["mb"] = (mu - cmo.month.mean()) / 12
    return cmo


def reml(y, X, groups, Z, a, b):
    """REML LMM, random effects (cols of Z) per group; residual variance s2 * a + b (a, b given per observation)."""
    q = Z.shape[1]
    idx = [np.flatnonzero(groups == g) for g in pd.unique(groups)]

    def unpack(th):
        L = np.zeros((q, q)); L[np.tril_indices(q)] = th[:-1]
        return L @ L.T, np.exp(2 * th[-1])

    def parts(th):
        G, s2 = unpack(th)
        A = np.zeros((X.shape[1],) * 2); c = np.zeros(X.shape[1]); ld = 0.0; Vis = []
        for i in idx:
            Vg = Z[i] @ G @ Z[i].T + np.diag(s2 * a[i] + b[i])
            Vi = np.linalg.inv(Vg); ld += np.linalg.slogdet(Vg)[1]
            A += X[i].T @ Vi @ X[i]; c += X[i].T @ Vi @ y[i]; Vis.append(Vi)
        return np.linalg.solve(A, c), A, ld, Vis

    def nll(th):
        beta, A, ld, Vis = parts(th)
        return 0.5 * (ld + np.linalg.slogdet(A)[1] + sum(float((y[i] - X[i] @ beta) @ Vi @ (y[i] - X[i] @ beta)) for i, Vi in zip(idx, Vis)))

    th0 = np.r_[0.7, 0.0, 0.3, np.log(0.5)]
    best = None
    for s in (1.0, 0.3, 2.0):
        r = optimize.minimize(nll, th0 * s, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-9, fatol=1e-12))
        r = optimize.minimize(nll, r.x, method="Powell", options=dict(maxiter=40000, xtol=1e-9, ftol=1e-12))
        best = r if best is None or r.fun < best.fun else best
    beta, A, _, _ = parts(best.x)
    G, s2 = unpack(best.x)
    return beta, np.sqrt(np.diag(np.linalg.inv(A))), G, s2


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def fit_measure(cmo, m, fit):
    d = cmo.dropna(subset=[m, m + "__se2"])
    sd = d[m].std()
    y = ((d[m] - d[m].mean()) / sd).to_numpy()                # z-scored for the optimizer; slopes rescaled below
    se2 = (d[m + "__se2"] / sd ** 2).to_numpy()
    X = np.column_stack([np.ones(len(d)), d.mw, d.mb])
    Z = np.column_stack([np.ones(len(d)), d.mw])
    one, zero = np.ones(len(d)), np.zeros(len(d))
    a, b = {"unweighted": (one, zero),
            "frame weights": (d[m + "__n"].mean() / d[m + "__n"].to_numpy(), zero),
            "meta-analytic": (one, se2)}[fit]
    beta, se, G, s2 = reml(y, X, d.unit.to_numpy(), Z, a, b)
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    out = dict(measure=m, label=M1[m], fit=fit, mean=d[m].mean(),
               within_per_year=beta[1] * sd, ci_lo=(beta[1] - 1.96 * se[1]) * sd, ci_hi=(beta[1] + 1.96 * se[1]) * sd,
               p_within=p[1], between_per_year=beta[2] * sd, p_between=p[2],
               sd_child_slope=np.sqrt(G[1, 1]) * sd, sd_residual=np.sqrt(s2) * sd,
               median_se=np.sqrt(np.median(d[m + "__se2"])))
    out["pct_change_per_year"] = out["within_per_year"] / out["mean"]
    return out


def child_trends(cmo, m):
    """Children with >= 3 child-months for measure m."""
    return {u: x for u, x in cmo.dropna(subset=[m]).groupby("unit") if len(x) >= 3}


def main():
    fr = pd.read_csv(D / "frames.csv.gz", dtype={"unit": str})
    fr["dep_share"] = fr.n_depicted / fr.n_obj.clip(lower=1)
    cmo = child_months(fr[fr.dataset == "bv"])
    vedb_mean = fr[fr.dataset == "vedb"].groupby("unit")[MW].mean().mean()
    print(f"labels: {V.LABELS}; {cmo.unit.nunique()} children, {len(cmo)} child-months; "
          f"frames per child-month {cmo.frames.min()}-{cmo.frames.max()} (median {cmo.frames.median():.0f}), "
          f"videos {cmo.videos.min()}-{cmo.videos.max()}")

    res = pd.DataFrame([fit_measure(cmo, m, f) for f in FITS for m in MW])
    res["q_within"] = res.groupby("fit").p_within.transform(bh)
    res["q_between"] = res.groupby("fit").p_between.transform(bh)
    res.to_csv(RES / "scene_statistics_within_child_density.csv", index=False)
    wide = res.pivot(index="label", columns="fit", values="within_per_year")[list(FITS)]
    wide.columns = [f"slope/yr {c}" for c in FITS]
    wide["q frame-wtd"] = res[res.fit == "frame weights"].set_index("label").q_within
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(wide.loc[[M1[m] for m in MW]].round(4))

    prim = res[res.fit == "frame weights"].set_index("measure")
    fmax = cmo.frames.max()
    fig, axs = plt.subplots(3, 3, figsize=(13, 11)); axs = axs.ravel()
    for ax, m in zip(axs, MW):
        r = prim.loc[m]
        for u, x in child_trends(cmo, m).items():
            ax.scatter(x.month, x[m], s=4 + 60 * x.frames / fmax, color=C_BV, alpha=.28, lw=0)
            b1, b0 = np.polyfit(x.month, x[m], 1, w=np.sqrt(x[m + "__n"]))
            xs = np.array([x.month.min(), x.month.max()])
            ax.plot(xs, b0 + b1 * xs, color=C_BV, alpha=.45, lw=.9)
        rest = cmo[~cmo.unit.isin(child_trends(cmo, m))]
        ax.scatter(rest.month, rest[m], s=4 + 60 * rest.frames / fmax, color=C_BV, alpha=.28, lw=0)
        xs = np.linspace(cmo.month.min(), cmo.month.max(), 50)
        ax.plot(xs, r["mean"] + r.within_per_year / 12 * (xs - cmo.month.mean()), color="#0d366b", lw=2.6)
        ax.axhline(vedb_mean[m], color=C_VEDB, lw=1.5)
        star = "**" if r.q_within < .01 else "*" if r.q_within < .05 else "n.s."
        ax.set_title(f"{M1[m]}\n{r.within_per_year:+.3g}/yr ({r.pct_change_per_year:+.0%}), {star}", loc="left", fontsize=9)
        ax.set_xlabel("age (months)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2, lw=.5)
    for f in (500, 2500, 10000):
        axs[0].scatter([], [], s=4 + 60 * f / fmax, color=C_BV, alpha=.5, lw=0, label=f"{f:,} frames")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Within-child change with age, weighted by frames. Points: child-month means (area by frames); "
                 "thin: each child's frame-weighted trend;\nthick: frame-weighted mixed-model within-child slope "
                 "(random intercept + age slope per child); orange: VEDB adults", x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIG / "scene_statistics_within_child_density.png", dpi=200, bbox_inches="tight")
    print("saved", FIG / "scene_statistics_within_child_density.png")


if __name__ == "__main__":
    main()
