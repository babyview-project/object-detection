#!/usr/bin/env python3
"""Figures for controls_analysis.py (results/controls[_renamed]/ -> figures/controls[_renamed]/).

  geometry_matching.png       BabyView and VEDB AUC before / after geometry matching, and their RDM correlation

  QOV_LABELS=renamed python controls_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import variant as V

HERE = Path(__file__).resolve().parent
import os as _os
_SFX = "_core" if _os.environ.get("QOV_CATS") == "core" else ""      # QOV_CATS=core: the core-list results
RES = V.out_dir(HERE / "results" / "controls"); RES = RES.with_name(RES.name + _SFX)
FIG = V.out_dir(HERE / "figures" / "controls"); FIG = FIG.with_name(FIG.name + _SFX); FIG.mkdir(parents=True, exist_ok=True)
import fig_style as FS                                   # manuscript style: Lato, bold labels / legends, dataset colours
FS.apply()
COLOR = dict(FS.DS_COLOR)
LABEL = {"bv": "BabyView", "bv_real": "BabyView, real", "vedb": "VEDB", "vedb_real": "VEDB, real", "things": "THINGS"}
MLABEL = {"clip": "CLIP", "dinov3": "DINOv3"}


def geom_fig():
    g = pd.read_csv(RES / "geom_summary.csv")
    g = g[(g.analysis == "matching")]
    gd = pd.read_csv(RES / "geom_differences.csv"); gd = gd[(gd.analysis == "matching") & (gd.metric == "auc") & (gd.pair == "bv - vedb")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for col, part in enumerate(("all", "real")):
        ax = axs[col]; d = g[g.part == part]
        for i, m in enumerate(("clip", "dinov3")):
            for j, cond in enumerate(("unmatched", "matched")):
                for k, ds in enumerate(("bv", "vedb")):
                    v = d[(d.model == m) & (d.level == cond) & (d.dataset == ds)].iloc[0]
                    x = i * 2.6 + j * 1.0 + (k - .5) * .3
                    ax.errorbar([x], [v.auc], yerr=[[v.auc - v.auc_ci_lo], [v.auc_ci_hi - v.auc]], fmt="o", color=COLOR[ds],
                                mfc=COLOR[ds] if cond == "matched" else "white", ms=6, capsize=2)
                x0 = i * 2.6 + j * 1.0; top = d[(d.model == m) & (d.level == cond)].auc_ci_hi.max() + .008
                r_ = gd[(gd.model == m) & (gd.part == part) & (gd.level == cond)]
                if len(r_): FS.bracket(ax, x0 - .15, x0 + .15, top, FS.stars(r_.p.iloc[0]), h=.003, fontsize=8)
        rho = {(m, c): d[(d.model == m) & (d.level == c)].bv_vedb_rho.iloc[0] for m in ("clip", "dinov3") for c in ("unmatched", "matched")}
        ax.set_xticks([0, 1, 2.6, 3.6], [f"{MLABEL[m]}\n{c}\nρ = {rho[m, c]:.2f}" for m in ("clip", "dinov3")
                                         for c in ("unmatched", "matched")], fontsize=8)
        ax.set_title(f"{'All formats' if part == 'all' else 'Real objects only'} ({int(d.n_categories.iloc[0])} categories)", loc="left", fontsize=10)
        ax.set_ylabel("domain clustering, AUC")
    for ds in ("bv", "vedb"):
        axs[0].errorbar([], [], fmt="o", color=COLOR[ds], label=LABEL[ds])
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Crop geometry: BabyView and VEDB before / after matching on size x vertical position x truncation "
                 "(ρ = BabyView-VEDB RDM correlation; brackets: BabyView vs VEDB, bootstrap p)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "geometry_matching.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def prevalence_fig():
    """Category frequency by rank: BabyView / VEDB frame prevalence, ImageNet / LVIS exemplar share (open categories)."""
    import controls_analysis as ca
    T = ca.tables("clip")
    def frame_prev(root):
        fbc = pd.read_csv(root / "frames_by_child.csv"); tot = pd.read_csv(root / "frames_total.csv").frames.sum()
        return fbc.groupby("cat").frames_all.sum() / tot
    P = {"bv": frame_prev(ca.OPEN), "vedb": frame_prev(ca.OPEN_VEDB)}
    for d in ("imagenet", "lvis"):
        n = T[d].attrs["n"].astype(float); P[d] = n / n.sum()
    rho = pd.read_csv(RES / "prevalence_spearman.csv").set_index("versus")
    col = {d: COLOR[d] for d in ("bv", "vedb", "imagenet", "lvis")}
    lab = {"bv": "BabyView (share of frames)", "vedb": "VEDB (share of frames)", "imagenet": "ImageNet (share of images)",
           "lvis": "LVIS (share of boxes)"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for d in P:
        v = np.sort(P[d][P[d] > 0].values)[::-1]
        axs[0].loglog(np.arange(1, len(v) + 1), v, color=col[d], lw=2, label=f"{lab[d]}, {len(v)} categories")
    axs[0].set_xlabel("rank"); axs[0].set_ylabel("frequency"); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("Category frequency by rank, each dataset's own ranking", loc="left", fontsize=10)
    for d in ("vedb", "lvis", "imagenet"):
        cs = sorted(set(P["bv"][P["bv"] > 0].index) & set(P[d][P[d] > 0].index))
        r = rho.loc[d]
        axs[1].scatter(P["bv"].loc[cs], P[d].loc[cs], s=6, alpha=.45, color=col[d], lw=0,
                       label=f"{lab[d].split(' (')[0]}: ρ = {r.spearman:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}], {len(cs)}")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("BabyView frame prevalence"); axs[1].set_ylabel("frequency in the other dataset")
    axs[1].legend(frameon=False, fontsize=8, markerscale=2.5, loc="lower right")
    axs[1].set_title("Same categories: BabyView vs each dataset (Spearman)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "prevalence_by_rank.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    prevalence_fig(); print("wrote prevalence_fig")
    for f, need in ((geom_fig, "geom_summary.csv"),):
        if (RES / need).exists():
            f(); print("wrote", f.__name__)
        else:
            print("skipped", f.__name__, "(no", need, ")")
