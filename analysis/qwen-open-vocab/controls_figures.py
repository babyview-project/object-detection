#!/usr/bin/env python3
"""Figures for controls_analysis.py (results/controls[_renamed]/ -> figures/controls[_renamed]/).

  split_within_vs_cross.png   domain-clustering AUC within half vs across halves (subject split), per dataset
  same_room_auc.png           AUC from exemplars of one room / activity: BabyView same label, pooled, THINGS
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


def split_fig():
    w = pd.read_csv(RES / "split_within_vs_cross.csv")
    s = pd.read_csv(RES / "split_summary.csv")
    w = w[w.comparison == "bv_vedb_things"]; s = s[s.comparison == "bv_vedb_things"]
    sd = pd.read_csv(RES / "split_differences.csv")
    sd = sd[(sd.comparison == "bv_vedb_things") & (sd.rdm == "cross-half") & (sd.metric == "auc")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axs, ("clip", "dinov3")):
        dss = [d for d in LABEL if d in set(w[w.model == m].dataset)]
        for i, d in enumerate(dss):
            r = w[(w.model == m) & (w.dataset == d)].iloc[0]
            c = s[(s.model == m) & (s.dataset == d)].iloc[0]
            ax.plot([i - .12, i + .12], [r.auc_within, r.auc_cross], color=COLOR[d], lw=1.5)
            ax.scatter([i - .12], [r.auc_within], s=40, color="white", edgecolor=COLOR[d], lw=1.5, zorder=3)
            ax.errorbar([i + .12], [r.auc_cross], yerr=[[r.auc_cross - c.auc_ci_lo], [c.auc_ci_hi - r.auc_cross]],
                        fmt="o", ms=6, color=COLOR[d], capsize=2, zorder=3)
        ax.axhline(.5, color="#8a8984", lw=.8, ls=":")
        top = s[s.model == m].auc_ci_hi.max() + .015
        for k, (a, b) in enumerate([(a, b) for i_, a in enumerate(dss) for b in dss[i_ + 1:]]):   # across-halves AUC, bootstrap p
            r_ = sd[(sd.model == m) & (sd.pair == f"{a} - {b}")]
            if len(r_): FS.bracket(ax, dss.index(a) + .12, dss.index(b) + .12, top + k * .025, FS.stars(r_.p.iloc[0]), h=.004)
        ax.set_xticks(range(len(dss)), [LABEL[d] for d in dss], rotation=20, ha="right")
        ax.set_title(f"{MLABEL[m]} ({int(s[s.model == m].n_categories.iloc[0])} categories)", loc="left")
    axs[0].set_ylabel("domain clustering, AUC")
    axs[0].scatter([], [], s=40, color="white", edgecolor="#52514e", label="within half")
    axs[0].scatter([], [], s=40, color="#52514e", label="across halves (95% CI)")
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Subject split: categories compared from different children (BabyView), sessions (VEDB) or images (THINGS)\n"
                 "brackets: across-halves AUC, bootstrap p (* < .05, ** < .01, *** < .001)",
                 x=.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "split_within_vs_cross.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def room_fig():
    r = pd.read_csv(RES / "room_summary.csv")
    r = r[(r.part == "all") & ~r.label.str.contains(" x ")]
    rd = pd.read_csv(RES / "room_differences.csv"); rd = rd[(rd.part == "all") & (rd.metric == "auc")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for row, kind in enumerate(("room", "act")):
        for col, m in enumerate(("clip", "dinov3")):
            ax = axs[row, col]; d = r[(r.model == m) & (r.kind == kind)]
            labels = list(dict.fromkeys(d.label))
            for i, L in enumerate(labels):
                g = d[d.label == L].set_index("dataset")
                for j, (ds, c, mk) in enumerate(((f"bv_all same {kind}", COLOR["bv"], "o"), ("bv_all pooled", FS.BASELINE, "s"),
                                                 ("things", COLOR["things"], "D"))):
                    if ds not in g.index: continue
                    x = i + (j - 1) * .2; v = g.loc[ds]
                    ax.errorbar([x], [v.auc], yerr=[[v.auc - v.auc_ci_lo], [v.auc_ci_hi - v.auc]], fmt=mk, color=c, ms=6, capsize=2)
                ax.text(i, .505, f"n={int(g.n_categories.iloc[0])}", ha="center", fontsize=7, color="#52514e")
                top = g.auc_ci_hi.max() + .01                  # same room vs pooled, same room vs THINGS (bootstrap p)
                for k, (other, x2) in enumerate(((f"bv_all pooled", i), ("things", i + .2))):
                    r_ = rd[(rd.model == m) & (rd.kind == kind) & (rd.label == L) & (rd.pair == f"bv_all same {kind} - {other}")]
                    if len(r_): FS.bracket(ax, i - .2, x2, top + k * .022, FS.stars(r_.p.iloc[0]), h=.004, fontsize=7)
            ax.set_xticks(range(len(labels)), labels, fontsize=8)
            ax.axhline(.5, color="#8a8984", lw=.8, ls=":")
            ax.set_title(f"{MLABEL[m]}, by {'room' if kind == 'room' else 'activity'}", loc="left", fontsize=10)
    for c, mk, l in ((COLOR["bv"], "o", "BabyView, same room / activity"), (FS.BASELINE, "s", "BabyView, pooled (same categories)"),
                     (COLOR["things"], "D", "THINGS")):
        axs[0, 0].errorbar([], [], fmt=mk, color=c, label=l)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3)
    for ax in axs[:, 0]: ax.set_ylabel("domain clustering, AUC")
    fig.suptitle("Room and activity held fixed: category means from one room / activity only (95% CIs; brackets: same room / activity "
                 "vs pooled and vs THINGS, * p < .05, ** < .01, *** < .001)", x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(FIG / "same_room_auc.png", dpi=200, bbox_inches="tight"); plt.close(fig)


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
    for f, need in ((split_fig, "split_summary.csv"), (room_fig, "room_summary.csv"), (geom_fig, "geom_summary.csv")):
        if (RES / need).exists():
            f(); print("wrote", f.__name__)
        else:
            print("skipped", f.__name__, "(no", need, ")")
