"""Overview figure for the response letter: concrete examples of what is shared with adults and what is child-specific.

A  Which categories: children/adults frequency ratio (BabyView vs VEDB) for example categories, 95% CIs.
B  Toys and pictures: share of objects that are depictions, BabyView vs VEDB, for common animals and vehicles.
C  What depictions add: domain clustering (Δ/SD, DINOv3) for furniture (shared), animals and vehicles, with all
   formats and real objects only.

Inputs: results/bv_vedb_renamed_core/category_frequency_ratio.csv,
        results/crossdataset_renamed_core/depiction_gap_bv_vs_vedb.csv,
        results/crossdataset_renamed_core/cluster_strength_by_domain_real.csv
Output: ../manuscript-2026/revision_private/figures/R0_overview_examples.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig_style as FS

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "manuscript-2026" / "revision_private" / "figures" / "R0_overview_examples"

CHILD_CATS = ["elephant", "block", "train", "cat", "dog", "teddy bear", "couch", "ball"]
ADULT_CATS = ["tree", "building", "backpack", "laptop", "desk", "keyboard"]
DEPICT_CATS = ["bus", "train", "truck", "car", "duck", "fish", "bird", "dog", "cat"]
CLUSTER_DOMAINS = [("furniture_rooms", "furniture"), ("animals", "animals"), ("vehicles", "vehicles")]
CLUSTER_SETS = ["bv", "bv_real", "vedb_real", "things"]
CLUSTER_LABEL = {"bv": "BabyView, all formats", "bv_real": "BabyView, real only", "vedb_real": "VEDB, real only",
                 "things": "THINGS"}


def times(log2):
    r = 2 ** abs(log2)
    return f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"


def panel_frequency(ax):
    fr = pd.read_csv(RES / "bv_vedb_renamed_core" / "category_frequency_ratio.csv", index_col=0)
    d = fr.loc[CHILD_CATS + ADULT_CATS].sort_values("log2_ratio")
    y = np.arange(len(d))
    col = [FS.DS_COLOR["bv"] if v > 0 else FS.DS_COLOR["vedb"] for v in d.log2_ratio]
    ax.barh(y, d.log2_ratio, color=col, height=0.62)
    ax.errorbar(d.log2_ratio, y, xerr=[d.log2_ratio - d.lo, d.hi - d.log2_ratio], fmt="none", ecolor=FS.MUTED,
                elinewidth=1, capsize=0)
    ax.axvline(0, color=FS.BASELINE, lw=1)
    for yi, (v, hi, lo) in enumerate(zip(d.log2_ratio, d.hi, d.lo)):
        x = hi + 0.25 if v > 0 else lo - 0.25
        ax.text(x, yi, times(v), va="center", ha="left" if v > 0 else "right", fontsize=8, color=FS.INK)
    ax.set_yticks(y, d.index)
    ticks = [-6, -4, -2, 0, 2, 4, 6]
    ax.set_xticks(ticks, [f"{2 ** abs(t):.0f}×" if t else "1×" for t in ticks])
    ax.set_xlim(-8.6, 8.6)
    ax.set_xlabel("← more often for adults      more often for children →", fontsize=9)
    ax.set_title("A  Which categories they see", loc="left")


def panel_depictions(ax):
    g = pd.read_csv(RES / "crossdataset_renamed_core" / "depiction_gap_bv_vs_vedb.csv").set_index("category")
    d = g.loc[DEPICT_CATS[::-1]]
    y = np.arange(len(d))
    for yi, (b, v) in enumerate(zip(d.bv_depicted, d.vedb_depicted)):
        ax.plot([v * 100, b * 100], [yi, yi], color=FS.FAINT, lw=2, zorder=1)
    ax.scatter(d.vedb_depicted * 100, y, s=46, color=FS.DS_COLOR["vedb"], edgecolor="white", lw=1.5, zorder=3,
               label="VEDB (adults)")
    ax.scatter(d.bv_depicted * 100, y, s=46, color=FS.DS_COLOR["bv"], edgecolor="white", lw=1.5, zorder=3,
               label="BabyView (children)")
    for yi, b in enumerate(d.bv_depicted):
        ax.text(b * 100 + 3.5, yi, f"{b * 100:.0f}%", va="center", fontsize=8, color=FS.INK)
    ax.set_yticks(y, d.index)
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("objects seen as toys or pictures", fontsize=9)
    ax.legend(loc="lower left", fontsize=8, handletextpad=0.2)
    ax.set_title("B  How often as toys and pictures", loc="left")


def panel_clustering(ax):
    c = pd.read_csv(RES / "crossdataset_renamed_core" / "cluster_strength_by_domain_real.csv")
    c = c[c.model == "dinov3"].set_index(["dataset", "domain"])
    w = 0.19
    for k, ds in enumerate(CLUSTER_SETS):
        xs = np.arange(len(CLUSTER_DOMAINS)) + (k - 1.5) * (w + 0.02)
        rows = [c.loc[(ds, dom)] for dom, _ in CLUSTER_DOMAINS]
        v = np.array([r.delta_sd for r in rows])
        lo = np.array([r.ci_low_sd for r in rows]); hi = np.array([r.ci_high_sd for r in rows])
        ax.bar(xs, v, width=w, color=FS.DS_COLOR[ds], label=CLUSTER_LABEL[ds])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=FS.MUTED, elinewidth=1, capsize=0)
        if ds in ("bv", "bv_real"):
            for x, val, h in zip(xs, v, hi):
                ax.text(x, h + 0.12, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color=FS.INK)
    ax.axhline(0, color=FS.BASELINE, lw=1)
    ax.set_xticks(range(len(CLUSTER_DOMAINS)), [lab for _, lab in CLUSTER_DOMAINS])
    ax.set_ylabel("domain clustering (Δ/SD)")
    ax.set_ylim(-0.3, 5.6)
    ax.legend(loc="upper left", fontsize=8, handlelength=1.0)
    ax.set_title("C  What depictions add", loc="left")


def main():
    FS.apply(base=10)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.1, 0.95, 1.05]})
    panel_frequency(axes[0])
    panel_depictions(axes[1])
    panel_clustering(axes[2])
    fig.tight_layout(w_pad=2.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
