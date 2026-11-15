#!/usr/bin/env python3
"""Depictions by category for one child: how often each Qwen category appears as a toy, drawing, photo or screen.

Every category with at least one depicted box (format toy / drawing / photo / screen), person excluded,
near-duplicate boxes dropped. Per category, left: depicted boxes stacked by format (counts); right: the
format mix of all its boxes, real included (shares). Sorted by depicted boxes.

Outputs (aggregate only, no frames) -> figures/<subject>/:
  depictions_by_category_all.pdf     every category with a depiction, 100 per page
  depictions_top100.png / .pdf       the 100 categories with the most depicted boxes
  depictions_by_domain.png / .pdf    format mix of all boxes per semantic domain
  depictions_domain_subplots.png/.pdf one panel per domain: its 20 most-depicted categories, stacked by format
  depictions_by_category.csv         counts and shares for every category with a depiction

  python make_depiction_figures.py [--subject S00320001 --tag subj_S00320001_t1_pp0 --per-page 100]
  python make_depiction_figures.py --children     # pooled over the complete children + one folder per child
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import sys
HERE = Path(__file__).resolve().parent
QP = HERE.parents[1] / "annotation" / "qwen_pilot"        # pipeline: loaders, embeddings, manifest
sys.path.insert(0, str(QP))
import qwen_data as qd  # noqa: E402
from make_subject_figures import COLORS, ORDER  # noqa: E402

HERE = Path(__file__).resolve().parent
DEPICT = ["toy", "drawing", "photo", "screen"]
import fig_style as FS  # noqa: E402  (make_subject_figures put this folder on the path and applied the style)
FMT = {k: FS.FMT_COLOR[k] for k in ("toy", "drawing", "photo", "screen", "real")}
INK, INK2 = FS.INK, FS.MUTED


def table(o: pd.DataFrame) -> pd.DataFrame:
    t = o.pivot_table(index="cat", columns="format", values="obj", aggfunc="size", fill_value=0)
    for f in DEPICT + ["real"]:
        if f not in t:
            t[f] = 0
    t = t[DEPICT + ["real"]]
    t["depicted"] = t[DEPICT].sum(axis=1)
    t["boxes"] = t.depicted + t.real
    t = t[t.depicted > 0]
    t["share_depicted"] = t.depicted / t.boxes
    t["frames"] = o.groupby("cat").unit_id.nunique().reindex(t.index)
    t["domain"] = o.groupby("cat").domain.first().reindex(t.index)
    return t.sort_values(["depicted", "boxes"], ascending=False)


def page(t: pd.DataFrame, title: str):
    n = len(t)
    h = max(4, .17 * n + 1.4)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, h), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(n)
    left = np.zeros(n)
    for f in DEPICT:
        a.barh(y, t[f].values, left=left, height=.78, color=FMT[f], edgecolor="white", linewidth=.4, label=f)
        left += t[f].values
    srt = np.sort(left)[::-1]
    xmax = srt[1] * 1.15 if n > 1 and srt[0] > 3 * srt[1] else srt[0]   # one runaway bar (bare "toy"): cut it at the edge
    for yi, (d, s) in enumerate(zip(t.depicted.values, t.share_depicted.values)):
        cut = d > xmax
        x = xmax if cut else d
        a.text(x + xmax * .01, yi, f"{d:,}{' (bar cut)' if cut else ''}  ({s:.0%} of its boxes)", va="center", fontsize=6.5,
               color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=.5) if cut else None)
        if cut:
            a.plot([xmax * .97, xmax * .99], [yi - .45, yi + .45], color="white", lw=2.2, clip_on=False)
    a.set_xlim(0, xmax * 1.45)
    a.set_xlabel("depicted boxes (stacked by format)", fontsize=9, color=INK2)
    left = np.zeros(n)
    for f in ["real"] + DEPICT:
        sh = (t[f] / t.boxes).values
        b.barh(y, sh, left=left, height=.78, color=FMT[f], edgecolor="white", linewidth=.4)
        left += sh
    b.set_xlim(0, 1)
    b.set_xlabel("share of all its boxes (real in grey)", fontsize=9, color=INK2)
    a.set_yticks(y, t.index, fontsize=7)
    for tk, d in zip(a.get_yticklabels(), t.domain.values):
        tk.set_color(COLORS.get(d, COLORS["other"]))
    a.invert_yaxis()
    for ax in (a, b):
        ax.grid(axis="x", color="#ececec", lw=.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=8, colors=INK2)
        ax.tick_params(axis="y", length=0)
    a.set_ylim(n - .5, -.5)
    handles = [Patch(color=FMT[f], label=f) for f in DEPICT + ["real"]]
    a.set_title(title, loc="left", fontsize=11, color=INK, pad=28)
    a.legend(handles=handles, loc="lower left", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0, 1.0))
    fig.text(.01, .2 / h, "Category names coloured by semantic domain (as in the long-tail figures).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, .35 / h, 1, 1))
    return fig


def write_all(o: pd.DataFrame, label: str, out: Path, per_page: int = 100) -> pd.DataFrame:
    """Every figure and the CSV for one set of boxes (one child, or several pooled); `label` heads the titles."""
    out.mkdir(parents=True, exist_ok=True)
    o = o[o.cat != "person"]
    t = table(o)
    if "subject" in o and o.subject.nunique() > 1:
        dep = o[o.format.isin(DEPICT)]
        t["children_depicted"] = dep.groupby("cat").subject.nunique().reindex(t.index).fillna(0).astype(int)
        t["children_seen"] = o.groupby("cat").subject.nunique().reindex(t.index).astype(int)
    t.to_csv(out / "depictions_by_category.csv")
    print(f"{label}: {len(t):,} categories with a depiction; {int(t.depicted.sum()):,} depicted boxes")

    with PdfPages(out / "depictions_by_category_all.pdf") as pdf:
        pages = int(np.ceil(len(t) / per_page))
        for k in range(pages):
            s = t.iloc[k * per_page:(k + 1) * per_page]
            fig = page(s, f"{label}: depictions by category, ranks {k * per_page + 1}-{k * per_page + len(s)} "
                          f"of {len(t):,} categories with a depiction (page {k + 1}/{pages})")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    fig = page(t.head(100), f"{label}: the 100 categories with the most depicted boxes "
                            f"({int(t.depicted.sum()):,} depicted boxes in {len(t):,} categories)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"depictions_top100.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # per domain: the 20 categories with the most depicted boxes, stacked by format
    doms = [x for x in ORDER if (t.domain == x).any()]
    fig, axs = plt.subplots(int(np.ceil(len(doms) / 2)), 2, figsize=(18, 3.3 * np.ceil(len(doms) / 2)), constrained_layout=True)
    for ax, dname in zip(axs.flat, doms):
        sd = t[t.domain == dname].head(20)
        x = np.arange(len(sd))
        bottom = np.zeros(len(sd))
        for f in DEPICT:
            ax.bar(x, sd[f].values, bottom=bottom, width=.78, color=FMT[f], edgecolor="white", linewidth=.4, label=f)
            bottom += sd[f].values
        ax.set_xticks(x, sd.index, rotation=50, ha="right", fontsize=8.5)
        ax.set_xlim(-.7, 19.7)
        n_dep = int((t.domain == dname).sum())
        ax.set_title(f"{dname.replace('_', ' ')}: top {len(sd)} of {n_dep:,} categories with a depiction "
                     f"({int(t.loc[t.domain == dname, 'depicted'].sum()):,} depicted boxes)",
                     loc="left", fontsize=10.5, color=COLORS.get(dname, INK), fontweight="bold")
        ax.set_ylabel("depicted boxes", fontsize=9, color=INK2)
        if sd.depicted.max() > 3 * sd.depicted.iloc[1:].max() if len(sd) > 1 else False:   # one runaway bar ("toy")
            top_ = sd.depicted.iloc[1:].max() * 1.25
            ax.set_ylim(0, top_)
            ax.text(0, top_ * .97, f"{int(sd.depicted.iloc[0]):,} (bar cut)", ha="center", va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.6))
        ax.grid(axis="y", color="#ececec", lw=.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=8, colors=INK2)
    for ax in list(axs.flat)[len(doms):]:
        ax.axis("off")
    fig.legend(handles=[Patch(color=FMT[f], label=f) for f in DEPICT], loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(f"{label}: depicted categories within each semantic domain, by format", fontsize=13, y=1.035)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"depictions_domain_subplots.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    d = o.groupby(["domain", "format"]).size().unstack(fill_value=0).reindex(columns=["real"] + DEPICT, fill_value=0)
    d = d.reindex([x for x in ORDER if x in d.index])
    sh = d.div(d.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    y = np.arange(len(sh))
    left = np.zeros(len(sh))
    for f in ["real"] + DEPICT:
        ax.barh(y, sh[f].values, left=left, height=.72, color=FMT[f], edgecolor="white", linewidth=.6, label=f)
        left += sh[f].values
    for yi, (dom_, row) in enumerate(d.iterrows()):
        ax.text(1.01, yi, f"{int(row[DEPICT].sum()):,} depicted of {int(row.sum()):,}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [FS.DOMAIN_LABEL.get(x, x.replace("_", " ")) for x in sh.index], fontsize=10)
    for tk, x in zip(ax.get_yticklabels(), sh.index):
        tk.set_color(COLORS.get(x, COLORS["other"]))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of boxes", color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=9)
    ax.set_title(f"{label}: visual format of Qwen's boxes by semantic domain", loc="left", fontsize=11, color=INK, pad=24)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"depictions_by_domain.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return t

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="S00320001")
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    ap.add_argument("--children", action="store_true",
                    help="all complete children from build_children_cache.py: pooled (figures/children/depictions_pooled/) "
                         "and each child (figures/children/<subject>/)")
    ap.add_argument("--per-page", type=int, default=100)
    args = ap.parse_args()
    if args.children:
        o = pd.read_pickle(HERE / "out" / "private" / "children_objects.pkl")
        o = o[~o.dup]
        kids = sorted(o.subject.unique())
        root = HERE / "figures" / "children"
        write_all(o, f"{len(kids)} children pooled", root / "depictions_pooled", args.per_page)
        for k in kids:
            write_all(o[o.subject == k], k, root / k, args.per_page)
        return
    o = qd.dedup(qd.objects_table(qd.load_records("detect_open_format", args.tag)))
    write_all(o, args.subject, HERE / "figures" / args.subject, args.per_page)


if __name__ == "__main__":
    main()
