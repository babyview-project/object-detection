#!/usr/bin/env python3
"""Every main-text figure of the revision, drawn like the UNR talk (talk-figures/unr-2026), DINOv3 only.

The talk's functions (fig_qwen_talk.py, fig_annotation_talk.py) are reused as they are; their output is redirected
here and renamed to the manuscript's panel names. New here: formats by domain for all 35 children, the five-dataset
percentile RDMs (with LVIS), real-only clustering with LVIS, and the individual-children panels for all 35 children.
Fig. 5C (agreement matrix), 8B (within-child) and 8C (age) come from make_dinov3_manuscript_figures.py and the
bv_vedb notebook and are copied in.

    python make_manuscript_figures.py      # -> figures/manuscript/ and manuscript_revision/figures/
"""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TALK = HERE.parent / "talk-figures" / "unr-2026"
sys.path.insert(0, str(TALK)); sys.path.insert(0, str(HERE))
import talk_style as ts  # noqa: E402
import fig_qwen_talk as Q  # noqa: E402
import fig_annotation_talk as A  # noqa: E402
import fig_style as FS  # noqa: E402

import variant as V  # noqa: E402  (QOV_DOMAINS=things27: THINGS-27 inputs, figures to *_things27 folders)
_T = Q._T
OUT = HERE / "figures" / ("manuscript" + _T); OUT.mkdir(parents=True, exist_ok=True)
DRAFT = V.manuscript_figures_dir(HERE.parent / "manuscript-2026" / "revision_private" / "manuscript_revision" / "figures")
RESP = V.manuscript_figures_dir(HERE.parent / "manuscript-2026" / "revision_private" / "figures")
for _d in (DRAFT, RESP):
    _d.mkdir(exist_ok=True)
RES_ = HERE / "results"
PRIV = HERE / "out" / "private"

# talk name -> manuscript panel name
NAMES = {"a01_scorecard": "fig1b_validation", "a03_format_confusion": "fig1c_format_validation",
         "a02_category_pr": "figS_category_pr",
         "q15c_long_tail_ccn_bv": "fig2a_long_tail", "q03_rank_frequency_bv_vedb": "fig2b_rank_frequency_bv_vedb",
         "q16_frequency_bv_vs_imagenet": "fig2c_frequency_imagenet", "q04_bv_vs_vedb_frequency": "fig2d_frequency_vedb",
         "q14_most_depicted_categories": "fig3b_most_depicted", "q05_depictions_bv_vedb": "fig3c_depictions_bv_vedb",
         "q13_similarity_to_things_by_domain": "fig4a_similarity_to_things",
         "q08e_ego_vs_curated_dinov3": "fig6a_cluster_strength", "q09e_ego_vs_curated_by_domain_dinov3": "fig6b_cluster_by_domain",
         "q11p_subject_rdms": "fig7a_subject_rdms", "q06_size_truncation_bv_vedb": "fig8a_size_truncation"}
_orig_save = ts.save


def save(fig, name, private=False, scale=ts.TEXT_SCALE):
    """Talk styling (Lato, bold), written to figures/manuscript/ under the manuscript name."""
    name = NAMES.get(name, name)
    ts.bolden(fig, scale)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    if DRAFT.exists():
        for ext in ("png", "pdf"):                         # PDF keeps text editable (pdf.fonttype 42)
            shutil.copy(OUT / f"{name}.{ext}", DRAFT / f"{name}.{ext}")
    print("wrote", name)


ts.save = save
Q.ts.save = save
A.ts.save = save
C, L, DC, DL, DOM = Q.C, Q.L, Q.DC, Q.DL, Q.DOM
for k in ("imagenet", "lvis"):
    C[k] = ts.DS_COLOR[k]
L.update(imagenet="ImageNet", lvis="LVIS")
FMT_COL = {"real": "#cfcfcf", "toy": "#7E57C2", "picture": "#26A69A", "screen": "#F9A825"}


# ---- Fig. 2A / SI: long tail, BabyView and VEDB -----------------------------------------------------------------------
def long_tail(key="bv", n_top=60, per_domain=10):
    """The talk's CCN layout (q15c), with every row spanning the same width: the top-n panel's edges line up with the
    first and last domain panels, row 2 (5 domains) and row 3 (4 domains) both fill the width. Domain legend (3 x 3) in
    the top panel's empty upper right."""
    root = PRIV / (f"open_renamed{_T}" if key == "bv" else f"open_vedb_renamed{_T}")
    tier1 = [l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()]
    prev = Q.prevalence(root).reindex(tier1).dropna()
    dom = pd.Series({c: DOM.get(c, "other") for c in prev.index})
    prev = prev[(prev > 0) & dom.isin(Q.CCN_DOMS)].sort_values(ascending=False); dom = dom[prev.index]
    n_frames = pd.read_csv(root / "frames_total.csv").frames.sum()
    fig = plt.figure(figsize=(ts.SLIDE_W, ts.SLIDE_H * 1.08))
    gs = fig.add_gridspec(3, 20, height_ratios=[1.15, 1, 1], hspace=0.95, wspace=0.0,
                          left=0.085, right=0.985, top=0.94, bottom=0.095)

    def bars(ax, v, fs):
        ax.bar(range(len(v)), v.values, color=[DC[dom[c]] for c in v.index], width=0.8)
        ax.set_xticks(range(len(v)), v.index, rotation=45, ha="right", rotation_mode="anchor", fontsize=fs)
        ax.set_xlim(-0.7, len(v) - 0.3); ax.tick_params(axis="y", labelsize=7.5, length=2, pad=1)
        ax.tick_params(axis="x", length=0, pad=1)

    ax = fig.add_subplot(gs[0, :]); bars(ax, prev.head(n_top), 6.5)
    ax.set_title(f"{L[key]}: top {n_top} of {len(prev)} categories ({n_frames:,} frames)", loc="left", fontsize=10,
                 color=C[key], fontweight="bold", pad=3)
    nd = len(Q.CCN_DOMS); r1 = -(-nd // 2); lc = -(-nd // 3)
    for i, d in enumerate(Q.CCN_DOMS):
        ax.text(0.60 - (lc - 3) * 0.135 + (i % lc) * 0.135, 0.95 - (i // lc) * 0.13, Q.CCN_DL[d], color=DC[d], ha="left", va="top",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
    gap = 0.9  # gap between panels, in grid columns
    for k, d in enumerate(Q.CCN_DOMS):
        n_in_row = r1 if k < r1 else nd - r1
        r, j = (1, k) if k < r1 else (2, k - r1)
        w = 20 / n_in_row
        x0, x1 = j * w, (j + 1) * w
        # place by position so all rows share the outer edges exactly
        a = fig.add_subplot(gs[r, 0])
        p0, pr = gs[r, 0].get_position(fig), gs[r, 19].get_position(fig)
        L0, W = p0.x0, pr.x1 - p0.x0
        cw = W / 20
        a.set_position([L0 + x0 * cw + (gap * cw / 2 if j else 0), p0.y0,
                        (x1 - x0) * cw - (gap * cw / 2 if j else 0) - (gap * cw / 2 if j < n_in_row - 1 else 0),
                        p0.height])
        bars(a, prev[dom == d].head(per_domain), 6.8)
        a.set_title(Q.CCN_DL[d], color=DC[d], fontsize=9.5, fontweight="bold", loc="left", pad=2)
    fig.text(0.012, 0.52, "Proportion of frames with object detection", rotation=90, va="center", fontsize=10,
             fontweight="bold")
    save(fig, "fig2a_long_tail" if key == "bv" else "figS_long_tail_vedb", scale=1.0)


# ---- Fig. 3A: format by domain, all 35 children ---------------------------------------------------------------------
def fig3a_formats_by_domain():
    fb = pd.read_csv(HERE / f"out/crossdataset_renamed{_T}/format_by_category.csv")
    fb = fb[fb.dataset == "bv"].copy()
    tier1 = {l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()}
    fb = fb[fb.category.isin(tier1)]
    fb["domain"] = fb.category.map(DOM).fillna("other")
    F = ["real", "toy", "picture", "screen"]
    tot = fb[F].mul(fb.n, axis=0).sum()
    # by domain, categories that are depictions by definition (toy, doll, teddy bear, picture, screen...) are left out,
    # as in format_depicted_by_domain.csv and the text
    INTRINSIC = {"picture", "photo", "drawing", "painting", "poster", "sticker", "screen", "television", "toy", "block",
                 "puzzle", "puzzle piece", "doll", "teddy bear", "lego", "lego block", "stuffed animal", "figurine", "cartoon",
                 "map", "calendar", "book", "picture book", "card", "tablet", "phone", "laptop", "computer monitor", "monitor"}
    fb = fb[~fb.category.isin(INTRINSIC)]
    cnt = fb[F].mul(fb.n, axis=0).groupby(fb.domain).sum()
    doms = [d for d in FS.DOMAIN_ORDER if d in cnt.index]
    sh = cnt.loc[doms].div(cnt.loc[doms].sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    y = np.arange(len(doms))[::-1]
    left = np.zeros(len(doms))
    for f in F:
        ax.barh(y, sh[f], left=left, color=FMT_COL[f], height=0.72, label=f)
        left += sh[f].values
    for yi, d in zip(y, doms):
        n = cnt.loc[d].sum(); dep = 1 - sh.loc[d, "real"]
        ax.text(1.01, yi, f"{dep:.0%} depicted  (n = {n / 1e6:.2f} M)" if n >= 1e6 else f"{dep:.0%} depicted  (n = {n / 1e3:.0f} k)",
                va="center", fontsize=10, color=ts.MUTED)
    ax.set_yticks(y, [DL.get(d, d) for d in doms])
    for tk, d in zip(ax.get_yticklabels(), doms):
        tk.set_color(DC.get(d, DC["other"])); tk.set_fontweight("bold")
    ax.set_xlim(0, 1); ax.set_xlabel("share of objects")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.set_title(f"All 35 children, {tot.sum() / 1e6:.1f} M objects in 828 categories: "
                 f"{1 - tot['real'] / tot.sum():.1%} depicted", loc="left", fontsize=12, pad=30)
    fig.tight_layout(); save(fig, "fig3a_formats_by_domain")


# ---- Fig. 5A: percentile RDMs, five datasets ------------------------------------------------------------------------
def fig5a_rdms():
    T = Q.tables()
    z = np.load(HERE / f"out/crossdataset_renamed{_T}/tables_dinov3.npz", allow_pickle=True)
    T["lvis"] = pd.DataFrame(z["lvis__means"], index=list(z["lvis__categories"]))
    order = Q._rdm_order(T); n = len(order); iu = np.triu_indices(n, 1)
    keys = ["bv", "vedb", "lvis", "things", "imagenet"]
    R = {k: Q.rdm(T[k], order) for k in keys}
    doms = [DOM[c] for c in order]
    starts = [0] + [i for i in range(1, n) if doms[i] != doms[i - 1]]; ends = starts[1:] + [n]
    fig = plt.figure(figsize=(15, 3.9))
    w = 0.158; x0 = 0.115
    axs = [fig.add_axes([x0 + i * (w + 0.012), 0.06, w, 0.78]) for i in range(5)]
    for ax, k in zip(axs, keys):
        im = ax.imshow(Q._pct(R[k]), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        Q._blocks(ax, starts, lw=0.6)
        name = {"bv": "BabyView (child)", "vedb": "VEDB (adult)"}.get(k, L[k])
        ax.set_title(name, fontsize=12, fontweight="bold", color=C[k] if k != "things" else "#3b7dc4", pad=18)
        if k != "bv":
            ax.text(0.5, 1.01, f"ρ with BabyView = {spearmanr(R[k][iu], R['bv'][iu]).statistic:.2f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9.5, color=ts.MUTED)
    cax = fig.add_axes([x0 + 5 * (w + 0.012), 0.2, 0.008, 0.5])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["0", "100"], fontsize=8); cb.outline.set_visible(False)
    cax.set_title("dissimilarity\npercentile", fontsize=8, loc="left")
    ax0 = axs[0]; mids = [(a + b - 1) / 2 for a, b in zip(starts, ends)]; ys = list(mids); gap = n * 0.07
    for _ in range(50):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                ys[i] = ys[i - 1] + gap
        if ys[-1] > n - 1:
            ys = [y - (ys[-1] - (n - 1)) for y in ys]
    for a, m, y in zip(starts, mids, ys):
        d = doms[a]
        ax0.annotate(DL.get(d, d), xy=(-0.5, m), xytext=(-n * 0.08, y), ha="right", va="center", fontsize=9,
                     color=DC.get(d, DC["other"]), fontweight="bold", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=DC.get(d, DC["other"]), lw=0.7, shrinkA=1, shrinkB=0))
    save(fig, "fig5a_rdms_five_datasets")


# ---- Fig. 6C: real objects only, with LVIS ---------------------------------------------------------------------------
def fig6c_real_only():
    RES = RES_ / f"crossdataset_renamed{_T}_core"
    dr = pd.read_csv(RES / "cluster_strength_by_domain_real.csv"); dr = dr[(dr.model == "dinov3") & (dr.domain != "ALL")]
    sg = pd.read_csv(RES / "cluster_strength_by_domain_real_differences.csv")
    sg = sg[(sg.model == "dinov3") & (sg.measure == "delta_sd")]
    doms = [d for d in Q._doms(["animals", "vehicles", "food_drink", "clothing", "furniture_rooms", "household", "toys"]) if d in set(dr.domain)]
    ks = ["things", "imagenet", "lvis", "vedb_real", "bv_real", "bv"]; w = 0.135
    off = {k: (j - 2.5) * w for j, k in enumerate(ks)}
    lab = dict(L, vedb_real="VEDB, real objects", bv_real="BabyView, real objects")
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    for k in ks:
        s = dr[dr.dataset == k].set_index("domain").reindex(doms); x = np.arange(len(doms)) + off[k]
        kw = dict(color="white", edgecolor=C["bv"], lw=1.5, hatch="//") if k == "bv" else dict(color=C[k])
        ax.bar(x, s.delta_sd, w * 0.92, label="BabyView, all formats" if k == "bv" else lab[k], **kw)
        ax.errorbar(x, s.delta_sd, yerr=[s.delta_sd - s.ci_low_sd, s.ci_high_sd - s.delta_sd], fmt="none", color=ts.INK,
                    capsize=1.5, lw=0.9)
        pair = {"things": "bv_real - things", "imagenet": "bv_real - imagenet", "lvis": "bv_real - lvis",
                "vedb_real": "bv_real - vedb_real", "bv": "bv - bv_real"}.get(k)
        if pair:
            st = sg[sg.pair == pair].set_index("domain").reindex(doms).stars.fillna("")
            for xi, hi, t in zip(x, s.ci_high_sd, st):
                if t and t != "n.s.":
                    ax.text(xi, hi + .04, t, ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)
    ax.set_xticks(range(len(doms)), [DL[d].replace(" & ", "\n& ") for d in doms])
    for tk, d in zip(ax.get_xticklabels(), doms):
        tk.set_color(DC[d]); tk.set_fontweight("bold")
    ax.set_ylabel("domain clustering (Δ / SD)"); ax.legend(loc="upper right", fontsize=10, ncol=2); ax.set_ylim(0, 5.2)
    ax.set_title(f"Real objects only (164 categories); stars: BabyView real vs each dataset", loc="left", fontsize=12)
    fig.tight_layout(); save(fig, "fig6c_real_only_by_domain")


# ---- Fig. 7B-C: all 35 children -------------------------------------------------------------------------------------
def domain_stats(D, cs, raw=False):
    sem = np.array([DOM.get(c, "other") for c in cs]); iu = np.triu_indices(len(cs), 1)
    x = D[iu]; same = sem[iu[0]] == sem[iu[1]]
    d = x[~same].mean() - x[same].mean()
    return d if raw else d / x.std()


def fig7_children(min_n=20, n_boot=500, min_shared=20):
    T = Q.tables(); TH = T["things"]; CORE = set(Q.CORE)
    z = np.load(PRIV / f"open_renamed{_T}/sums_dinov3.npz", allow_pickle=True)
    subj, cats = list(z["subjects"]), list(z["categories"]); S, N = z["sum_all"], z["n_all"]
    ft = pd.read_csv(PRIV / f"open_renamed{_T}/frames_total.csv").set_index("subject").frames
    rng = np.random.default_rng(0)
    CM, rows = {}, []
    for i, sb in enumerate(subj):
        keep = N[i] >= min_n
        CM[sb] = pd.DataFrame(S[i][keep] / N[i][keep, None], index=np.array(cats)[keep])
        sh = [c for c in CM[sb].index if c in TH.index and c in CORE]   # the 183 core categories, as in Figs. 5-6
        Dc, Dt = Q.rdm(CM[sb], sh), Q.rdm(TH, sh)
        d = domain_stats(Dc, sh) - domain_stats(Dt, sh)
        bs = []
        for _ in range(n_boot):
            b = rng.choice(len(sh), len(sh), replace=True); b = np.unique(b)          # unique to avoid zero distances
            cs_ = [sh[j] for j in b]
            if len({DOM.get(c) for c in cs_}) < 2: continue
            bs.append(domain_stats(Dc[np.ix_(b, b)], cs_) - domain_stats(Dt[np.ix_(b, b)], cs_))
        rows.append(dict(subject=sb, frames=int(ft[sb]), categories=len(sh), child=domain_stats(Dc, sh),
                         delta_diff=domain_stats(Dc, sh, raw=True) - domain_stats(Dt, sh, raw=True),
                         things=domain_stats(Dt, sh), diff=d, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    r = pd.DataFrame(rows).sort_values("frames").reset_index(drop=True)
    r.to_csv(RES_ / f"open_renamed{_T}" / "per_child_clustering_things_all35.csv", index=False)
    pairs = []
    for x in range(len(subj)):
        for y in range(x + 1, len(subj)):
            a, b = CM[subj[x]], CM[subj[y]]; sh = a.index.intersection(b.index)
            if len(sh) < min_shared: continue
            pairs.append(dict(a=subj[x], b=subj[y], n_categories=len(sh),
                              spearman=spearmanr(Q.rdm(a, sh)[np.triu_indices(len(sh), 1)],
                                                 Q.rdm(b, sh)[np.triu_indices(len(sh), 1)]).statistic))
    P = pd.DataFrame(pairs); P.to_csv(RES_ / f"open_renamed{_T}" / "individual_rdm_pairs_all35.csv", index=False)

    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    y = np.arange(len(r))
    ax.hlines(y, r.things, r.child, color=ts.FAINT, lw=2)
    ax.scatter(r.things, y, s=40, color=C["things"], label="THINGS, same categories", zorder=3)
    ax.scatter(r.child, y, s=50, color=C["bv"], label="each child", zorder=3)
    sig = r.lo > 0
    ax.set_yticks(y, [f"{f / 1000:.1f}k · {n}" if f < 10_000 else f"{f / 1000:.0f}k · {n}" for f, n in zip(r.frames, r.categories)],
                  fontsize=7.5)
    for tk, s_ in zip(ax.get_yticklabels(), sig):
        tk.set_color(ts.INK if s_ else ts.MUTED)
    ax.set_ylabel("children (frames · core categories)", fontsize=11)
    ax.set_xlabel("domain clustering (Δ / SD)"); ax.legend(loc="lower right", fontsize=10)
    ax.set_title(f"All {len(r)} children: child > THINGS for {(r['diff'] > 0).sum()} of {len(r)} "
                 f"({sig.sum()} with 95% CI above 0)", loc="left", fontsize=12)
    fig.tight_layout(); save(fig, "fig7b_children_vs_things")

    fig, ax = plt.subplots(figsize=(ts.SLIDE_W * 0.75, ts.SLIDE_H * 0.85))
    ax.hist(P.spearman, bins=np.arange(0.44, 0.82, 0.015), color=C["bv"], edgecolor="white")
    ax.axvline(P.spearman.mean(), color=ts.INK, lw=1.8)
    ax.text(0.03, 0.95, f"mean ρ = {P.spearman.mean():.2f}\nrange {P.spearman.min():.2f}–{P.spearman.max():.2f}\n"
            f"{len(P)} pairs, {P.n_categories.min()}–{P.n_categories.max()} shared categories",
            transform=ax.transAxes, va="top", fontsize=11)
    ax.set_xlabel("Spearman ρ between two children's RDMs"); ax.set_ylabel("pairs of children")
    fig.tight_layout(); save(fig, "fig7c_children_pairs")
    print(r[["frames", "categories", "child", "things", "diff", "lo", "hi", "delta_diff"]].round(3).to_string()); print("dsd>0", (r["diff"] > 0).sum(), "ci>0", (r.lo > 0).sum(), "delta>0", (r.delta_diff > 0).sum(), "big dsd>0", (r[r.frames >= 10000]["diff"] > 0).sum())
    print("pairs:", len(P), P.spearman.describe().round(3).to_dict(), "shared", P.n_categories.min(), P.n_categories.max())
    big = set(ft[ft >= 10_000].index)
    Pb = P[P.a.isin(big) & P.b.isin(big)]
    print("pairs among >=10k children:", len(Pb), round(Pb.spearman.mean(), 3), round(Pb.spearman.min(), 3), round(Pb.spearman.max(), 3))
    return r, P


def fig7a_all_children(min_n=20, ncol=7):
    """Every child's RDM on the core categories with >= min_n of the child's boxes, BabyView's pooled order, percentile
    colour within each RDM; ordered by number of annotated frames."""
    T = Q.tables(); order = Q._rdm_order(T)
    z = np.load(PRIV / f"open_renamed{_T}/sums_dinov3.npz", allow_pickle=True)
    subj, cats = list(z["subjects"]), list(z["categories"]); S, N = z["sum_all"], z["n_all"]
    ft = pd.read_csv(PRIV / f"open_renamed{_T}/frames_total.csv").set_index("subject").frames
    kids = sorted(subj, key=lambda s_: -ft[s_])
    nrow = int(np.ceil(len(kids) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ts.SLIDE_W, ts.SLIDE_W * nrow / ncol * 1.08))
    for ax in axs.flat:
        ax.axis("off")
    for j, (ax, sb) in enumerate(zip(axs.flat, kids)):
        i = subj.index(sb)
        cs = [c for c in order if c in cats and N[i, cats.index(c)] >= min_n]
        M = pd.DataFrame(S[i, [cats.index(c) for c in cs]] / N[i, [cats.index(c) for c in cs], None], index=cs)
        doms = [DOM[c] for c in cs]; starts = [0] + [q for q in range(1, len(cs)) if doms[q] != doms[q - 1]]
        ax.axis("on"); ax.imshow(Q._pct(Q.rdm(M, cs)), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        Q._blocks(ax, starts, lw=0.35)
        f = ft[sb]
        ax.set_title(f"{f / 1000:.0f}k fr · {len(cs)}" if f >= 10_000 else f"{f / 1000:.1f}k fr · {len(cs)}", fontsize=7.5,
                     pad=2, color=ts.INK if f >= 10_000 else ts.MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.08, hspace=0.25)
    save(fig, "fig7a_subject_rdms", scale=1.0)


def resp_blur():
    """Response Figure 7: crop sharpness (left) and domain clustering on all / sharpest half / sharpest third (right)."""
    D_ = PRIV / f"bv_vedb_renamed{_T}"                       # the renamed run's box sample (the core run reuses it)
    sh = pd.read_csv(D_ / "sharpness.csv.gz"); col = "sharpness"
    sh["dataset"] = np.load(D_ / "sample_dinov3.npz", allow_pickle=True)["dataset"]
    sh = sh[np.isfinite(sh[col]) & (sh[col] > 0)]
    B = pd.read_csv(RES_ / f"bv_vedb_renamed{_T}_core/blur_control.csv"); B = B[B.model == "dinov3"]
    Dd = pd.read_csv(RES_ / f"bv_vedb_renamed{_T}_core/blur_control_differences.csv"); Dd = Dd[Dd.model == "dinov3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.8), gridspec_kw={"width_ratios": [1.2, 1]})
    bins = np.linspace(-1, 4, 70)
    for d in ("bv", "vedb"):
        v = np.log10(sh.loc[sh.dataset == d, col].clip(lower=1e-3))
        a.hist(v, bins=bins, density=True, histtype="step", lw=2.2, color=C[d], label=L[d])
    bvv = np.log10(sh.loc[sh.dataset == "bv", col].clip(lower=1e-3))
    for q in (50, 67):
        a.axvline(np.percentile(bvv, q), color=ts.MUTED, ls="--", lw=1.2)
    a.set_xlabel("crop sharpness (log10 Laplacian variance)"); a.set_ylabel("density"); a.legend(loc="upper right", fontsize=10)
    a.set_title("dashed: BabyView 50th / 67th percentile", loc="left", fontsize=10, color=ts.MUTED)
    cuts = ["all crops", "> BV p50", "> BV p67"]; lab = ["all crops", "sharpest\nhalf", "sharpest\nthird"]
    for j, d in enumerate(("things", "vedb", "bv")):
        y = B[B.dataset == d].set_index("sharp_cut").loc[cuts].delta
        b.plot(range(3), y, "-o", color=C[d], lw=2.2, ms=8)
        b.text(2.12, y.iloc[-1], {"bv": "BabyView", "vedb": "VEDB", "things": "THINGS"}[d], color=C[d] if d != "things" else "#3b7dc4",
               va="center", fontsize=10.5, fontweight="bold")
    for k, c in enumerate(cuts):
        r = Dd[(Dd.sharp_cut == c) & (Dd.pair == "bv - things")].iloc[0]
        b.text(k, 0.135, f"+{r['diff']:.3f}", ha="center", fontsize=9.5, color=ts.INK)
    b.text(-0.3, 0.147, "BabyView − THINGS:", fontsize=9.5, color=ts.MUTED)
    b.set_xticks(range(3), lab); b.set_ylim(0, 0.16); b.set_xlim(-0.35, 2.75)
    b.set_ylabel("domain clustering (Δ, DINOv3)")
    n = int(B.categories.iloc[0])
    b.set_title(f"{n} categories", loc="left", fontsize=10, color=ts.MUTED)
    fig.tight_layout()
    ts.bolden(fig, ts.TEXT_SCALE)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"R3a_blur_sharpness.{ext}", bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    if RESP.exists():
        for ext in ("png", "pdf"):
            shutil.copy(OUT / f"R3a_blur_sharpness.{ext}", RESP / f"R3a_blur_sharpness.{ext}")
    print("wrote R3a_blur_sharpness")


MONT = TALK / "figures" / "private" / "montages"
# talk montage -> manuscript panel. These show family homes (BabyView frames), so they are copied ONLY into the
# git-ignored manuscript draft folder, never into figures/manuscript/. Made by talk-figures/unr-2026/make_talk_montages.py.
MONTAGES = {"pipeline_crop_embed_average": "fig1a_pipeline_example",
            "var_format": "fig3d_animals_by_format", "grid_duck_bird": "fig3e_duck_bird_child_adult",
            "strategy_similar_dissimilar": "fig4b_similar_dissimilar_things",
            "strategy_similar_dissimilar_imagenet": "figS_similar_dissimilar_imagenet",
            "strategy_similar_dissimilar_vedb": "figS_similar_dissimilar_vedb",
            "strategy_similar_dissimilar_lvis": "figS_similar_dissimilar_lvis",
            "var_exemplar": "figS_variation_exemplars", "var_viewpoint": "figS_variation_viewpoints",
            "grid_bus_truck": "figS_bus_truck_child_adult", "grid_car_tractor": "figS_car_tractor_child_adult",
            "grid_sun_fire_hydrant": "figS_sun_fire_hydrant_child_adult",
            "frames_chair": "figS_frames_chair", "frames_table": "figS_frames_table"}


def montages():
    """Copy the talk's exemplar montages (private) into the manuscript draft's figures folder."""
    if not DRAFT.exists():
        return
    for src, dst in MONTAGES.items():
        shutil.copy(MONT / f"{src}.jpg", DRAFT / f"{dst}.jpg")
        print("montage", dst)


def copy_in():
    """Panels made elsewhere, already DINOv3-only and current."""
    for src, dst in ((HERE / f"figures/manuscript_dinov3{_T}/fig4a_rdm_agreement.png", "fig5b_rdm_agreement"),
                     (HERE / f"figures/bv_vedb_renamed{_T}_core/category_frequency_ratio.png", "fig2e_frequency_ratio"),
                     (HERE / f"figures/bv_vedb_renamed{_T}/scene_statistics_within_child_density.png", "fig8b_within_child"),
                     (HERE / f"figures/manuscript_dinov3{_T}/fig7c_age_trends.png", "fig8c_age_trends")):
        for s in (src, src.with_suffix(".pdf")):
            if s.exists():
                shutil.copy(s, OUT / f"{dst}{s.suffix}")
                if DRAFT.exists():
                    shutil.copy(s, DRAFT / f"{dst}{s.suffix}")
        print("copied", dst)


if __name__ == "__main__":
    ts.setup()
    only = sys.argv[1:]
    steps = {
        "validation": lambda: (A.a01_scorecard(), A.a02_category_pr(), A.a03_format_confusion()),
        "long_tail": lambda: (long_tail("bv"), long_tail("vedb"), Q.q03_rank_frequency(), Q.q16_frequency_imagenet(), Q.q04_scatter()),
        "formats": lambda: (fig3a_formats_by_domain(), Q.q14_depicted_categories(), Q.q05_formats()),
        "similarity": Q.q13_similarity,
        "rdms": fig5a_rdms,
        "clustering": lambda: (Q.q08e_ego_vs_curated("dinov3"), Q.q09e_ego_vs_curated_by_domain("dinov3"), fig6c_real_only()),
        "children": lambda: (fig7a_all_children(), fig7_children()),
        "age": Q.q06_size,
        "copy": copy_in,
        "montages": montages,
        "resp": resp_blur,
    }
    for k, f in steps.items():
        if not only or k in only:
            f()
