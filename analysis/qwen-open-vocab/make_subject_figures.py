#!/usr/bin/env python3
"""Manuscript-style figures for one child's Qwen annotations (long tail + individual RDMs).

Styles follow analysis/manuscript-2026: notebook 01 (CDI_SEMANTIC_COLORS, top-50 bars + per-domain
subplots, power-law panel) and notebooks 03/04 (viridis, cosine distance 0-2, domains in
CDI_SEMANTIC_ORDER, categories alphabetical within a domain, white domain boundaries).

Categories are Qwen's own (qwen_data `cat`), person excluded; frame prevalence = share of the child's
frames with the category. RDMs: z-scored category means of the CLIP / DINOv3 crop embeddings from
embed_qwen_crops.py (categories with >= 30 crops), cosine distance.

Figures (aggregate only, no frames) -> figures/<subject>/ as .png and .pdf:
  long_tailed_top50_plus_domain_subplots[_real]   top 50 overall + top 10 per domain (all formats / real only)
  long_tailed_top150                              top 150 categories
  long_tailed_powerlaw_overall                    rank-frequency with the power-law fit (>= 30 frames)
  long_tailed_top50_plus_domain_subplots_depicted depictions only (a frame counts when the category is a toy,
  long_tailed_top150_depicted                     drawing, photo or screen there)
  long_tailed_by_depiction_format                 top 30 per format (toy, drawing, photo, screen)
  long_tailed_powerlaw_real_vs_depicted           both tails with their power-law fits
  individual_rdm_<model>_all                      all categories with embeddings, domain blocks
  individual_rdm_<model>_top100                   the 100 most frequent categories, labelled
  individual_rdm_<model>_valid129_vs_yoloe        the categories shared with the manuscript's 129, manuscript
                                                  order: Qwen boxes | this child's YOLOE + CLIP crops
  ..._clustered                                   each RDM again with categories ordered by average-linkage
                                                  hierarchical clustering (optimal leaf order) within each domain;
                                                  z-scoring and distances are unchanged, only the order differs

  python make_subject_figures.py [--subject S00320001 --tag subj_S00320001_t1_pp0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform

import sys
HERE = Path(__file__).resolve().parent
QP = HERE.parents[1] / "annotation" / "qwen_pilot"        # pipeline: loaders, embeddings, manifest
sys.path.insert(0, str(QP))
import qwen_data as qd  # noqa: E402
sys.path.insert(0, str(HERE))
import fig_style as FS  # noqa: E402
FS.apply()

HERE = Path(__file__).resolve().parent
PRIV = QP / "out" / "private"      # crop-embedding means (embed_qwen_crops.py) live with the pipeline
MS_RES = qd.OD_ROOT / "analysis" / "manuscript-2026" / "main_results_valid129s_2025.2" / "results"
ORDER = list(FS.DOMAIN_ORDER)                                          # manuscript notebook 01 (CDI) or THINGS-27
COLORS = FS.DOMAIN_COLOR                                               # manuscript CDI domain colours
MIN_FRAMES, MIN_CROPS = 30, 30


def style(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", labelsize=11, width=1.0)


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def bars(ax, prev: pd.Series, dom: pd.Series, rot=55, fs=9):
    ax.bar(range(len(prev)), prev.values, color=[COLORS.get(dom.get(c, "other"), COLORS["other"]) for c in prev.index], width=.8)
    ax.set_xticks(range(len(prev)), prev.index, rotation=rot, ha="right", fontsize=fs)
    ax.set_xlim(-.7, len(prev) - .3)
    style(ax)


def top50_plus_domains(prev, dom, title, out, name, n_frames):
    doms = [d for d in ORDER if (dom.reindex(prev.index) == d).any()]
    ncol = 5
    nrow = int(np.ceil(len(doms) / ncol))
    fig = plt.figure(figsize=(20, 4.6 + 3.4 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.35] + [1] * nrow)
    ax = fig.add_subplot(gs[0, :])
    bars(ax, prev.head(50), dom, fs=10)
    ax.set_ylabel("Frame prevalence", fontsize=12)
    ax.set_title(title, fontsize=13)
    for k, d in enumerate(doms):
        a = fig.add_subplot(gs[1 + k // ncol, k % ncol])
        s = prev[dom.reindex(prev.index) == d].head(10)
        bars(a, s, dom, rot=50, fs=9)
        n_common = int(((dom.reindex(prev.index) == d) & (prev * n_frames >= MIN_FRAMES)).sum())
        a.set_title(f"{d.replace('_', ' ')} ({n_common})", color=COLORS[d],
                    fontsize=12, fontweight="bold")
        if k % ncol == 0:
            a.set_ylabel("Frame prevalence", fontsize=11)
    save(fig, out, name)


def fit(v):
    v = np.sort(v[v > 0])[::-1]
    x, y = np.log(np.arange(1, len(v) + 1)), np.log(v)
    s, i = np.polyfit(x, y, 1)
    yh = s * x + i
    return v, -s, i, 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def zrows(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)


def load_means(path, min_crops=0):
    d = pd.read_csv(path)
    d["category"] = d.category.astype(str).str.strip().str.lower()
    d = d.set_index("category")
    if "n_crops" in d and min_crops:
        d = d[d.n_crops >= min_crops]
    return d.filter(like="dim_")


def domain_sorted(cats, dom):
    return sorted(cats, key=lambda c: (ORDER.index(dom.get(c, "other")) if dom.get(c, "other") in ORDER else 99, c))


def cluster_within(cats, D, dom):
    """Domains in ORDER; within a domain, average-linkage hierarchical clustering of the RDM rows/cols
    with optimal leaf ordering (scipy). Returns the new index order."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    idx = []
    for d in ORDER + sorted({dom.get(c, "other") for c in cats} - set(ORDER)):
        k = [i for i, c in enumerate(cats) if dom.get(c, "other") == d]
        if len(k) > 2:
            y = squareform(D[np.ix_(k, k)], checks=False)
            Z = optimal_leaf_ordering(linkage(y, method="average"), y)
            k = [k[j] for j in leaves_list(Z)]
        idx += k
    return idx


def ordered_rdm(E, cats, dom, within):
    D = squareform(pdist(zrows(E.loc[cats].to_numpy(float)), metric="cosine"))
    if within == "cluster":
        idx = cluster_within(cats, D, dom)
        cats, D = [cats[i] for i in idx], D[np.ix_(idx, idx)]
    return cats, D


def rdm_panel(ax, D, cats, dom, title, labels=False):
    im = ax.imshow(D, cmap="viridis", vmin=0, vmax=2, interpolation="nearest", aspect="equal")
    ds = [dom.get(c, "other") for c in cats]
    starts = [0] + [i for i in range(1, len(ds)) if ds[i] != ds[i - 1]]
    ends = starts[1:] + [len(ds)]
    for b in starts[1:]:
        ax.axhline(b - .5, color="white", lw=.8, alpha=.8)
        ax.axvline(b - .5, color="white", lw=.8, alpha=.8)
    if labels:
        ax.set_yticks(range(len(cats)), cats, fontsize=6)
        ax.set_xticks(range(len(cats)), cats, fontsize=6, rotation=90)
        for t in ax.get_yticklabels() + ax.get_xticklabels():
            t.set_color(COLORS.get(dom.get(t.get_text(), "other"), COLORS["other"]))
    else:
        ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [ds[a].replace("_", " ") for a in starts],
                      fontsize=10, fontweight="bold")
        for t, a in zip(ax.get_yticklabels(), starts):
            t.set_color(COLORS.get(ds[a], COLORS["other"]))
        ax.set_xticks([])
    ax.set_title(title, fontsize=12)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="S00320001")
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    args = ap.parse_args()
    out = HERE / "figures" / args.subject
    out.mkdir(parents=True, exist_ok=True)

    rec = qd.load_records("detect_open_format", args.tag)
    N = len(rec)
    o = qd.dedup(qd.objects_table(rec))
    o = o[o.cat != "person"]
    dom = o.groupby("cat").domain.first()
    prev = o.groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
    prev_real = o[o.format == "real"].groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
    DEP = ["toy", "drawing", "photo", "screen"]
    prev_dep = o[o.format.isin(DEP)].groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
    print(f"{N:,} frames, {len(prev):,} categories")

    # ---- long tail
    top50_plus_domains(prev, dom, f"{args.subject}: top 50 Qwen categories (coloured by CDI semantic domain; "
                       f"(n) = categories in >= {MIN_FRAMES} frames)", out, "long_tailed_top50_plus_domain_subplots", N)
    top50_plus_domains(prev_real, dom, f"{args.subject}: top 50 Qwen categories, real objects only (no toys, pictures, screens)",
                       out, "long_tailed_top50_plus_domain_subplots_real", N)
    fig, ax = plt.subplots(figsize=(26, 7), constrained_layout=True)
    bars(ax, prev.head(150), dom, rot=75, fs=8)
    ax.set_ylabel("Frame prevalence", fontsize=12)
    ax.set_title(f"{args.subject}: top 150 of {len(prev):,} Qwen categories", fontsize=13)
    ax.legend(handles=[Patch(facecolor=COLORS[d], label=d.replace("_", " ")) for d in ORDER if d in set(dom)],
              frameon=False, ncol=6, fontsize=10, loc="upper right")
    save(fig, out, "long_tailed_top150")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    common = prev[prev * N >= MIN_FRAMES]
    v, a, i, r2 = fit(prev.values)
    ax.loglog(np.arange(1, len(v) + 1), v, "o", ms=2.5, color=FS.FAINT, label=f"all {len(v):,} categories: alpha {a:.2f}, R² {r2:.2f}")
    v, a, i, r2 = fit(common.values)
    r = np.arange(1, len(v) + 1)
    ax.loglog(r, v, "o", ms=3, color=FS.DS_COLOR["bv"], label=f"{len(v)} categories in >= {MIN_FRAMES} frames: alpha {a:.2f}, R² {r2:.2f}")
    ax.loglog(r, np.exp(i) * r ** -a, color=FS.INK, lw=1.2, ls="--")
    ax.set_xlabel("Rank", fontsize=12); ax.set_ylabel("Frame prevalence", fontsize=12)
    ax.set_title(f"{args.subject}: rank-frequency of Qwen categories", fontsize=12)
    ax.legend(frameon=False, fontsize=9); style(ax)
    save(fig, out, "long_tailed_powerlaw_overall")

    # ---- depictions only (frame prevalence counts a frame when the category appears as a toy, drawing, photo or screen)
    top50_plus_domains(prev_dep, dom, f"{args.subject}: top 50 Qwen categories as depictions only (toy, drawing, photo, screen); "
                       f"(n) = categories depicted in >= {MIN_FRAMES} frames", out, "long_tailed_top50_plus_domain_subplots_depicted", N)
    fig, ax = plt.subplots(figsize=(26, 7), constrained_layout=True)
    bars(ax, prev_dep.head(150), dom, rot=75, fs=8)
    ax.set_ylabel("Frame prevalence (as a depiction)", fontsize=12)
    ax.set_title(f"{args.subject}: top 150 of {len(prev_dep):,} categories seen as depictions", fontsize=13)
    ax.legend(handles=[Patch(facecolor=COLORS[d], label=d.replace("_", " ")) for d in ORDER if d in set(dom)],
              frameon=False, ncol=6, fontsize=10, loc="upper right")
    save(fig, out, "long_tailed_top150_depicted")

    fig, axs = plt.subplots(2, 2, figsize=(20, 9), constrained_layout=True)
    for ax, f in zip(axs.flat, DEP):
        pf = o[o.format == f].groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
        bars(ax, pf.head(30), dom, rot=60, fs=9)
        ax.set_title(f"{f}: top 30 of {len(pf):,} categories ({int((pf * N >= MIN_FRAMES).sum())} in >= {MIN_FRAMES} frames)",
                     loc="left", fontsize=12)
        ax.set_ylabel("Frame prevalence", fontsize=10)
    fig.suptitle(f"{args.subject}: long tail within each depiction format", fontsize=13)
    save(fig, out, "long_tailed_by_depiction_format")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for pv, col, name in ((prev_real, FS.BASELINE, "real"), (prev_dep, FS.FMT_COLOR["toy"], "depicted")):
        c = pv[pv * N >= MIN_FRAMES]
        v, a, i, r2 = fit(c.values)
        r = np.arange(1, len(v) + 1)
        ax.loglog(r, v, "o", ms=3, color=col, label=f"{name}: {len(v)} categories in >= {MIN_FRAMES} frames, alpha {a:.2f}, R² {r2:.2f}")
        ax.loglog(r, np.exp(i) * r ** -a, color=col, lw=1.2, ls="--")
    ax.set_xlabel("Rank", fontsize=12); ax.set_ylabel("Frame prevalence", fontsize=12)
    ax.set_title(f"{args.subject}: rank-frequency, real vs depicted", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5); style(ax)
    save(fig, out, "long_tailed_powerlaw_real_vs_depicted")

    # ---- individual RDMs (alphabetical within domain, and hierarchical clustering within domain)
    for m, label in (("clip", "CLIP"), ("dinov3", "DINOv3")):
        p = PRIV / f"qwen_{m}_exemplar_avg_raw_{args.tag}_all.csv"
        if not p.exists():
            print(f"no {p.name}: run embed_qwen_crops.py first"); continue
        E = load_means(p, MIN_CROPS).drop(index="person", errors="ignore")
        yp = PRIV / f"yoloe_{m}_exemplar_avg_raw_{args.subject}.csv"
        Y = load_means(yp) if yp.exists() else None
        order = pd.read_csv(MS_RES / f"bv_things_rdm_order_bv_semantic_{m}_filtered-0.27_valid129.csv")
        mdom = order.set_index("category").cdi_semantic.to_dict()
        for within, suf, how in (("alpha", "", "alphabetical within domain"),
                                 ("cluster", "_clustered", "hierarchical clustering within domain")):
            cats, D = ordered_rdm(E, domain_sorted(list(E.index), dom), dom, within)
            fig, ax = plt.subplots(figsize=(10, 9.4), constrained_layout=True)
            im = rdm_panel(ax, D, cats, dom, f"{args.subject} individual RDM ({label}, Qwen boxes), n={len(cats)}; {how}")
            fig.colorbar(im, ax=ax, shrink=.75, label="Cosine distance")
            save(fig, out, f"individual_rdm_{m}_all{suf}")

            top = domain_sorted([c for c in prev.index if c in E.index][:100], dom)
            top, D = ordered_rdm(E, top, dom, within)
            fig, ax = plt.subplots(figsize=(13, 12.4), constrained_layout=True)
            im = rdm_panel(ax, D, top, dom, f"{args.subject} individual RDM ({label}, Qwen boxes): the 100 most frequent "
                           f"categories; {how}", labels=True)
            fig.colorbar(im, ax=ax, shrink=.6, label="Cosine distance")
            save(fig, out, f"individual_rdm_{m}_top100{suf}")

            shared = [c for c in order.category if c in E.index and (Y is None or c in Y.index)]
            if within == "cluster":                  # order from Qwen's RDM, applied to both panels
                shared, Dq = ordered_rdm(E, shared, mdom, "cluster")
            else:                                    # manuscript notebook-03 order
                Dq = squareform(pdist(zrows(E.loc[shared].to_numpy(float)), metric="cosine"))
            fig, axs = plt.subplots(1, 2 if Y is not None else 1, figsize=(17 if Y is not None else 9, 8.2), constrained_layout=True)
            axs = np.atleast_1d(axs)
            im = rdm_panel(axs[0], Dq, shared, mdom, f"Qwen boxes (n={len(shared)})", labels=within == "cluster")
            if Y is not None:
                from scipy.stats import spearmanr
                Dy = squareform(pdist(zrows(Y.loc[shared].to_numpy(float)), metric="cosine"))
                il = np.tril_indices(len(shared), -1)
                rho = spearmanr(Dq[il], Dy[il]).statistic
                rdm_panel(axs[1], Dy, shared, mdom, f"YOLOE + CLIP crops, CLIP >= 0.27 (n={len(shared)}); Spearman with Qwen {rho:.2f}",
                          labels=within == "cluster")
            sub = "notebook-03 order" if within == "alpha" else "clustered within domain on the Qwen RDM (same order in both panels)"
            fig.suptitle(f"{args.subject} individual RDMs ({label}) on the manuscript's categories, {sub}", fontsize=13)
            fig.colorbar(im, ax=list(axs), shrink=.75, label="Cosine distance")
            save(fig, out, f"individual_rdm_{m}_valid129_vs_yoloe{suf}")


if __name__ == "__main__":
    main()
