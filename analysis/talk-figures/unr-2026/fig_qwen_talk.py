"""UNR talk (final deck): figures from the Qwen3.8-27B open-vocabulary pipeline, DINOv3 only, BabyView vs VEDB vs THINGS.

Every number comes from analysis/qwen-open-vocab results on renamed labels; cross-dataset comparisons use the 183 core
categories (164 for real objects only), as in the revision. Domain clustering is Δ/SD (between - within cosine
distance divided by the SD of all pair distances).

    python fig_qwen_talk.py            # -> figures/q*.png ; figures/private/q01_qwen_examples.png (BabyView frames)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import talk_style as ts

Q = ts.ANALYSIS / "qwen-open-vocab"
sys.path.insert(0, str(Q))
import fig_style as FS  # noqa: E402  (manuscript domain colours incl. people / other)

import os as _os
_T = "_things27" if _os.environ.get("QOV_DOMAINS", "cdi") == "things27" else ""   # THINGS-27 domain variant
_THINGS = bool(_T)
_DISPLAY = [d for d in FS.DOMAIN_ORDER if d not in ("people", "other")]


def _doms(cdi_list):
    """A figure's CDI domain list, or the THINGS-27 display domains in that variant."""
    return list(_DISPLAY) if _THINGS else cdi_list


RES = Q / "results" / f"crossdataset_renamed{_T}_core"
PRIV = Q / "out" / "private"
CORE = [l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_core.txt") if l.strip()]
DOM = pd.read_csv(PRIV / f"open_renamed{_T}" / "categories.csv", index_col=0).domain.to_dict()
DC = FS.DOMAIN_COLOR
DL = FS.DOMAIN_LABEL
C = {"bv": ts.DS_COLOR["bv"], "vedb": ts.DS_COLOR["vedb"], "things": ts.DS_COLOR["things"], "bv_real": ts.DS_COLOR["bv"],
     "vedb_real": ts.DS_COLOR["vedb"]}
L = {"bv": "BabyView (child)", "vedb": "VEDB (adult)", "things": "THINGS", "bv_real": "BabyView, real objects",
     "vedb_real": "VEDB, real objects"}


def prevalence(root):
    f = pd.read_csv(root / "frames_by_child.csv"); n = pd.read_csv(root / "frames_total.csv").frames.sum()
    return f.groupby("cat").frames_all.sum() / n


def q01_examples():
    import os
    d = Q.parents[1] / "annotation/qwen_pilot/out/private/overlays_gold_v2/qwen"
    fs = {int(f.split("_")[0]): f for f in os.listdir(d)}
    ims = []
    for n in (17, 24, 36):
        im = Image.open(d / fs[n]).convert("RGB"); w, h = im.size
        ims.append(im.crop((0, int(h * 0.035), w, h)))            # drop the overlay's text header
    fig, axs = plt.subplots(1, 3, figsize=(ts.SLIDE_W, ts.SLIDE_H))
    for ax, im in zip(axs, ims):
        ax.imshow(im); ax.axis("off")
    fig.tight_layout(w_pad=0.6); ts.save(fig, "q01_qwen_examples", private=True)


def q02_long_tail():
    p = prevalence(PRIV / f"open_renamed{_T}").sort_values(ascending=False)
    top = p.head(40)
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    ax.bar(range(len(top)), top.values, color=[DC.get(DOM.get(c, "other"), DC["other"]) for c in top.index], width=0.8)
    ax.set_xticks(range(len(top)), top.index, rotation=60, ha="right", fontsize=10)
    ax.set_xlim(-0.7, len(top) - 0.3); ax.set_ylabel("proportion of frames\nwith object detection")
    doms = [d for d in FS.DOMAIN_ORDER if d in {DOM.get(c) for c in top.index}]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DC[d]) for d in doms], labels=[DL[d] for d in doms],
              ncol=2, loc="upper right", fontsize=11)
    ia = ax.inset_axes([0.42, 0.40, 0.25, 0.42])
    ia.loglog(np.arange(1, len(p) + 1), p.values, color=C["bv"], lw=2)
    ia.set_xlabel("rank", fontsize=10); ia.set_ylabel("share", fontsize=10); ia.tick_params(labelsize=8)
    ia.set_title(f"all {len(p)} categories", fontsize=10)
    fig.tight_layout(); ts.save(fig, "q02_long_tail_bv")


def q03_rank_frequency():
    pb, pv = prevalence(PRIV / f"open_renamed{_T}"), prevalence(PRIV / f"open_vedb_renamed{_T}")
    shared = pb.index.intersection(pv.index[pv > 0])
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W * 0.62, ts.SLIDE_H))
    for k, p in (("bv", pb[shared]), ("vedb", pv[shared])):
        v = np.sort(p.values)[::-1]; r = np.arange(1, len(v) + 1)
        ax.loglog(r, v, color=C[k], lw=2.6, label=L[k])
    ax.set_xlabel("rank (each dataset's own order)"); ax.set_ylabel("proportion of frames\nwith object detection")
    ax.legend(loc="lower left"); ax.set_title(f"{len(shared)} categories in both datasets", loc="left")
    fig.tight_layout(); ts.save(fig, "q03_rank_frequency_bv_vedb")
    return len(shared), spearmanr(pb[shared], pv[shared]).statistic


def q04_scatter():
    pb, pv = prevalence(PRIV / f"open_renamed{_T}"), prevalence(PRIV / f"open_vedb_renamed{_T}")
    s = pd.DataFrame({"bv": pb, "vedb": pv}).dropna(); s = s[(s.bv > 0) & (s.vedb > 0)]
    s["dom"] = [DOM.get(c, "other") for c in s.index]; s["r"] = np.log2(s.bv / s.vedb)
    rho = spearmanr(s.bv, s.vedb).statistic
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    lim = [s[["bv", "vedb"]].min().min() * 0.7, s[["bv", "vedb"]].max().max() * 1.5]
    ax.plot(lim, lim, color=ts.FAINT, lw=1.2, ls="--", zorder=0)
    ax.scatter(s.vedb, s.bv, s=14, c=[DC.get(d, DC["other"]) for d in s.dom], alpha=0.8, lw=0)
    # hand-picked, recognisable examples (all >= 7x more common in one view), labelled in the empty corners with leader
    # lines so no label sits on the data: child-heavy stacked top-left, adult-heavy stacked bottom-right
    # examples across the range, not only the extremes (ratio child/adult in brackets)
    CHILD = ["crib", "rattle", "block", "dog", "stroller", "ball", "spoon"]          # 202x ... 3x
    SHARED = ["table", "door", "cup"]                                              # ~1x
    ADULT = ["bottle", "car", "phone", "tree", "laptop", "keyboard"]               # 0.7x ... 0.03x
    # label slots (x, top y, bottom y; data units) in the two empty corners; labels assigned to slots so that no leader
    # lines cross (fewest crossings, then shortest total length, in log space)
    from itertools import permutations

    def _cross(p1, p2, q1, q2):
        d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return d(p1, p2, q1) * d(p1, p2, q2) < 0 and d(q1, q2, p1) * d(q1, q2, p2) < 0

    def _assign(tg, sl):
        best = None
        for perm in permutations(range(len(tg))):
            segs = [(sl[i], tg[j]) for i, j in enumerate(perm)]
            k = sum(_cross(*segs[i], *segs[j]) for i in range(len(segs)) for j in range(i + 1, len(segs)))
            L = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in segs)
            if best is None or (k, L) < best[0]:
                best = ((k, L), perm)
        return best[1]

    spots = ((CHILD, C["bv"], 7.0e-5, 1.6e-1, 8.0e-3), (SHARED, ts.INK, 3.0e-3, 3.9e-1, 1.5e-1),
             (ADULT, C["vedb"], 2.6e-2, 2.8e-4, 1.2e-5))                              # right-aligned columns
    for names, col, x_t, y0, y1 in spots:
        names = [c for c in names if c in s.index]
        slots = [(x_t, y) for y in np.geomspace(y0, y1, len(names))]
        tg = [np.log10([s.loc[c, "vedb"], s.loc[c, "bv"]]) for c in names]
        sl = [np.log10([x, y]) for x, y in slots]                                  # lines start at the right edge
        perm = _assign(tg, sl)
        for (xs, ys), j in zip(slots, perm):
            c = names[j]; r = s.loc[c]
            ax.annotate(c, (r.vedb, r.bv), xytext=(xs, ys), textcoords="data", fontsize=11, color=col,
                        fontweight="bold", ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=4, shrinkB=3, relpos=(1, 0.5)))
            ax.scatter(r.vedb, r.bv, s=24, color=col, zorder=3, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("proportion of frames with object detection, VEDB (adult)")
    ax.set_ylabel("proportion of frames with object\ndetection, BabyView (child)")
    ax.set_title(f"{len(s)} categories, Spearman ρ = {rho:.2f}", loc="left")
    ax.text(0.03, 0.97, "child-heavy", transform=ax.transAxes, va="top", fontsize=12, color=C["bv"], fontweight="bold")
    ax.text(0.97, 0.03, "adult-heavy", transform=ax.transAxes, ha="right", fontsize=12, color=C["vedb"], fontweight="bold")
    fig.tight_layout(); ts.save(fig, "q04_bv_vs_vedb_frequency")
    return len(s), rho


def q05_formats():
    """Depicted objects, every child and session: left, share of each format over all Tier 1 objects; right, share
    depicted per domain (categories that are depictions by definition left out). Dots = children / sessions with
    >= 20 boxes; line = mean, whisker = ±1 SE (unit_shares.py)."""
    import unit_shares as US
    allu = US.per_unit()
    domu = US.per_unit(DOM, exclude_intrinsic=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(ts.SLIDE_W, ts.SLIDE_H), gridspec_kw={"width_ratios": [1.5, 2.3]})
    F = [("depicted", "all\ndepicted"), ("toy", "toy"), ("picture", "pic-\nture"), ("screen", "screen")]
    for i, (f, lab) in enumerate(F):
        for j, k in enumerate(("bv", "vedb")):
            u = allu[allu.dataset == k]; x = i + (j - .5) * .38
            FS.units(a, x, u[f] * 100, C[k], width=.3, s=9 if k == "bv" else 4, seed=i * 2 + j)
            FS.mean_se(a, x, u[f].mean() * 100, u[f].std() / np.sqrt(len(u)) * 100, width=.3)
    a.set_xticks(range(len(F)), [l for _, l in F]); a.set_ylabel("% of objects")
    a.set_ylim(bottom=0)
    n = allu.groupby("dataset").unit.nunique()
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C[k], label=f"{L[k]}, n = {n[k]}") for k in ("bv", "vedb")],
             loc="upper right", fontsize=10)
    doms = _doms(["animals", "vehicles", "toys", "food_drink", "outside", "household", "clothing", "furniture_rooms", "body_parts"])
    for i, d in enumerate(doms):
        for j, k in enumerate(("bv", "vedb")):
            u = domu[(domu.dataset == k) & (domu.domain == d)]; x = i + (j - .5) * .38
            FS.units(b, x, u.depicted * 100, C[k], width=.3, s=7 if k == "bv" else 3, seed=10 + i * 2 + j)
            if len(u) > 1:
                FS.mean_se(b, x, u.depicted.mean() * 100, u.depicted.std() / np.sqrt(len(u)) * 100, width=.3)
    b.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=40, ha="right")
    for t, d in zip(b.get_xticklabels(), doms):
        t.set_color(DC[d])
    b.set_ylabel("% depicted, by domain"); b.set_ylim(0, 100)
    b.set_title("dots: children / sessions (≥ 20 objects); line: mean ± 1 SE", loc="left", fontsize=10, color=ts.MUTED)
    fig.tight_layout(); ts.save(fig, "q05_depictions_bv_vedb")


def q06_size():
    s = pd.read_csv(Q / f"results/bv_vedb_renamed{_T}_core/scene_statistics_se.csv")   # uncertainty_bv_vedb.py
    s = s[s.scope == "all frames"].set_index(["group", "measure"])
    um = pd.read_csv(Q / f"out/private/bv_vedb_renamed{_T}/scene_unit_means.csv")
    um = um[um.scope == "all frames"]
    rng = np.random.default_rng(0)
    g = pd.read_csv(Q / f"results/controls_renamed{_T}_core/geom_describe.csv")
    g = g[(g.model == "dinov3") & (g.part == "all")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ts.SLIDE_W, ts.SLIDE_H), gridspec_kw={"width_ratios": [1, 2.3]})
    for j, (grp, k) in enumerate((("all ages", "bv"), ("VEDB", "vedb"))):
        r = s.loc[(grp, "median_area")]
        v = um[um.group == grp].median_area * 100                      # one dot per child / session
        a.scatter(j * 1.4 + rng.uniform(-.22, .22, len(v)), v, s=9 if k == "bv" else 3, color=C[k], alpha=.45 if k == "bv" else .2,
                  lw=0, zorder=1)
        a.errorbar(j * 1.4, r["mean"] * 100, yerr=r.se * 100, fmt="_", ms=26, mew=2.5, color=ts.INK, capsize=5, lw=2, zorder=3)
        a.text(j * 1.4 + .3, r["mean"] * 100, f"{r['mean'] * 100:.1f}%", ha="left", va="center", fontsize=13, fontweight="bold")
    a.set_xticks([0, 1.4], ["BabyView\n(n = 35)", "VEDB\n(n = 709)"]); a.set_xlim(-.5, 2.15)
    a.set_ylabel("median object size (% of frame)"); a.set_ylim(bottom=0)
    import unit_shares as US                                          # every child / session, core categories
    core = [l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_core.txt") if l.strip()]
    tu = US.per_unit(DOM, categories=core)
    doms = _doms(["animals", "clothing", "food_drink", "furniture_rooms", "household", "toys", "vehicles"])
    for i, d in enumerate(doms):
        for j, k in enumerate(("bv", "vedb")):
            u = tu[(tu.dataset == k) & (tu.domain == d)]; x = i + (j - .5) * .38
            FS.units(b, x, u.edge * 100, C[k], width=.3, s=7 if k == "bv" else 3, seed=20 + i * 2 + j)
            FS.mean_se(b, x, u.edge.mean() * 100, u.edge.std() / np.sqrt(len(u)) * 100, width=.3)
    b.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=40, ha="right")
    for tk, d in zip(b.get_xticklabels(), doms):
        tk.set_color(DC[d])
    b.set_ylabel("% cut off by frame edge"); b.set_ylim(0, 100)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=C[k], label=L[k]) for k in ("bv", "vedb")], loc="upper left")
    b.set_title("dots: children / sessions (≥ 20 objects); line: mean ± 1 SE", loc="left", fontsize=10, color=ts.MUTED)
    fig.tight_layout(); ts.save(fig, "q06_size_truncation_bv_vedb")


def tables():
    z = np.load(Q / f"out/crossdataset_renamed{_T}/tables_dinov3.npz", allow_pickle=True)
    return {k: pd.DataFrame(z[f"{k}__means"], index=list(z[f"{k}__categories"])) for k in ("bv", "vedb", "things", "imagenet")}


def rdm(df, cs):
    X = df.loc[cs].values.astype(float); X = (X - X.mean(0)) / (X.std(0) + 1e-10)
    return squareform(pdist(X, "cosine"))


C["imagenet"] = ts.DS_COLOR["imagenet"]; L["imagenet"] = "ImageNet"
TITLE_C = {"things": ts.DS_COLOR["things"], "imagenet": ts.DS_COLOR["imagenet"], "vedb": C["vedb"], "bv": C["bv"]}


def q07_rdms(keys=("things", "imagenet", "bv"), name="q07_rdms_things_imagenet_bv_dinov3"):
    from scipy.cluster.hierarchy import leaves_list, linkage
    T = tables()
    order = []
    for d in FS.DOMAIN_ORDER:
        cs = [c for c in CORE if DOM.get(c) == d]
        if len(cs) > 2:
            D = rdm(T["bv"], CORE)[np.ix_([CORE.index(c) for c in cs], [CORE.index(c) for c in cs])]
            cs = [cs[i] for i in leaves_list(linkage(squareform(D, checks=False), "average"))]
        order += cs
    R = {k: rdm(T[k], order) for k in keys}
    iu = np.triu_indices(len(order), 1)
    n = len(order)
    fig, axs = plt.subplots(1, len(keys), figsize=(2.75 * len(keys) + 1.4, 3.5))
    doms = [DOM[c] for c in order]
    starts = [0] + [i for i in range(1, n) if doms[i] != doms[i - 1]]; ends = starts[1:] + [n]
    for ax, k in zip(axs, keys):
        D = R[k].copy(); v = D[iu]
        D = np.searchsorted(np.sort(v), D) / len(v); np.fill_diagonal(D, 0)      # percentile within each RDM
        ax.imshow(D, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        for s in starts[1:]:
            ax.axhline(s - .5, color="white", lw=.6); ax.axvline(s - .5, color="white", lw=.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(L[k], fontsize=12, color=TITLE_C[k], fontweight="bold", pad=16)
        if k != "bv":
            ax.text(0.5, 1.015, f"ρ with BabyView = {spearmanr(R[k][iu], R['bv'][iu]).statistic:.2f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5, color=ts.MUTED)
    # every domain label, spread apart with leader lines where blocks are small
    ax0 = axs[0]; mids = [(a + b - 1) / 2 for a, b in zip(starts, ends)]; ys = list(mids); gapmin = n * 0.06
    for _ in range(50):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gapmin:
                ys[i] = ys[i - 1] + gapmin
        over = ys[-1] - (n - 1)
        if over > 0:
            ys = [y - over for y in ys]
    for a, m, y in zip(starts, mids, ys):
        d = doms[a]
        ax0.annotate(DL.get(d, d), xy=(-0.5, m), xytext=(-n * 0.09, y), textcoords="data", ha="right", va="center",
                     fontsize=8.5, color=DC.get(d, DC["other"]), fontweight="bold", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=DC.get(d, DC["other"]), lw=0.8, shrinkA=1, shrinkB=0))
    fig.tight_layout(); fig.subplots_adjust(left=0.2); ts.save(fig, name)


def q07b_rdms_things_bv():
    q07_rdms(("things", "imagenet", "bv"), "q07b_rdms_things_imagenet_bv_dinov3")


def q07c_rdms_all():
    q07_rdms(("things", "imagenet", "vedb", "bv"), "q07c_rdms_things_imagenet_vedb_bv_dinov3")


def ci(s):
    return [float(x) for x in s.strip("[]").split(",")]


def q08_cluster_overall():
    cs = pd.read_csv(RES / "cluster_strength_core.csv"); cs = cs[cs.model == "dinov3"].set_index("dataset")
    df = pd.read_csv(RES / "cluster_strength_differences_core.csv")
    df = df[(df.model == "dinov3") & (df.measure == "delta_sd")].set_index("pair")
    slots = {"things": 0, "imagenet": 1, "vedb": 2, "bv": 3}
    lab = {"things": "THINGS", "imagenet": "ImageNet", "vedb": "VEDB\n(adult)", "bv": "BabyView\n(child)"}
    for build, ks in (("1_curated_bv", ["things", "imagenet", "bv"]), ("2_vedb", ["things", "imagenet", "vedb", "bv"]), ("3_things_vedb_bv", ["things", "vedb", "bv"])):
        slots = ({k: j for j, k in enumerate(ks)} if build.startswith("3") else
                 {"things": 0, "imagenet": 1, "vedb": 2, "bv": 3})
        fig, ax = plt.subplots(figsize=(ts.SLIDE_W * (0.6 if build.startswith("3") else 0.7), ts.SLIDE_H))
        for k in ks:
            v = cs.loc[k, "delta_sd"]; lo, hi = ci(cs.loc[k, "delta_sd_ci"])
            ax.bar(slots[k], v, color=C[k], width=0.62)
            ax.errorbar(slots[k], v, yerr=[[v - lo], [hi - v]], color=ts.INK, capsize=4)
            ax.text(slots[k], hi + 0.02, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")
        def br(a, y):
            q = df.loc[f"bv - {a}", "q_bh"]; st = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."
            ax.plot([slots[a], slots[a], slots["bv"], slots["bv"]], [y - .015, y, y, y - .015], color=ts.INK, lw=1)
            ax.text((slots[a] + slots["bv"]) / 2, y + .005, st, ha="center", va="bottom", fontsize=12,
                    fontweight="normal" if st == "n.s." else "bold")
        if "imagenet" in ks:
            br("things", 1.06); br("imagenet", 0.98)
            if "vedb" in ks:
                br("vedb", 0.90)
        else:
            br("things", 1.07); br("vedb", 0.97)
        ax.set_xticks([slots[k] for k in slots], [lab[k] for k in slots]); ax.set_xlim(-0.6, len(slots) - 0.4)
        ax.set_ylim(0, 1.14); ax.set_ylabel("domain clustering (Δ / SD)")
        fig.tight_layout(); ts.save(fig, f"q08_cluster_overall_{build}")


def q09_cluster_by_domain():
    dd = pd.read_csv(RES / "cluster_strength_by_domain_core.csv"); dd = dd[dd.model == "dinov3"]
    sg = pd.read_csv(RES / "cluster_strength_by_domain_differences_core.csv")
    sg = sg[(sg.model == "dinov3") & (sg.measure == "delta_sd")]
    doms = _doms(["animals", "vehicles", "food_drink", "clothing", "furniture_rooms", "household", "toys"])
    for build, ks in (("1_curated_bv", ["things", "imagenet", "bv"]), ("2_vedb", ["things", "imagenet", "vedb", "bv"]), ("3_things_vedb_bv", ["things", "vedb", "bv"])):
        fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
        w = 0.8 / len(ks); off = {k: (j - (len(ks) - 1) / 2) * w for j, k in enumerate(ks)}
        for k in ks:
            s = dd[dd.dataset == k].set_index("domain").loc[doms]; x = np.arange(len(doms)) + off[k]
            ax.bar(x, s.delta_sd, w * 0.92, color=C[k], label=L[k])
            ax.errorbar(x, s.delta_sd, yerr=[s.delta_sd - s.ci_low_sd, s.ci_high_sd - s.delta_sd], fmt="none", color=ts.INK, capsize=2, lw=1)
            if k != "bv":
                st = sg[sg.dataset == k].set_index("domain").reindex(doms).stars.fillna("")
                for xi, hi, t in zip(x, s.ci_high_sd, st):
                    if t:
                        ax.text(xi, hi + .05, t, ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=30, ha="right", rotation_mode="anchor")
        for tk, d in zip(ax.get_xticklabels(), doms):
            tk.set_color(DC[d]); tk.set_fontweight("bold")
        ax.set_ylabel("domain clustering (Δ / SD)"); ax.set_ylim(0, 4.4); ax.legend(loc="upper right", ncol=2 if len(ks) > 3 else 1)
        fig.tight_layout(); ts.save(fig, f"q09_cluster_by_domain_{build}")


def q10_real_only():
    dr = pd.read_csv(RES / "cluster_strength_by_domain_real.csv"); dr = dr[(dr.model == "dinov3") & (dr.domain != "ALL")]
    doms = _doms(["animals", "vehicles", "food_drink", "clothing", "furniture_rooms", "household", "toys"])
    doms = [d for d in doms if d in set(dr.domain)]
    ks = ["things", "imagenet", "vedb_real", "bv_real", "bv"]; w = 0.16
    off = {k: (j - 2) * w for j, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    for k in ks:
        s = dr[dr.dataset == k].set_index("domain").reindex(doms); x = np.arange(len(doms)) + off[k]
        kw = dict(color="white", edgecolor=C["bv"], lw=1.5, hatch="//") if k == "bv" else dict(color=C[k])
        ax.bar(x, s.delta_sd, w * 0.92, label="BabyView, all formats (toys, pictures)" if k == "bv" else L[k], **kw)
        ax.errorbar(x, s.delta_sd, yerr=[s.delta_sd - s.ci_low_sd, s.ci_high_sd - s.delta_sd], fmt="none", color=ts.INK, capsize=2, lw=1)
    ax.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=30, ha="right", rotation_mode="anchor")
    for tk, d in zip(ax.get_xticklabels(), doms):
        tk.set_color(DC[d]); tk.set_fontweight("bold")
    ax.set_ylabel("domain clustering (Δ / SD)"); ax.legend(loc="upper right", fontsize=10, ncol=2); ax.set_ylim(0, 4.8)
    fig.tight_layout(); ts.save(fig, "q10_real_only_by_domain")


def q16_frequency_imagenet():
    z = np.load(Q / f"out/crossdataset_renamed{_T}/tables_dinov3.npz", allow_pickle=True)
    ni = pd.Series(z["imagenet__counts"].astype(float), index=list(z["imagenet__categories"])); ni = ni / ni.sum()
    pb = prevalence(PRIV / f"open_renamed{_T}")
    s = pd.DataFrame({"bv": pb, "im": ni}).dropna(); s = s[(s.bv > 0) & (s.im > 0)]
    s["dom"] = [DOM.get(c, "other") for c in s.index]
    r = pd.read_csv(Q / f"results/controls_renamed{_T}/prevalence_spearman.csv").set_index("versus").loc["imagenet"]
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    ax.scatter(s.bv, s.im, s=26, c=[DC.get(d, DC["other"]) for d in s.dom], alpha=.85, lw=0)
    s["r"] = np.log10(s.bv) - np.log10(s.im) * 1.0
    placed = []
    for side in (s.sort_values("r", ascending=False), s.sort_values("r")):
        k = 0
        for c, row in side.iterrows():
            xy = np.log10([row.bv, row.im])
            if k >= 5 or any(np.hypot(*(xy - q)) < 0.45 for q in placed):
                continue
            placed.append(xy); k += 1
            ax.annotate(c, (row.bv, row.im), xytext=(4, 2), textcoords="offset points", fontsize=11, color=ts.INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("proportion of frames with object detection, BabyView"); ax.set_ylabel("proportion of images, ImageNet")
    ax.set_title(f"{len(s)} shared categories, Spearman ρ = {r.spearman:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", loc="left")
    fig.tight_layout(); ts.save(fig, "q16_frequency_bv_vs_imagenet")


def q11_children(min_n=30):
    T = tables()
    z = np.load(PRIV / f"open_renamed{_T}/sums_dinov3.npz", allow_pickle=True)
    subj, cats = list(z["subjects"]), list(z["categories"])
    S, N = z["sum_all"], z["n_all"]
    ft = pd.read_csv(PRIV / f"open_renamed{_T}/frames_total.csv").set_index("subject").frames
    rows = []
    for i, sb in enumerate(subj):
        if ft.get(sb, 0) < 10_000:
            continue
        cs = [c for c in CORE if c in cats and N[i, cats.index(c)] >= min_n and c in T["things"].index]
        idx = [cats.index(c) for c in cs]
        M = pd.DataFrame(S[i, idx] / N[i, idx, None], index=cs)
        sem = np.array([DOM[c] for c in cs]); iu = np.triu_indices(len(cs), 1); same = sem[iu[0]] == sem[iu[1]]
        out = {}
        for k, df in (("child", M), ("things", T["things"])):
            x = rdm(df, cs)[iu]; out[k] = (x[~same].mean() - x[same].mean()) / x.std()
        rows.append(dict(subject=sb, n=len(cs), child=out["child"], things=out["things"]))
    r = pd.DataFrame(rows).sort_values("child")
    r.to_csv(ts.OUT / "q11_children_delta_sd.csv", index=False)
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    y = np.arange(len(r))
    ax.hlines(y, r.things, r.child, color=ts.FAINT, lw=2)
    ax.scatter(r.things, y, s=45, color=C["things"], label="THINGS, same categories", zorder=3)
    ax.scatter(r.child, y, s=55, color=C["bv"], label="each child", zorder=3)
    ax.set_yticks(y, [f"child {k + 1}" for k in range(len(r))], fontsize=9); ax.set_xlabel("domain clustering (Δ / SD)")
    ax.legend(loc="lower right"); ax.set_title(f"{len(r)} children with ≥ 10,000 frames; {int(r.n.min())}–{int(r.n.max())} core categories each",
                                               loc="left", fontsize=13)
    fig.tight_layout(); ts.save(fig, "q11_children_vs_things")
    return r


def q12_age():
    R = pd.read_csv(Q / f"results/bv_vedb_renamed{_T}_core/rdm_split_half.csv")
    R = R[(R.model == "dinov3") & (~R.real)]
    J = pd.read_csv(Q / f"results/bv_vedb_renamed{_T}_core/js_divergence.csv")
    J = J[(J.scope == "home-scene frames") & (J.level == "category") & J.comparison.str.startswith("excess")]
    G = ["<12 mo", "12-18 mo", "18-24 mo", "24+ mo"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.85))
    for comp, col, lab in (("BabyView vs VEDB", C["vedb"], "children vs adults"), ("within BabyView", "#9e9e9e", "children vs children (reliability)")):
        s = R[R.comparison == comp].set_index("group").reindex(G)
        a.plot(range(4), s.rho, "-o", color=col, lw=2.4, ms=8, label=lab)
    a.set_xticks(range(4), G); a.set_ylim(0, 1); a.set_ylabel("RDM correlation (ρ)"); a.legend(loc="lower left", fontsize=11)
    a.set_title("Category structure", loc="left")
    s = J.set_index("group").reindex(G)
    b.errorbar(range(4), s.js, yerr=[s.js - s.lo, s.hi - s.js], fmt="-o", color=C["bv"], lw=2.4, ms=8, capsize=4)
    b.set_xticks(range(4), G); b.set_ylim(0, 0.2); b.axhline(0, color=ts.FAINT, lw=1)
    b.set_ylabel("divergence from adults (JS, excess)"); b.set_title("Which categories (home scenes)", loc="left")
    fig.tight_layout(); ts.save(fig, "q12_age_structure_frequency")


def q13_similarity():
    c = pd.read_csv(Q / f"results/open_renamed{_T}/open_categorywise_bv_things.csv"); c = c[c.model == "dinov3"]
    doms = c.groupby("domain").cosine.median().sort_values().index.tolist()
    doms = [d for d in doms if d in DC]
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    rng = np.random.default_rng(0)
    for i, d in enumerate(doms):
        v = c[c.domain == d].cosine.values
        ax.scatter(rng.uniform(-.28, .28, len(v)) + i, v, s=10, color=DC[d], alpha=.6, lw=0)
        ax.hlines(np.median(v), i - .32, i + .32, color=ts.INK, lw=2.2)
    ax.set_xticks(range(len(doms)), [DL.get(d, d) for d in doms], rotation=35, ha="right")
    for tk, d in zip(ax.get_xticklabels(), doms):
        tk.set_color(DC[d]); tk.set_fontweight("bold")
    ax.set_ylabel("similarity to THINGS\n(DINOv3 cosine)")
    ax.set_title(f"{len(c)} categories with a THINGS match; line = median", loc="left", fontsize=13)
    fig.tight_layout(); ts.save(fig, "q13_similarity_to_things_by_domain")


def q14_depicted_categories(n_show=30, min_bv=1000):
    g = pd.read_csv(Q / f"results/crossdataset_renamed{_T}/depiction_gap_bv_vs_vedb.csv")
    g = g[g.bv_n >= min_bv].sort_values("bv_depicted", ascending=False).head(n_show)
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    x = np.arange(len(g))
    ax.bar(x, g.bv_depicted * 100, color=[DC.get(d, DC["other"]) for d in g.domain], width=0.75, label="BabyView (child), bar colour = domain")
    v = g[g.vedb_n >= 30]
    ax.scatter(x[g.vedb_n.values >= 30], v.vedb_depicted * 100, color=C["vedb"], s=55, zorder=3, edgecolor="white",
               label="VEDB (adult), where ≥ 30 boxes")
    ax.set_xticks(x, g.category, rotation=60, ha="right", fontsize=11)
    for tk, d in zip(ax.get_xticklabels(), g.domain):
        tk.set_color(DC.get(d, DC["other"]))
    ax.set_ylabel("% depicted"); ax.set_ylim(0, 105); ax.set_xlim(-0.7, len(g) - 0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.32), ncol=2, fontsize=11)
    ax.set_title(f"Most-depicted categories with ≥ {min_bv:,} boxes in BabyView", loc="left", fontsize=13)
    fig.tight_layout(); ts.save(fig, "q14_most_depicted_categories")


def q15_long_tail_manuscript(key="bv", ncol=4, per_domain=10):
    """Manuscript Figure 2 layout: top 50 categories, then the top categories within each domain."""
    root = PRIV / (f"open_renamed{_T}" if key == "bv" else f"open_vedb_renamed{_T}")
    tier1 = [l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()]
    prev = prevalence(root).reindex(tier1).dropna()
    prev = prev[prev > 0].sort_values(ascending=False)
    dom = pd.Series({c: DOM.get(c, "other") for c in prev.index})
    doms = [d for d in FS.DOMAIN_ORDER if (dom == d).any()]
    nrow = int(np.ceil(len(doms) / ncol))
    fig = plt.figure(figsize=(15, 4.4 + 2.75 * nrow), constrained_layout=True)
    gs = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.5] + [1] * nrow)

    def bars(ax, v, fs, rot):
        ax.bar(range(len(v)), v.values, color=[DC.get(dom[c], DC["other"]) for c in v.index], width=0.8)
        ax.set_xticks(range(len(v)), v.index, rotation=rot, ha="right", fontsize=fs)
        ax.set_xlim(-0.7, len(v) - 0.3); ax.tick_params(axis="y", labelsize=10)

    ax = fig.add_subplot(gs[0, :]); bars(ax, prev.head(50), 10.5, 60)
    ax.set_ylabel("proportion of frames\nwith object detection", fontsize=12)
    n_frames = pd.read_csv(root / "frames_total.csv").frames.sum()
    ax.set_title(f"{L[key]}: top 50 of {len(prev)} categories ({n_frames:,} frames)", loc="left", fontsize=15,
                 color=C[key], fontweight="bold")
    for k, d in enumerate(doms):
        a = fig.add_subplot(gs[1 + k // ncol, k % ncol])
        bars(a, prev[dom == d].head(per_domain), 9.5, 55)
        a.set_title(f"{DL.get(d, d)} ({int((dom == d).sum())})", color=DC[d], fontsize=13, fontweight="bold")
        if k % ncol == 0:
            a.set_ylabel("proportion of frames\nwith object detection", fontsize=9.5)
    ts.save(fig, f"q15_long_tail_top50_domains_{key}")


CCN_DOMS = ["furniture_rooms", "household", "toys", "body_parts", "food_drink", "clothing", "outside", "vehicles", "animals"]
CCN_DOMS = _doms(CCN_DOMS)
CCN_DL = {"furniture_rooms": "furniture", "household": "household objects", "toys": "toys", "body_parts": "body parts",
          "food_drink": "food&drinks", "clothing": "clothing", "outside": "outside", "vehicles": "vehicles",
          "animals": "animals"}
if _THINGS:
    CCN_DL = {d: FS.DOMAIN_LABEL[d] for d in CCN_DOMS}


def q15c_long_tail_ccn(key="bv", n_top=60, per_domain=10):
    """CCN-deck layout (CCN slide 20): top-n bars with a domain legend in coloured text at the right, then the top
    categories of the nine CDI domains in two rows (5 + 4), one shared y label. People and 'other' left out, as at CCN."""
    root = PRIV / (f"open_renamed{_T}" if key == "bv" else f"open_vedb_renamed{_T}")
    tier1 = [l.strip() for l in open(ts.ANALYSIS.parent / "data" / "included_categories_qwen_open_tier1.txt") if l.strip()]
    prev = prevalence(root).reindex(tier1).dropna()
    dom = pd.Series({c: DOM.get(c, "other") for c in prev.index})
    prev = prev[(prev > 0) & dom.isin(CCN_DOMS)].sort_values(ascending=False); dom = dom[prev.index]
    fig = plt.figure(figsize=(ts.SLIDE_W, ts.SLIDE_H * 1.02))
    gs = fig.add_gridspec(3, 10, height_ratios=[1.15, 1, 1], hspace=0.95, wspace=1.6,
                          left=0.085, right=0.985, top=0.975, bottom=0.095)

    def bars(ax, v, fs):
        ax.bar(range(len(v)), v.values, color=[DC[dom[c]] for c in v.index], width=0.8)
        ax.set_xticks(range(len(v)), v.index, rotation=45, ha="right", rotation_mode="anchor", fontsize=fs)
        ax.set_xlim(-0.7, len(v) - 0.3); ax.tick_params(axis="y", labelsize=7.5, length=2, pad=1)
        ax.tick_params(axis="x", length=0, pad=1)

    ax = fig.add_subplot(gs[0, :8]); bars(ax, prev.head(n_top), 6.5)
    lax = fig.add_subplot(gs[0, 8:]); lax.axis("off")
    for i, d in enumerate(CCN_DOMS):
        lax.text(1.0, 1 - i * 0.12, CCN_DL[d], color=DC[d], ha="right", va="top", fontsize=8.5, fontweight="bold",
                 transform=lax.transAxes)
    for k, d in enumerate(CCN_DOMS):
        r, c0 = (1, 2 * k) if k < 5 else (2, 2 * (k - 5))
        a = fig.add_subplot(gs[r, c0:c0 + 2]); bars(a, prev[dom == d].head(per_domain), 6.8)
        a.set_title(CCN_DL[d], color=DC[d], fontsize=9.5, fontweight="bold", loc="left", pad=2)
    fig.text(0.012, 0.55, "Proportion of frames with object detection", rotation=90, va="center", fontsize=10,
             fontweight="bold")
    ts.save(fig, f"q15c_long_tail_ccn_{key}", scale=1.0)


def _rdm_order(T):
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = []
    for d in FS.DOMAIN_ORDER:
        cs = [c for c in CORE if DOM.get(c) == d]
        if len(cs) > 2:
            D = rdm(T["bv"], CORE)[np.ix_([CORE.index(c) for c in cs], [CORE.index(c) for c in cs])]
            cs = [cs[i] for i in leaves_list(linkage(squareform(D, checks=False), "average"))]
        order += cs
    return order


def _pct(D):
    iu = np.triu_indices(len(D), 1); v = np.sort(D[iu]); P = np.searchsorted(v, D) / len(v); np.fill_diagonal(P, 0)
    return P


def _blocks(ax, starts, lw=0.7):
    for s_ in starts[1:]:
        ax.axhline(s_ - .5, color="white", lw=lw); ax.axvline(s_ - .5, color="white", lw=lw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


DS_NAME = {"bv": "BabyView Dataset", "things": "THINGS Dataset", "imagenet": "ImageNet Dataset", "vedb": "VEDB Dataset"}


def q07p_rdm_pair(other="things"):
    """CCN-deck layout (CCN slide 47): BabyView and one other dataset side by side, colour bar and domain labels at the
    left, Spearman's rho between the two RDMs underneath."""
    T = tables(); order = _rdm_order(T); n = len(order); iu = np.triu_indices(n, 1)
    R = {k: rdm(T[k], order) for k in ("bv", other)}
    doms = [DOM[c] for c in order]
    starts = [0] + [i for i in range(1, n) if doms[i] != doms[i - 1]]; ends = starts[1:] + [n]
    fig = plt.figure(figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.92))
    cax = fig.add_axes([0.045, 0.2, 0.018, 0.55])
    axs = [fig.add_axes([0.25 + i * 0.37, 0.12, 0.36, 0.36 * ts.SLIDE_W / (ts.SLIDE_H * 0.92)]) for i in range(2)]
    for ax, k in zip(axs, ("bv", other)):
        im = ax.imshow(_pct(R[k]), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        _blocks(ax, starts); ax.set_title(DS_NAME[k], fontsize=12, fontweight="bold", pad=4)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["0", "1"], fontsize=8); cb.outline.set_visible(False)
    cax.text(0.5, 1.04, "Dissimilar", transform=cax.transAxes, ha="center", va="bottom", fontsize=9, color="#E0B040",
             fontweight="bold")
    cax.text(0.5, -0.04, "Similar", transform=cax.transAxes, ha="center", va="top", fontsize=9, color="#4B2C83",
             fontweight="bold")
    ax0 = axs[0]; mids = [(a + b - 1) / 2 for a, b in zip(starts, ends)]; ys = list(mids); gap = n * 0.065
    for _ in range(50):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                ys[i] = ys[i - 1] + gap
        if ys[-1] > n - 1:
            ys = [y - (ys[-1] - (n - 1)) for y in ys]
    for a, m, y in zip(starts, mids, ys):
        d = doms[a]
        ax0.annotate(DL.get(d, d), xy=(-0.5, m), xytext=(-n * 0.06, y), ha="right", va="center", fontsize=10,
                     color=DC.get(d, DC["other"]), fontweight="bold", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=DC.get(d, DC["other"]), lw=0.7, shrinkA=1, shrinkB=0))
    rho = spearmanr(R["bv"][iu], R[other][iu]).statistic
    from matplotlib.lines import Line2D
    x1, x2, y0 = 0.25 + 0.18, 0.62 + 0.18, 0.105                      # bracket joining the two panel centres
    fig.add_artist(Line2D([x1, x1, x2, x2], [y0, y0 - 0.02, y0 - 0.02, y0], color=ts.INK, lw=1.2))
    fig.add_artist(Line2D([(x1 + x2) / 2] * 2, [y0 - 0.02, y0 - 0.035], color=ts.INK, lw=1.2))
    fig.text((x1 + x2) / 2, 0.02, f"Spearman's ρ = {rho:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ts.save(fig, f"q07p_rdm_bv_{other}", scale=1.0)


def q11p_subject_rdms(k=8):
    """CCN-deck layout (CCN slide 48): RDMs of the k children with the most frames, BabyView's category order."""
    T = tables(); order = _rdm_order(T)
    z = np.load(PRIV / f"open_renamed{_T}/sums_dinov3.npz", allow_pickle=True)
    subj, cats = list(z["subjects"]), list(z["categories"]); S, N = z["sum_all"], z["n_all"]
    ft = pd.read_csv(PRIV / f"open_renamed{_T}/frames_total.csv").set_index("subject").frames
    top = [sb for sb in ft.sort_values(ascending=False).index if sb in subj][:k]
    fig, axs = plt.subplots(2, 4, figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.95))
    for j, (ax, sb) in enumerate(zip(axs.flat, top)):
        i = subj.index(sb)
        cs = [c for c in order if c in cats and N[i, cats.index(c)] >= 10]
        M = pd.DataFrame(S[i, [cats.index(c) for c in cs]] / N[i, [cats.index(c) for c in cs], None], index=cs)
        doms = [DOM[c] for c in cs]; starts = [0] + [q for q in range(1, len(cs)) if doms[q] != doms[q - 1]]
        ax.imshow(_pct(rdm(M, cs)), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        _blocks(ax, starts, lw=0.5); ax.set_title(f"Subject {j + 1}", fontsize=11, fontweight="bold", pad=3)
    fig.text(0.01, 0.01, f"RDMs from the top {k} children with the densest data (categories with ≥10 crops each)",
             fontsize=8.5, style="italic", color=ts.MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05, wspace=0.08, hspace=0.18)
    ts.save(fig, "q11p_subject_rdms", scale=1.0)


def q08e_ego_vs_curated(model="dinov3"):
    """Domain clustering (Δ / SD) per dataset on the core categories: point ± 1 SE (within-domain category bootstrap,
    10,000 draws); short dashed lines = mean of the three curated and of the two first-person datasets; bracket = mean
    first-person minus mean curated on the same draws, 95% CI and two-sided bootstrap p (ego_vs_curated.py)."""
    cs = pd.read_csv(RES / "cluster_strength_core.csv"); cs = cs[cs.model == model].set_index("dataset")
    ev = pd.read_csv(RES / "ego_vs_curated.csv")
    ev = ev[(ev.model == model) & (ev.measure == "delta_sd")].set_index("pair")
    order = [("things", "THINGS"), ("imagenet", "ImageNet"), ("lvis", "LVIS"), ("vedb", "VEDB\n(adult)"), ("bv", "BabyView\n(child)")]
    col = {k: ts.DS_COLOR[k] for k, _ in order}
    x = [0, 1, 2, 3.5, 4.5]
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W * 0.8, ts.SLIDE_H))
    hi, lo = 0, np.inf
    for xi, (k, _) in zip(x, order):
        v, se = cs.loc[k, "delta_sd"], cs.loc[k, "delta_sd_se"]
        ax.errorbar(xi, v, yerr=se, fmt="o", ms=11, color=col[k], mec="white", mew=1.2, capsize=5, lw=2, zorder=3)
        ax.text(xi + 0.17, v, f"{v:.2f}", va="center", ha="left", fontsize=12, color=ts.INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        hi = max(hi, v + se); lo = min(lo, v - se)
    mc = np.mean([cs.loc[k, "delta_sd"] for k in ("things", "imagenet", "lvis")])
    me = np.mean([cs.loc[k, "delta_sd"] for k in ("vedb", "bv")])
    ax.hlines(mc, -0.35, 2.35, color=ts.MUTED, lw=1.2, ls="--", zorder=1)
    ax.hlines(me, 3.15, 4.85, color=ts.MUTED, lw=1.2, ls="--", zorder=1)
    r = ev.loc["ego - curated"]; st = FS.stars(r.q_bh)
    y = hi + (hi - lo) * .12; tk = (hi - lo) * .04
    ax.plot([1, 1, 4, 4], [y - tk, y, y, y - tk], color=ts.INK, lw=1.2)
    ax.text(2.5, y + tk / 2, f"first-person − curated: {r['diff']:+.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] {st}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x, [l for _, l in order]); ax.set_xlim(-0.6, 5.1)
    ax.set_ylim(max(0, lo - (y - lo) * .25), y + (y - lo) * .22)                  # dot plot: zoom to the data
    ax.set_ylabel("domain clustering (Δ / SD)")
    for xc, lab in ((1, "curated photographs"), (4, "first-person video")):
        ax.annotate(lab, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -58), textcoords="offset points",
                    ha="center", va="top", fontsize=11, color=ts.MUTED)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.25); ts.save(fig, f"q08e_ego_vs_curated_{model}")


def q09b_bv_vs_each_by_domain(model="dinov3"):
    """BabyView as the anchor: per domain, BabyView's domain clustering (Δ / SD) minus each other dataset's (THINGS,
    ImageNet, LVIS, VEDB), on the core categories. Point = difference, whisker = ±1 SE of the difference (within-domain
    category bootstrap, 10,000 draws shared by all datasets); stars = two-sided bootstrap test, BH across
    domains x datasets (crossdataset_analysis.ipynb section 4b). Above 0: BabyView clusters the domain more."""
    sg = pd.read_csv(RES / "cluster_strength_by_domain_differences_core.csv")
    sg = sg[(sg.model == model) & (sg.measure == "delta_sd")]
    doms = _doms(["animals", "vehicles", "food_drink", "furniture_rooms", "household", "clothing", "toys"])
    doms = [d for d in doms if d in set(sg.domain)]
    ks = ["things", "imagenet", "lvis", "vedb"]
    lab = {"things": "THINGS", "imagenet": "ImageNet", "lvis": "LVIS", "vedb": "VEDB (adult)"}
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    w = 0.21
    ax.axhline(0, color=ts.MUTED, lw=1.2, ls="--", zorder=1)
    shown = sg[sg.domain.isin(doms) & sg.dataset.isin(ks)]
    top = (shown["diff"] + shown.se).max(); bot = min(0, (shown["diff"] - shown.se).min()); rng_ = top - bot
    row = top + rng_ * .07                                               # one row of stars above the data
    for j, k in enumerate(ks):
        s_ = sg[sg.dataset == k].set_index("domain").reindex(doms); x = np.arange(len(doms)) + (j - 1.5) * w
        ax.errorbar(x, s_["diff"], yerr=s_.se, fmt="o", ms=8, color=ts.DS_COLOR[k], mec="white", mew=1, capsize=3, lw=1.6,
                    label=f"BabyView − {lab[k]}", zorder=3)
        for xi, q_ in zip(x, s_.q_bh):
            st = FS.stars(q_, "")
            if st:
                ax.text(xi, row + (j % 2) * rng_ * .05, st, ha="center", va="center", fontsize=8, fontweight="bold",
                        color=ts.DS_COLOR[k])                                 # two staggered rows: neighbours never touch
    ax.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=30, ha="right", rotation_mode="anchor")
    for tk, d in zip(ax.get_xticklabels(), doms):
        tk.set_color(DC[d]); tk.set_fontweight("bold")
    ax.set_xlim(-0.6, len(doms) - 0.4); ax.set_ylim(bot - rng_ * .05, row + rng_ * .10)
    ax.set_ylabel("BabyView − dataset\ndomain clustering (Δ / SD)")
    ax.set_title("above 0: BabyView clusters the domain more; stars (top row): BH q across domains × datasets",
                 loc="left", fontsize=10, color=ts.MUTED)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)
    fig.tight_layout(); ts.save(fig, f"q09b_bv_vs_each_by_domain_{model}")


def q09e_ego_vs_curated_by_domain(model="dinov3"):
    """Per CDI domain: delta / SD (±1 bootstrap SE) for THINGS, ImageNet, LVIS (curated) and VEDB, BabyView (first-person); above
    each domain the pooled egocentric - curated difference (ego_vs_curated.py, BH across domains). 'outside' (3 core
    categories) left out, as in q09."""
    dd = pd.read_csv(RES / "cluster_strength_by_domain_core.csv"); dd = dd[dd.model == model]
    ev = pd.read_csv(RES / "ego_vs_curated_by_domain.csv")
    ev = ev[(ev.model == model) & (ev.pair == "ego - curated")].set_index("domain")
    doms = _doms(["animals", "vehicles", "food_drink", "furniture_rooms", "household", "clothing", "toys"])
    ks = ["things", "imagenet", "lvis", "vedb", "bv"]
    col = {"things": C["things"], "imagenet": ts.DS_COLOR["imagenet"], "lvis": ts.DS_COLOR["lvis"], "vedb": C["vedb"], "bv": C["bv"]}
    lab = {"things": "THINGS", "imagenet": "ImageNet", "lvis": "LVIS", "vedb": "VEDB (adult)", "bv": "BabyView (child)"}
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    w = 0.16; off = {k: (j - 2) * w + (0.04 if j >= 3 else 0) for j, k in enumerate(ks)}
    top = np.zeros(len(doms))
    for k in ks:
        s_ = dd[dd.dataset == k].set_index("domain").loc[doms]; x = np.arange(len(doms)) + off[k]
        ax.bar(x, s_.delta_sd, w * 0.92, color=col[k], label=lab[k])
        ax.errorbar(x, s_.delta_sd, yerr=s_.se_sd, fmt="none", color=ts.INK, capsize=1.5, lw=0.9)   # ±1 bootstrap SE
        top = np.maximum(top, (s_.delta_sd + s_.se_sd).values)
    for i, d in enumerate(doms):
        r = ev.loc[d]; q = r.q_bh; st = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."
        ax.text(i + 0.02, top[i] + 0.12, f"{r['diff']:+.2f}\n{st}", ha="center", va="bottom", fontsize=10,
                fontweight="bold" if st != "n.s." else "normal", color=ts.INK if st != "n.s." else ts.MUTED)
    ax.set_xticks(range(len(doms)), [DL[d] for d in doms], rotation=30, ha="right", rotation_mode="anchor")
    for tk, d in zip(ax.get_xticklabels(), doms):
        tk.set_color(DC[d]); tk.set_fontweight("bold")
    ax.set_ylabel("domain clustering (Δ / SD)"); ax.set_ylim(0, top.max() + 0.9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)
    ax.set_title("number above each domain: first-person − curated (mean of 2 vs mean of 3)", loc="left", fontsize=11,
                 color=ts.MUTED)
    fig.tight_layout(); ts.save(fig, f"q09e_ego_vs_curated_by_domain_{model}")


if __name__ == "__main__":
    ts.setup()
    if len(sys.argv) > 1:                           # e.g. python fig_qwen_talk.py q04_scatter q07_rdms
        for f in sys.argv[1:]:
            globals()[f]()
        sys.exit()
    q01_examples(); q02_long_tail()
    print("rank-frequency categories, rho:", q03_rank_frequency())
    print("scatter categories, rho:", q04_scatter())
    q05_formats(); q06_size(); q07b_rdms_things_bv(); q07c_rdms_all(); q08_cluster_overall(); q09_cluster_by_domain(); q10_real_only(); q16_frequency_imagenet()
    r = q11_children(); print(r.assign(diff=r.child - r.things).describe().round(3).to_string())
    q12_age(); q13_similarity(); q14_depicted_categories()
    q15_long_tail_manuscript("bv"); q15_long_tail_manuscript("vedb")
