#!/usr/bin/env python3
"""Response-letter figures for the two background tests, drawn the same way: domain clustering (delta / SD, DINOv3) as
the amount of background around each object changes, with BabyView in both.

  R1d  LVIS crops vs whole LVIS images, on the current pipeline's core categories (Qwen renamed labels, WordNet mapping
       mapping/map_lvis.csv), with BabyView crops on the same categories. An LVIS image counts for a category when it
       has a box of an LVIS category mapped to it; "single category": images with no box of another compared category.
       Category means = mean of L2-normalized vectors; z-scored over the categories compared; cosine RDMs; domains with
       >= 3 categories; delta = mean between-domain - mean within-domain distance, / SD of all distances. CIs: 1,000
       within-domain category bootstrap draws, shared across versions (so differences are paired).
  R1j  BabyView and VEDB with Qwen's box, SAM 3's box, and SAM 3's box with the background greyed out; plotted from
       analyze_background_control.py results (QOV_CATS=core), which uses the same delta / SD and bootstrap.

Domains follow QOV_DOMAINS (cdi: categories.csv; things27: data/category_domains_things27.csv domain_analysis).
Outputs: results/background_response[_things27]/lvis_whole_vs_crop{,_differences}.csv; figures to
analysis/manuscript-2026/revision_private/figures[_things27]/R1d_background.{png,pdf}, R1j_background_control.{png,pdf}

  QOV_LABELS=renamed [QOV_DOMAINS=things27] python background_response_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
import variant as V
import fig_style as FS
FS.apply()

REPO = HERE.parents[1]
CD = REPO / "analysis" / "comparison-datasets" / "outputs"
LVIS_IMG = Path("/ccn2a/dataset/babyview/related_data/lvis/outputs/image_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m")
CACHE = HERE / "out" / "crossdataset" / "lvis_whole_image_means_qwen_core_dinov3.npz"
TABLES = V.out_dir(HERE / "out" / "crossdataset")
RES = HERE / "results" / ("background_response" + ("_things27" if V.DOMAINS == "things27" else ""))
BGC = V.out_dir(HERE / "results" / "bv_vedb"); BGC = BGC.with_name(BGC.name + "_core")
FIGDIR = V.manuscript_figures_dir(REPO / "analysis" / "manuscript-2026" / "revision_private" / "figures")
CORE = [l.strip() for l in open(REPO / "data" / "included_categories_qwen_core.txt") if l.strip()]
MIN_IMG, N_BOOT = 10, 10000          # p floor 2 / N_BOOT
C_BV, C_VEDB, C_LVIS = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"], FS.DS_COLOR["lvis"]


def domains():
    if V.DOMAINS == "things27":
        return pd.read_csv(REPO / "data" / "category_domains_things27.csv").set_index("category").domain_analysis.to_dict()
    return pd.read_csv(HERE / "out" / "private" / "open_renamed" / "categories.csv", index_col=0).domain.to_dict()


def lvis_whole_means():
    """Per core category: sum of L2-normalized whole-image vectors over images with a mapped box (all; single category)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return list(z["cats"]), z["s_all"], z["n_all"], z["s_one"], z["n_one"]
    mp = pd.read_csv(HERE / "mapping" / "map_lvis.csv")
    mp = mp[mp.category.isin(CORE)].set_index("id").category
    meta = pd.read_csv(CD / "mapping" / "lvis_crop_metadata.csv.gz", usecols=["coco_image_id", "category_id", "coco_folder"])
    meta = meta[meta.category_id.isin(mp.index)]
    meta["cat"] = meta.category_id.map(mp)
    meta["f"] = meta.coco_folder + "_" + meta.coco_image_id.astype(int).map("{:012d}".format) + ".npy"
    img = meta.groupby("f").cat.agg(frozenset)
    cats = sorted(set(meta.cat)); ci = {c: i for i, c in enumerate(cats)}
    S1 = np.zeros((len(cats), 768)); S2 = np.zeros_like(S1); n1 = np.zeros(len(cats)); n2 = np.zeros(len(cats)); miss = 0
    for k, (f, cs) in enumerate(img.items()):
        p = LVIS_IMG / f
        if not p.exists():
            miss += 1; continue
        v = np.load(p).astype(np.float64).ravel(); v /= max(np.linalg.norm(v), 1e-12)
        for c in cs:
            S1[ci[c]] += v; n1[ci[c]] += 1
        if len(cs) == 1:
            c = next(iter(cs)); S2[ci[c]] += v; n2[ci[c]] += 1
        if k % 20000 == 0:
            print(f"  LVIS whole images {k:,}/{len(img):,}", flush=True)
    print(f"{len(img):,} images, {miss} without embeddings", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, cats=np.array(cats), s_all=S1, n_all=n1, s_one=S2, n_one=n2)
    return cats, S1, n1, S2, n2


def zrdm(M):
    M = (M - M.mean(0)) / (M.std(0) + 1e-10); return squareform(pdist(M, "cosine"))


def delta_sd(D, sem, ix=None):
    ix = np.arange(len(sem)) if ix is None else ix
    i, j = np.triu_indices(len(ix), 1); ok = ix[i] != ix[j]
    x = D[ix[i][ok], ix[j][ok]]; same = sem[ix][i][ok] == sem[ix][j][ok]
    return (x[~same].mean() - x[same].mean()) / x.std()


def p_two(x):
    return max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / len(x))


def lvis_analysis(DOM):
    cats, S1, n1, S2, n2 = lvis_whole_means()
    T = np.load(TABLES / "tables_dinov3.npz", allow_pickle=True)
    tab = {d: pd.DataFrame(T[f"{d}__means"], index=[str(c) for c in T[f"{d}__categories"]]) for d in ("bv", "lvis")}
    cnt = {d: pd.Series(T[f"{d}__counts"], index=[str(c) for c in T[f"{d}__categories"]]) for d in ("bv", "lvis")}
    cs = [c for i, c in enumerate(cats) if n2[i] >= MIN_IMG and c in tab["bv"].index and c in tab["lvis"].index
          and cnt["lvis"][c] >= MIN_IMG and c in DOM]
    vc = pd.Series([DOM[c] for c in cs]).value_counts(); keep = set(vc[vc >= 3].index)
    cs = [c for c in cs if DOM[c] in keep]; sem = np.array([DOM[c] for c in cs])
    ci = [cats.index(c) for c in cs]
    M = {"bv": tab["bv"].loc[cs].values, "lvis_crop": tab["lvis"].loc[cs].values,
         "lvis_whole": S1[ci] / n1[ci, None], "lvis_whole_one": S2[ci] / n2[ci, None]}
    R = {k: zrdm(v) for k, v in M.items()}
    rng = np.random.default_rng(0); by = {s: np.flatnonzero(sem == s) for s in sorted(set(sem))}
    B = [np.concatenate([rng.choice(v, len(v)) for v in by.values()]) for _ in range(N_BOOT)]
    boot = {k: np.array([delta_sd(D, sem, ix) for ix in B]) for k, D in R.items()}
    rows = [dict(version=k, categories=len(cs), domains=len(keep), delta_sd=delta_sd(D, sem),
                 ci_low=np.percentile(boot[k], 2.5), ci_high=np.percentile(boot[k], 97.5), se=boot[k].std(ddof=1))
            for k, D in R.items()]
    obs = {r["version"]: r["delta_sd"] for r in rows}
    diffs = []
    for a, b in (("lvis_whole", "lvis_crop"), ("lvis_whole_one", "lvis_crop"), ("bv", "lvis_crop"), ("bv", "lvis_whole")):
        x = boot[a] - boot[b]
        diffs.append(dict(comparison=f"{a} - {b}", diff=obs[a] - obs[b], ci_low=np.percentile(x, 2.5),
                          ci_high=np.percentile(x, 97.5), p_boot=p_two(x)))
    RES.mkdir(parents=True, exist_ok=True)
    diffs = pd.DataFrame(diffs); diffs["q_bh"] = FS.bh(diffs.p_boot)             # BH across the four comparisons
    out = pd.DataFrame(rows); out.to_csv(RES / "lvis_whole_vs_crop.csv", index=False)
    dif = pd.DataFrame(diffs); dif.to_csv(RES / "lvis_whole_vs_crop_differences.csv", index=False)
    pd.Series(cs, name="category").to_frame().assign(domain=sem, n_whole=n1[ci], n_whole_one=n2[ci],
                                                    n_crops=[cnt["lvis"][c] for c in cs]).to_csv(RES / "lvis_whole_vs_crop_categories.csv", index=False)
    print(out.round(3).to_string(index=False)); print(dif.round(3).to_string(index=False))
    print("domains:", vc[vc >= 3].to_dict())
    return out.set_index("version"), dif.set_index("comparison")


def fmt_diff(r):
    return f"{r['diff']:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}], {FS.ptext(r.q_bh, 'q')}".replace("-", "−")


def style(ax, ylim):
    ax.set_ylim(*ylim); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("domain clustering (Δ / SD, DINOv3)")
    ax.annotate("", xy=(.93, -.30), xytext=(.07, -.30), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#8a8984", lw=1.2))
    ax.text(.5, -.33, "more background around the object", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#5f5e5a")


def plot_lvis(L, Ld):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = [0, 1, 2]; ver = ["lvis_crop", "lvis_whole", "lvis_whole_one"]
    y = [L.loc[v, "delta_sd"] for v in ver]
    ax.plot(xs, y, "-", color=C_LVIS, lw=2, zorder=2)
    for x, v in zip(xs, ver):
        r = L.loc[v]
        ax.errorbar(x, r.delta_sd, yerr=r.se, fmt="o", color=C_LVIS,                    # ±1 bootstrap SE
                    ms=9, mec="white", capsize=3, lw=1.5, zorder=3)
    b = L.loc["bv"]
    ax.errorbar(-0.12, b.delta_sd, yerr=b.se, fmt="o", color=C_BV, ms=9,
                mec="white", capsize=3, lw=1.5, zorder=3)
    ax.axhline(b.delta_sd, color=C_BV, lw=1, ls=":", zorder=1)
    ax.text(2.15, b.delta_sd, "BabyView\n(object crops)", color=C_BV, va="center", fontsize=9, fontweight="bold")
    ax.text(2.15, L.loc["lvis_whole_one", "delta_sd"], "LVIS", color=C_LVIS, va="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs, ["object crop", "whole image", "whole image,\nonly this category"])
    ax.set_xlim(-.4, 2.7)
    lo = min((L.delta_sd - L.se).min(), 0); hi = (L.delta_sd + L.se).max()
    style(ax, (max(0, lo - .1), hi + .35))
    t = (f"whole image − crop: {fmt_diff(Ld.loc['lvis_whole - lvis_crop'])}\n"
         f"single-category image − crop: {fmt_diff(Ld.loc['lvis_whole_one - lvis_crop'])}\n"
         f"BabyView − LVIS crop: {fmt_diff(Ld.loc['bv - lvis_crop'])}")
    ax.text(.02, .98, t, transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set_title(f"More background around LVIS objects ({int(L.categories.iloc[0])} categories)", loc="left")
    return fig


def plot_sam():
    T = pd.read_csv(BGC / "background_control.csv"); Df = pd.read_csv(BGC / "background_control_differences.csv")
    T = T[T.model == "dinov3"]; Df = Df[(Df.model == "dinov3") & (Df.measure == "delta_sd")].set_index(["subset", "comparison"])
    ver = ["original", "box", "masked"]; xs = [2, 1, 0]          # x: least background (left) to most (right)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ylim = (max(0, (T.delta_sd - T.delta_sd_se).min() - .2), (T.delta_sd + T.delta_sd_se).max() + .3)
    for ax, sub in zip(axs, ("all formats", "real objects")):
        s = T[T.subset == sub]
        n = int(s.categories.iloc[0])
        lines = []
        yb, yv = (s[(s.dataset == d) & (s.version == "original")].delta_sd.iloc[0] for d in ("bv", "vedb"))
        gap = max(0, .09 * (ylim[1] - ylim[0]) - abs(yb - yv)) / 2       # keep the two end labels apart
        for d, col, lab, off in (("bv", C_BV, "BabyView", gap), ("vedb", C_VEDB, "VEDB", -gap)):
            y = [s[(s.dataset == d) & (s.version == v)].delta_sd.iloc[0] for v in ver]
            se = [s[(s.dataset == d) & (s.version == v)].delta_sd_se.iloc[0] for v in ver]
            dx = -.04 if d == "bv" else .04                                   # dodge so the two series' whiskers stay apart
            ax.errorbar([x + dx for x in xs], y, yerr=se, fmt="-o", color=col, lw=2, ms=9, mec="white", capsize=3, zorder=3)   # ±1 SE
            ax.text(2.12, y[0] + off, lab, color=col, va="center", fontsize=9, fontweight="bold")
            r = Df.loc[(sub, f"{d}: masked - original")]
            lines.append(f"{lab}, background removed − Qwen box: {fmt_diff(r)}")
        rho = s[s.dataset == "BabyView-VEDB RDM rho"].set_index("version").rho
        ax.set_xticks([0, 1, 2], [f"object only\n(SAM 3 mask)", "SAM 3 box", "Qwen box"])
        for x, v in zip(xs, ver):
            ax.text(x, ylim[0] + .02, f"ρ = {rho[v]:.2f}", ha="center", va="bottom", fontsize=8.5, color="#5f5e5a")
        ax.set_xlim(-.3, 2.6)
        style(ax, ylim)
        if ax is not axs[0]:
            ax.set_ylabel("")
        ax.text(.02, .98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_title(f"{sub} ({n} categories)", loc="left")
    fig.suptitle("Less background around BabyView and VEDB objects (± 1 SE; ρ: BabyView–VEDB RDM correlation;\n"
                 "differences: 95% CI, BH q across datasets × versions × subsets)", x=.01, ha="left")
    return fig


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    FS.text_overlaps(fig, name)
    for e in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{e}", dpi=200, bbox_inches="tight")
    print("saved", FIGDIR / f"{name}.png")


def main():
    DOM = domains()
    L, Ld = lvis_analysis(DOM)
    save(plot_lvis(L, Ld), "R1d_background")
    save(plot_sam(), "R1j_background_control")


if __name__ == "__main__":
    main()
