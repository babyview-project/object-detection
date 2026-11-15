#!/usr/bin/env python3
"""Background control (Reviewer 1): BabyView vs VEDB domain clustering with the object's background removed.

Three versions of every sampled box where SAM 3 found the object (annotation/qwen_pilot/sam3_crop_sample.py,
embed_masked_sample.py), compared on the same boxes and the same categories:
  original  Qwen's box (the sample's own embeddings)
  box       SAM 3's corrected box (recovers parts of the object outside Qwen's box), background kept
  masked    SAM 3's corrected box with everything outside the object mask set to grey
Category means = mean of L2-normalized vectors, per dataset (BabyView all ages, VEDB), categories with >= MIN_N (20) boxes in
both; z-scored within the categories compared; cosine RDMs. Domain clustering: raw delta (between - within), delta / SD
and AUC; BabyView - VEDB differences with a within-domain category bootstrap (1,000 draws, shared across versions);
BabyView-VEDB RDM Spearman; per domain. For all formats and real objects only. THINGS / LVIS / ImageNet are not
included (their images are not masked).

Outputs: results/bv_vedb_renamed/background_control{,_differences,_by_domain}.csv, figures/bv_vedb_renamed/background_control.png
  python analyze_background_control.py           # analysis env
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
import variant as V
SAMPLE = V.out_dir(HERE / "out" / "private" / "bv_vedb")
SAM = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb_adults")
import os as _os
CORE_MODE = _os.environ.get("QOV_CATS", "tier1")   # core: restrict to the 183-category core (164 for real objects)
_DATA = Path(__file__).resolve().parents[2] / "data"
CORE = {l.strip() for l in open(_DATA / "included_categories_qwen_core.txt") if l.strip()}
CORE_REAL = {l.strip() for l in open(_DATA / "included_categories_qwen_core_real.txt") if l.strip()}
SFX = "_core" if CORE_MODE == "core" else ""
def in_core(c, real=False):
    return CORE_MODE != "core" or c in (CORE_REAL if real else CORE)
RES = V.out_dir(HERE / "results" / "bv_vedb"); RES = RES.with_name(RES.name + SFX)
FIG = V.out_dir(HERE / "figures" / "bv_vedb"); FIG = FIG.with_name(FIG.name + SFX)
MIN_N, N_BOOT = 20, 1000
VERSIONS = ["original", "box", "masked"]


def l2(X):
    X = X.astype(np.float32); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
    rows, diffs, doms = [], [], []
    for m in ("clip", "dinov3"):
        z = np.load(SAMPLE / f"sample_{m}.npz", allow_pickle=True)
        e = np.load(SAM / f"embeddings_{m}.npz")
        ok = np.isfinite(e["masked"][:, 0].astype(float)) & np.isfinite(e["box"][:, 0].astype(float))
        idx = e["idx"][ok]
        E = {"original": l2(z["emb"][idx]), "box": l2(e["box"][ok]), "masked": l2(e["masked"][ok])}
        ds, cat, fmt = z["dataset"][idx], z["category"][idx], z["format"][idx].astype(str)
        for subset in ("all formats", "real objects"):
            sel = np.ones(len(idx), bool) if subset == "all formats" else (fmt == "real")
            cnt = pd.crosstab(cat[sel], ds[sel])
            cats = sorted(c for c in cnt.index[(cnt.get("bv", 0) >= MIN_N) & (cnt.get("vedb", 0) >= MIN_N)]
                          if c in DOM and in_core(c, subset == "real objects"))
            sem = np.array([DOM[c] for c in cats])
            vc = pd.Series(sem).value_counts(); keepd = [d for d in vc.index if vc[d] >= 3]
            cs = [c for c, s in zip(cats, sem) if s in keepd]; sem = np.array([DOM[c] for c in cs])
            iu = np.triu_indices(len(cs), 1); same = sem[iu[0]] == sem[iu[1]]
            rng = np.random.default_rng(0); by = {s: np.flatnonzero(sem == s) for s in set(sem)}
            B = [np.concatenate([rng.choice(v, len(v)) for v in by.values()]) for _ in range(N_BOOT)]
            R = {}
            for ver in VERSIONS:
                for d in ("bv", "vedb"):
                    s2 = sel & (ds == d)
                    M = pd.DataFrame(E[ver][s2]).groupby(cat[s2]).mean().loc[cs].values
                    M = (M - M.mean(0)) / (M.std(0) + 1e-10)
                    R[ver, d] = squareform(pdist(M, "cosine"))
                    dd = R[ver, d][iu]; w, bt = dd[same], dd[~same]
                    rows.append(dict(model=m, subset=subset, version=ver, dataset=d, categories=len(cs), delta=bt.mean() - w.mean(),
                                     delta_sd=(bt.mean() - w.mean()) / dd.std(), auc=stats.mannwhitneyu(bt, w).statistic / (len(bt) * len(w))))
                    for s in keepd:
                        a = sem == s; ww = R[ver, d][np.ix_(a, a)][np.triu_indices(a.sum(), 1)]; bb = R[ver, d][np.ix_(a, ~a)].ravel()
                        doms.append(dict(model=m, subset=subset, version=ver, dataset=d, domain=s, n=int(a.sum()), delta=bb.mean() - ww.mean()))
                rho = stats.spearmanr(R[ver, "bv"][iu], R[ver, "vedb"][iu]).statistic
                rows.append(dict(model=m, subset=subset, version=ver, dataset="BabyView-VEDB RDM rho", categories=len(cs), delta=np.nan,
                                 delta_sd=np.nan, auc=np.nan, rho=rho))

            def delta_draw(D, ix):
                ii, jj = np.triu_indices(len(ix), 1); okp = ix[ii] != ix[jj]
                sm = sem[ix][ii][okp] == sem[ix][jj][okp]; x = D[ix[ii][okp], ix[jj][okp]]
                return x[~sm].mean() - x[sm].mean(), (x[~sm].mean() - x[sm].mean()) / x.std()   # raw delta, delta / SD
            boot = {(ver, d): np.array([delta_draw(R[ver, d], ix) for ix in B]) for ver in VERSIONS for d in ("bv", "vedb")}
            for k_, meas in enumerate(("delta", "delta_sd")):
                for ver in VERSIONS:
                    x = boot[ver, "bv"][:, k_] - boot[ver, "vedb"][:, k_]
                    obs = [r for r in rows if r["model"] == m and r["subset"] == subset and r["version"] == ver and r["dataset"] in ("bv", "vedb")]
                    diff = obs[0][meas] - obs[1][meas]
                    diffs.append(dict(model=m, subset=subset, measure=meas, comparison=f"BabyView - VEDB, {ver}", diff=diff, ci_low=np.percentile(x, 2.5),
                                      ci_high=np.percentile(x, 97.5), p_boot=max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / N_BOOT)))
                for d in ("bv", "vedb"):
                    for ver in ("box", "masked"):
                        x = boot[ver, d][:, k_] - boot["original", d][:, k_]
                        diffs.append(dict(model=m, subset=subset, measure=meas, comparison=f"{d}: {ver} - original", diff=np.mean(x),
                                          ci_low=np.percentile(x, 2.5), ci_high=np.percentile(x, 97.5),
                                          p_boot=max(2 * min((x <= 0).mean(), (x >= 0).mean()), 2 / N_BOOT)))
            print(f"{m}, {subset}: {len(cs)} categories, {sel.sum():,} boxes", flush=True)
    T = pd.DataFrame(rows); T.to_csv(RES / "background_control.csv", index=False)
    Df = pd.DataFrame(diffs); Df.to_csv(RES / "background_control_differences.csv", index=False)
    Dm = pd.DataFrame(doms); Dm.to_csv(RES / "background_control_by_domain.csv", index=False)
    pd.set_option("display.width", 200)
    print(T[T.dataset != "BabyView-VEDB RDM rho"].pivot_table(index=["model", "subset", "dataset"], columns="version", values=["delta", "auc"])
          .round(3).to_string())
    print(T[T.dataset == "BabyView-VEDB RDM rho"].pivot_table(index=["model", "subset"], columns="version", values="rho").round(3).to_string())
    print(Df.round(3).to_string())
    print(Dm[~Dm.domain.astype(str).str.startswith("other")]
          .pivot_table(index=["model", "subset", "domain"], columns=["dataset", "version"], values="delta").round(3).to_string())
    plot(T, Df)


def plot(T, Df=None):
    """Zero-based grouped bars: BabyView and VEDB domain clustering for each version of the boxes."""
    import matplotlib.pyplot as plt
    import fig_style as FS
    FS.apply()
    VL = ["Qwen box", "SAM 3 box", "SAM 3 box,\nbackground removed"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (m, sub) in zip(axs, [(m, x) for m in ("clip", "dinov3") for x in ("all formats", "real objects")]):
        for j, d in enumerate(("bv", "vedb")):
            t = T[(T.model == m) & (T.subset == sub) & (T.dataset == d)].set_index("version").loc[VERSIONS]
            ax.bar(np.arange(3) + (j - .5) * .36, t.delta_sd, .34, color=FS.DS_COLOR[d], label=FS.label(d), edgecolor="white", linewidth=1.5)
        n = int(T[(T.model == m) & (T.subset == sub)].categories.iloc[0])
        if Df is not None and "measure" in Df:          # BabyView vs VEDB per version (Δ / SD, category bootstrap)
            top = T[(T.model == m) & (T.subset == sub) & T.dataset.isin(["bv", "vedb"])].delta_sd.max()
            for k, ver in enumerate(VERSIONS):
                r_ = Df[(Df.model == m) & (Df.subset == sub) & (Df.measure == "delta_sd") & (Df.comparison == f"BabyView - VEDB, {ver}")]
                FS.bracket(ax, k - .18, k + .18, top * 1.06, FS.stars(r_.p_boot.iloc[0]), h=top * .02)
        ax.set_xticks(range(3), VL); ax.set_ylim(0, None)
        ax.set_title(f"{'CLIP' if m == 'clip' else 'DINOv3'}, {sub} ({n} categories)", loc="left")
    axs[0].set_ylabel("domain clustering (Δ / SD)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(.99, .99))
    fig.suptitle("Background control: the same boxes with Qwen's box, SAM 3's corrected box, and the background removed\n"
                 "brackets: BabyView vs VEDB, category bootstrap (* p < .05, ** < .01, *** < .001)", x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(FIG / "background_control_bars.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:                       # redraw from the saved table
        plot(pd.read_csv(RES / "background_control.csv"), pd.read_csv(RES / "background_control_differences.csv"))
    else:
        main()
