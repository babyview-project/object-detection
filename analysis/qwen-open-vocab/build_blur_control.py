#!/usr/bin/env python3
"""Blur control (Reviewer 3): does BabyView's domain clustering survive on sharp crops only?

measure: every box of the BabyView / VEDB embedding sample (build_bv_vedb_data.py, up to 150 per category and age bin) is
re-cut from its frame, resized to 224 x 224 (the models' input size, so sharpness is judged at the scale the embeddings
see) and scored by the variance of the Laplacian of the grey image (higher = sharper; Pech-Pacheco et al., 2000).
analyze: category means from the sharpest crops only (sharpness above the BabyView 50th / 67th percentile, the same
absolute threshold for VEDB), z-scored within the categories compared, cosine RDMs, domain clustering (raw delta,
delta / SD, AUC) against all crops of the same sample and against THINGS; BabyView - VEDB / THINGS differences with a
within-domain category bootstrap. Categories with >= 10 sharp crops in BabyView and VEDB and present in THINGS.

Outputs: out/private/bv_vedb/sharpness.csv.gz (per sampled box); results/bv_vedb/blur_control.csv,
blur_control_differences.csv; figures/bv_vedb/blur_sharpness.png
  [QOV_LABELS=renamed] python build_blur_control.py measure [--workers 8]          # analysis env, CPU
  [QOV_LABELS=renamed] python build_blur_control.py analyze
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variant as V                                     # QOV_LABELS=renamed: renamed-label sample, parallel folders
D = V.out_dir(HERE / "out" / "private" / "bv_vedb")
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
BV_FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
VEDB_FRAMES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_adults")
SIDE, MIN_SHARP = 224, 10


def frame_path(video_id: str, frame: int) -> Path:
    if video_id.startswith("session_"):
        return VEDB_FRAMES / video_id / f"{frame:08d}.jpg"
    return BV_FRAMES / video_id / f"{frame:05d}.jpg"


def score_frame(args):
    """Laplacian variance of each box of one frame: [(row, sharpness)]."""
    from PIL import Image
    from scipy.ndimage import laplace
    (vid, fn), rows, boxes = args
    try:
        im = Image.open(frame_path(vid, int(fn))).convert("L")
    except Exception:
        return [(r, np.nan) for r in rows]
    out = []
    for r, (x0, y0, x1, y1) in zip(rows, boxes):
        c = im.crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))))
        if c.width < 2 or c.height < 2:
            out.append((r, np.nan)); continue
        a = np.asarray(c.resize((SIDE, SIDE), Image.BILINEAR), np.float32)
        out.append((r, float(laplace(a).var())))
    return out


def measure(workers: int):
    z = np.load(D / "sample_dinov3.npz", allow_pickle=True)
    df = pd.DataFrame({"video_id": z["video_id"], "frame_number": z["frame_number"]})
    b = z["bbox"]
    jobs = [((v, f), g.index.values, b[g.index.values]) for (v, f), g in df.groupby(["video_id", "frame_number"])]
    sharp = np.full(len(df), np.nan, np.float32)
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(score_frame, jobs, chunksize=64)):
            for r, s in res:
                sharp[r] = s
            if i % 50000 == 0:
                print(f"  {i:,}/{len(jobs):,} frames", flush=True)
    pd.DataFrame({"sharpness": sharp}).to_csv(D / "sharpness.csv.gz", index=False)
    print(f"done: {np.isfinite(sharp).sum():,} of {len(sharp):,} boxes scored")


def analyze():
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
    sh = pd.read_csv(D / "sharpness.csv.gz").sharpness.values
    zr = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-10)
    l2 = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rows, diffs = [], []
    for m in ("clip", "dinov3"):
        z = np.load(D / f"sample_{m}.npz", allow_pickle=True)
        E, ds, cat = l2(z["emb"].astype(np.float32)), z["dataset"], z["category"]
        tz = np.load(V.out_dir(HERE / "out" / "crossdataset") / f"tables_{m}.npz", allow_pickle=True)
        things = pd.DataFrame(tz["things__means"], index=list(tz["things__categories"]))
        bv_s = sh[(ds == "bv") & np.isfinite(sh)]
        strict = np.isfinite(sh) & (sh > np.percentile(bv_s, 67))
        cnt = pd.crosstab(cat[strict], ds[strict])          # same categories at every cut: those that survive the strictest
        cs = sorted(c for c in cnt.index[(cnt["bv"] >= MIN_SHARP) & (cnt["vedb"] >= MIN_SHARP)] if c in things.index and in_core(c))
        for cut in (0, 50, 67):
            thr = np.percentile(bv_s, cut) if cut else -np.inf
            keep = np.isfinite(sh) & (sh > thr)
            means = {}
            for d in ("bv", "vedb"):
                sel = keep & (ds == d)
                M = pd.DataFrame(E[sel]).groupby(cat[sel]).mean()
                means[d] = M
            means["things"] = things
            sem = np.array([DOM.get(c, "other") for c in cs])
            i, j = np.triu_indices(len(cs), 1); same = sem[i] == sem[j]
            def measures(Dm):
                d = Dm[i, j]; w, bt = d[same], d[~same]
                return bt.mean() - w.mean(), (bt.mean() - w.mean()) / d.std(), stats.mannwhitneyu(bt, w).statistic / (len(bt) * len(w))
            R = {d: squareform(pdist(zr(means[d].loc[cs].values.astype(float)), "cosine")) for d in means}
            for d, Dm in R.items():
                dl, sd, auc = measures(Dm)
                rows.append(dict(model=m, sharp_cut=f"> BV p{cut}" if cut else "all crops", dataset=d, categories=len(cs),
                                 delta=dl, delta_sd=sd, auc=auc, median_sharpness=float(np.median(sh[keep & (ds == d)])) if d != "things" else np.nan))
            rng = np.random.default_rng(0); by = {s: np.flatnonzero(sem == s) for s in set(sem)}
            for a, b in (("bv", "vedb"), ("bv", "things"), ("vedb", "things")):
                bs = []
                for _ in range(1000):
                    ix = np.concatenate([rng.choice(v, len(v)) for v in by.values()])
                    ii, jj = np.triu_indices(len(ix), 1); ok = ix[ii] != ix[jj]; sm = sem[ix][ii][ok] == sem[ix][jj][ok]
                    def dd(Dm):
                        d = Dm[ix[ii][ok], ix[jj][ok]]; return d[~sm].mean() - d[sm].mean()
                    bs.append(dd(R[a]) - dd(R[b]))
                bs = np.array(bs)
                diffs.append(dict(model=m, sharp_cut=f"> BV p{cut}" if cut else "all crops", pair=f"{a} - {b}",
                                  diff=measures(R[a])[0] - measures(R[b])[0], ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                                  p_boot=max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 2 / len(bs))))
    out = pd.DataFrame(rows); out.to_csv(RES / "blur_control.csv", index=False)
    dif = pd.DataFrame(diffs); dif.to_csv(RES / "blur_control_differences.csv", index=False)
    print(out.round(3).to_string()); print(dif.round(3).to_string())
    z = np.load(D / "sample_dinov3.npz", allow_pickle=True); ds = z["dataset"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    import fig_style as FS
    FS.apply()
    for d, col in (("bv", FS.DS_COLOR["bv"]), ("vedb", FS.DS_COLOR["vedb"])):
        x = np.log10(sh[(ds == d) & np.isfinite(sh) & (sh > 0)])
        ax.hist(x, bins=80, density=True, histtype="step", lw=1.6, color=col, label={"bv": "BabyView", "vedb": "VEDB"}[d])
    for cut in (50, 67):
        ax.axvline(np.log10(np.percentile(sh[(ds == "bv") & np.isfinite(sh)], cut)), color="#8a8984", ls="--", lw=1)
    ax.set_xlabel("crop sharpness, log10 Laplacian variance (dashed: BabyView 50th / 67th percentile)")
    ax.set_ylabel("density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "blur_sharpness.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if step == "measure":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        measure(w)
    else:
        analyze()
