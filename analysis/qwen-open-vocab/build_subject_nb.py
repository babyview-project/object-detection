#!/usr/bin/env python3
"""Write qwen_subject_analysis.ipynb (code only). Execute into out/private/ (it reads private data):

  python build_subject_nb.py
  jupyter nbconvert --to notebook --execute qwen_subject_analysis.ipynb \
      --ExecutePreprocessor.kernel_name=vislearnlabpy --ExecutePreprocessor.timeout=3600 \
      --output-dir out/private --output qwen_subject_analysis_executed.ipynb
Set SUBJECT / TAG in the first cell for another child or run. Safe to run while the shards are
still writing (it reads whatever frames are finished). Section 7 needs embed_qwen_crops.py outputs.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Qwen3.8-27B open-vocabulary detections for one child (S00320001)

Every 5th 1 fps frame of the child's 2025.2 videos, setup B (open label + visual format + box,
at most 30 objects), temperature 1.0. **Private:** frames are not shown, but the data name videos;
keep the executed notebook in `out/private/`.

**Categories are Qwen's own**, not the manuscript's 129: `cat` = Qwen's label with light
normalization only (lowercase, plurals → singular, a few exact synonyms such as couch → sofa, and
depiction words dropped because `format` records them: "toy cars" → car (format toy), "picture of a
dog" → dog; a bare "toy" stays toy). `domain` groups categories into the CDI semantic domains (the CDI
word's domain when the category is one, else the majority of its 5 nearest CDI words by CLIP text
embedding, plus overrides for frequent misses; approximate, see `results/category_domains.csv`).

Sections: 1 coverage · 2 objects per frame by age · 3 long tail of Qwen's categories · 4 visual
formats by age · 5 cross-check against YOLOE + CLIP · 6 co-occurrence RDM · 7 embedding RDMs
(CLIP / DINOv3 of Qwen's boxes), with a small cross-check against the manuscript's RDMs."""),
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path.cwd()))                                   # this folder
sys.path.insert(0, str(Path.cwd().parents[1] / "annotation" / "qwen_pilot"))   # pipeline loaders
import qwen_data as qd

SUBJECT, TASK, TAG = "S00320001", "detect_open_format", "subj_S00320001_t1_pp0"
EXCLUDE = {"person"}            # as in the YOLOE pipeline
rec = qd.load_records(TASK, TAG)
fr = qd.frames_table(rec)
obj_all = qd.objects_table(rec)
obj = qd.dedup(obj_all).merge(fr[["unit_id", "age_months", "date"]], on="unit_id")
o = obj[~obj.cat.isin(EXCLUDE)]
manifest = pd.read_csv(qd.HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv")
n_total = int(manifest.video_id.str.startswith(SUBJECT + "_").sum())
N = len(fr)
PRIV = qd.HERE / "out" / "private"
MS = qd.OD_ROOT / "analysis" / "manuscript-2026"
MS_RES = MS / "main_results_valid129s_2025.2" / "results"
VALID129 = [l.strip() for l in open(qd.OD_ROOT / "data" / "included_categories_valid129.txt") if l.strip()]
DOMAINS = ["furniture_rooms", "household", "toys", "food_drink", "clothing", "body_parts", "animals", "vehicles", "outside", "people"]
import fig_style as FS; FS.apply()                              # manuscript style: Lato, bold labels and legends
DOM_COLOR = {d: FS.DOMAIN_COLOR[d] for d in DOMAINS}             # manuscript CDI domain colours
DOM_OF = o.groupby("cat").domain.first()

BLUE, ORANGE, AQUA, YELLOW, GREY = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"], FS.DS_COLOR["lvis"], FS.FMT_COLOR["screen"], FS.BASELINE
INK, INK2, GRID = FS.INK, FS.MUTED, "#ececec"
plt.rcParams.update({
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110})

def fit_power_law_rank(values, min_points=3):            # manuscript notebook 01
    v = np.asarray(values, float); v = np.sort(v[np.isfinite(v) & (v > 0)])[::-1]
    if v.size < min_points:
        return {"n_points": int(v.size), "alpha": np.nan, "intercept_log": np.nan, "r2_log": np.nan}
    x, y = np.log(np.arange(1, v.size + 1)), np.log(v)
    slope, icpt = np.polyfit(x, y, 1); yh = slope * x + icpt
    return {"n_points": int(v.size), "alpha": -slope, "intercept_log": icpt,
            "r2_log": 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()}

def lower(D): return D[np.tril_indices(len(D), -1)]

def domain_stats(D, cats):
    sem = np.array([DOM_OF.get(c, "other") for c in cats]); i, j = np.tril_indices(len(cats), -1)
    same = sem[i] == sem[j]; d = D[i, j]; w, b = d[same], d[~same]
    return {"n_categories": len(cats), "within": w.mean(), "between": b.mean(), "delta": b.mean() - w.mean(),
            "auc": stats.mannwhitneyu(b, w).statistic / (len(b) * len(w))}

def domain_order(cats, key=None):
    key = key or {}
    return sorted(range(len(cats)), key=lambda i: (DOMAINS.index(DOM_OF.get(cats[i])) if DOM_OF.get(cats[i]) in DOMAINS else 99,
                                                   -key.get(cats[i], 0), cats[i]))

def rdm_heatmap(ax, D, cats, title, cmap="viridis", label_every=0):
    order = domain_order(cats, PREV.to_dict())
    M = D[np.ix_(order, order)]
    im = ax.imshow(M, cmap=cmap, vmin=np.percentile(lower(M), 2), vmax=np.percentile(lower(M), 98))
    ax.set_title(title, loc="left", fontsize=10, color=INK); ax.grid(False)
    doms = [DOM_OF.get(cats[i], "other") for i in order]
    edges = [i for i in range(1, len(doms)) if doms[i] != doms[i - 1]]
    for e in edges:
        ax.axhline(e - .5, color="white", lw=.8); ax.axvline(e - .5, color="white", lw=.8)
    starts = [0] + edges; ends = edges + [len(doms)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [doms[a] for a in starts], fontsize=8); ax.set_xticks([])
    return im

print(f"{N:,} of {n_total:,} frames done ({N/n_total:.0%}); {fr.video_id.nunique()} videos; "
      f"age {fr.age_months.min():.1f}-{fr.age_months.max():.1f} months")"""),

md("## 1. Coverage and reply quality"),
code("""q = pd.Series({
    "frames": len(fr), "parsed": fr.parsed.mean(), "boxes (before dedup)": len(obj_all),
    "near-duplicate boxes dropped": int(obj_all.dup.sum()), "objects per frame": len(obj) / len(fr),
    "frames at the 30-box cap": fr.at_cap.mean(), "frames with a marching loop": fr.marching.mean(),
    "distinct labels": obj.label.nunique(), "distinct categories (normalized)": obj.cat.nunique(), "output tokens per frame": fr.out_tok.mean()})
display(q.to_frame("value").style.format("{:,.3f}"))"""),

md("## 2. Objects per frame by age\nMonthly means (frames binned by age at recording); band = middle 50% of frames."),
code("""fr["age_bin"] = np.floor(fr.age_months).astype(int)
g = fr.groupby("age_bin").n_objects.agg(["mean", lambda x: x.quantile(.25), lambda x: x.quantile(.75), "size"])
g.columns = ["mean", "q25", "q75", "frames"]
g = g[g.frames >= 200]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.fill_between(g.index, g.q25, g.q75, color=BLUE, alpha=.15, lw=0)
ax.plot(g.index, g["mean"], color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5)
ax.set_xlabel("age (months)"); ax.set_ylabel("objects per frame")
ax.set_title("Objects Qwen lists per frame, by age (months with >= 200 frames)", loc="left", fontsize=11, color=INK)
fig.tight_layout(); plt.show()
display(g.round(2).T)"""),

md("""## 3. The long tail of Qwen's categories
Prevalence = share of frames in which a category appears (person excluded). Power law fitted as in
manuscript notebook 01 (log share vs log rank; `alpha` = −slope). Categories seen in a handful of
frames make a flat tail (many ties at 1-2 frames), so fits are given for all categories and for
categories in ≥ 30 frames; per domain on the latter. "Real only" counts only boxes with format real."""),
code("""PREV = o.groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
PREV_REAL = o[o.format == "real"].groupby("cat").unit_id.nunique().div(N).sort_values(ascending=False)
nfr = (PREV * N).round().astype(int)
print(f"{len(PREV):,} categories from {o.label.nunique():,} distinct labels; "
      + ", ".join(f"{(nfr >= k).sum():,} in >= {k} frames" for k in (10, 30, 100, 1000)) + f"; {(nfr == 1).sum():,} in one frame")
cum = o.cat.value_counts(normalize=True).cumsum()
print("share of boxes from the top 10 / 50 / 200 categories:", [round(float(cum.iloc[k - 1]), 3) for k in (10, 50, 200)])
print(f"{o.cdi.notna().mean():.0%} of boxes have a label that maps to a CDI word")

common = PREV[nfr >= 30]
fits = [{"set": "all categories", "domain": "all", **fit_power_law_rank(PREV)},
        {"set": ">= 30 frames", "domain": "all", **fit_power_law_rank(common)},
        {"set": ">= 30 frames, real only", "domain": "all", **fit_power_law_rank(PREV_REAL[PREV_REAL * N >= 30])}]
for d in DOMAINS:
    fits.append({"set": ">= 30 frames", "domain": d, **fit_power_law_rank(common[DOM_OF.reindex(common.index) == d])})
fits = pd.DataFrame(fits)
display(fits.round(3))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
r = np.arange(1, len(PREV) + 1)
axs[0].loglog(r, PREV.values, color=GREY, lw=1.2, zorder=1)
for d in DOMAINS:
    m = (DOM_OF.reindex(PREV.index) == d).values & (r <= len(common))
    axs[0].scatter(r[m], PREV.values[m], s=12, color=DOM_COLOR[d], label=d, zorder=3, edgecolor="white", linewidth=.3)
f = fits.iloc[1]; rr = np.arange(1, len(common) + 1)
axs[0].loglog(rr, np.exp(f.intercept_log) * rr ** -f.alpha, color=INK, lw=1, ls="--")
axs[0].text(rr[-1], np.exp(f.intercept_log) * rr[-1] ** -f.alpha, f"  alpha {f.alpha:.2f} (>= 30 frames)", fontsize=8, color=INK2, va="bottom")
for k, c in enumerate(PREV.index[:12]):
    axs[0].annotate(c, (k + 1, PREV.iloc[k]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2)
axs[0].set_xlabel("category rank"); axs[0].set_ylabel("share of frames")
axs[0].set_title("Rank-frequency of Qwen's categories", loc="left", fontsize=11, color=INK)
axs[0].legend(frameon=False, fontsize=7, ncol=2)
for d in DOMAINS:
    v = common[DOM_OF.reindex(common.index) == d].values
    if len(v) >= 3:
        axs[1].loglog(np.arange(1, len(v) + 1), v, "-o", ms=3, lw=1.2, color=DOM_COLOR[d], label=f"{d} ({len(v)})")
axs[1].set_xlabel("rank within domain"); axs[1].set_ylabel("share of frames")
axs[1].set_title("Within each domain (categories in >= 30 frames)", loc="left", fontsize=11, color=INK)
axs[1].legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout(); plt.show()

fm = o.assign(fmt=o.format.replace({"photo": "picture", "drawing": "picture"})).groupby("cat").fmt.value_counts(normalize=True).unstack().fillna(0)
top = pd.DataFrame({"share_of_frames": PREV.head(50), "domain": DOM_OF.reindex(PREV.head(50).index)}).join(fm[["real", "toy", "picture", "screen"]])
display(top.round(3).style.format(precision=3).set_caption("top 50 categories, with the format mix of their boxes"))
print("Most frequent categories per domain:")
for d in DOMAINS:
    s_ = common[DOM_OF.reindex(common.index) == d]
    print(f"  {d} ({len(s_)}): " + ", ".join(f"{c} {v:.3f}" for c, v in s_.head(15).items()))"""),

md("""### Categories mostly seen as depictions
Categories (≥ 30 frames) whose boxes are mostly toys, pictures or screens: what this child sees
mainly in depicted form."""),
code("""dep = fm.reindex(common.index)
dep = dep.assign(depicted=1 - dep.real, frames=nfr.reindex(common.index), domain=DOM_OF.reindex(common.index))
display(dep[dep.depicted >= .5].sort_values("frames", ascending=False).head(40).round(2)[["frames", "domain", "real", "toy", "picture", "screen"]])
print("share of categories (>= 30 frames) mostly depicted, by domain:")
display(dep.groupby("domain").depicted.apply(lambda x: (x >= .5).mean()).reindex(DOMAINS).round(2).to_frame("mostly_depicted"))"""),

md("""## 4. Visual formats by age
Share of boxes (people excluded) per format; pictures = photo + drawing."""),
code("""o2 = o.assign(age_bin=np.floor(o.age_months).astype(int),
             fmt=o.format.replace({"photo": "picture", "drawing": "picture"}))
share = o2.groupby("age_bin").fmt.value_counts(normalize=True).unstack().fillna(0)
share = share[o2.groupby("age_bin").unit_id.nunique().reindex(share.index) >= 200]
print("overall:", o2.fmt.value_counts(normalize=True).round(3).to_dict())
fig, ax = plt.subplots(figsize=(8, 3.6))
for f, col in (("toy", YELLOW), ("picture", ORANGE), ("screen", BLUE)):
    if f in share:
        ax.plot(share.index, share[f], color=col, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=f)
        ax.text(share.index[-1] + .2, share[f].iloc[-1], f, color=INK2, fontsize=9, va="center")
ax.set_xlabel("age (months)"); ax.set_ylabel("share of boxes")
ax.set_title("Depicted objects by age: toys, pictures, screens", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False); fig.tight_layout(); plt.show()
display(share.round(3).T)
for f in ("toy", "picture", "screen"):
    print(f"top categories as {f}:", o2[o2.fmt == f].cat.value_counts().head(12).to_dict())"""),

md("""## 5. Cross-check: Qwen vs YOLOE + CLIP on the same frames
YOLOE + CLIP (2025.2 crop index, CLIP ≥ 0.27 = `passed`) covers this child's videos from 2024-11-10
on, so this uses only frames both have. Category prevalence = share of those frames with at least
one detection of the category. This is the one place Qwen labels are mapped to CDI words, because
YOLOE + CLIP only knows that vocabulary (person excluded)."""),
code("""yo = qd.yoloe_table(fr)                                    # YOLOE + CLIP rows on these frames
both = sorted(set(fr.unit_id) & set(yo.unit_id))
print(f"frames with YOLOE + CLIP rows: {len(both):,} of {len(fr):,}")
if len(both) >= 500:
    nb = len(both)
    yq = yo[yo.unit_id.isin(both) & yo.passed & ~yo.class_name.isin(EXCLUDE | {"picture"})]
    ya = yo[yo.unit_id.isin(both) & ~yo.class_name.isin(EXCLUDE | {"picture"})]
    qq = o[o.unit_id.isin(both)].dropna(subset=["cdi"])
    P = pd.DataFrame({"qwen": qq.groupby("cdi").unit_id.nunique() / nb,
                      "yoloe_clip": yq.groupby("class_name").unit_id.nunique() / nb,
                      "yoloe_raw": ya.groupby("class_name").unit_id.nunique() / nb}).fillna(0)
    P = P[(P.qwen > 0) | (P.yoloe_clip > 0)]
    sh = P[["qwen", "yoloe_clip"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman, category prevalence Qwen vs YOLOE + CLIP: {sh:.2f} over {len(P)} categories")
    fig, ax = plt.subplots(figsize=(6, 6))
    eps = .5 / nb
    ax.loglog(P.yoloe_clip + eps, P.qwen + eps, "o", ms=5, color=BLUE, mec="white", mew=.8)
    lim = [eps, 1]; ax.plot(lim, lim, color=GREY, lw=1, ls="--")
    for name, r_ in P.assign(ratio=(P.qwen + eps) / (P.yoloe_clip + eps)).sort_values("ratio").iloc[np.r_[0:6, -6:0]].iterrows():
        ax.annotate(name, (r_.yoloe_clip + eps, r_.qwen + eps), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("YOLOE + CLIP: share of frames"); ax.set_ylabel("Qwen (mapped to CDI): share of frames")
    ax.set_title("Category prevalence on the same frames", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); plt.show()
    P["log_ratio_qwen_over_yoloe"] = np.log10((P.qwen + eps) / (P.yoloe_clip + eps))
    display(P.sort_values("qwen", ascending=False).head(30).round(4))
    print("Qwen much more often than YOLOE + CLIP:"); display(P.sort_values("log_ratio_qwen_over_yoloe").tail(15).round(4))
    print("YOLOE + CLIP much more often than Qwen:"); display(P.sort_values("log_ratio_qwen_over_yoloe").head(15).round(4))
else:
    print("Not enough overlapping frames yet (the run reaches videos from 2024-11-10 later); rerun when it is further along.")"""),

md("""## 6. Co-occurrence RDM
Which categories appear in the same frames. As in comparison-datasets notebook 12: distance between
two categories = 1 − Jaccard over frames (frames with both / frames with either), here on Qwen's
categories in ≥ 100 frames. Domain clustering: mean between-domain minus within-domain distance (Δ)
and AUC = P(a between-domain pair is farther apart than a within-domain pair)."""),
code("""import scipy.sparse as sp

def jaccard_rdm(C):
    n = np.diag(C).astype(float); U = n[:, None] + n[None, :] - C
    D = 1 - np.divide(C, U, out=np.zeros_like(U, dtype=float), where=U > 0); np.fill_diagonal(D, 0); return D

CC = list(PREV[nfr >= 100].index)
ix = {c: i for i, c in enumerate(CC)}
d_ = o[o.cat.isin(ix)]
f_ = pd.factorize(d_.unit_id)[0]
B = sp.csr_matrix((np.ones(len(d_)), (f_, d_.cat.map(ix).values)), shape=(f_.max() + 1, len(CC)))
B.sum_duplicates(); B.data[:] = 1
Dq = jaccard_rdm((B.T @ B).toarray())
print(f"{len(CC)} categories in >= 100 frames")
display(pd.Series(domain_stats(Dq, CC)).round(3).to_frame("co-occurrence"))
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = rdm_heatmap(ax, Dq, CC, f"Co-occurrence RDM (1 − Jaccard), {len(CC)} categories", cmap="viridis_r")
fig.colorbar(im, ax=ax, shrink=.7); plt.show()
i, j = np.tril_indices(len(CC), -1)
pairs = pd.DataFrame({"a": np.array(CC)[i], "b": np.array(CC)[j], "jaccard": 1 - Dq[i, j]})
print("Most co-occurring pairs:"); display(pairs.sort_values("jaccard", ascending=False).head(20).round(3))
SHOW = [c for c in ["toy", "book", "cup", "ball", "dog", "car", "spoon", "shoe", "television", "high chair", "bottle", "banana"] if c in ix]
print("Top co-occurring categories:")
for c in SHOW:
    nn = np.argsort(Dq[ix[c]])[1:9]
    print(f"  {c}: " + ", ".join(f"{CC[k]} {1 - Dq[ix[c], k]:.2f}" for k in nn))"""),

md("""## 7. Embedding RDMs: CLIP and DINOv3 of Qwen's boxes
`embed_qwen_crops.py` crops boxes of every category in ≥ 30 frames (up to 2,000 per category, one
per frame first) and embeds them with the manuscript's models: CLIP ViT-B/32 image features and DINOv3
ViT-B/16 CLS. Following manuscript notebooks 03 and 06, each crop is L2-normalised, crops are
averaged per category, each dimension is z-scored across the categories compared, and RDM = cosine
distance. Categories need ≥ 30 crops. "Real only" averages only boxes with format real."""),
code("""def load_means(path):
    d = pd.read_csv(path); d["category"] = d.category.astype(str).str.strip().str.lower()
    return d.set_index("category")

def zrows(X): return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
def cos_rdm(X): return squareform(pdist(X, metric="cosine"))     # manuscript notebook 03

MIN_CROPS = 30
QE = {}
for m in ("clip", "dinov3"):
    for part in ("all", "real"):
        p = PRIV / f"qwen_{m}_exemplar_avg_raw_{TAG}_{part}.csv"
        if p.exists():
            d = load_means(p); QE[m, part] = d[d.n_crops >= MIN_CROPS].filter(like="dim_")
HAVE_QWEN = ("clip", "all") in QE
if not HAVE_QWEN:
    print("No Qwen crop embeddings yet: embed_after_subject.sh runs embed_qwen_crops.py after the subject run.")
else:
    print({f"{m} {p}": len(v) for (m, p), v in QE.items()}, "categories with >=", MIN_CROPS, "crops")"""),
code("""if HAVE_QWEN:
    rows, RD = [], {}
    for (m, part), E in QE.items():
        cats = [c for c in E.index if c not in EXCLUDE]
        RD[m, part] = (cats, cos_rdm(zrows(E.loc[cats].to_numpy(float))))
        rows.append({"model": m, "boxes": part, **domain_stats(RD[m, part][1], cats)})
    dom_t = pd.DataFrame(rows)
    display(dom_t.round(3).style.set_caption("domain clustering of category embeddings (Δ = between − within cosine distance)"))
    cats_c, Dc = RD["clip", "all"]; cats_d, Dd = RD["dinov3", "all"]
    sh = [c for c in cats_c if c in cats_d]
    a = Dc[np.ix_([cats_c.index(c) for c in sh], [cats_c.index(c) for c in sh])]
    b = Dd[np.ix_([cats_d.index(c) for c in sh], [cats_d.index(c) for c in sh])]
    print(f"CLIP vs DINOv3 RDM, {len(sh)} categories: Spearman {stats.spearmanr(lower(a), lower(b)).statistic:.2f}")
    cc = [c for c in cats_c if c in ix]
    a = Dc[np.ix_([cats_c.index(c) for c in cc], [cats_c.index(c) for c in cc])]
    b = Dq[np.ix_([ix[c] for c in cc], [ix[c] for c in cc])]
    print(f"CLIP distance vs co-occurrence distance, {len(cc)} categories: Spearman {stats.spearmanr(lower(a), lower(b)).statistic:.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, key, t in ((axs[0], ("clip", "all"), "CLIP"), (axs[1], ("dinov3", "all"), "DINOv3")):
        im = rdm_heatmap(ax, RD[key][1], RD[key][0], f"{t} RDM, {len(RD[key][0])} Qwen categories")
    fig.colorbar(im, ax=axs, shrink=.7, label="cosine distance"); plt.show()

    from sklearn.manifold import MDS
    top = [c for c in PREV.index if c in cats_c][:120]
    sub = Dc[np.ix_([cats_c.index(c) for c in top], [cats_c.index(c) for c in top])]
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4).fit_transform(sub)
    fig, ax = plt.subplots(figsize=(11, 9))
    for d in DOMAINS:
        k = [n for n, c in enumerate(top) if DOM_OF.get(c) == d]
        ax.scatter(xy[k, 0], xy[k, 1], s=26, color=DOM_COLOR[d], label=d, edgecolor="white", linewidth=.5)
    for n, c in enumerate(top):
        ax.annotate(c, xy[n], xytext=(3, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("CLIP category map (MDS), the 120 most frequent Qwen categories", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1)); fig.tight_layout(); plt.show()

    print("Nearest categories in CLIP space:")
    for c in [c for c in SHOW if c in cats_c]:
        nn = np.argsort(Dc[cats_c.index(c)])[1:9]
        print(f"  {c}: " + ", ".join(f"{cats_c[k]} {Dc[cats_c.index(c), k]:.2f}" for k in nn))"""),

md("""### Cross-check: categories shared with the manuscript's RDMs
Qwen categories whose name is exactly one of the manuscript's 129 categories, compared on those
categories (every source re-z-scored within them): this child's Qwen boxes, this child's YOLOE + CLIP
crops, all of BabyView (YOLOE + CLIP), and THINGS."""),
code("""if HAVE_QWEN:
    out = []
    for m in ("clip", "dinov3"):
        src = {"qwen (this child)": QE[m, "all"],
               "yoloe+clip (this child)": load_means(PRIV / f"yoloe_{m}_exemplar_avg_raw_{SUBJECT}.csv").filter(like="dim_"),
               "yoloe+clip (all children)": load_means(MS / "exemplar_set_embeddings" / "2025.2" / "valid129" / f"bv_{m}_exemplar_avg_raw_within_valid129.csv").filter(like="dim_"),
               "THINGS": load_means(MS / "exemplar_set_embeddings" / "valid129" / f"things_{m}_exemplar_avg_raw_within_valid129.csv").filter(like="dim_")}
        cats = [c for c in VALID129 if all(c in v.index for v in src.values()) and c not in EXCLUDE]
        R = {k: cos_rdm(zrows(v.loc[cats].to_numpy(float))) for k, v in src.items()}
        names = list(R)
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                out.append({"model": m, "a": names[x], "b": names[y], "n_categories": len(cats),
                            "spearman": stats.spearmanr(lower(R[names[x]]), lower(R[names[y]])).statistic})
    display(pd.DataFrame(out).round(3))
    mc = pd.read_csv(MS_RES / "bv_things_rdm_comparison_summary_v2_lowertri_filtered-0.27_valid129.csv")
    print("manuscript, all children vs THINGS on 129 categories:", mc.drop_duplicates("model").set_index("model").spearman_r.round(3).to_dict())"""),

md("""## Notes
* Categories: Qwen's labels, normalized only for plurals, exact synonyms and depiction words; "toy"
  without an object name stays "toy" and is one of the most frequent categories. Near-synonyms Qwen
  uses side by side (cup / sippy cup / mug) stay separate categories.
* Domains are approximate for categories outside the CDI list (CLIP-text nearest CDI words);
  check `results/category_domains.csv` and add overrides in `qwen_data.DOMAIN_OVERRIDES`.
* Temperature 1.0: individual frames are not exactly reproducible; aggregates are stable.
* Age = video date − rounded birth date (metadata 2025.1). One child, every 5th frame.
* Qwen boxes are not CLIP-filtered (YOLOE's are, at 0.27); toys and pictures are included unless
  "real only"."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "qwen_subject_analysis.ipynb")
print("wrote", HERE / "qwen_subject_analysis.ipynb")
