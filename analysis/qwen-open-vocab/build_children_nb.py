#!/usr/bin/env python3
"""Write children_analysis.ipynb: Qwen's open-vocabulary detections across the children annotated so far.

Reads out/private/children_{objects,frames}.pkl from build_children_cache.py (rerun it as the full run finishes
more children). Execute into out/private/ (it reads private data); figures (aggregate only) go to figures/children/.

  python build_children_nb.py
  jupyter nbconvert --to notebook --execute children_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \\
      --ExecutePreprocessor.timeout=7200 --output-dir out/private --output children_analysis_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Qwen3.8-27B open-vocabulary detections across children (2025.2, every 5th 1 fps frame)

Children whose frames are all annotated so far (`build_children_cache.py`; the full run finishes children in
ID order). Setup B (open label + format + box, at most 30 objects), temperature 1.0. Categories are Qwen's own
(`cat`, lightly normalized), person excluded, near-duplicate boxes dropped. Domains: the hierarchy table
(Qwen judged the whole name; CDI convention for CDI words) where available, else the older automatic rule.
**Private:** keep the executed notebook in `out/private/`; saved figures are aggregate only.

Sections: 1 overview · 2 long tail (per child, equal-size samples, shared vs child-specific categories,
between-child agreement) · 3 depictions (per child, by age, by domain) · 4 co-occurrence across children.
Children with < 10,000 frames are shown but left out of fits and between-child comparisons (`BIG`)."""),
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path.cwd()))                                   # this folder
sys.path.insert(0, str(Path.cwd().parents[1] / "annotation" / "qwen_pilot"))   # pipeline loaders
import qwen_data as qd
from make_subject_figures import COLORS, ORDER, fit

PRIV = Path.cwd() / "out" / "private"          # children_*.pkl from build_children_cache.py
FIG = Path.cwd() / "figures" / "children"; FIG.mkdir(parents=True, exist_ok=True)
fr = pd.read_pickle(PRIV / "children_frames.pkl")
obj = pd.read_pickle(PRIV / "children_objects.pkl")
o = obj[~obj.dup & (obj.cat != "person")].copy()
o["fmt"] = o.format.replace({"photo": "picture", "drawing": "picture"})
o["depicted"] = o.format != "real"
NF = fr.groupby("subject").size().sort_values(ascending=False)
KIDS = list(NF.index); BIG = [k for k in KIDS if NF[k] >= 10_000]
import fig_style as FS; FS.apply()                              # manuscript style: Lato, bold labels and legends
FMT = {k: FS.FMT_COLOR[k] for k in ("real", "toy", "picture", "screen")}
INK, INK2, GRID = FS.INK, FS.MUTED, "#ececec"
plt.rcParams.update({
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True, "figure.dpi": 110})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")

def prev_by_child(d, unit_col="unit_id"):                    # category x child: share of the child's frames
    p = d.groupby(["cat", "subject"])[unit_col].nunique().unstack(fill_value=0)
    return p.div(NF.reindex(p.columns), axis=1)

print(f"{len(KIDS)} children, {len(fr):,} frames, {len(o):,} boxes, {o.cat.nunique():,} categories; "
      f"{len(BIG)} children with >= 10,000 frames")
print(f"domain from the hierarchy table for {(o.domain_source == 'hierarchy').mean():.1%} of boxes")"""),

md("## 1. Overview"),
code("""ov = pd.DataFrame({
    "frames": NF, "videos": fr.groupby("subject").video_id.nunique(),
    "age_from": fr.groupby("subject").age_months.min(), "age_to": fr.groupby("subject").age_months.max(),
    "objects_per_frame": o.groupby("subject").size() / NF,
    "categories": o.groupby("subject").cat.nunique(),
    "categories_30fr": (o.groupby(["subject", "cat"]).unit_id.nunique() >= 30).groupby(level=0).sum(),
    "share_depicted": o.groupby("subject").depicted.mean(),
    "frames_at_cap": fr.groupby("subject").at_cap.mean(), "marching_loops": fr.groupby("subject").marching.mean()})
display(ov.round(3))"""),

md("""## 2. Long tail
### 2a. Rank-frequency per child
Frame prevalence of each category within a child, ranked. Power law fitted on categories in >= 30 of the
child's frames (manuscript notebook 01 fit). Larger samples reach further into the tail, so 2b compares
children at the same number of frames."""),
code("""P = prev_by_child(o)
fits = []
fig, ax = plt.subplots(figsize=(8, 5.5))
for k in KIDS:
    v = np.sort(P[k].values[P[k].values > 0])[::-1]
    ax.loglog(np.arange(1, len(v) + 1), v, color=FS.MUTED if k in BIG else FS.FAINT, lw=.9, alpha=.75 if k in BIG else .5)
    c = P[k][P[k] * NF[k] >= 30].values
    if len(c) >= 10:
        _, a, _, r2 = fit(c)
        fits.append({"subject": k, "frames": NF[k], "categories_30fr": len(c), "alpha": a, "r2_log": r2})
pooled = o.groupby("cat").unit_id.nunique().div(len(fr)).sort_values(ascending=False)
ax.loglog(np.arange(1, len(pooled) + 1), pooled.values, color=FS.DS_COLOR["bv"], lw=2.2, label="all children pooled")
ax.set_xlabel("category rank"); ax.set_ylabel("share of the child's frames")
ax.set_title("Rank-frequency of Qwen categories, one line per child (grey: < 10,000 frames)", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False); fig.tight_layout(); save(fig, "long_tail_rank_frequency_by_child"); plt.show()
fits = pd.DataFrame(fits).set_index("subject")
display(fits.round(3))"""),

md("""### 2b. Equal-size samples
Each child with >= 10,000 frames, resampled to 10,000 random frames (5 draws): distinct categories, categories
in >= 30 frames, and the power-law alpha. Accumulation curves: distinct categories as frames are added in
random order."""),
code("""N_EQ, rng = 10_000, np.random.default_rng(0)
fid = {u: i for i, u in enumerate(fr.unit_id)}                 # integer frame ids: fast set operations
o["fid"] = o.unit_id.map(fid).astype(np.int64)
fr["fid"] = fr.unit_id.map(fid).astype(np.int64)
frames_of = fr.groupby("subject").fid.apply(np.array)
by_kid = {k: o.loc[o.subject == k, ["fid", "cat"]].drop_duplicates() for k in KIDS}
rows = []
for k in BIG:
    ok = by_kid[k]
    for d in range(5):
        s = rng.choice(frames_of[k], N_EQ, replace=False)
        sub = ok[np.isin(ok.fid.values, s)]
        pv = sub.groupby("cat").size() / N_EQ
        c = pv[pv * N_EQ >= 30].values
        _, a, _, _ = fit(c)
        rows.append({"subject": k, "draw": d, "categories": len(pv), "categories_30fr": len(c), "alpha": a})
eq = pd.DataFrame(rows).groupby("subject")[["categories", "categories_30fr", "alpha"]].agg(["mean", "std"]).round(3)
display(eq)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for k in KIDS:
    ok = by_kid[k]
    order = rng.permutation(frames_of[k])
    rank = pd.Series(np.arange(len(order)), index=order)          # position of each frame in the random order
    first = ok.assign(r=rank.reindex(ok.fid.values).values).groupby("cat").r.min()
    curve = np.bincount(first.values.astype(int), minlength=len(order)).cumsum()
    xs = np.unique(np.geomspace(1, len(order), 200).astype(int)) - 1
    axs[0].semilogx(xs + 1, curve[xs], color=FS.MUTED if k in BIG else FS.FAINT, lw=1)
axs[0].set_xlabel("frames (random order)"); axs[0].set_ylabel("distinct categories seen")
axs[0].set_title("Accumulation of categories per child", loc="left", fontsize=11, color=INK)
m_ = eq["alpha"]["mean"].sort_values()
axs[1].errorbar(m_.values, range(len(m_)), xerr=eq["alpha"]["std"].reindex(m_.index).values, fmt="o", color=FS.DS_COLOR["bv"], ms=6, mec="white")
axs[1].set_yticks(range(len(m_)), m_.index, fontsize=8); axs[1].set_xlabel(f"alpha at {N_EQ:,} frames (mean ± sd of 5 draws)")
axs[1].set_title("Steepness of the long tail at equal sample size", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "long_tail_equal_samples"); plt.show()"""),

md("""### 2c. Shared vs child-specific categories
A category is "present" for a child when it is in >= 0.1% of the child's frames (children with >= 10,000
frames). How many children each category reaches, which are universal, and which are frequent for one child
but rare elsewhere."""),
code("""PB = P[BIG]
present = PB >= .001
reach = present.sum(axis=1)
pooled_big = o[o.subject.isin(BIG)].groupby("cat").unit_id.nunique().div(NF[BIG].sum())
fig, ax = plt.subplots(figsize=(8, 3.6))
h = reach[reach > 0].value_counts().sort_index()
ax.bar(h.index, h.values, color=FS.DS_COLOR["bv"], width=.8)
ax.set_xlabel(f"number of children (of {len(BIG)}) where the category is in >= 0.1% of frames"); ax.set_ylabel("categories")
ax.set_title("How widely categories are shared across children", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "categories_shared_across_children"); plt.show()
univ = reach[reach == len(BIG)].index
print(f"{len(univ)} categories present for every child; top 60 by pooled prevalence:")
print(", ".join(pooled_big.reindex(univ).sort_values(ascending=False).head(60).index))
spec = pd.DataFrame({"max_child": PB.idxmax(axis=1), "max_share": PB.max(axis=1),
                     "median_other": PB.apply(lambda r: r.drop(r.idxmax()).median(), axis=1)})
spec = spec[(spec.max_share >= .005)].assign(ratio=lambda d: d.max_share / (d.median_other + 1e-4))
print("Frequent for one child, rare for the others (>= 0.5% of that child's frames):")
display(spec.sort_values("ratio", ascending=False).head(30).round(4))

top = pooled_big.head(40).index
fig, ax = plt.subplots(figsize=(14, 4.8))
for j, c in enumerate(top):
    ax.scatter(np.full(len(BIG), j) + rng.uniform(-.2, .2, len(BIG)), PB.loc[c].values, s=10, color=FS.BASELINE, zorder=2)
    ax.scatter(j, pooled_big[c], s=40, color=COLORS.get(o[o.cat == c].domain.iloc[0], COLORS["other"]), zorder=3, edgecolor="white")
ax.set_xticks(range(len(top)), top, rotation=60, ha="right", fontsize=8); ax.set_ylabel("share of frames")
ax.set_title("Top 40 categories: pooled (coloured by domain) and each child (grey)", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "top40_categories_by_child"); plt.show()"""),

md("""### 2d. How similar are children's category distributions?
Spearman correlation of category prevalence between pairs of children, over categories in >= 0.1% of frames
for at least one of the two."""),
code("""R = pd.DataFrame(np.nan, index=BIG, columns=BIG)
for a in BIG:
    for b in BIG:
        keep = (PB[a] >= .001) | (PB[b] >= .001)
        R.loc[a, b] = stats.spearmanr(PB.loc[keep, a], PB.loc[keep, b]).statistic
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(R.values, cmap="viridis", vmin=np.nanmin(R.values), vmax=1); ax.grid(False)
ax.set_xticks(range(len(BIG)), BIG, rotation=90, fontsize=8); ax.set_yticks(range(len(BIG)), BIG, fontsize=8)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman, category prevalence")
ax.set_title("Between-child agreement of category distributions", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "between_child_prevalence_agreement"); plt.show()
off = R.values[~np.eye(len(BIG), dtype=bool)]
print(f"pairwise Spearman: median {np.median(off):.2f}, range {off.min():.2f}-{off.max():.2f}")"""),

md("""## 3. Depictions
### 3a. Format mix per child"""),
code("""mix = o.groupby("subject").fmt.value_counts(normalize=True).unstack(fill_value=0)[["real", "toy", "picture", "screen"]]
mix = mix.loc[mix.real.sort_values().index]
fig, ax = plt.subplots(figsize=(9, .32 * len(mix) + 1.4))
left = np.zeros(len(mix))
for f in ["real", "toy", "picture", "screen"]:
    ax.barh(range(len(mix)), mix[f].values, left=left, color=FMT[f], edgecolor="white", linewidth=.6, label=f, height=.75)
    left += mix[f].values
ax.set_yticks(range(len(mix)), [f"{k} ({NF[k]:,} fr)" for k in mix.index], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("share of boxes")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0))
ax.set_title("Visual format of Qwen's boxes per child", loc="left", fontsize=11, color=INK, pad=24)
fig.tight_layout(); save(fig, "format_mix_by_child"); plt.show()
display(mix.round(3))"""),

md("""### 3b. By age
Share of boxes that are toys, pictures and screens per month of age; thin lines = children with >= 500 frames
that month, thick = pooled over children (months with >= 3 children)."""),
code("""o["age_bin"] = np.floor(o.age_months).astype("Int64")
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, f in zip(axs, ["toy", "picture", "screen"]):
    per = o.groupby(["subject", "age_bin"]).fmt.apply(lambda x: (x == f).mean()).unstack(0)
    nfr = o.groupby(["subject", "age_bin"]).unit_id.nunique().unstack(0)
    per = per.where(nfr >= 500)
    for k in per.columns:
        ax.plot(per.index, per[k], color=FMT[f], lw=.8, alpha=.35)
    pooled_m = per.mean(axis=1)[per.notna().sum(axis=1) >= 3]
    ax.plot(pooled_m.index, pooled_m.values, color=FMT[f], lw=2.6, marker="o", ms=4, mec="white")
    ax.set_title(f"{f}", loc="left", fontsize=11, color=INK); ax.set_xlabel("age (months)")
axs[0].set_ylabel("share of boxes")
fig.suptitle("Depicted objects by age across children", x=.01, ha="left", fontsize=12)
fig.tight_layout(); save(fig, "depictions_by_age_across_children"); plt.show()"""),

md("""### 3c. Which domains are seen as depictions, child by child
Share of a domain's boxes that are depictions (toy, picture or screen), per child."""),
code("""dm = o[o.subject.isin(BIG)].groupby(["subject", "domain"]).depicted.mean().unstack().reindex(columns=[d for d in ORDER if d in set(o.domain)])
fig, ax = plt.subplots(figsize=(10, .35 * len(dm) + 1.6))
im = ax.imshow(dm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
ax.set_xticks(range(dm.shape[1]), [c.replace("_", " ") for c in dm.columns], rotation=35, ha="right")
ax.set_yticks(range(len(dm)), dm.index, fontsize=8)
for i in range(dm.shape[0]):
    for j in range(dm.shape[1]):
        v = dm.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > .55 else INK)
fig.colorbar(im, ax=ax, shrink=.8, label="share of the domain's boxes that are depictions")
ax.set_title("Depicted share by domain and child", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "depicted_share_domain_by_child"); plt.show()
dc = o[o.subject.isin(BIG)].groupby(["cat", "subject"]).depicted.agg(["mean", "size"]).reset_index()
dc = dc[dc["size"] >= 20]
cons = dc.groupby("cat").agg(children=("subject", "nunique"), mostly_depicted=("mean", lambda x: (x >= .5).sum()),
                             median_depicted=("mean", "median"))
cons = cons[cons.children >= len(BIG) // 2]
print("Categories seen (>= 20 boxes) by at least half the children, most consistently depicted:")
display(cons.sort_values(["median_depicted", "children"], ascending=False).head(30).round(2))"""),

md("""## 4. Co-occurrence across children
Distance = 1 − Jaccard over frames, per child, on the categories in >= 0.5% of frames pooled. For each pair of
children, Spearman between their co-occurrence RDMs over categories in >= 20 frames for both."""),
code("""from scipy.spatial.distance import squareform
CC = list(pooled_big[pooled_big >= .005].index.drop("person", errors="ignore"))
ix = {c: i for i, c in enumerate(CC)}
def jac(k):
    d = o[(o.subject == k) & o.cat.isin(ix)]
    f = pd.factorize(d.unit_id)[0]
    B = sp.csr_matrix((np.ones(len(d)), (f, d.cat.map(ix).values)), shape=(f.max() + 1, len(CC)))
    B.sum_duplicates(); B.data[:] = 1
    C = (B.T @ B).toarray(); n = np.diag(C).astype(float); U = n[:, None] + n[None, :] - C
    J = np.divide(C, U, out=np.zeros_like(U, dtype=float), where=U > 0)
    return J, n
JN = {k: jac(k) for k in BIG}
print(f"{len(CC)} categories in >= 0.5% of pooled frames")
CR = pd.DataFrame(np.nan, index=BIG, columns=BIG)
for a in BIG:
    for b in BIG:
        if a == b: CR.loc[a, b] = 1; continue
        keep = np.flatnonzero((JN[a][1] >= 20) & (JN[b][1] >= 20))
        i, j = np.tril_indices(len(keep), -1)
        CR.loc[a, b] = stats.spearmanr(JN[a][0][keep][:, keep][i, j], JN[b][0][keep][:, keep][i, j]).statistic
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(CR.values, cmap="viridis", vmin=np.nanmin(CR.values), vmax=1); ax.grid(False)
ax.set_xticks(range(len(BIG)), BIG, rotation=90, fontsize=8); ax.set_yticks(range(len(BIG)), BIG, fontsize=8)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman, co-occurrence (Jaccard)")
ax.set_title("Between-child agreement of which objects appear together", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "between_child_cooccurrence_agreement"); plt.show()
off = CR.values[~np.eye(len(BIG), dtype=bool)]
print(f"pairwise Spearman: median {np.nanmedian(off):.2f}, range {np.nanmin(off):.2f}-{np.nanmax(off):.2f}")
Js = np.stack([JN[k][0] for k in BIG]); i, j = np.tril_indices(len(CC), -1)
pairs = pd.DataFrame({"a": np.array(CC)[i], "b": np.array(CC)[j], "median_jaccard": np.median(Js[:, i, j], axis=0),
                      "min_jaccard": Js[:, i, j].min(axis=0), "max_jaccard": Js[:, i, j].max(axis=0)})
print("Pairs that co-occur for every child (highest minimum Jaccard):")
display(pairs.sort_values("min_jaccard", ascending=False).head(20).round(3))
print("Pairs that co-occur strongly for some children only (largest max − median):")
display(pairs.assign(spread=pairs.max_jaccard - pairs.median_jaccard).sort_values("spread", ascending=False).head(20).round(3))"""),

md("""## Notes
* Children finish in ID order, so this set is not a random sample of children; rerun `build_children_cache.py`
  and this notebook as more finish.
* Domains for categories not yet in the hierarchy table use the older automatic rule (see the coverage line in
  the setup cell); run `qwen_hierarchy.py` on the new categories when GPUs are free.
* Generic labels ("toy", "animal") are kept as Qwen wrote them; the crop pass will resolve some of them.
* Formats are Qwen's own; "real" was the least reliable label in the montages (toys sometimes called real)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "children_analysis.ipynb")
print("wrote", HERE / "children_analysis.ipynb")
