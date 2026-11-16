#!/usr/bin/env python3
"""Write open_category_analysis.ipynb: the manuscript's analyses on Qwen's open categories (beyond valid129).

Inputs: out/private/open/ (build_open_means.py: 844 basic-level categories with >= 300 boxes from >= 10 children, no
generic labels, person or surfaces), THINGS embeddings (annotation/qwen_pilot/embed_things.py), the manuscript's
129-category results for the side-by-side. Methods follow the manuscript: mean of L2-normalized crop vectors per category, z-score each dimension
across the categories compared, RDM = cosine distance, power law on frame prevalence (notebook 01), domain clustering as
between - within mean distance with a category bootstrap. Figures (aggregate) -> figures/; tables -> results/.

  python build_open_nb.py
  [QOV_LABELS=renamed] jupyter nbconvert --to notebook --execute open_category_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \\
      --ExecutePreprocessor.timeout=7200 --output-dir out/private --output open_category_analysis_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Manuscript analyses on Qwen's open categories (2025.2, every 5th 1 fps frame)

Qwen3.8-27B detections of the full 2025.2 run, grouped by Qwen's basic-level category: the Tier 1 list (828 categories
with >= 300 boxes from >= 10 children; generic labels, person and surfaces left out; synonyms merged;
data/included_categories_qwen_open_tier1.txt). THINGS is matched by the WordNet mapping used for every comparison
dataset (mapping/map_things.csv: each THINGS concept counts for its closest Tier 1 category; build_crossdataset_mapping.py),
read from the cross-dataset tables (out/crossdataset*/tables_*.npz, build_crossdataset_tables.py). Sections: 0 the 129-category replication (Qwen vs YOLOE + CLIP) · 1 the open set · 2 long tail ·
3 BabyView RDMs and domain clustering · 4 individual children (pairwise agreement, each child vs THINGS, top-8) · 5 BabyView vs THINGS on the shared concepts ·
6 category-wise BabyView-THINGS similarity. "all" = every format (as YOLOE + CLIP counted depictions); "real" = format
real only. Domains are the CDI semantic domains from the hierarchy table (approximate outside the CDI list)."""),
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.spatial.distance import pdist, squareform
HERE = Path.cwd()                                              # analysis/qwen-open-vocab
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
from make_subject_figures import COLORS, ORDER, fit, cluster_within

import variant as V                                            # QOV_LABELS=renamed (environment): renamed-label means
print("labels:", V.LABELS)
OPEN = V.out_dir(HERE / "out" / "private" / "open")
FIG = HERE / "figures" if V.SUFFIX == "" else HERE / "figures" / ("open" + V.SUFFIX); FIG.mkdir(exist_ok=True)
RES = HERE / "results" if V.SUFFIX == "" else HERE / "results" / ("open" + V.SUFFIX); RES.mkdir(exist_ok=True)
MS = HERE.parents[1] / "analysis" / "manuscript-2026"
THINGS = Path("/data2/dataset/babyview/related_data/things/embeddings")
import fig_style as FS; FS.apply()                              # manuscript style: Lato, bold labels and legends
INK, INK2 = FS.INK, FS.MUTED
plt.rcParams.update({"figure.dpi": 110})

cats = pd.read_csv(OPEN / "categories.csv", index_col=0)
cats["things_concept"] = cats.things_concept.fillna("")
CATS = list(cats.index); DOM = cats.domain.to_dict()
S = {m: np.load(OPEN / f"sums_{m}.npz", allow_pickle=True) for m in ("clip", "dinov3")}
SUBJ = list(S["clip"]["subjects"])
assert list(S["clip"]["categories"]) == CATS
fbc = pd.read_csv(OPEN / "frames_by_child.csv").set_index(["subject", "cat"])
ftot = pd.read_csv(OPEN / "frames_total.csv").set_index("subject").frames

def pooled(model, part="all", min_n=1):
    s, n = S[model][f"sum_{part}"].sum(0), S[model][f"n_{part}"].sum(0)
    keep = n >= min_n
    return pd.DataFrame(s[keep] / n[keep, None], index=np.array(CATS)[keep])

def things_wordnet(m):
    \"\"\"THINGS category means on the Tier 1 categories, WordNet mapping (same rule as ImageNet and LVIS).\"\"\"
    z = np.load(V.out_dir(HERE / "out" / "crossdataset") / f"tables_{m}.npz", allow_pickle=True)
    return pd.DataFrame(z["things__means"], index=list(z["things__categories"]))
THW = {m: things_wordnet(m) for m in ("clip", "dinov3")}

def zrows(X): return (X - X.mean(0)) / (X.std(0) + 1e-10)
def rdm(M): return squareform(pdist(zrows(np.asarray(M, float)), metric="cosine"))
def lower(D): return D[np.tril_indices(len(D), -1)]

def domain_stats(D, cs, n_boot=0, seed=0):
    sem = np.array([DOM.get(c, "other") for c in cs]); i, j = np.tril_indices(len(cs), -1)
    same = sem[i] == sem[j]; d = D[i, j]
    out = {"n": len(cs), "within": d[same].mean(), "between": d[~same].mean(), "delta": d[~same].mean() - d[same].mean(),
           "auc": stats.mannwhitneyu(d[~same], d[same]).statistic / ((~same).sum() * same.sum())}
    return out

def boot_auc(Ds, cs, n_boot=1000, seed=0):
    \"\"\"Same within-domain category bootstrap as boot_delta, for AUC = P(between > within).\"\"\"
    rng = np.random.default_rng(seed); sem = np.array([DOM.get(c, "other") for c in cs])
    doms = sorted(set(sem)); idx_by = {d: np.flatnonzero(sem == d) for d in doms}
    res = {k: [] for k in Ds}
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(idx_by[d], len(idx_by[d])) for d in doms])
        i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]; ok = a != b
        same = sem[a][ok] == sem[b][ok]
        for k, D in Ds.items():
            d = D[a[ok], b[ok]]; r = stats.rankdata(np.concatenate([d[same], d[~same]]))
            res[k].append((r[same.sum():].sum() - (~same).sum() * ((~same).sum() + 1) / 2) / (same.sum() * (~same).sum()))
    return {k: np.array(v) for k, v in res.items()}

def boot_delta(Ds, cs, n_boot=1000, seed=0):
    \"\"\"Bootstrap over categories (resampled within domain, self-pairs dropped): delta for each RDM in Ds on the same draws.\"\"\"
    rng = np.random.default_rng(seed); sem = np.array([DOM.get(c, "other") for c in cs])
    doms = sorted(set(sem)); idx_by = {d: np.flatnonzero(sem == d) for d in doms}
    res = {k: [] for k in Ds}
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(idx_by[d], len(idx_by[d])) for d in doms])
        i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]; ok = a != b
        same = sem[a][ok] == sem[b][ok]
        for k, D in Ds.items():
            d = D[a[ok], b[ok]]; res[k].append(d[~same].mean() - d[same].mean())
    return {k: np.array(v) for k, v in res.items()}

print(f"{len(CATS)} categories, {len(SUBJ)} children, {int(S['clip']['n_all'].sum()):,} boxes "
      f"({int(S['clip']['n_real'].sum()):,} real); THINGS (WordNet mapping): {len(THW['clip'])} categories")"""),

md("## 1. The open category set"),
code("""t = cats.groupby("domain").agg(categories=("boxes", "size"), boxes=("boxes", "sum"),
                               in_things=("boxes", lambda s: s.index.isin(THW["clip"].index).sum()))
display(t.reindex([d for d in ORDER if d in t.index]))
print("most frequent categories not in THINGS:", ", ".join(cats[~cats.index.isin(THW["clip"].index)].sort_values("boxes", ascending=False).index[:40]))"""),

md("""## 2. Long tail of the open categories
Frame prevalence = share of all frames (pooled over children) with the category; power law fitted as in manuscript
notebook 01, overall and per domain, for all formats and real only. Per child: alpha on the child's own prevalences."""),
code("""N = ftot.sum()
prev = {p: fbc[f"frames_{p}"].groupby(level="cat").sum().reindex(CATS).fillna(0) / N for p in ("all", "real")}
fits = []
for p in ("all", "real"):
    v = prev[p][prev[p] > 0]
    _, a, i, r2 = fit(v.values); fits.append({"formats": p, "domain": "all", "n": len(v), "alpha": a, "r2_log": r2})
    for d in ORDER:
        vv = v[[DOM.get(c) == d for c in v.index]]
        if len(vv) >= 5:
            _, a, i, r2 = fit(vv.values); fits.append({"formats": p, "domain": d, "n": len(vv), "alpha": a, "r2_log": r2})
fits = pd.DataFrame(fits); fits.to_csv(RES / "open_long_tail_powerlaw.csv", index=False)
display(fits.pivot(index="domain", columns="formats", values=["alpha", "n"]).round(3))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for p, col in (("all", FS.DS_COLOR["bv"]), ("real", FS.DS_COLOR["bv_real"])):
    v = np.sort(prev[p].values[prev[p].values > 0])[::-1]; r = np.arange(1, len(v) + 1)
    _, a, i, r2 = fit(v)
    axs[0].loglog(r, v, "o", ms=2.5, color=col, mfc=col if p == "all" else "white", label=f"{p}: alpha {a:.2f}, R² {r2:.2f}")
    axs[0].loglog(r, np.exp(i) * r ** -a, color=col, lw=1, ls="--")
axs[0].set_xlabel("rank"); axs[0].set_ylabel("frame prevalence"); axs[0].legend(frameon=False)
axs[0].set_title(f"{len(CATS)} open categories, all children pooled", loc="left", fontsize=11)
top = prev["all"].sort_values(ascending=False).head(40)
axs[1].bar(range(40), top.values, color=[COLORS.get(DOM.get(c), COLORS["other"]) for c in top.index])
axs[1].set_xticks(range(40), top.index, rotation=70, ha="right", fontsize=8); axs[1].set_ylabel("frame prevalence")
axs[1].set_title("Top 40 open categories (coloured by domain)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "open_long_tail.png", dpi=200, bbox_inches="tight"); plt.show()
pc = []
for s in SUBJ:
    if ftot[s] < 10_000: continue
    v = fbc.loc[s, "frames_all"].reindex(CATS).fillna(0) / ftot[s]
    v = v[v * ftot[s] >= 30]
    _, a, _, r2 = fit(v.values); pc.append({"subject": s, "frames": ftot[s], "categories_30fr": len(v), "alpha": a, "r2_log": r2})
pc = pd.DataFrame(pc).set_index("subject"); pc.to_csv(RES / "open_long_tail_by_child.csv")
print("per child (>= 10,000 frames; categories in >= 30 of the child's frames):"); display(pc.round(3))"""),

md("""## 3. BabyView RDMs on the open set and domain clustering
Pooled category means (categories with >= 30 boxes), z-scored across the categories, cosine RDM. Domain clustering:
between - within distance (delta) and AUC; 95% CI from a category bootstrap within domains."""),
code("""BV = {}
rows = []
for m in ("clip", "dinov3"):
    for p in ("all", "real"):
        M = pooled(m, p, min_n=30); D = rdm(M.values); cs = list(M.index); BV[m, p] = (cs, D)
        b = boot_delta({"bv": D}, cs, n_boot=500, seed=1)["bv"]
        rows.append({"model": m, "formats": p, **domain_stats(D, cs), "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
dc = pd.DataFrame(rows); dc.to_csv(RES / "open_bv_domain_clustering.csv", index=False)
display(dc.round(3))
print("CLIP vs DINOv3 RDM (all formats):",
      round(stats.spearmanr(lower(BV['clip','all'][1]), lower(BV['dinov3','all'][1])).statistic, 3))
for m in ("clip", "dinov3"):
    cs, D = BV[m, "all"]
    order = sorted(range(len(cs)), key=lambda k: (ORDER.index(DOM.get(cs[k], "other")) if DOM.get(cs[k]) in ORDER else 99, cs[k]))
    csd = [cs[k] for k in order]; Dd = D[np.ix_(order, order)]
    idx = cluster_within(csd, Dd, DOM); csd = [csd[k] for k in idx]; Dd = Dd[np.ix_(idx, idx)]
    fig, ax = plt.subplots(figsize=(10, 9.4))
    im = ax.imshow(Dd, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    ds = [DOM.get(c, "other") for c in csd]; starts = [0] + [k for k in range(1, len(ds)) if ds[k] != ds[k - 1]]
    for b in starts[1:]:
        ax.axhline(b - .5, color="white", lw=.8); ax.axvline(b - .5, color="white", lw=.8)
    ends = starts[1:] + [len(ds)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [ds[a].replace("_", " ") for a in starts], fontsize=9)
    for tk, a in zip(ax.get_yticklabels(), starts): tk.set_color(COLORS.get(ds[a], COLORS["other"]))
    ax.set_xticks([]); ax.set_title(f"BabyView {m.upper() if m == 'clip' else 'DINOv3'} RDM, {len(cs)} open categories "
                                    "(clustered within domain)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=.7, label="cosine distance")
    fig.savefig(FIG / f"open_bv_rdm_{m}.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 4. Individual children's RDMs
Per child: category means from the child's own boxes (categories with >= 20 of the child's boxes). For each pair of
children with >= 10,000 frames: Spearman between their RDMs over the categories both have (each child z-scored within
those categories). Compared with each child's agreement with the pooled RDM."""),
code("""big = [s for s in SUBJ if ftot[s] >= 10_000]
def child_means(m, s, part="all", min_n=20):
    k = SUBJ.index(s); n = S[m][f"n_{part}"][k]; keep = n >= min_n
    return pd.DataFrame(S[m][f"sum_{part}"][k][keep] / n[keep, None], index=np.array(CATS)[keep])
out = []
for m in ("clip", "dinov3"):
    CM = {s: child_means(m, s) for s in big}
    P = pooled(m, "all", 30)
    for x in range(len(big)):
        for y in range(x + 1, len(big)):
            a, b = CM[big[x]], CM[big[y]]
            sh = a.index.intersection(b.index)
            if len(sh) < 50: continue
            out.append({"model": m, "a": big[x], "b": big[y], "n_categories": len(sh),
                        "spearman": stats.spearmanr(lower(rdm(a.loc[sh].values)), lower(rdm(b.loc[sh].values))).statistic})
pairs = pd.DataFrame(out); pairs.to_csv(RES / "open_individual_rdm_pairs.csv", index=False)
display(pairs.groupby("model").agg(pairs=("spearman", "size"), median=("spearman", "median"), min=("spearman", "min"),
                                   max=("spearman", "max"), median_categories=("n_categories", "median")).round(3))"""),

md("""### 4b. Each child on their own: domain clustering and agreement with THINGS
For every child with >= 10,000 frames: the child's RDM (categories with >= 20 of the child's boxes), its domain
clustering (delta, AUC), and its Spearman with THINGS on the child's categories that THINGS has (WordNet mapping), next to
THINGS' own delta on exactly those categories (child - THINGS > 0 = the child's view clusters by domain more than THINGS)."""),
code("""rows = []
for m in ("clip", "dinov3"):
    for s in big:
        cm = child_means(m, s)
        Dc = rdm(cm.values); dsc = domain_stats(Dc, list(cm.index))
        sh = [c for c in cm.index if c in THW[m].index]
        Dcs = rdm(cm.loc[sh].values); Dts = rdm(THW[m].loc[sh].values)
        dct, dtt = domain_stats(Dcs, sh), domain_stats(Dts, sh)
        rows.append({"model": m, "subject": s, "frames": int(ftot[s]), "categories": len(cm), "delta": dsc["delta"], "auc": dsc["auc"],
                     "things_categories": len(sh), "rdm_spearman_things": stats.spearmanr(lower(Dcs), lower(Dts)).statistic,
                     "child_delta_shared": dct["delta"], "things_delta_shared": dtt["delta"],
                     "child_minus_things": dct["delta"] - dtt["delta"]})
pcs = pd.DataFrame(rows); pcs.to_csv(RES / "open_per_child_clustering_things.csv", index=False)
display(pcs.groupby("model")[["categories", "delta", "auc", "rdm_spearman_things", "child_minus_things"]].describe().T.round(3))
print("children whose view clusters by domain more than THINGS (child - THINGS > 0):",
      pcs.groupby("model").child_minus_things.apply(lambda x: f"{(x > 0).sum()} of {len(x)}").to_dict())
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, col, lab in ((axs[0], "child_minus_things", "child - THINGS domain clustering (delta)"),
                     (axs[1], "rdm_spearman_things", "Spearman with THINGS RDM")):
    for k, (m, colr) in enumerate((("clip", FS.DS_COLOR["bv"]), ("dinov3", FS.DS_COLOR["things"]))):
        v = pcs[pcs.model == m][col].values
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.12, .12, len(v)), v, s=22, color=colr, edgecolor="white")
        ax.hlines(np.median(v), k - .25, k + .25, color=INK, lw=2)
    ax.set_xticks([0, 1], ["CLIP", "DINOv3"]); ax.set_ylabel(lab)
    if col == "child_minus_things": ax.axhline(0, color=FS.BASELINE, lw=1, ls="--")
axs[0].set_title("Per child (>= 10,000 frames), open categories", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "open_per_child_clustering_things.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""### 4c. The 8 densest children
Children with the most Qwen boxes in the open set; pairwise RDM agreement among them (each pair on the categories both
have, >= 20 boxes each), as in the manuscript's top-8 analysis."""),
code("""dens = pd.Series(S["clip"]["n_all"].sum(1), index=SUBJ).sort_values(ascending=False)
top8 = list(dens.index[:8]); print("densest children:", {s: int(dens[s]) for s in top8})
for m in ("clip", "dinov3"):
    CM = {s: child_means(m, s) for s in top8}
    M = pd.DataFrame(np.eye(8), index=top8, columns=top8)
    for a in range(8):
        for b in range(a + 1, 8):
            sh = CM[top8[a]].index.intersection(CM[top8[b]].index)
            M.iloc[a, b] = M.iloc[b, a] = stats.spearmanr(lower(rdm(CM[top8[a]].loc[sh].values)), lower(rdm(CM[top8[b]].loc[sh].values))).statistic
    M.to_csv(RES / f"open_top8_rdm_agreement_{m}.csv")
    off = M.values[~np.eye(8, dtype=bool)]
    print(f"{m}: top-8 pairwise Spearman median {np.median(off):.3f} (range {off.min():.3f}-{off.max():.3f})")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M.values, cmap="viridis", vmin=np.floor(off.min() * 20) / 20, vmax=1); ax.set_xticks(range(8), top8, rotation=90, fontsize=8)
    ax.set_yticks(range(8), top8, fontsize=8); fig.colorbar(im, ax=ax, shrink=.8, label="Spearman between RDMs")
    ax.set_title(f"Top-8 children, {m.upper() if m == 'clip' else 'DINOv3'}", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"open_top8_rdm_agreement_{m}.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 5. BabyView vs THINGS on the shared categories
Categories THINGS has under the WordNet mapping, with >= 30 Qwen boxes. THINGS category means from all images mapped to the
category (mean of L2-normalized vectors), BabyView means as above; each z-scored within the shared set. Differences in delta
and AUC with the same within-domain category bootstrap. Check first: our THINGS embeddings reproduce the manuscript's
THINGS tables on the 129 categories (exact concept names, as the manuscript)."""),
code("""TH = {m: pd.read_csv(THINGS / f"things_{m}_concept_mean_raw.csv").set_index("concept").filter(like="dim_") for m in ("clip", "dinov3")}
for m in ("clip", "dinov3"):
    ms = pd.read_csv(MS / "exemplar_set_embeddings" / "valid129" / f"things_{m}_exemplar_avg_raw_within_valid129.csv").set_index("category").filter(like="dim_")
    sh = [c for c in ms.index if c in TH[m].index]
    r = stats.spearmanr(lower(rdm(ms.loc[sh].values)), lower(rdm(TH[m].loc[sh].values))).statistic
    print(f"check {m}: manuscript THINGS vs our THINGS RDM on {len(sh)} of the 129: Spearman {r:.3f}")
rows, bootrows = [], []
for m in ("clip", "dinov3"):
    for p in ("all", "real"):
        B = pooled(m, p, 30)
        shared = [c for c in B.index if c in THW[m].index]
        Db = rdm(B.loc[shared].values); Dt = rdm(THW[m].loc[shared].values)
        bt = boot_delta({"bv": Db, "things": Dt}, shared, n_boot=1000, seed=3)
        ba = boot_auc({"bv": Db, "things": Dt}, shared, n_boot=1000, seed=4)
        diff = bt["bv"] - bt["things"]; da = ba["bv"] - ba["things"]
        rows.append({"model": m, "formats": p, "n_categories": len(shared),
                     "rdm_spearman": stats.spearmanr(lower(Db), lower(Dt)).statistic,
                     "rdm_pearson": stats.pearsonr(lower(Db), lower(Dt)).statistic,
                     "bv_delta": domain_stats(Db, shared)["delta"], "things_delta": domain_stats(Dt, shared)["delta"],
                     "bv_auc": domain_stats(Db, shared)["auc"], "things_auc": domain_stats(Dt, shared)["auc"],
                     "bv_minus_things": domain_stats(Db, shared)["delta"] - domain_stats(Dt, shared)["delta"],
                     "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5),
                     "p_boot": 2 * min((diff <= 0).mean(), (diff >= 0).mean()),
                     "auc_bv_minus_things": domain_stats(Db, shared)["auc"] - domain_stats(Dt, shared)["auc"],
                     "auc_ci_low": np.percentile(da, 2.5), "auc_ci_high": np.percentile(da, 97.5),
                     "auc_p_boot": 2 * min((da <= 0).mean(), (da >= 0).mean())})
        if (m, p) == ("clip", "all"):
            SH_CLIP = (shared, Db, Dt)
bt = pd.DataFrame(rows); bt.to_csv(RES / "open_bv_vs_things.csv", index=False)
display(bt.round(3))
shared, Db, Dt = SH_CLIP
fig, axs = plt.subplots(1, 2, figsize=(16, 7.6))
order = sorted(range(len(shared)), key=lambda k: (ORDER.index(DOM.get(shared[k], "other")) if DOM.get(shared[k]) in ORDER else 99, shared[k]))
csd = [shared[k] for k in order]; idx = cluster_within(csd, Db[np.ix_(order, order)], DOM)
fin = [order[k] for k in idx]
for ax, D, t in ((axs[0], Db, "BabyView (Qwen boxes)"), (axs[1], Dt, "THINGS")):
    im = ax.imshow(D[np.ix_(fin, fin)], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    ds = [DOM.get(shared[k], "other") for k in fin]; starts = [0] + [k for k in range(1, len(ds)) if ds[k] != ds[k - 1]]
    for b in starts[1:]:
        ax.axhline(b - .5, color="white", lw=.7); ax.axvline(b - .5, color="white", lw=.7)
    ends = starts[1:] + [len(ds)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [ds[a].replace("_", " ") for a in starts], fontsize=9)
    ax.set_xticks([]); ax.set_title(f"{t}, CLIP, {len(shared)} shared categories", loc="left", fontsize=11)
fig.colorbar(im, ax=list(axs), shrink=.7, label="cosine distance")
fig.savefig(FIG / "open_bv_vs_things_rdm_clip.png", dpi=200, bbox_inches="tight"); plt.show()"""),

md("""## 6. Category-wise BabyView-THINGS similarity
Per shared category (WordNet mapping): cosine similarity between its BabyView and THINGS means, both z-scored within the shared set
(manuscript notebook 02's measure). Which categories look most and least alike in children's views and in THINGS."""),
code("""rows = []
for m in ("clip", "dinov3"):
    B = pooled(m, "all", 30)
    shared = [c for c in B.index if c in THW[m].index]
    Zb = zrows(B.loc[shared].values); Zt = zrows(THW[m].loc[shared].values)
    cos = (Zb * Zt).sum(1) / (np.linalg.norm(Zb, axis=1) * np.linalg.norm(Zt, axis=1))
    rows.append(pd.DataFrame({"category": shared, "model": m, "cosine": cos, "domain": [DOM.get(c) for c in shared],
                              "boxes": cats.boxes.reindex(shared).values, "real_share": cats.real_share.reindex(shared).values}))
cw = pd.concat(rows); cw.to_csv(RES / "open_categorywise_bv_things.csv", index=False)
w = cw.pivot(index="category", columns="model", values="cosine")
print(f"CLIP vs DINOv3 category-wise similarity: Spearman {stats.spearmanr(w["clip"], w["dinov3"]).statistic:.2f}")
c = cw[cw.model == "clip"]
display(c.groupby("domain").cosine.agg(["size", "median", "mean"]).reindex([d for d in ORDER if d in set(c.domain)]).round(3))
print("most alike (CLIP):", ", ".join(c.sort_values("cosine", ascending=False).category.head(20)))
print("least alike (CLIP):", ", ".join(c.sort_values("cosine").category.head(20)))
r = stats.spearmanr(c.cosine, c.real_share).statistic
print(f"similarity vs share of boxes that are real objects: Spearman {r:.2f}")"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "open_category_analysis.ipynb")
print("wrote", HERE / "open_category_analysis.ipynb")
