#!/usr/bin/env python3
"""Write cooccurrence_analysis.ipynb: which objects appear together in the child's view (BabyView) and the adult's
(VEDB), how similar the two co-occurrence structures are, which pairs differ, and whether children's structure
approaches adults' with age.

Input: build_cooccurrence_data.py (out/private/cooccurrence/: frame x category presence, both datasets, Qwen Tier 1
categories); the embedding sample of build_bv_vedb_data.py for section 7.
Measure: normalized PMI over frames (-1 never together, 0 independent, 1 always together), which, unlike Jaccard, does
not rank pairs by how common the two categories are. Agreement between datasets uses the split-half design of
bv_vedb_analysis.ipynb (random halves of children / sessions), with the four halves also matched in frame count.
Sections: 1 measure · 2 what goes with what · 3 domain structure and scene clusters · 4 BabyView vs VEDB agreement
(by age, home scenes, real objects only) · 5 pairs that differ · 6 per child: closer to other children or to adults? ·
7 co-occurrence vs visual similarity · 8 time scale (same frame vs 30 s / 2 min windows).

  python build_cooccurrence_nb.py
  jupyter nbconvert --to notebook --execute cooccurrence_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \\
      --ExecutePreprocessor.timeout=14400 --output-dir out/private --output cooccurrence_analysis_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Object co-occurrence: the child's view (BabyView) and the adult's (VEDB)

Both datasets went through the same Qwen3.8-27B pipeline (every 5th 1 fps frame, open labels mapped to basic-level
categories, Tier 1 list). A category is *present* in a frame if Qwen boxed it at least once. Categories: those in
>= 0.1% of frames in both datasets. Units: BabyView children (four age bins) and VEDB sessions.

**Measure.** Normalized PMI over frames, NPMI = log[p(a,b) / p(a)p(b)] / -log p(a,b): 1 = always together, 0 =
independent, -1 = never together. Jaccard (used in the subject and children notebooks) mostly ranks pairs by how
common they are, which differs a lot between the datasets (hands are in most BabyView frames). Pairs are compared only
when their expected count under independence, n·p(a)p(b), is >= 5 in every sample compared; this does not depend
on the observed co-occurrence, so it does not favour positive associations.

**Agreement** (sections 4, 8) uses the split-half design of `bv_vedb_analysis.ipynb`: children and sessions are split
into random halves; BabyView half vs VEDB half is the comparison, half vs half within each dataset the reliability,
and the four halves are subsampled to the same number of frames. *Corrected* agreement = cross-dataset rho /
sqrt(within-BabyView rho x within-VEDB rho): 1 means the two datasets agree as well as their own halves do.

Caveats: VEDB sessions span many settings (12% home-scene frames vs 42% in BabyView), so every comparison is repeated on
home-scene frames; Qwen labels slightly more objects per VEDB frame (higher resolution); VEDB units are sessions, not
people."""),
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
HERE = Path.cwd()
sys.path.insert(0, str(HERE)); import variant as V   # QOV_LABELS=renamed (environment): parallel folders
print("labels:", V.LABELS)
D = V.out_dir(HERE / "out" / "private" / "cooccurrence")
BVV = V.out_dir(HERE / "out" / "private" / "bv_vedb")
FIG = V.out_dir(HERE / "figures" / "cooccurrence"); FIG.mkdir(parents=True, exist_ok=True)
RES = V.out_dir(HERE / "results" / "cooccurrence"); RES.mkdir(parents=True, exist_ok=True)
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
DOMAINS = ["people", "body_parts", "animals", "toys", "food_drink", "clothing", "household", "furniture_rooms",
           "vehicles", "outside", "other"]
BINS = ["<12 mo", "12-18 mo", "18-24 mo", "24+ mo"]
GROUPS = BINS + ["all ages"]
import fig_style as FS; FS.apply()                              # manuscript style: Lato, bold labels and legends
C_BV, C_VEDB, C_BASE, INK = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"], FS.BASELINE, FS.INK
MIN_PREV, MIN_EXP = .001, 5
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")

fr = pd.read_csv(D / "frames.csv.gz", dtype={"unit": str, "video_id": str})
X = sp.load_npz(D / "presence.npz").tocsr()
Xd = sp.load_npz(D / "depicted.npz").tocsr()
CATS = np.array((D / "categories.txt").read_text().splitlines())
keep = ((fr.dataset == "vedb") | fr.age_bin.isin(BINS)).values
fr = fr[keep].reset_index(drop=True); X = X[keep]; Xd = Xd[keep]
ISBV = (fr.dataset == "bv").values; HOME = fr.home.values.astype(bool)
prev_bv = np.asarray(X[ISBV].mean(0)).ravel(); prev_v = np.asarray(X[~ISBV].mean(0)).ravel()
A = np.flatnonzero((prev_bv >= MIN_PREV) & (prev_v >= MIN_PREV))
CC = CATS[A]; K = len(CC); CI = {c: i for i, c in enumerate(CC)}
XA = X[:, A].astype(np.float32).tocsr()                          # presence, any format
XR = (X - Xd)[:, A].astype(np.float32).tocsr(); XR.eliminate_zeros()   # present as a real object
UCODE, UNITS = pd.factorize(fr.unit)
IL, JL = np.tril_indices(K, -1)
SEM = np.array([DOM.get(c, "other") for c in CC])
print(f"{K} categories in >= {MIN_PREV:.1%} of frames in both datasets (of {len(CATS)} Tier 1)")
display(fr.groupby(["dataset", "age_bin"]).agg(units=("unit", "nunique"), frames=("unit", "size"), home_share=("home", "mean")).round(3))
display(pd.Series(SEM).value_counts().to_frame("categories").T)"""),
code("""def counts(M, rows=None):
    S = M if rows is None else M[rows]
    return (S.T @ S).toarray().astype(np.float64), S.shape[0]

def npmi(C, n):
    \"\"\"NPMI and expected co-count from a co-count matrix C (diagonal = frames with the category) over n frames.\"\"\"
    N = np.diag(C).copy(); NN = np.outer(N, N); E = NN / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = C / n
        P = np.log(C * n / NN) / -np.log(p)
    P[C == 0] = -1; P[p >= 1] = 1; P[NN == 0] = np.nan
    np.fill_diagonal(P, 1)
    return P, E

def lower(M): return M[IL, JL]
def rho(a, b): return stats.spearmanr(a, b).statistic
def halves(u, rng):
    u = np.array(u); rng.shuffle(u); h = len(u) // 2
    return u[:h], u[h:2 * h]
def group_mask(meta, g):
    return (meta.dataset == "vedb").values if g == "VEDB" else \\
           ((meta.dataset == "bv") & ((meta.age_bin == g) | (g == "all ages"))).values

# pooled co-occurrence, all frames and home-scene frames
POOL = {}
for scope, m in (("all frames", np.ones(len(fr), bool)), ("home-scene frames", HOME)):
    for ds, dm in (("bv", ISBV), ("vedb", ~ISBV)):
        C, n = counts(XA, np.flatnonzero(m & dm)); P, E = npmi(C, n)
        POOL[scope, ds] = dict(C=C, n=n, P=P, E=E)
pairs = pd.DataFrame({"a": CC[IL], "b": CC[JL], "domain_a": SEM[IL], "domain_b": SEM[JL]})
for (scope, ds), d in POOL.items():
    s = "" if scope == "all frames" else "_home"
    pairs[f"frames_both_{ds}{s}"] = lower(d["C"]); pairs[f"expected_{ds}{s}"] = lower(d["E"]).round(2)
    pairs[f"npmi_{ds}{s}"] = lower(d["P"]).round(4)
pairs.to_csv(RES / "pairs_pooled.csv.gz", index=False)
print({k: v["n"] for k, v in POOL.items()})"""),

md("""## 1. Why NPMI rather than Jaccard
Top pairs by Jaccard are pairs of very common categories; NPMI asks whether two categories appear together more
often than their frequencies predict. Below: the same BabyView pairs ranked both ways, and how strongly each
measure tracks the frequency of the rarer category."""),
code("""d = POOL["all frames", "bv"]; C, N = d["C"], np.diag(d["C"])
jac = lower(C) / (N[IL] + N[JL] - lower(C))
t = pairs.assign(jaccard=jac)
ok = t.expected_bv >= MIN_EXP
print("BabyView, top 12 pairs by Jaccard:"); display(t[ok].nlargest(12, "jaccard")[["a", "b", "frames_both_bv", "jaccard", "npmi_bv"]].round(3))
print("BabyView, top 12 pairs by NPMI (>= 50 frames together):")
display(t[ok & (t.frames_both_bv >= 50)].nlargest(12, "npmi_bv")[["a", "b", "frames_both_bv", "jaccard", "npmi_bv"]].round(3))
minf = np.minimum(N[IL], N[JL])[ok]
print(f"Spearman with the rarer category's frame count: Jaccard {rho(jac[ok], minf):.2f}, NPMI {rho(t.npmi_bv[ok], minf):.2f}")"""),

md("""## 2. What goes with what
Strongest associations in each dataset (pairs together in >= 50 frames), and the top associates of a few
categories side by side."""),
code("""TOP = []
for ds, name in (("bv", "BabyView"), ("vedb", "VEDB")):
    t = pairs[(pairs[f"expected_{ds}"] >= MIN_EXP) & (pairs[f"frames_both_{ds}"] >= 50)].nlargest(30, f"npmi_{ds}")
    TOP.append(t.assign(pair=t.a + " + " + t.b)[["pair", f"npmi_{ds}"]].reset_index(drop=True).rename(
        columns={"pair": f"{name} pair", f"npmi_{ds}": f"{name} NPMI"}))
display(pd.concat(TOP, axis=1).round(3))
SHOW = [c for c in ["cup", "spoon", "bottle", "ball", "book", "dog", "car", "shoe", "television", "laptop", "toothbrush",
                    "banana", "pillow", "tree", "stroller", "teddy bear", "phone", "keys"] if c in CI]
def associates(P, C, c, k=8, min_both=20):
    i = CI[c]; s = pd.Series(P[i], index=CC).drop(c); s = s[C[i][[CI[x] for x in s.index]] >= min_both]
    return ", ".join(f"{x} {v:.2f}" for x, v in s.nlargest(k).items())
display(pd.DataFrame({ds: {c: associates(POOL["all frames", ds]["P"], POOL["all frames", ds]["C"], c) for c in SHOW}
                      for ds in ("bv", "vedb")}).rename(columns={"bv": "BabyView", "vedb": "VEDB"}).style.set_properties(
                      **{"text-align": "left", "white-space": "normal"}))"""),

md("""## 3. Domain structure and scene clusters
Do objects of one domain (food, toys, clothing, ...) appear together? Mean NPMI within vs between domains, and
AUC = P(a within-domain pair has higher NPMI than a between-domain pair); pairs with expected count >= 5.
Then the NPMI matrices ordered by domain, and data-driven *scene clusters*: average-linkage clustering of
1 - NPMI cut into 25 clusters, compared between datasets (adjusted Rand index, ARI) and with the domain labels."""),
code("""def domain_stats(P, E):
    ok = lower(E) >= MIN_EXP; p = lower(P)[ok]; same = (SEM[IL] == SEM[JL])[ok]
    w, b = p[same], p[~same]
    return {"pairs": ok.sum(), "within": w.mean(), "between": b.mean(), "delta": w.mean() - b.mean(),
            "auc": stats.mannwhitneyu(w, b).statistic / (len(w) * len(b))}
dst = pd.DataFrame({f"{ds}, {scope}": domain_stats(POOL[scope, ds]["P"], POOL[scope, ds]["E"])
                    for scope in ("all frames", "home-scene frames") for ds in ("bv", "vedb")}).T
dst.to_csv(RES / "domain_clustering.csv"); display(dst.round(3))
rows = []
for ds in ("bv", "vedb"):
    P, E = POOL["all frames", ds]["P"], POOL["all frames", ds]["E"]
    for d in DOMAINS:
        k = SEM == d
        if k.sum() < 3: continue
        sub = P[np.ix_(k, k)][np.tril_indices(k.sum(), -1)]; ok = E[np.ix_(k, k)][np.tril_indices(k.sum(), -1)] >= MIN_EXP
        rows.append(dict(dataset=ds, domain=d, categories=k.sum(), within_npmi=np.nanmean(sub[ok])))
print("Mean within-domain NPMI, by domain:")
display(pd.DataFrame(rows).pivot(index="domain", columns="dataset", values="within_npmi").round(3))"""),
code("""order = sorted(range(K), key=lambda i: (DOMAINS.index(SEM[i]) if SEM[i] in DOMAINS else 99, -prev_bv[A[i]]))
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
for ax, ds, name in ((axs[0], "bv", "BabyView"), (axs[1], "vedb", "VEDB")):
    M = np.where(POOL["all frames", ds]["E"] >= MIN_EXP, POOL["all frames", ds]["P"], np.nan)[np.ix_(order, order)]
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#dddddd")
    im = ax.imshow(M, cmap=cm, vmin=-.4, vmax=.4, interpolation="nearest"); ax.grid(False)
    sem_o = SEM[order]; edges = [0] + [i for i in range(1, K) if sem_o[i] != sem_o[i - 1]] + [K]
    for e in edges[1:-1]:
        ax.axhline(e - .5, color=INK, lw=.4); ax.axvline(e - .5, color=INK, lw=.4)
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    labs = [sem_o[a].replace("_", " ") for a in edges[:-1]]
    ax.set_xticks(mids, labs, rotation=90, fontsize=7); ax.set_yticks(mids, labs, fontsize=7)
    ax.set_title(f"{name}: NPMI over frames, {K} categories", loc="left", fontsize=11, color=INK)
fig.colorbar(im, ax=axs, shrink=.7, label="NPMI (red = together more than chance; grey = expected count < 5)")
save(fig, "npmi_matrices"); plt.show()"""),
code("""def clusters(P, k=25):
    Dm = (1 - np.nan_to_num(P, nan=0)) / 2; np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform(Dm, checks=False), "average"), k, "maxclust")
CL = {ds: clusters(POOL["all frames", ds]["P"]) for ds in ("bv", "vedb")}
# split-half baseline: two halves of BabyView children, two halves of VEDB sessions
rng = np.random.default_rng(1); ari = []
for _ in range(10):
    lab = {}
    for ds, dm in (("bv", ISBV), ("vedb", ~ISBV)):
        h1, h2 = halves(np.unique(UCODE[dm]), rng)
        lab[ds] = [clusters(npmi(*counts(XA, np.flatnonzero(np.isin(UCODE, h))))[0]) for h in (h1, h2)]
    ari.append((adjusted_rand_score(lab["bv"][0], lab["vedb"][0]), adjusted_rand_score(*lab["bv"]), adjusted_rand_score(*lab["vedb"])))
ari = np.array(ari).mean(0)
print(f"ARI, pooled BabyView vs pooled VEDB clusters: {adjusted_rand_score(CL['bv'], CL['vedb']):.2f}")
print(f"ARI, half vs half (10 draws): BabyView vs VEDB {ari[0]:.2f}, within BabyView {ari[1]:.2f}, within VEDB {ari[2]:.2f}")
print(f"ARI with domain labels: BabyView {adjusted_rand_score(SEM, CL['bv']):.2f}, VEDB {adjusted_rand_score(SEM, CL['vedb']):.2f}")
rows = []
for ds, name in (("bv", "BabyView"), ("vedb", "VEDB")):
    prev = prev_bv[A] if ds == "bv" else prev_v[A]
    for c in np.unique(CL[ds]):
        k = np.flatnonzero(CL[ds] == c)
        if len(k) < 3: continue
        k = k[np.argsort(-prev[k])]
        rows.append(dict(dataset=name, size=len(k), members=", ".join(CC[k[:14]]) + (" ..." if len(k) > 14 else "")))
cl = pd.DataFrame(rows).sort_values(["dataset", "size"], ascending=[True, False])
pd.DataFrame({"category": CC, "domain": SEM, "cluster_bv": CL["bv"], "cluster_vedb": CL["vedb"]}).to_csv(RES / "scene_clusters.csv", index=False)
display(cl.style.hide(axis="index").set_properties(**{"text-align": "left", "white-space": "normal"}))"""),

md("""## 4. How similar are the two co-occurrence structures?
Split-half agreement (50 draws), per BabyView age bin and pooled: Spearman between NPMI vectors over category pairs,
halves matched in frame count. Scopes: all frames; home-scene frames; real objects only (a category counts as present
only if Qwen called at least one of its boxes *real*, so toys and pictures do not carry the structure)."""),
code("""def split_half(M, meta, mask, groups, n_draw=50, seed=2):
    rng = np.random.default_rng(seed); ucode = pd.factorize(meta.unit)[0]; out = []
    vm = mask & group_mask(meta, "VEDB"); uv = np.unique(ucode[vm])
    for g in groups:
        bm = mask & group_mask(meta, g); ub = np.unique(ucode[bm]); r = []
        for _ in range(n_draw):
            (b1, b2), (v1, v2) = halves(ub, rng), halves(uv, rng)
            idx = [np.flatnonzero(bm & np.isin(ucode, h)) for h in (b1, b2)] + [np.flatnonzero(vm & np.isin(ucode, h)) for h in (v1, v2)]
            m = min(len(x) for x in idx)
            PE = [npmi(*counts(M, rng.choice(x, m, replace=False))) for x in idx]
            ok = np.all([lower(E) >= MIN_EXP for _, E in PE], axis=0)
            L = [lower(P)[ok] for P, _ in PE]
            cross = np.mean([rho(L[a], L[b]) for a in (0, 1) for b in (2, 3)])
            wb, wv = rho(L[0], L[1]), rho(L[2], L[3])
            r.append((cross, wb, wv, cross / np.sqrt(wb * wv), ok.sum(), m))
        r = np.array(r)
        for k, name in enumerate(("BabyView vs VEDB", "within BabyView", "within VEDB", "corrected")):
            out.append(dict(group=g, comparison=name, rho=r[:, k].mean(), lo=np.percentile(r[:, k], 2.5),
                            hi=np.percentile(r[:, k], 97.5), pairs=int(r[:, 4].mean()), frames_per_half=int(r[:, 5].mean())))
    return pd.DataFrame(out)
ALL = np.ones(len(fr), bool)
SH = pd.concat([split_half(M, fr, m, GROUPS).assign(scope=s) for s, M, m in
                (("all frames", XA, ALL), ("home-scene frames", XA, HOME), ("real objects only", XR, ALL))], ignore_index=True)
SH.to_csv(RES / "split_half_agreement.csv", index=False)
display(SH.pivot_table(index=["scope", "comparison"], columns="group", values="rho", sort=False)[GROUPS].round(3))
display(SH.drop_duplicates(["scope", "group"]).pivot(index="scope", columns="group", values="frames_per_half")[GROUPS])"""),
code("""fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True); x = np.arange(len(GROUPS))
for ax, scope in zip(axs, ("all frames", "home-scene frames", "real objects only")):
    t = SH[SH.scope == scope]
    for comp, col, ls, mk in (("BabyView vs VEDB", C_VEDB, "-", "o"), ("within BabyView", C_BV, "-", "s"),
                              ("within VEDB", C_BASE, "--", "^")):
        tt = t[t.comparison == comp].set_index("group").loc[GROUPS]
        ax.errorbar(x, tt.rho, yerr=[tt.rho - tt.lo, tt.hi - tt.rho], fmt=mk + ls, color=col, ms=6, capsize=2, lw=1.5, label=comp)
    ax.set_xticks(x, GROUPS, fontsize=8); ax.set_title(scope, loc="left", fontsize=10, color=INK)
axs[0].set_ylabel("Spearman between NPMI vectors")
fig.suptitle("Co-occurrence structure: BabyView vs VEDB, split halves matched in frames", x=.01, ha="left")
fig.tight_layout(); fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, frameon=False,
                               bbox_to_anchor=(.5, -.06)); save(fig, "split_half_agreement"); plt.show()
print("Corrected agreement (1 = as similar as each dataset's own halves):")
display(SH[SH.comparison == "corrected"].pivot(index="scope", columns="group", values="rho")[GROUPS].round(3))"""),

md("""## 5. Which pairs differ?
NPMI difference (BabyView - VEDB) per pair from the pooled data, with 95% CIs from resampling children and sessions
(500 draws). Pairs need expected count >= 10 in both datasets and >= 30 frames together in at least one. Then the
mean difference per domain x domain block."""),
code("""def unit_counts(M, mask):
    rows = np.flatnonzero(mask); u = UCODE[rows]; out = np.zeros((len(UNITS), K, K), np.float32); n = np.zeros(len(UNITS))
    for k in np.unique(u):
        r = rows[u == k]; S = M[r]; out[k] = (S.T @ S).toarray(); n[k] = len(r)
    return out, n
def boot_diff(mask, n_boot=500, seed=3):
    UC, un = unit_counts(XA, mask); flat = UC.reshape(len(UNITS), -1)
    ub = np.flatnonzero((un > 0) & np.isin(np.arange(len(UNITS)), UCODE[ISBV]))
    uv = np.flatnonzero((un > 0) & np.isin(np.arange(len(UNITS)), UCODE[~ISBV]))
    rng = np.random.default_rng(seed); diffs = np.zeros((n_boot, len(IL)), np.float32)
    for b in range(n_boot):
        P = []
        for us in (ub, uv):
            w = np.bincount(rng.choice(us, len(us)), minlength=len(UNITS)).astype(np.float32)
            P.append(lower(npmi((w @ flat).reshape(K, K).astype(np.float64), w @ un)[0]))
        diffs[b] = P[0] - P[1]
    return np.percentile(diffs, 2.5, axis=0), np.percentile(diffs, 97.5, axis=0)
DIFF = {}
for scope, m, s in (("all frames", ALL, ""), ("home-scene frames", HOME, "_home")):
    lo, hi = boot_diff(m)
    t = pairs.assign(diff=pairs[f"npmi_bv{s}"] - pairs[f"npmi_vedb{s}"], lo=lo, hi=hi)
    t = t[(t[f"expected_bv{s}"] >= 10) & (t[f"expected_vedb{s}"] >= 10) &
          (np.maximum(t[f"frames_both_bv{s}"], t[f"frames_both_vedb{s}"]) >= 30)]
    DIFF[scope] = t
    t.to_csv(RES / f"pair_differences{s}.csv.gz", index=False)
    cols = ["a", "b", f"frames_both_bv{s}", f"frames_both_vedb{s}", f"npmi_bv{s}", f"npmi_vedb{s}", "diff", "lo", "hi"]
    print(f"{scope}: {len(t):,} pairs; CI excludes 0 for {((t.lo > 0) | (t.hi < 0)).mean():.0%}")
    # NPMI jumps to -1 at zero co-count, so ranked lists use pairs seen together in >= 10 frames in both datasets
    both = t[(t[f"frames_both_bv{s}"] >= 10) & (t[f"frames_both_vedb{s}"] >= 10)]
    print("  more together for children (BabyView > VEDB):"); display(both[both.lo > 0].nlargest(20, "diff")[cols].round(3))
    print("  more together for adults (VEDB > BabyView):"); display(both[both.hi < 0].nsmallest(20, "diff")[cols].round(3))
    for ds, other in (("vedb", "bv"), ("bv", "vedb")):
        z = t[(t[f"frames_both_{ds}{s}"] == 0) & (t[f"frames_both_{other}{s}"] >= 30)].nlargest(10, f"expected_{ds}{s}")
        print(f"  never together in {'VEDB' if ds == 'vedb' else 'BabyView'} despite the highest expected count:")
        display(z[["a", "b", f"expected_{ds}{s}", f"frames_both_{other}{s}", f"npmi_{other}{s}"]].round(3))"""),
code("""fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, scope in zip(axs, ("all frames", "home-scene frames")):
    t = DIFF[scope]
    dd = [d for d in DOMAINS if d in set(t.domain_a) | set(t.domain_b)]
    B = np.full((len(dd), len(dd)), np.nan)
    for i, a in enumerate(dd):
        for j, b in enumerate(dd):
            s = t[((t.domain_a == a) & (t.domain_b == b)) | ((t.domain_a == b) & (t.domain_b == a))]
            if len(s) >= 5: B[i, j] = s["diff"].mean()
    v = np.nanmax(np.abs(B))
    im = ax.imshow(B, cmap="RdBu_r", vmin=-v, vmax=v); ax.grid(False)
    ax.set_xticks(range(len(dd)), [d.replace("_", " ") for d in dd], rotation=90, fontsize=8)
    ax.set_yticks(range(len(dd)), [d.replace("_", " ") for d in dd], fontsize=8)
    ax.set_title(f"{scope}: mean NPMI difference", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, shrink=.75, label="BabyView - VEDB (red = more together for children)")
fig.tight_layout(); save(fig, "domain_block_difference"); plt.show()"""),

md("""## 6. Per child: closer to other children or to adults?
For each child x age bin with >= 5,000 frames: NPMI from the child's frames vs (a) the other children, same age bin,
(b) the other children, all ages, (c) VEDB, each subsampled to the child's frame count (capped at 30,000; 10 draws;
pairs with expected count >= 5 in all four). If a child's co-occurrence structure is closer to other children's than to
adults', (a) and (b) exceed (c); if children approach adults with age, the gap shrinks across bins."""),
code("""rng = np.random.default_rng(5); rows = []
fbv = fr[ISBV]
for (u, b), g in fbv.groupby(["unit", "age_bin"]):
    if len(g) < 5000: continue
    m = min(len(g), 30000)
    pools = {"other children, same age": np.flatnonzero(ISBV & (fr.unit != u).values & (fr.age_bin == b).values),
             "other children, all ages": np.flatnonzero(ISBV & (fr.unit != u).values),
             "VEDB": np.flatnonzero(~ISBV)}
    r = []
    for _ in range(10):
        PE = [npmi(*counts(XA, rng.choice(g.index.values, m, replace=False)))] + \\
             [npmi(*counts(XA, rng.choice(p, m, replace=False))) for p in pools.values()]
        ok = np.all([lower(E) >= MIN_EXP for _, E in PE], axis=0)
        r.append([rho(lower(PE[0][0])[ok], lower(P)[ok]) for P, _ in PE[1:]])
    rows.append(dict(unit=u, age_bin=b, age_months=g.age_months.mean(), frames=len(g), **dict(zip(pools, np.mean(r, 0)))))
PC = pd.DataFrame(rows); PC.to_csv(D / "per_child_agreement.csv", index=False)     # private: subject ids
PC["children minus VEDB"] = PC["other children, same age"] - PC["VEDB"]
s = PC.groupby("age_bin")[["other children, same age", "other children, all ages", "VEDB", "children minus VEDB"]].agg(["mean", "count"])
s = s.loc[[b for b in BINS if b in s.index]]
s.to_csv(RES / "per_child_agreement_by_age.csv"); display(s.round(3))
w = stats.wilcoxon(PC["other children, same age"], PC["VEDB"])
print(f"{len(PC)} child x age-bin units; closer to other children than to VEDB in {(PC['children minus VEDB'] > 0).mean():.0%} "
      f"(Wilcoxon p = {w.pvalue:.2g})")
print(f"Spearman, age vs agreement with VEDB: {rho(PC.age_months, PC.VEDB):.2f}; age vs (children - VEDB): {rho(PC.age_months, PC['children minus VEDB']):.2f}")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axs[0]
ax.scatter(PC.age_months, PC["other children, same age"], color=C_BV, s=28, label="vs other children, same age", edgecolor="white", lw=.5)
ax.scatter(PC.age_months, PC.VEDB, color=C_VEDB, s=28, marker="s", label="vs VEDB adults", edgecolor="white", lw=.5)
ax.set_xlabel("age (months)"); ax.set_ylabel("Spearman between NPMI vectors"); ax.legend(fontsize=8, frameon=False)
ax.set_title("each child x age bin, frame-matched", loc="left", fontsize=10, color=INK)
ax = axs[1]
ax.scatter(PC.age_months, PC["children minus VEDB"], color=INK, s=28, edgecolor="white", lw=.5); ax.axhline(0, color=C_BASE, lw=1, ls=":")
ax.set_xlabel("age (months)"); ax.set_ylabel("rho (other children) - rho (VEDB)")
ax.set_title("how much closer to other children than to adults", loc="left", fontsize=10, color=INK)
fig.tight_layout(); save(fig, "per_child_agreement"); plt.show()"""),

md("""## 7. Co-occurrence vs visual similarity
Do objects that appear together also look alike? Category means of CLIP / DINOv3 crop vectors from the embedding
sample of `build_bv_vedb_data.py` (L2-normalized, averaged per category within a dataset, z-scored across categories,
cosine similarity; categories with >= 30 exemplars), against NPMI in the same dataset. All pairs, and within-domain
pairs only (which removes the shared domain effect)."""),
code("""rows = []
for m in ("clip", "dinov3"):
    s = dict(np.load(BVV / f"sample_{m}.npz", allow_pickle=True))
    E0 = s["emb"].astype(np.float32); E0 /= np.maximum(np.linalg.norm(E0, axis=1, keepdims=True), 1e-12)
    for ds in ("bv", "vedb"):
        sel = s["dataset"] == ds; cat = s["category"][sel]; E = E0[sel]
        cnt = pd.Series(cat).value_counts()
        cats = [c for c in CC if cnt.get(c, 0) >= 30]
        Mn = np.stack([E[cat == c].mean(0) for c in cats]); Mn = (Mn - Mn.mean(0)) / (Mn.std(0) + 1e-10)
        Sim = 1 - squareform(pdist(Mn, "cosine"))
        k = np.array([CI[c] for c in cats]); i, j = np.tril_indices(len(k), -1)
        P, Ex = POOL["all frames", ds]["P"], POOL["all frames", ds]["E"]
        npm, ok = P[k[i], k[j]], Ex[k[i], k[j]] >= MIN_EXP
        same = SEM[k[i]] == SEM[k[j]]
        rows.append(dict(model=m, dataset=ds, categories=len(cats), rho_all=rho(Sim[i, j][ok], npm[ok]),
                         rho_within_domain=rho(Sim[i, j][ok & same], npm[ok & same]), pairs=ok.sum()))
VS = pd.DataFrame(rows); VS.to_csv(RES / "cooccurrence_vs_visual.csv", index=False); display(VS.round(3))"""),

md("""## 8. Time scale: same frame vs nearby frames
Objects that are not in the same frame can still be in the same scene. Presence is pooled over windows of 30 s (6
sampled frames) and 2 min (24) within a video, and the split-half comparison (all ages) is repeated on windows.
If the BabyView-VEDB difference is about the narrow field of view of a child's camera, it should shrink at longer
windows; if it is about which places and activities children and adults are in, it should not."""),
code("""rows = []
for w, name in ((5, "same frame"), (30, "30 s window"), (120, "2 min window")):
    key = fr.video_id + "_" + (fr.frame_number // w).astype(str)
    wc, wk = pd.factorize(key)
    G = sp.csr_matrix((np.ones(len(fr), np.float32), (wc, np.arange(len(fr)))), shape=(len(wk), len(fr)))
    MW = (G @ XA).tocsr(); MW.data[:] = 1
    meta = fr.groupby(wc).agg(dataset=("dataset", "first"), unit=("unit", "first"), age_bin=("age_bin", "first"),
                              home=("home", "any")).reset_index(drop=True)
    t = split_half(MW, meta, np.ones(len(meta), bool), ["all ages"], n_draw=30)
    rows.append(t.assign(window=name, windows_bv=int((meta.dataset == "bv").sum()), windows_vedb=int((meta.dataset == "vedb").sum())))
TW = pd.concat(rows, ignore_index=True); TW.to_csv(RES / "time_scale_agreement.csv", index=False)
display(TW.pivot(index="window", columns="comparison", values="rho").loc[["same frame", "30 s window", "2 min window"]].round(3))"""),

md("""## Summary"""),
code("""c = SH[SH.comparison == "corrected"].set_index(["scope", "group"]).rho
print(f"Categories: {K}. Pairs compared (all frames, all ages): {SH[(SH.scope == 'all frames') & (SH.group == 'all ages')].pairs.iloc[0]:,}")
print(f"BabyView vs VEDB, corrected agreement (all ages): all frames {c['all frames', 'all ages']:.2f}, "
      f"home scenes {c['home-scene frames', 'all ages']:.2f}, real objects {c['real objects only', 'all ages']:.2f}")
print("By age (all frames):", ", ".join(f"{b} {c['all frames', b]:.2f}" for b in BINS))
print("Domain AUC (within > between):", ", ".join(f"{k} {v:.2f}" for k, v in dst.auc.items()))
print(f"Per child: closer to other children than to VEDB in {(PC['children minus VEDB'] > 0).mean():.0%} of {len(PC)} child x age units")"""),

md("""## Notes
* NPMI depends on frame density: a frame with many objects contributes many pairs. VEDB frames are larger (2048 x 1536
  vs 512 x 910) and Qwen boxes more Tier 1 objects per frame there, which raises co-occurrence counts but not NPMI's
  correction for frequency; the frame-matched split halves keep sample size equal, not objects per frame.
* Home-scene frames are defined by content (a bed, couch, crib, fridge, ...), so they select scenes with furniture; they
  match setting only roughly. VEDB setting labels exist for 80 of 717 sessions.
* The per-child section counts child x age-bin units, which are not independent within a child; the Wilcoxon test is
  descriptive.
* Labels are Qwen's; set QOV_LABELS=renamed to rerun on the second-pass names (parallel output folders)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "cooccurrence_analysis.ipynb")
print("wrote", HERE / "cooccurrence_analysis.ipynb")
