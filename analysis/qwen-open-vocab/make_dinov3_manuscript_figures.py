#!/usr/bin/env python3
"""DINOv3-only versions of the revision's main-text figures that showed CLIP next to DINOv3 (Figs. 4A, 5A-C, 6, 7C).

The revision reports DINOv3 only (decided 2026-11-13). Every number is read from the result files the notebooks already
wrote (renamed labels, 183-category core); nothing is recomputed. BH corrections in those files are per model, so the
DINOv3 stars are unchanged by dropping CLIP.

  python make_dinov3_manuscript_figures.py            # writes figures/manuscript_dinov3/ and copies into the draft
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig_style as FS

FS.apply()
HERE = Path(__file__).resolve().parent
import os
os.environ.setdefault("QOV_LABELS", "renamed")
import variant as V                     # QOV_DOMAINS=things27: THINGS-27 results, figures to *_things27 folders
XD = V.out_dir(HERE / "results" / "crossdataset"); XD = XD.with_name(XD.name + "_core")
BVV = V.out_dir(HERE / "results" / "bv_vedb"); BVV = BVV.with_name(BVV.name + "_core")
OPEN = HERE / "results" / ("open" + V.SUFFIX)
OUT = HERE / "figures" / ("manuscript_dinov3" + ("" if V.DOMAINS == "cdi" else "_" + V.DOMAINS)); OUT.mkdir(parents=True, exist_ok=True)
DRAFT = V.manuscript_figures_dir(HERE.parents[0] / "manuscript-2026" / "revision_private" / "manuscript_revision" / "figures")
DRAFT.mkdir(exist_ok=True)
M = "dinov3"
LABEL = FS.DS_LABEL
ZERO = dict(color="#8a8984", lw=1, ls="--")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    if DRAFT.exists():
        for ext in ("png", "pdf"):
            shutil.copy(OUT / f"{name}.{ext}", DRAFT / f"{name}.{ext}")
    print("wrote", name)


# ---- Fig. 4A: dataset x dataset RDM agreement ------------------------------------------------------------------------
pr = pd.read_csv(XD / "pairwise_rdm_correlations.csv"); pr = pr[pr.model == M]
DS = ["bv", "bv_real", "vedb", "vedb_real", "things", "lvis", "imagenet"]
mat = pd.DataFrame(np.eye(len(DS)), index=DS, columns=DS)
for r in pr.itertuples():
    if r.a in DS and r.b in DS:
        mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.spearman
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
ax.set_xticks(range(len(DS)), [LABEL[d] for d in DS], rotation=40, ha="right")
ax.set_yticks(range(len(DS)), [LABEL[d] for d in DS])
for i in range(len(DS)):
    for j in range(len(DS)):
        if i != j:
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > .6 else FS.INK)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman ρ between RDMs")
ax.set_title("RDM agreement between datasets (DINOv3)", loc="left", fontsize=11)
fig.tight_layout(); save(fig, "fig4a_rdm_agreement")

# ---- Fig. 5A: domain clustering per dataset ---------------------------------------------------------------------------
cs = pd.read_csv(XD / "cluster_strength_core.csv"); cs = cs[cs.model == M].reset_index(drop=True)
df = pd.read_csv(XD / "cluster_strength_differences_core.csv")
df = df[(df.model == M) & (df.measure == "delta_sd")].set_index("pair")
ci = cs.delta_sd_ci.map(lambda s: [float(x) for x in s.strip("[]").split(",")])
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for k, (d, v, (lo, hi)) in enumerate(zip(cs.dataset, cs.delta_sd, ci)):
    ax.errorbar([k], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=FS.DS_COLOR[d], ms=8, mec="white", capsize=3, lw=1.6)
    if d != "bv":
        q = df.loc[f"bv - {d}", "q_bh"]
        ax.text(k, hi + .02, FS.stars(q), ha="center", va="bottom", fontsize=9, fontweight="bold" if q < .05 else "normal")
ax.set_xticks(range(len(cs)), [LABEL[d] for d in cs.dataset], rotation=30, ha="right")
ax.axhline(0, **ZERO); ax.set_ylim(bottom=-.05)
ax.set_ylabel("domain clustering (Δ / SD)")
ax.set_title(f"Domain clustering, {int(cs.n.iloc[0])} shared categories (DINOv3)", loc="left", fontsize=11)
fig.tight_layout(); save(fig, "fig5a_cluster_strength")


# ---- Fig. 5B / 5C: by domain ------------------------------------------------------------------------------------------
def by_domain(dd, sig, show, star_pair, title, name, n_col="n_categories"):
    dd = dd[(dd.model == M) & (dd.domain != "ALL")]
    doms = list(dict.fromkeys(dd.domain))
    n = dd.drop_duplicates("domain").set_index("domain")[n_col].astype(int)
    sig = sig[(sig.model == M) & (sig.measure == "delta_sd")]
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(doms)), 4.2))
    w = .8 / len(show)
    for j, d in enumerate(show):
        s = dd[dd.dataset == d].set_index("domain").reindex(doms)
        x = np.arange(len(doms)) - .4 + w * (j + .5)
        col = FS.DS_COLOR[d]
        ax.errorbar(x, s.delta_sd, yerr=[s.delta_sd - s.ci_low_sd, s.ci_high_sd - s.delta_sd], fmt="o", color=col, ms=6,
                    mfc="white" if (d == "bv" and "bv_real" in show) else col, mec=col if "bv_real" in show else "white",
                    capsize=2, lw=1.2, label=LABEL[d] + (" (all formats)" if d == "bv" and "bv_real" in show else ""))
        pair = star_pair(d)
        if pair:
            st = sig[sig.pair == pair].set_index("domain").reindex(doms).stars.fillna("")
            for xi, hi, t in zip(x, s.ci_high_sd, st):
                if t:
                    ax.text(xi, hi + .05, t, ha="center", va="bottom", fontsize=8, color=FS.INK)
    ax.axhline(0, **ZERO); ax.set_ylabel("Δ / SD")
    ax.set_ylim(top=ax.get_ylim()[1] + .5)
    ax.legend(ncol=len(show), frameon=False, loc="upper right", fontsize=8.5)
    ax.set_xticks(range(len(doms)), [f"{FS.DOMAIN_LABEL.get(d, d)}\n({n[d]})" for d in doms], fontsize=9)
    FS.color_domain_ticks(ax.get_xticklabels(), doms)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout(); save(fig, name)


by_domain(pd.read_csv(XD / "cluster_strength_by_domain_core.csv"),
          pd.read_csv(XD / "cluster_strength_by_domain_differences_core.csv"),
          ["bv", "vedb", "things", "lvis", "imagenet"], lambda d: None if d == "bv" else f"bv - {d}",
          "Domain clustering by domain, 183 shared categories (DINOv3; n per domain in brackets)",
          "fig5b_cluster_by_domain")
by_domain(pd.read_csv(XD / "cluster_strength_by_domain_real.csv"),
          pd.read_csv(XD / "cluster_strength_by_domain_real_differences.csv"),
          ["bv", "bv_real", "vedb_real", "things", "lvis", "imagenet"],
          lambda d: None if d == "bv_real" else ("bv - bv_real" if d == "bv" else f"bv_real - {d}"),
          "Domain clustering with real objects only, 164 categories (DINOv3; open circles: BabyView all formats)",
          "fig5c_cluster_by_domain_real")

# ---- Fig. 6: individual children --------------------------------------------------------------------------------------
pc = pd.read_csv(OPEN / "open_per_child_clustering_things.csv"); pc = pc[pc.model == M].sort_values("frames")
pairs = pd.read_csv(OPEN / "open_individual_rdm_pairs.csv"); pairs = pairs[pairs.model == M]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
x = np.arange(len(pc))
ax.vlines(x, 0, pc.child_minus_things, color=FS.DS_COLOR["things"], lw=1.2)
ax.scatter(x, pc.child_minus_things, s=34, color=FS.DS_COLOR["bv"], edgecolor="white", zorder=3)
ax.axhline(0, **ZERO)
ax.set_xticks(x, [f"{f / 1000:.0f}k" for f in pc.frames], rotation=90, fontsize=7)
ax.set_xlabel("children, ordered by annotated frames")
ax.set_ylabel("child − THINGS domain clustering (Δ)")
ax.set_title(f"(A) Each child vs THINGS on the same categories ({(pc.child_minus_things > 0).sum()} of {len(pc)} > 0)",
             loc="left", fontsize=10)
ax = axs[1]
ax.hist(pairs.spearman, bins=np.arange(.5, .81, .0125), color=FS.DS_COLOR["bv"], alpha=.85, edgecolor="white")
med = pairs.spearman.mean()
ax.axvline(med, color=FS.INK, lw=1.6); ax.text(.03, .95, f"mean ρ = {med:.2f}\n(range {pairs.spearman.min():.2f}–{pairs.spearman.max():.2f})", transform=ax.transAxes, va="top", fontsize=9)
ax.set_xlabel("Spearman ρ between two children's RDMs"); ax.set_ylabel("pairs of children")
ax.set_title(f"(B) Agreement between children ({len(pairs)} pairs)", loc="left", fontsize=10)
fig.tight_layout(); save(fig, "fig6_children")

# ---- Fig. 7C: child vs adult divergence by age ------------------------------------------------------------------------
BINS = ["<12 mo", "12-18 mo", "18-24 mo", "24+ mo"]
C_BV, C_VEDB, C_BASE = FS.DS_COLOR["bv"], FS.DS_COLOR["vedb"], FS.BASELINE
J = pd.read_csv(BVV / "js_divergence.csv")
sc = pd.read_csv(BVV / "scene_statistics.csv")
AG = pd.read_csv(BVV / "appearance_mmd_by_age.csv"); AG = AG[AG.model == M]
RR = pd.read_csv(BVV / "rdm_split_half.csv"); RR = RR[(RR.model == M) & ~RR.real]
xb = np.arange(len(BINS))


def band(ax, t, col, ycol, label, ls="-", invert=False):
    t = t.set_index("group").loc[BINS]
    y, lo, hi = (1 - t[ycol], 1 - t.hi, 1 - t.lo) if invert else (t[ycol], t.lo, t.hi)
    ax.plot(xb, y, ls, color=col, marker="o", ms=5, lw=1.5, label=label)
    ax.fill_between(xb, lo, hi, color=col, alpha=.15, lw=0)


fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 6)
axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
       fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
for ax, level in zip(axs[:2], ("category", "domain")):
    j = J[(J.level == level) & (J.scope == "all frames")]
    band(ax, j[j.comparison == "BabyView vs VEDB"], C_VEDB, "js", "BabyView vs VEDB")
    band(ax, j[j.comparison == "within BabyView"], C_BASE, "js", "within BabyView (baseline)")
    jh = J[(J.level == level) & (J.scope == "home-scene frames") & (J.comparison == "BabyView vs VEDB")]
    band(ax, jh, C_VEDB, "js", "BabyView vs VEDB, home scenes", ls="--")
    ax.set_title(f"what is seen: JS divergence ({level})", loc="left", fontsize=10)
ax = axs[2]
s = sc[(sc.scope == "all frames") & (sc.measure == "dep_share")].set_index("group")
ax.errorbar(xb, s.loc[BINS, "mean"], yerr=[s.loc[BINS, "mean"] - s.loc[BINS, "lo"], s.loc[BINS, "hi"] - s.loc[BINS, "mean"]],
            fmt="o-", color=C_BV, capsize=2, label="BabyView")
ax.axhline(s.loc["VEDB", "mean"], color=C_VEDB, lw=1.5, label="VEDB")
ax.set_ylim(bottom=0)
ax.set_title("share of objects depicted (toy / picture / screen)", loc="left", fontsize=10)
ax = axs[3]
for r, ls in ((False, "-"), (True, "--")):
    band(ax, AG[AG.real == r].rename(columns={"median_excess": "y"}), C_VEDB, "y",
         "real objects only" if r else "all formats", ls)
ax.axhline(0, color=C_BASE, lw=1, ls=":")
ax.set_title("how things look: MMD² excess over within-dataset baseline (DINOv3)", loc="left", fontsize=10)
ax = axs[4]
band(ax, RR[RR.comparison == "BabyView vs VEDB"].rename(columns={"rho": "y"}), C_VEDB, "y", "BabyView vs VEDB", invert=True)
band(ax, RR[RR.comparison == "within BabyView"].rename(columns={"rho": "y"}), C_BASE, "y", "within BabyView (baseline)",
     invert=True)
ax.set_ylim(bottom=0)
ax.set_title("between-category structure: 1 − RDM ρ (DINOv3)", loc="left", fontsize=10)
for ax in axs:
    ax.set_xticks(xb, BINS, fontsize=8); ax.legend(fontsize=7.5, frameon=False)
fig.suptitle("Does children's visual experience approach adults' with age?", x=.01, ha="left")
fig.tight_layout(); save(fig, "fig7c_age_trends")


# =====================================================================================================================
# Response-letter figures (revision_private/figures/): DINOv3-only redraws of the controls, plus copies of the
# main-text figures above under the letter's names.
# =====================================================================================================================
RESP = V.manuscript_figures_dir(HERE.parents[0] / "manuscript-2026" / "revision_private" / "figures"); RESP.mkdir(exist_ok=True)
CTRL = V.out_dir(HERE / "results" / "controls"); CTRL = CTRL.with_name(CTRL.name + "_core")
CMP = HERE.parents[0] / "comparison-datasets" / "outputs"


def save_resp(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)   # editable text (pdf.fonttype 42)
    if RESP.exists():
        for ext in ("png", "pdf"):
            shutil.copy(OUT / f"{name}.{ext}", RESP / f"{name}.{ext}")
    print("wrote", name)


if RESP.exists():
    for src, dst in (("fig4a_rdm_agreement", "E1_rdm_matrix_open"), ("fig5b_cluster_by_domain", "E3_cluster_by_domain_open"),
                     ("fig5c_cluster_by_domain_real", "R1f_real_only_by_domain"), ("fig7c_age_trends", "R1i_age_trends")):
        for ext in ("png", "pdf"):
            shutil.copy(OUT / f"{src}.{ext}", RESP / f"{dst}.{ext}")
        print("copied", src, "->", dst)

# ---- R1a: crop-size floors and the original CLIP filter applied to LVIS (DINOv3 clustering) --------------------------
sm = pd.read_csv(CMP / "matched_controls" / "results" / "size_clip_matching_summary.csv"); sm = sm[sm.model == M]
sdf = pd.read_csv(CMP / "matched_controls" / "results" / "size_clip_matching_differences.csv"); sdf = sdf[sdf.model == M]
VAR = [("none", "unmatched"), ("ge32", "≥32 px"), ("ge64", "≥64 px"), ("clipfilter", "CLIP filter\non LVIS"),
       ("ge32_clipfilter", "≥32 px +\nCLIP filter"), ("ge64_clipfilter", "≥64 px +\nCLIP filter")]
DSA = [("bv", "BabyView"), ("vedb", "VEDB"), ("lvis", "LVIS crops"), ("things", "THINGS")]
fig, ax = plt.subplots(figsize=(10, 4.2))
for i, (v, _) in enumerate(VAR):
    for k, (d, lab) in enumerate(DSA):
        r = sm[(sm.variant == v) & (sm.dataset == d)].iloc[0]
        x = i + (k - 1.5) * .17
        ax.errorbar([x], [r.auc], yerr=[[r.auc - r.auc_ci_lo], [r.auc_ci_hi - r.auc]], fmt="o", color=FS.DS_COLOR[d], ms=7,
                    mec=FS.INK, mew=.8, capsize=0, lw=2, label=lab if i == 0 else None)
        if d != "bv":
            p = sdf[(sdf.variant == v) & (sdf.pair == (f"bv_minus_{d}" if d != "things" else "bv_minus_things"))].p_boot.iloc[0]
            t = FS.stars(p, "")
            if t:
                ax.text(x, r.auc_ci_hi + .004, t, ha="center", va="bottom", fontsize=8)
    lv = sm[(sm.variant == v) & (sm.dataset == "lvis")].iloc[0]
    if v != "none":
        ax.text(i, .62, f"LVIS {lv.change_vs_unmatched:+.3f}\np = {lv.change_p:.2f}", ha="center", fontsize=7.5, color=FS.MUTED)
ax.set_xticks(range(len(VAR)), [l for _, l in VAR], fontsize=9)
ax.set_ylim(.6, .86); ax.set_ylabel("domain clustering, AUC (DINOv3)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=4)
ax.set_title("Crop-size floors on every dataset and the original pipeline's CLIP filter on LVIS (≈104 categories; 95% CIs)\n"
             "stars: differs from BabyView (category bootstrap); grey: change in LVIS AUC vs unmatched",
             loc="left", fontsize=10)
fig.tight_layout(); save_resp(fig, "R1a_clip_filter_lvis")

# ---- R1b: subject split -----------------------------------------------------------------------------------------------
LABC = {"bv": "BabyView", "bv_real": "BabyView, real", "vedb": "VEDB", "vedb_real": "VEDB, real", "things": "THINGS"}
w = pd.read_csv(CTRL / "split_within_vs_cross.csv"); s = pd.read_csv(CTRL / "split_summary.csv")
w = w[(w.comparison == "bv_vedb_things") & (w.model == M)]; s = s[(s.comparison == "bv_vedb_things") & (s.model == M)]
sd = pd.read_csv(CTRL / "split_differences.csv")
sd = sd[(sd.comparison == "bv_vedb_things") & (sd.rdm == "cross-half") & (sd.metric == "auc") & (sd.model == M)]
fig, ax = plt.subplots(figsize=(5.6, 3.9))
dss = [d for d in LABC if d in set(w.dataset)]
for i, d in enumerate(dss):
    r = w[w.dataset == d].iloc[0]; c = s[s.dataset == d].iloc[0]
    ax.plot([i - .12, i + .12], [r.auc_within, r.auc_cross], color=FS.DS_COLOR[d], lw=1.5)
    ax.scatter([i - .12], [r.auc_within], s=40, color="white", edgecolor=FS.DS_COLOR[d], lw=1.5, zorder=3)
    ax.errorbar([i + .12], [r.auc_cross], yerr=[[r.auc_cross - c.auc_ci_lo], [c.auc_ci_hi - r.auc_cross]], fmt="o", ms=6,
                color=FS.DS_COLOR[d], capsize=2, zorder=3)
ax.axhline(.5, color="#8a8984", lw=.8, ls=":")
top = s.auc_ci_hi.max() + .015
for k, (a, b) in enumerate([(a, b) for i_, a in enumerate(dss) for b in dss[i_ + 1:]]):
    r_ = sd[sd.pair == f"{a} - {b}"]
    if len(r_):
        FS.bracket(ax, dss.index(a) + .12, dss.index(b) + .12, top + k * .025, FS.stars(r_.p.iloc[0]), h=.004)
ax.set_xticks(range(len(dss)), [LABC[d] for d in dss], rotation=20, ha="right")
ax.set_ylabel("domain clustering, AUC (DINOv3)")
ax.scatter([], [], s=40, color="white", edgecolor="#52514e", label="within half")
ax.scatter([], [], s=40, color="#52514e", label="across halves (95% CI)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title(f"Subject split ({int(s.n_categories.iloc[0])} categories): categories compared across children\n"
             "(BabyView), sessions (VEDB) or images (THINGS); brackets: across halves, bootstrap p", loc="left", fontsize=10)
fig.tight_layout(); save_resp(fig, "R1b_subject_split")

# ---- R1c: room and activity held fixed --------------------------------------------------------------------------------
r = pd.read_csv(CTRL / "room_summary.csv"); r = r[(r.part == "all") & ~r.label.str.contains(" x ") & (r.model == M)]
rd = pd.read_csv(CTRL / "room_differences.csv"); rd = rd[(rd.part == "all") & (rd.metric == "auc") & (rd.model == M)]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, kind in zip(axs, ("room", "act")):
    d = r[r.kind == kind]; labels = list(dict.fromkeys(d.label))
    for i, L in enumerate(labels):
        g = d[d.label == L].set_index("dataset")
        for j, (ds, c, mk, lab) in enumerate(((f"bv_all same {kind}", FS.DS_COLOR["bv"], "o", "BabyView, same room / activity"),
                                              ("bv_all pooled", FS.BASELINE, "s", "BabyView, pooled (same categories)"),
                                              ("things", FS.DS_COLOR["things"], "D", "THINGS"))):
            if ds not in g.index: continue
            v = g.loc[ds]
            ax.errorbar([i + (j - 1) * .2], [v.auc], yerr=[[v.auc - v.auc_ci_lo], [v.auc_ci_hi - v.auc]], fmt=mk, color=c, ms=6,
                        capsize=2, label=lab if (i == 0 and kind == "room") else None)
        ax.text(i, .535, f"n={int(g.n_categories.iloc[0])}", ha="center", fontsize=7, color="#52514e")
        top = g.auc_ci_hi.max() + .01
        for k, (other, x2) in enumerate((("bv_all pooled", i), ("things", i + .2))):
            r_ = rd[(rd.kind == kind) & (rd.label == L) & (rd.pair == f"bv_all same {kind} - {other}")]
            if len(r_):
                FS.bracket(ax, i - .2, x2, top + k * .022, FS.stars(r_.p.iloc[0]), h=.004, fontsize=7)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_title(f"by {'room' if kind == 'room' else 'activity'}", loc="left", fontsize=10)
axs[0].set_ylabel("domain clustering, AUC (DINOv3)"); axs[0].set_ylim(bottom=.53)
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3)
fig.suptitle("Room and activity held fixed: category means from one room / activity only (DINOv3; 95% CIs;\n"
             "brackets: same room / activity vs pooled and vs THINGS, * p < .05, ** < .01, *** < .001)", x=.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, .06, 1, 1)); save_resp(fig, "R1c_same_room")

# ---- R1d: background amount (LVIS crops vs whole images; ImageNet) ----------------------------------------------------
BA = CMP / "background_amount" / "results"
ls_ = pd.read_csv(BA / "lvis_whole_vs_crop_summary.csv"); ls_ = ls_[(ls_.model == M) & (ls_.category_set == "exclusive_ok")].set_index("matrix")
ld = pd.read_csv(BA / "lvis_whole_vs_crop_differences.csv")
ld = ld[(ld.model == M) & (ld.category_set == "exclusive_ok") & (ld.metric == "auc")].set_index("pair")
im = pd.read_csv(BA / "imagenet_comparison_summary.csv"); im = im[im.model == M].set_index("matrix")
idf = pd.read_csv(BA / "imagenet_comparison_differences.csv"); idf = idf[(idf.model == M) & (idf.metric == "auc")].set_index("pair")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
rows_l = [("bv", "BabyView", FS.DS_COLOR["bv"], "o", "vs whole", "bv_minus_lvis_whole"),
          ("things", "THINGS", FS.DS_COLOR["things"], "o", "vs whole", "things_minus_lvis_whole"),
          ("lvis_crop", "LVIS crops", "white", "o", None, None),
          ("lvis_whole", "LVIS whole images", FS.DS_COLOR["lvis"], "o", "vs crop", "lvis_whole_minus_lvis_crop"),
          ("lvis_whole_excl", "LVIS whole images (exclusive)", FS.DS_COLOR["lvis"], "s", "vs crop", "lvis_whole_excl_minus_lvis_crop")]
rows_i = [("bv", "BabyView", FS.DS_COLOR["bv"], "o", "vs ImageNet", "bv_minus_imagenet"),
          ("vedb", "VEDB", FS.DS_COLOR["vedb"], "o", "vs ImageNet", "vedb_minus_imagenet"),
          ("things", "THINGS", FS.DS_COLOR["things"], "o", "vs ImageNet", "imagenet_minus_things"),
          ("imagenet", "ImageNet", FS.DS_COLOR["imagenet"], "o", None, None)]
for ax, rows, S_, D_, title in ((axs[0], rows_l, ls_, ld, "LVIS object crops vs whole images (92 categories)"),
                                (axs[1], rows_i, im, idf, "ImageNet whole photographs (75 categories)")):
    for k, (key, lab, col, mk, vs, pair) in enumerate(rows):
        y = len(rows) - 1 - k; v = S_.loc[key]
        ax.plot([v.auc_ci_lo, v.auc_ci_hi], [y, y], color="#52514e", lw=1.6)
        ax.scatter([v.auc], [y], s=70, color=col, edgecolor="#333333", zorder=3, marker=mk)
        txt = f"{v.auc:.3f}" + (f"   {vs}: {FS.stars(D_.loc[pair].p_boot)}" if pair else "")
        ax.text(v.auc_ci_hi + .005, y, txt, va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(range(len(rows)), [r_[1] for r_ in rows][::-1])
    ax.set_xlabel("AUC: P(between > within distance), DINOv3"); ax.set_title(title, loc="left", fontsize=10)
axs[0].set_xlim(.6, .86)
fig.tight_layout(); save_resp(fig, "R1d_background")

# ---- R1e: crop geometry matching --------------------------------------------------------------------------------------
g = pd.read_csv(CTRL / "geom_summary.csv"); g = g[(g.analysis == "matching") & (g.model == M)]
gd = pd.read_csv(CTRL / "geom_differences.csv")
gd = gd[(gd.analysis == "matching") & (gd.metric == "auc") & (gd.pair == "bv - vedb") & (gd.model == M)]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, part in zip(axs, ("all", "real")):
    d = g[g.part == part]
    for j, cond in enumerate(("unmatched", "matched")):
        for k, ds in enumerate(("bv", "vedb")):
            v = d[(d.level == cond) & (d.dataset == ds)].iloc[0]
            ax.errorbar([j + (k - .5) * .3], [v.auc], yerr=[[v.auc - v.auc_ci_lo], [v.auc_ci_hi - v.auc]], fmt="o", color=FS.DS_COLOR[ds],
                        mfc=FS.DS_COLOR[ds] if cond == "matched" else "white", ms=7, capsize=2,
                        label=LABC[ds] if (j == 1 and part == "all") else None)
        top = d[d.level == cond].auc_ci_hi.max() + .006
        r_ = gd[(gd.part == part) & (gd.level == cond)]
        if len(r_):
            FS.bracket(ax, j - .15, j + .15, top, FS.stars(r_.p.iloc[0]), h=.003, fontsize=8)
    rho = {c: d[d.level == c].bv_vedb_rho.iloc[0] for c in ("unmatched", "matched")}
    ax.set_xticks([0, 1], [f"{c}\nρ = {rho[c]:.2f}" for c in ("unmatched", "matched")], fontsize=9)
    ax.set_xlim(-.6, 1.6)
    ax.set_title(f"{'All formats' if part == 'all' else 'Real objects only'} ({int(d.n_categories.iloc[0])} categories)", loc="left", fontsize=10)
    ax.set_ylabel("domain clustering, AUC (DINOv3)")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Crop geometry: BabyView and VEDB before / after matching on size x vertical position x truncation\n"
             "(ρ = BabyView-VEDB RDM correlation; brackets: BabyView vs VEDB, bootstrap p)", x=.01, ha="left", fontsize=10)
fig.tight_layout(); save_resp(fig, "R1e_geometry_matching")

# ---- R1j: background removed (SAM 3) ----------------------------------------------------------------------------------
T = pd.read_csv(BVV / "background_control.csv"); T = T[T.model == M]
Df = pd.read_csv(BVV / "background_control_differences.csv"); Df = Df[Df.model == M]
VERS = ["original", "box", "masked"]; VL = ["Qwen box", "SAM 3 box", "SAM 3 box,\nbackground removed"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, sub in zip(axs, ("all formats", "real objects")):
    for j, d in enumerate(("bv", "vedb")):
        t = T[(T.subset == sub) & (T.dataset == d)].set_index("version").loc[VERS]
        ax.bar(np.arange(3) + (j - .5) * .36, t.delta_sd, .34, color=FS.DS_COLOR[d], label=FS.label(d), edgecolor="white", linewidth=1.5)
    rho = T[(T.subset == sub) & (T.dataset == "BabyView-VEDB RDM rho")].set_index("version").loc[VERS].rho
    top = T[(T.subset == sub) & T.dataset.isin(["bv", "vedb"])].delta_sd.max()
    for k, ver in enumerate(VERS):
        r_ = Df[(Df.subset == sub) & (Df.measure == "delta_sd") & (Df.comparison == f"BabyView - VEDB, {ver}")]
        FS.bracket(ax, k - .18, k + .18, top * 1.06, FS.stars(r_.p_boot.iloc[0]), h=top * .02)
    ax.set_xticks(range(3), [f"{v}\nρ = {x:.2f}" for v, x in zip(VL, rho)], fontsize=8.5); ax.set_ylim(0, top * 1.2)
    n = int(T[T.subset == sub].categories.iloc[0])
    ax.set_title(f"{sub} ({n} categories)", loc="left", fontsize=10)
axs[0].set_ylabel("domain clustering (Δ / SD, DINOv3)")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=2)
fig.suptitle("Background control: Qwen's box, SAM 3's corrected box, and the background removed\n"
             "brackets: BabyView vs VEDB (category bootstrap); ρ = BabyView-VEDB RDM correlation", x=.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .93)); save_resp(fig, "R1j_background_control")
