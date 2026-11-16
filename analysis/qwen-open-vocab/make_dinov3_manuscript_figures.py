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
# Main-text panels are copied into manuscript_revision/figures/ by make_manuscript_figures.py (copy_in) under their
# manuscript names; this script writes only to figures/manuscript_dinov3*/ and the response-letter folder.
M = "dinov3"
LABEL = FS.DS_LABEL
ZERO = dict(color="#8a8984", lw=1, ls="--")


def save(fig, name):
    FS.text_overlaps(fig, name)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
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
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for k, (d, v, se) in enumerate(zip(cs.dataset, cs.delta_sd, cs.delta_sd_se)):          # ±1 bootstrap SE
    hi = v + se
    ax.errorbar([k], [v], yerr=se, fmt="o", color=FS.DS_COLOR[d], ms=8, mec="white", capsize=3, lw=1.6)
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
        ax.errorbar(x, s.delta_sd, yerr=s.se_sd, fmt="o", color=col, ms=6,                       # ±1 bootstrap SE
                    mfc="white" if (d == "bv" and "bv_real" in show) else col, mec=col if "bv_real" in show else "white",
                    capsize=2, lw=1.2, label=LABEL[d] + (" (all formats)" if d == "bv" and "bv_real" in show else ""))
        pair = star_pair(d)
        if pair:
            st = sig[sig.pair == pair].set_index("domain").reindex(doms).stars.fillna("")
            for xi, hi, t in zip(x, s.delta_sd + s.se_sd, st):
                if t:
                    ax.text(xi, hi + .05, t, ha="center", va="bottom", fontsize=8, color=FS.INK)
    ax.axhline(0, **ZERO); ax.set_ylabel("Δ / SD")
    ax.set_ylim(top=ax.get_ylim()[1] + .5)
    ax.legend(ncol=len(show), frameon=False, loc="upper right", fontsize=8.5)
    ax.set_xticks(range(len(doms)), [f"{FS.DOMAIN_LABEL.get(d, d)}\n({n[d]})" for d in doms], fontsize=9)
    FS.color_domain_ticks(ax.get_xticklabels(), doms)
    ax.set_title(title.format(n=int(n.sum())), loc="left", fontsize=11)
    fig.tight_layout(); save(fig, name)


by_domain(pd.read_csv(XD / "cluster_strength_by_domain_core.csv"),
          pd.read_csv(XD / "cluster_strength_by_domain_differences_core.csv"),
          ["bv", "vedb", "things", "lvis", "imagenet"], lambda d: None if d == "bv" else f"bv - {d}",
          "Domain clustering by domain, {n} shared categories (DINOv3; n per domain in brackets; ± 1 SE)\n"
          "stars: vs BabyView, BH across domains × datasets (* q < .05, ** < .01, *** < .001)",
          "fig5b_cluster_by_domain")
by_domain(pd.read_csv(XD / "cluster_strength_by_domain_real.csv"),
          pd.read_csv(XD / "cluster_strength_by_domain_real_differences.csv"),
          ["bv", "bv_real", "vedb_real", "things", "lvis", "imagenet"],
          lambda d: None if d == "bv_real" else ("bv - bv_real" if d == "bv" else f"bv_real - {d}"),
          "Domain clustering with real objects only, {n} categories (DINOv3; open circles: BabyView all formats; ± 1 SE)\n"
          "stars: vs BabyView real (on open circles: all formats vs real), BH across domains × comparisons (* q < .05, ** < .01, *** < .001)",
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
# Every interval here is ±1 SE from resampling children and VEDB sessions (uncertainty_bv_vedb.py), except the MMD panel,
# whose point is a median over categories (SE from resampling categories).
J = pd.read_csv(BVV / "js_divergence_se.csv")
sc = pd.read_csv(BVV / "scene_statistics_se.csv")
UM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "bv_vedb") / "scene_unit_means.csv")
AG = pd.read_csv(BVV / "appearance_mmd_by_age_se.csv"); AG = AG[AG.model == M]
RR = pd.read_csv(BVV / "rdm_split_half_se.csv"); RR = RR[(RR.model == M) & ~RR.real]
xb = np.arange(len(BINS))


def series(ax, t, col, ycol, label, ls="-", invert=False, dx=0.0, mfc=None):
    """Point estimate per age bin with ±1 SE whiskers."""
    t = t.set_index("group").loc[BINS]
    y = 1 - t[ycol] if invert else t[ycol]
    ax.errorbar(xb + dx, y, yerr=t.se, fmt="o" + ls, color=col, ms=5, lw=1.5, capsize=2.5, elinewidth=1.1,
                mfc=mfc or col, label=label)


fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 6)
axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
       fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
for ax, level in zip(axs[:2], ("category", "domain")):
    j = J[(J.level == level) & (J.scope == "all frames")]
    series(ax, j[j.comparison == "BabyView vs VEDB"], C_VEDB, "js", "BabyView vs VEDB", dx=-.08)
    jh = J[(J.level == level) & (J.scope == "home-scene frames") & (J.comparison == "BabyView vs VEDB")]
    series(ax, jh, C_VEDB, "js", "BabyView vs VEDB, home scenes", ls="--", mfc="white")
    series(ax, j[j.comparison == "within BabyView"], C_BASE, "js", "within BabyView (baseline)", dx=.08)
    ax.set_ylim(bottom=0)
    ax.set_title(f"what is seen: JS divergence ({level})", loc="left", fontsize=10)
ax = axs[2]
u = UM[(UM.scope == "all frames") & UM.group.isin(BINS)]
rng_ = np.random.default_rng(0)
for k, g in enumerate(BINS):
    v = u[u.group == g].dep_share
    ax.scatter(k + rng_.uniform(-.13, .13, len(v)), v, s=11, color=C_BV, alpha=.35, lw=0, zorder=1)
s = sc[(sc.scope == "all frames") & (sc.measure == "dep_share")].set_index("group")
ax.errorbar(xb, s.loc[BINS, "mean"], yerr=s.loc[BINS, "se"], fmt="o-", color=C_BV, ms=6, capsize=3, lw=1.6, zorder=3,
            label="BabyView: mean of children (dots: each child)")
vm, vs = s.loc["VEDB", "mean"], s.loc["VEDB", "se"]
ax.axhspan(vm - vs, vm + vs, color=C_VEDB, alpha=.25, lw=0)
ax.axhline(vm, color=C_VEDB, lw=1.5, label="VEDB: mean of sessions")
ax.set_ylim(bottom=0)
ax.set_title("share of objects depicted (toy / picture / screen)", loc="left", fontsize=10)
ax = axs[3]
for r, ls, dx in ((False, "-", -.05), (True, "--", .05)):
    series(ax, AG[AG.real == r].rename(columns={"median_excess": "y"}), C_VEDB, "y",
           "real objects only" if r else "all formats", ls, dx=dx, mfc="white" if r else None)
ax.axhline(0, color=C_BASE, lw=1, ls=":")
ax.set_title("how things look: MMD² excess over within-dataset baseline (DINOv3)", loc="left", fontsize=10)
ax = axs[4]
series(ax, RR[RR.comparison == "BabyView vs VEDB"].rename(columns={"rho": "y"}), C_VEDB, "y", "BabyView vs VEDB", invert=True)
series(ax, RR[RR.comparison == "within BabyView"].rename(columns={"rho": "y"}), C_BASE, "y", "within BabyView (baseline)",
       invert=True)
ax.set_ylim(bottom=0)
ax.set_title("between-category structure: 1 − RDM ρ (DINOv3)", loc="left", fontsize=10)
for ax in axs:
    ax.set_xticks(xb, BINS, fontsize=8); ax.legend(fontsize=7.5, frameon=False)
axs[0].set_ylim(top=axs[0].get_ylim()[1] * 1.25); axs[0].legend(fontsize=7.5, frameon=False, loc="upper right", ncol=1)
fig.suptitle("Does children's visual experience approach adults' with age?  (error bars: ±1 SE)", x=.01, ha="left")
fig.tight_layout(); save(fig, "fig7c_age_trends")


# =====================================================================================================================
# Response-letter figures (revision_private/figures/): DINOv3-only redraws of the controls, plus copies of the
# main-text figures above under the letter's names.
# =====================================================================================================================
RESP = V.manuscript_figures_dir(HERE.parents[0] / "manuscript-2026" / "revision_private" / "figures"); RESP.mkdir(exist_ok=True)
CTRL = V.out_dir(HERE / "results" / "controls"); CTRL = CTRL.with_name(CTRL.name + "_core")
CMP = HERE.parents[0] / "comparison-datasets" / "outputs"


def save_resp(fig, name):
    FS.text_overlaps(fig, name)
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
lo_, hi_ = (sm.delta_sd - sm.delta_sd_se).min(), (sm.delta_sd + sm.delta_sd_se).max()
ylo, yhi = lo_ - .3 * (hi_ - lo_), hi_ + .3 * (hi_ - lo_)   # room for the LVIS notes below and the legend above
fig, ax = plt.subplots(figsize=(10, 4.2))
for i, (v, _) in enumerate(VAR):
    for k, (d, lab) in enumerate(DSA):
        r = sm[(sm.variant == v) & (sm.dataset == d)].iloc[0]
        x = i + (k - 1.5) * .17
        ax.errorbar([x], [r.delta_sd], yerr=[r.delta_sd_se], fmt="o", color=FS.DS_COLOR[d], ms=7,
                    mec=FS.INK, mew=.8, capsize=0, lw=2, label=lab if i == 0 else None)
        if d != "bv":
            p = sdf[(sdf.variant == v) & (sdf.pair == f"bv_minus_{d}")].q_bh_sd.iloc[0]
            t = FS.stars(p, "")
            if t:
                ax.text(x, r.delta_sd + r.delta_sd_se + .01, t, ha="center", va="bottom", fontsize=8)
    lv = sm[(sm.variant == v) & (sm.dataset == "lvis")].iloc[0]
    if v != "none":
        ax.text(i, ylo + .02, f"LVIS {lv.delta_sd_change_vs_unmatched:+.2f}\n{FS.ptext(lv.delta_sd_change_q, 'q')}", ha="center", fontsize=7.5, color=FS.MUTED)
ax.set_xticks(range(len(VAR)), [l for _, l in VAR], fontsize=9)
ax.set_ylim(ylo, yhi); ax.set_ylabel("domain clustering, Δ / SD (DINOv3)")
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
ax.set_title("Crop-size floors on every dataset and the original pipeline's CLIP filter on LVIS (≈104 categories; ± SE)\n"
             "stars: differs from BabyView (* q < .05, ** < .01, *** < .001; BH across the 18 comparisons); grey: change in LVIS Δ / SD vs unmatched (BH across 5)",
             loc="left", fontsize=10)
fig.tight_layout(); save_resp(fig, "R1a_clip_filter_lvis")

LABC = {"bv": "BabyView", "bv_real": "BabyView, real", "vedb": "VEDB", "vedb_real": "VEDB, real", "things": "THINGS"}

# ---- R1e: crop geometry matching --------------------------------------------------------------------------------------
g = pd.read_csv(CTRL / "geom_summary.csv"); g = g[(g.analysis == "matching") & (g.model == M)]
gd = pd.read_csv(CTRL / "geom_differences.csv")
gd = gd[(gd.analysis == "matching") & (gd.metric == "delta_sd") & (gd.model == M)
        & ((gd.pair == "bv - vedb") | (gd.pair == "(bv - vedb): matched - unmatched"))].copy()
gd["q"] = FS.bh(gd.p)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, part in zip(axs, ("all", "real")):
    d = g[g.part == part]
    for j, cond in enumerate(("unmatched", "matched")):
        for k, ds in enumerate(("bv", "vedb")):
            v = d[(d.level == cond) & (d.dataset == ds)].iloc[0]
            ax.errorbar([j + (k - .5) * .3], [v.delta_sd], yerr=[v.delta_sd_se], fmt="o", color=FS.DS_COLOR[ds],
                        mfc=FS.DS_COLOR[ds] if cond == "matched" else "white", ms=7, capsize=0, lw=2,
                        label=LABC[ds] if (j == 1 and part == "all") else None)
        top = (d[d.level == cond].delta_sd + d[d.level == cond].delta_sd_se).max() + .02
        r_ = gd[(gd.part == part) & (gd.level == cond) & (gd.pair == "bv - vedb")]
        if len(r_):
            FS.bracket(ax, j - .15, j + .15, top, FS.stars(r_.q.iloc[0]), h=.01, fontsize=8)
    ch = gd[(gd.part == part) & (gd.pair == "(bv - vedb): matched - unmatched")].iloc[0]
    note = (f"change in BabyView − VEDB gap after matching:\n{ch['diff']:+.2f} [{ch.ci_lo:+.2f}, {ch.ci_hi:+.2f}], "
            f"{FS.ptext(ch.q, 'q')}").replace("-", "−")
    rho = {c: d[d.level == c].bv_vedb_rho.iloc[0] for c in ("unmatched", "matched")}
    ax.set_xticks([0, 1], [f"{c}\nρ = {rho[c]:.2f}" for c in ("unmatched", "matched")], fontsize=9)
    ax.set_xlim(-.6, 1.6)
    ax.set_title(f"{'All formats' if part == 'all' else 'Real objects only'} ({int(d.n_categories.iloc[0])} categories)\n{note}",
                 loc="left", fontsize=9)
    ax.set_ylabel("domain clustering, Δ / SD (DINOv3)")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Crop geometry: BabyView and VEDB before / after matching on size x vertical position x truncation\n"
             "(± SE; ρ = BabyView-VEDB RDM correlation; brackets: BabyView vs VEDB; all tests BH across the six shown)", x=.01, ha="left", fontsize=10)
fig.tight_layout(); save_resp(fig, "R1e_geometry_matching")

# R1j (background removed, SAM 3) is drawn by background_response_figures.py (plot_sam).

