#!/usr/bin/env python3
"""One table of the numbers the manuscript and response letter report, read from the result tables (never typed in).

Outputs (results/numbers/):
  numbers_key.csv   curated numbers with stable keys (cite the key in the text): key, section, description, variant,
                    value, ci_low, ci_high, p, q, n, unit, source (file), selector (row filter / column)
  numbers_all.csv.gz every numeric cell of every aggregate result table: source, row (its text columns joined), column,
                    value. For numbers that are not in numbers_key.csv.
Domain-dependent numbers appear once per domain scheme (variant = cdi / things27); the rest have variant = all.
DINOv3 throughout (the revision reports DINOv3 only). Rerun after any analysis rerun:

  QOV_LABELS=renamed /data/j7yang/anaconda3/bin/python build_numbers_table.py [--skip-counts]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
MS = REPO / "analysis" / "manuscript-2026"
CMP = REPO / "analysis" / "comparison-datasets" / "outputs"
GOLD = REPO / "annotation" / "gold_set" / "out" / "private" / "ra_scores"
OUT = RES / "numbers"
M = "dinov3"
VARIANTS = {"cdi": "", "things27": "_things27"}
ROWS: list[dict] = []


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def ci(s):
    """'[a, b]' -> (a, b)."""
    if isinstance(s, str) and s.startswith("["):
        a, b = (float(x) for x in s.strip("[]").split(","))
        return a, b
    return np.nan, np.nan


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def slug(*parts) -> str:
    s = ".".join(str(p) for p in parts if p not in (None, ""))
    s = s.replace(" - ", "_minus_").replace(" vs ", "_vs_").replace("<", "lt").replace(">", "gt").replace("+", "plus")
    return re.sub(r"[^A-Za-z0-9_.]+", "_", s).strip("_").lower()


def add(key, section, desc, value, variant="all", lo=np.nan, hi=np.nan, p=np.nan, q=np.nan, n=np.nan, unit="",
        source="", selector=""):
    ROWS.append(dict(key=key, section=section, description=desc, variant=variant, value=num(value), ci_low=num(lo),
                     ci_high=num(hi), p=num(p), q=num(q), n=num(n), unit=unit, source=source, selector=selector))


def rows(path, section, base, desc, label, val, *, where=None, lo=None, hi=None, ci_col=None, p=None, q=None, n=None,
         unit="", variant="all", index_col=None):
    """One key per row of `path` passing `where` (dict col -> value); key = base + the row's `label` columns."""
    path = Path(path)
    if not path.exists():
        print("missing", rel(path) if str(path).startswith(str(REPO)) else path, file=sys.stderr)
        return
    d = pd.read_csv(path, index_col=index_col)
    if index_col is not None:
        d = d.rename_axis("_row").reset_index()
    for c, v in (where or {}).items():
        d = d[d[c].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    label = [label] if isinstance(label, str) else list(label)
    for _, r in d.iterrows():
        tag = [r[c] for c in label]
        a, b = ci(r[ci_col]) if ci_col else (r[lo] if lo else np.nan, r[hi] if hi else np.nan)
        add(slug(base, *tag), section, desc.format(**{str(c).lstrip("_"): r[c] for c in r.index}), r[val], variant, a, b,
            r[p] if p else np.nan, r[q] if q else np.nan, r[n] if n else np.nan, unit, rel(path),
            "; ".join(f"{c}={v}" for c, v in list((where or {}).items()) + list(zip(label, tag))) + f"; column={val}")


def counts():
    sys.path.insert(0, str(HERE))
    import variant as V
    from build_bv_vedb_data import STORES, unit_of
    for ds, store in STORES.items():
        ix = V.read_index(store, usecols=["video_id", "frame_number", "label"])
        src = rel(V.index_path(store)) if str(V.index_path(store)).startswith(str(REPO)) else str(V.index_path(store))
        units = ix.video_id.drop_duplicates().map(unit_of)
        lab = "children" if ds == "bv" else "recording sessions"
        add(f"data.{ds}.units", "data", f"{ds.upper()}: {lab}", units.nunique(), source=src, selector="unique unit_of(video_id)")
        add(f"data.{ds}.videos", "data", f"{ds.upper()}: videos", ix.video_id.nunique(), source=src, selector="unique video_id")
        add(f"data.{ds}.frames", "data", f"{ds.upper()}: frames in the Qwen box index", len(ix[["video_id", "frame_number"]].drop_duplicates()),
            source=src, selector="unique (video_id, frame_number)")
        add(f"data.{ds}.objects", "data", f"{ds.upper()}: Qwen objects (boxes)", len(ix), source=src, selector="rows")
        add(f"data.{ds}.labels", "data", f"{ds.upper()}: distinct (renamed) labels", ix.label.nunique(), source=src, selector="unique label")
    for name, f in (("tier1", "included_categories_qwen_open_tier1.txt"), ("core", "included_categories_qwen_core.txt"),
                    ("core_real", "included_categories_qwen_core_real.txt")):
        p = REPO / "data" / f
        add(f"data.categories.{name}", "data", f"categories: {name}", sum(1 for l in open(p) if l.strip()), source=rel(p), selector="lines")


def key_numbers():
    # ---- validation (gold set v2, RA-corrected Qwen boxes) -------------------------------------------------------
    rows(GOLD / "summary.csv", "validation", "validation.gold.precision", "gold set ({set}): precision", "set", "precision", n="frames")
    for col, what in (("recall", "recall"), ("name_kept", "share of matched boxes whose name raters kept"),
                      ("format_kept", "share of matched boxes whose format raters kept"), ("iou_median", "median IoU, Qwen vs corrected box")):
        rows(GOLD / "summary.csv", "validation", f"validation.gold.{col}", f"gold set ({{set}}): {what}", "set", col, n="frames")
    ir = GOLD / "interrater.csv"
    if ir.exists():
        r = pd.read_csv(ir).iloc[0]
        add("validation.interrater.box_f1", "validation", "inter-rater box agreement (F1)", r.box_agreement_f1, n=r.shared_frames_both_reviewed,
            source=rel(ir), selector="box_agreement_f1")
        add("validation.interrater.name_same", "validation", "inter-rater: same name", r.name_same, source=rel(ir), selector="name_same")
    fc = GOLD / "format_confusion.csv"
    if fc.exists():
        c = pd.read_csv(fc, index_col=0).drop(index="All", columns="All")
        for f in c.columns:
            add(f"validation.format.recall.{f}", "validation", f"format: share of rater-'{f}' boxes Qwen called '{f}'",
                c.loc[f, f] / c[f].sum(), n=c[f].sum(), source=rel(fc), selector=f"diag/column {f}")

    # ---- long tail, formats, frequencies (domain-independent) ----------------------------------------------------
    rows(RES / "open_renamed" / "open_long_tail_powerlaw.csv", "long_tail", "long_tail.alpha", "power-law exponent ({formats}, {domain})",
         ["formats", "domain"], "alpha", n="n")
    rows(RES / "open_renamed" / "open_long_tail_powerlaw.csv", "long_tail", "long_tail.r2", "power-law fit R² (log) ({formats}, {domain})",
         ["formats", "domain"], "r2_log", where={"domain": "all"})
    for f in ("real", "toy", "picture", "screen"):
        rows(RES / "crossdataset_renamed_core" / "format_prevalence.csv", "formats", f"formats.share.{f}",
             f"share of exemplars that are {f} ({{dataset}}, {{categories}} categories)", ["dataset", "categories"], f, n="exemplars")
    for ds in ("bv", "vedb", "things"):
        rows(RES / "crossdataset_renamed_core" / "format_depicted_by_domain.csv", "formats", f"formats.depicted_by_domain.{ds}",
             f"{ds}: share depicted (toy / picture / screen), {{domain}}", "domain", ds)
    rows(RES / "controls_renamed_core" / "prevalence_spearman.csv", "frequency", "frequency.prevalence_rho",
         "category frequency, BabyView vs {versus}: Spearman ρ", "versus", "spearman", lo="ci_lo", hi="ci_hi", p="p", n="n_categories")
    rows(RES / "bv_vedb_renamed_core" / "domain_frequency_ratio.csv", "frequency", "frequency.domain_log2_ratio",
         "BabyView / VEDB frequency ratio (log2), {row}", "_row", "log2_ratio", lo="lo", hi="hi", n="n", index_col=0)

    # ---- category geometry: RDM correlations (domain-independent) ------------------------------------------------
    rows(RES / "crossdataset_renamed_core" / "pairwise_rdm_correlations.csv", "rdm", "rdm.rho", "RDM Spearman ρ, {a} vs {b}",
         ["a", "b"], "spearman", where={"model": M}, lo="ci_low", hi="ci_high", n="n_categories")
    rows(RES / "controls_renamed_core" / "perm_rdm_null.csv", "rdm", "rdm.perm_null", "RDM ρ with BabyView vs {versus}, null = {null}",
         ["versus", "null"], "rho", where={"model": M}, p="p", n="n_categories")
    rows(RES / "crossdataset_renamed_core" / "things_readout_check.csv", "rdm", "rdm.things_readout.rho_bv",
         "THINGS readout {variant}: RDM ρ with BabyView", "variant", "rho_bv", where={"model": M})

    # ---- domain clustering and controls, per domain scheme -------------------------------------------------------
    for v, s in VARIANTS.items():
        X = RES / f"crossdataset_renamed{s}_core"
        C = RES / f"controls_renamed{s}_core"
        B = RES / f"bv_vedb_renamed{s}_core"
        rows(X / "cluster_strength_core.csv", "clustering", "clustering.dsd", "domain clustering Δ/SD, {dataset}", "dataset", "delta_sd",
             where={"model": M}, ci_col="delta_sd_ci", n="n", variant=v)
        rows(X / "cluster_strength_core.csv", "clustering", "clustering.auc", "domain clustering AUC, {dataset}", "dataset", "auc",
             where={"model": M}, n="n", variant=v)
        rows(X / "cluster_strength_differences_core.csv", "clustering", "clustering.diff.dsd", "domain clustering Δ/SD difference, {pair}",
             "pair", "diff", where={"model": M, "measure": "delta_sd"}, lo="ci_low", hi="ci_high", p="p_boot", q="q_bh", variant=v)
        rows(X / "ego_vs_curated.csv", "clustering", "clustering.ego_vs_curated.dsd", "egocentric − curated Δ/SD, {pair}", "pair", "diff",
             where={"model": M, "measure": "delta_sd"}, lo="ci_low", hi="ci_high", p="p_boot", q="q_bh", n="n_categories", variant=v)
        rows(X / "cluster_strength_by_domain_core.csv", "clustering", "clustering.by_domain.dsd", "Δ/SD within {domain}, {dataset}",
             ["dataset", "domain"], "delta_sd", where={"model": M}, lo="ci_low_sd", hi="ci_high_sd", n="n_categories", variant=v)
        rows(X / "cluster_strength_by_domain_real.csv", "clustering", "clustering.real_only.dsd", "real objects only: Δ/SD, {domain}, {dataset}",
             ["dataset", "domain"], "delta_sd", where={"model": M}, lo="ci_low_sd", hi="ci_high_sd", n="n_categories", variant=v)
        rows(X / "cluster_strength_by_domain_real_differences.csv", "clustering", "clustering.real_only.diff",
             "real objects only: difference, {domain}, {pair}", ["domain", "pair"], "diff", where={"model": M, "measure": "delta_sd"},
             lo="ci_low", hi="ci_high", p="p_boot", variant=v) if (X / "cluster_strength_by_domain_real_differences.csv").exists() else None
        O = RES / f"open_renamed{s}"
        rows(O / "open_bv_vs_things.csv", "clustering", "clustering.open_bv_vs_things.diff", "open set ({formats}): BabyView − THINGS Δ",
             "formats", "bv_minus_things", where={"model": M}, lo="ci_low", hi="ci_high", p="p_boot", n="n_categories", variant=v)
        rows(O / "open_bv_vs_things.csv", "rdm", "rdm.open_bv_vs_things.rho", "open set ({formats}): BabyView–THINGS RDM ρ",
             "formats", "rdm_spearman", where={"model": M}, n="n_categories", variant=v)
        pc = O / "per_child_clustering_things_all35.csv"
        if pc.exists():
            d = pd.read_csv(pc)
            add("children.clustering.n_above_things", "children", "children whose clustering exceeds THINGS (point estimate)",
                (d["diff"] > 0).sum(), v, n=len(d), source=rel(pc), selector="diff > 0")
            add("children.clustering.n_sig_above_things", "children", "children whose clustering exceeds THINGS (CI above 0)",
                (d["lo"] > 0).sum(), v, n=len(d), source=rel(pc), selector="lo > 0")
        tp = C / "things_per_child.csv"
        if tp.exists():
            d = pd.read_csv(tp); d = d[d.model == M]
            add("children.clustering.n_raw_delta_above_things", "children", "children whose raw Δ exceeds THINGS",
                (d.delta_child > d.delta_things).sum(), v, n=len(d), source=rel(tp), selector=f"model={M}; delta_child > delta_things")
        # controls
        rows(C / "split_summary.csv", "controls", "controls.split.auc", "subject split ({comparison}), {rdm}: AUC {dataset}",
             ["comparison", "rdm", "dataset"], "auc", where={"model": M}, lo="auc_ci_lo", hi="auc_ci_hi", n="n_categories", variant=v)
        rows(C / "split_differences.csv", "controls", "controls.split.diff", "subject split ({comparison}), {rdm}: {pair}",
             ["comparison", "rdm", "pair"], "diff", where={"model": M, "metric": "auc"}, lo="ci_lo", hi="ci_hi", p="p", variant=v)
        rows(C / "room_differences.csv", "controls", "controls.room.diff", "{kind} held fixed ({label}): {pair}, AUC",
             ["kind", "label", "pair"], "diff", where={"model": M, "metric": "auc", "part": "all"}, lo="ci_lo", hi="ci_hi", p="p",
             n="n_categories", variant=v)
        rows(C / "cooc_coefficients.csv", "controls", "controls.cooc.retained", "co-occurrence control ({comparison}, covariates {covariates_from}): share retained, {dataset}",
             ["comparison", "covariates_from", "dataset"], "retained", where={"model": M}, p="p_perm", n="n_categories", variant=v)
        rows(C / "geom_summary.csv", "controls", "controls.geom.auc", "crop geometry ({analysis}, {level}): AUC {dataset}",
             ["analysis", "level", "dataset"], "auc", where={"model": M, "part": "all", "analysis": ["matching", "level"]},
             lo="auc_ci_lo", hi="auc_ci_hi", n="n_categories", variant=v)
        rows(C / "geom_differences.csv", "controls", "controls.geom.diff", "crop geometry ({analysis}, {level}): {pair}, AUC",
             ["analysis", "level", "pair"], "diff", where={"model": M, "part": "all", "metric": "auc"}, lo="ci_lo", hi="ci_hi", p="p", variant=v)
        rows(C / "size_summary.csv", "controls", "controls.size.auc", "size floor {floor} (vs {versus}, {part}): AUC {dataset}",
             ["versus", "floor", "part", "dataset"], "auc", where={"model": M}, lo="auc_ci_lo", hi="auc_ci_hi", n="n_categories", variant=v)
        rows(C / "size_differences.csv", "controls", "controls.size.diff", "size floor {floor} (vs {versus}, {part}): {pair}, AUC",
             ["versus", "floor", "part", "pair"], "diff", where={"model": M, "metric": "auc"}, lo="ci_lo", hi="ci_hi", p="p", variant=v)
        rows(B / "blur_control.csv", "controls", "controls.blur.dsd", "blur ({sharp_cut}): Δ/SD {dataset}", ["sharp_cut", "dataset"], "delta_sd",
             where={"model": M}, n="categories", variant=v)
        rows(B / "blur_control_differences.csv", "controls", "controls.blur.diff", "blur ({sharp_cut}): {pair}", ["sharp_cut", "pair"], "diff",
             where={"model": M}, lo="ci_low", hi="ci_high", p="p_boot", variant=v)
        rows(B / "background_control.csv", "controls", "controls.sam3.dsd", "SAM 3 background ({subset}, {version}): Δ/SD {dataset}",
             ["subset", "version", "dataset"], "delta_sd", where={"model": M, "dataset": ["bv", "vedb"]}, n="categories", variant=v)
        rows(B / "background_control.csv", "controls", "controls.sam3.rho", "SAM 3 background ({subset}, {version}): BabyView–VEDB RDM ρ",
             ["subset", "version"], "rho", where={"model": M, "dataset": "BabyView-VEDB RDM rho"}, n="categories", variant=v)
        rows(B / "background_control_differences.csv", "controls", "controls.sam3.diff", "SAM 3 background ({subset}): {comparison}",
             ["subset", "comparison"], "diff", where={"model": M, "measure": "delta_sd"}, lo="ci_low", hi="ci_high", p="p_boot", variant=v)
        L = RES / ("background_response" + s)
        rows(L / "lvis_whole_vs_crop.csv", "controls", "controls.lvis_whole.dsd", "LVIS background test: Δ/SD {version}", "version", "delta_sd",
             lo="ci_low", hi="ci_high", n="categories", variant=v)
        rows(L / "lvis_whole_vs_crop_differences.csv", "controls", "controls.lvis_whole.diff", "LVIS background test: {comparison}",
             "comparison", "diff", lo="ci_low", hi="ci_high", p="p_boot", variant=v)
        rows(X / "things_readout_check.csv", "rdm", "rdm.things_readout.dsd", "THINGS readout {variant}: Δ/SD", "variant", "delta_sd",
             where={"model": M}, variant=v)
        D = RES / f"bv_vedb_renamed{s}"
        rows(D / "dev_within_child_slopes.csv", "development", "development.within_child.slope", "within-child slope per year: {label}",
             "measure", "slope", lo="lo", hi="hi", p="p", q="q", n="children", unit="per year", variant=v)
        rows(D / "dev_within_child_slopes.csv", "development", "development.between_child.slope", "between-child (age) effect: {label}",
             "measure", "between", p="p_between", unit="per year", variant=v)

    # ---- children, co-occurrence, development (domain-independent) -----------------------------------------------
    ip = RES / "open_renamed" / "individual_rdm_pairs_all35.csv"
    if ip.exists():
        d = pd.read_csv(ip)
        per = pd.concat([d[["a", "spearman"]].rename(columns={"a": "s"}), d[["b", "spearman"]].rename(columns={"b": "s"})]).groupby("s").spearman.mean()
        add("children.rdm.mean_pair_rho", "children", "child–child RDM ρ, mean over all pairs", d.spearman.mean(), n=len(d), source=rel(ip), selector="mean spearman")
        add("children.rdm.child_mean_rho_min", "children", "per-child mean RDM ρ with the other children: lowest", per.min(), n=len(per), source=rel(ip), selector="min of per-child means")
        add("children.rdm.child_mean_rho_max", "children", "per-child mean RDM ρ with the other children: highest", per.max(), n=len(per), source=rel(ip), selector="max of per-child means")
    rows(RES / "cooccurrence_renamed" / "split_half_agreement.csv", "cooccurrence", "cooccurrence.split_half_rho",
         "co-occurrence agreement ({scope}, {group}): {comparison}", ["scope", "group", "comparison"], "rho", lo="lo", hi="hi", n="pairs")
    rows(RES / "bv_vedb_renamed" / "dev_size_summary.csv", "development", "development.size.median_pct_per_year",
         "object size ({format}): median change per year, all boxes", "format", "overall_median_pct", p="overall_p", unit="% per year")
    rows(RES / "bv_vedb_renamed" / "dev_size_summary.csv", "development", "development.size.within_category_pct",
         "object size ({format}): pooled within-category change per year", "format", "pooled_within_cat_pct", lo="pooled_lo",
         hi="pooled_hi", p="pooled_p", n="categories", unit="% per year")
    rows(RES / "bv_vedb_renamed" / "dev_size_summary.csv", "development", "development.size.share_decreasing",
         "object size ({format}): share of categories getting smaller", "format", "share_decreasing", p="sign_test_p", n="categories")

    # ---- response figure 1: LVIS crop-size floors and CLIP filter (original pipeline) ----------------------------
    mc = CMP / "matched_controls" / "results"
    rows(mc / "size_clip_matching_summary.csv", "controls", "controls.lvis_clip_filter.auc",
         "LVIS controls ({variant}, floor {floor_px}, CLIP filter {lvis_clip_filter}): AUC {dataset}",
         ["variant", "floor_px", "lvis_clip_filter", "dataset"], "auc", where={"model": M}, lo="auc_ci_lo", hi="auc_ci_hi", n="n_categories")
    rows(mc / "size_clip_matching_differences.csv", "controls", "controls.lvis_clip_filter.diff", "LVIS controls ({variant}): {pair}",
         ["variant", "pair"], "diff", where={"model": M}, lo="ci_lo", hi="ci_hi", p="p_boot", n="n_categories")

    # ---- SI 1: original YOLOE + CLIP pipeline on 2025.2 (1 fps and one frame per 5 s) ----------------------------
    for tag in ("2025.2", "2025.2_5s"):
        R = MS / f"main_results_valid129s_{tag}" / "results"
        t = slug(tag)
        rows(R / "long_tailed_powerlaw_fits_filtered-0.27_valid129.csv", "si1", f"si1.{t}.long_tail.alpha",
             f"SI 1 ({tag}): power-law exponent, {{semantic_category}}", "semantic_category", "alpha", where={"semantic_category": "all"})
        rows(R / "bv_things_rdm_comparison_summary_v2_lowertri_filtered-0.27_valid129.csv", "si1", f"si1.{t}.rdm_rho",
             f"SI 1 ({tag}): BabyView–THINGS RDM ρ ({{model}})", "model", "spearman_r",
             where={"model": M, "ordering": "bv_first_semantic"}, n="n_categories")
        f = R / "bv_vs_things_cluster_strength_valid129.csv"
        if f.exists():
            d = pd.read_csv(f).set_index("model").loc[M]
            for col, what in (("bv_delta_between_minus_within", "BabyView Δ"), ("things_delta_between_minus_within", "THINGS Δ")):
                add(f"si1.{t}.{col}", "si1", f"SI 1 ({tag}): {what}", d[col], source=rel(f), selector=f"model={M}; column={col}")
            add(f"si1.{t}.bv_minus_things", "si1", f"SI 1 ({tag}): BabyView − THINGS Δ", d.delta_diff_bv_minus_things, lo=d.boot_ci_low,
                hi=d.boot_ci_high, source=rel(f), selector=f"model={M}; column=delta_diff_bv_minus_things")


def dump_all():
    """Every numeric cell of every aggregate result table (per-category tables included; per-box / per-frame excluded)."""
    roots = [RES, MS / "main_results_valid129s_2025.2" / "results", MS / "main_results_valid129s_2025.2_5s" / "results",
             MS / "supplemental_results_valid85cats_2025.2" / "results", MS / "supplemental_results_valid85cats_2025.2_5s" / "results",
             CMP / "matched_controls" / "results", GOLD]
    skip = re.compile(r"(boxes\.csv|reviewed_frames|name_changes|interrater_pairs|sam3?_box_check|/numbers/|subject_|cache)")
    out = []
    for root in roots:
        for f in sorted(root.rglob("*.csv")) if root.exists() else []:
            if skip.search(str(f)):
                continue
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            numc = d.select_dtypes("number").columns
            txt = [c for c in d.columns if c not in numc]
            rowid = d[txt].astype(str).agg(" | ".join, axis=1) if txt else pd.Series(d.index.astype(str), index=d.index)
            for c in numc:
                out.append(pd.DataFrame({"source": rel(f), "row": rowid.values, "column": c, "value": d[c].values}))
    a = pd.concat(out, ignore_index=True)
    a.to_csv(OUT / "numbers_all.csv.gz", index=False)
    print(f"numbers_all.csv.gz: {len(a):,} values from {a.source.nunique()} tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-counts", action="store_true", help="skip the dataset counts (reads both box indexes, ~1 min)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_counts:
        counts()
    key_numbers()
    k = pd.DataFrame(ROWS)
    dup = k[k.duplicated(["key", "variant"], keep=False)]
    if len(dup):
        print("duplicate keys:", dup[["key", "variant"]].drop_duplicates().head(20).to_string(), file=sys.stderr)
    k.to_csv(OUT / "numbers_key.csv", index=False)
    print(f"numbers_key.csv: {len(k)} numbers, {k.key.nunique()} keys; by section: {k.section.value_counts().to_dict()}")
    dump_all()


if __name__ == "__main__":
    main()
