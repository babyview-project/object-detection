#!/usr/bin/env python3
"""Regenerate manuscript number autofill files from main_results_valid129 outputs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PREPRINT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PREPRINT_DIR.parents[1]
sys.path.insert(0, str(PREPRINT_DIR / "not_in_manuscript"))

from _paths import MANUSCRIPT_DIR  # noqa: E402

RESULTS_DIR = next(
    p
    for p in sorted(
        MANUSCRIPT_DIR.glob("main_results_valid129*/results"),
        key=lambda x: x.parent.name,
        reverse=True,
    )
    if p.exists()
)
TEX_PATH = MANUSCRIPT_DIR / "results_preprint.tex"
OUT_PATH = MANUSCRIPT_DIR / "results_preprint_numbers_autofill.txt"
OUT_TABLE_PATH = MANUSCRIPT_DIR / "results_preprint_numbers_table.csv"
OUT_PARAGRAPHS_PATH = MANUSCRIPT_DIR / "results_preprint_paragraphs_autofill.txt"


def pick_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in candidates]}")


def main() -> int:
    paths = {
        "long_tail": pick_existing(
            RESULTS_DIR / "long_tailed_powerlaw_fits_filtered-0.27_valid129.csv",
            RESULTS_DIR / "long_tailed_power_law_fit_by_semantic.csv",
        ),
        "long_tail_freq": RESULTS_DIR / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
        "precision_frame": pick_existing(
            RESULTS_DIR / "precision_vs_frame_prevalence_correlation_filtered-0.27_valid129.csv",
        ),
        "category_cos_clip": RESULTS_DIR / "category_wise_cosine_similarity_clip_filtered-0.27_valid129.csv",
        "category_cos_clip_dino": RESULTS_DIR / "category_wise_cosine_similarity_clip_dinov3_filtered-0.27_valid129.csv",
        "rdm_bv_things": pick_existing(
            RESULTS_DIR / "bv_things_rdm_comparison_summary_filtered-0.27_valid129.csv",
            RESULTS_DIR / "bv_things_rdm_comparison_summary_v2_lowertri_filtered-0.27_valid129.csv",
        ),
        "cluster_strength": RESULTS_DIR / "bv_vs_things_cluster_strength_valid129.csv",
        "cluster_strength_rankcorr": RESULTS_DIR / "bv_vs_things_cluster_strength_rankcorr_valid129.csv",
        "binary_template_corr": RESULTS_DIR / "binary_template_vs_real_rdm_correlations_valid129.csv",
        "pairwise_subject_summary": RESULTS_DIR / "individual_rdm_pairwise_correlation_summary_clip_dinov3_filtered-0.27_valid129.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected result files: {missing}")

    print(f"Using RESULTS_DIR: {RESULTS_DIR}")

    df_long = pd.read_csv(paths["long_tail"])
    sem_col = "cdi_semantic" if "cdi_semantic" in df_long.columns else "semantic_category"
    df_long[sem_col] = df_long[sem_col].astype(str).str.replace("^semantic_", "", regex=True)

    df_freq = pd.read_csv(paths["long_tail_freq"])
    df_prec = pd.read_csv(paths["precision_frame"])
    df_cos_clip = pd.read_csv(paths["category_cos_clip"])
    df_cos_both = pd.read_csv(paths["category_cos_clip_dino"])
    df_rdm = pd.read_csv(paths["rdm_bv_things"])
    df_cluster = pd.read_csv(paths["cluster_strength"])
    df_cluster_rank = pd.read_csv(paths["cluster_strength_rankcorr"])
    df_binary = pd.read_csv(paths["binary_template_corr"])
    df_pair_summary = pd.read_csv(paths["pairwise_subject_summary"])

    semantic_order = [
        "clothing", "furniture_rooms", "household", "toys", "body_parts",
        "food_drink", "outside", "vehicles", "animals",
    ]
    overall_row = df_long[df_long[sem_col] == "all"]
    values: dict = {}
    values["long_tail_alpha_overall"] = float(overall_row.iloc[0]["alpha"])
    alpha_by_sem = df_long.set_index(sem_col)["alpha"].reindex(semantic_order)
    values["long_tail_alpha_by_semantic"] = {
        k: float(v) for k, v in alpha_by_sem.dropna().items()
    }
    values["long_tail_alpha_semantic_mean"] = float(alpha_by_sem.dropna().mean())

    top5 = df_freq.sort_values("proportion", ascending=False).head(5)
    for i, row in enumerate(top5.itertuples(), start=1):
        values[f"frame_prevalence_top{i}_category"] = row.category
        values[f"frame_prevalence_top{i}_proportion"] = float(row.proportion)

    prec_row = df_prec.iloc[0]
    values["precision_vs_frame_prevalence_pearson_r"] = float(prec_row["pearson_r"])
    values["precision_vs_frame_prevalence_spearman_rho"] = float(prec_row["spearman_rho"])
    values["frame_prevalence_total_frames"] = int(df_freq["total_frames"].iloc[0])

    values["cos_clip_mean"] = float(df_cos_clip["cosine_similarity"].mean())
    values["cos_clip_min"] = float(df_cos_clip["cosine_similarity"].min())
    values["cos_clip_max"] = float(df_cos_clip["cosine_similarity"].max())
    values["cos_clip_n"] = int(df_cos_clip.shape[0])

    clip_col = "cosine_similarity_clip" if "cosine_similarity_clip" in df_cos_both.columns else "clip_cosine_similarity"
    dino_col = "cosine_similarity_dinov3" if "cosine_similarity_dinov3" in df_cos_both.columns else "dinov3_cosine_similarity"
    values["clip_vs_dino_categorywise_pearson_r"] = float(
        np.corrcoef(df_cos_both[clip_col], df_cos_both[dino_col])[0, 1]
    )

    rdm_lookup = df_rdm.set_index("model")
    values["rdm_bv_vs_things_spearman_clip"] = float(rdm_lookup.loc["clip", "spearman_r"])
    values["rdm_bv_vs_things_spearman_dinov3"] = float(rdm_lookup.loc["dinov3", "spearman_r"])
    if "babydinov3" in rdm_lookup.index:
        values["rdm_bv_vs_things_spearman_babydinov3"] = float(rdm_lookup.loc["babydinov3", "spearman_r"])

    bin_sel = df_binary[["model", "source", "spearman_rho_template_dissim_vs_rdm"]].copy()
    for model in ["clip", "dinov3", "babydinov3"]:
        for source in ["babyview", "things"]:
            key = f"binary_template_rho_{source}_{model}"
            values[key] = float(
                bin_sel[(bin_sel.model == model) & (bin_sel.source == source)][
                    "spearman_rho_template_dissim_vs_rdm"
                ].iloc[0]
            )

    for _, row in df_cluster.iterrows():
        model = str(row["model"]).lower()
        values[f"cluster_delta_diff_{model}"] = float(row["delta_diff_bv_minus_things"])
        values[f"cluster_delta_ci_low_{model}"] = float(row["boot_ci_low"])
        values[f"cluster_delta_ci_high_{model}"] = float(row["boot_ci_high"])

    for _, row in df_cluster_rank.iterrows():
        model = str(row["model"]).lower()
        values[f"cluster_rank_spearman_{model}"] = float(row["spearman_rho"])
        values[f"cluster_rank_spearman_p_{model}"] = float(row["spearman_p"])
        values[f"cluster_rank_n_clusters_{model}"] = int(row["n_clusters_used"])

    pair_idx = df_pair_summary.set_index("group")
    values["top8_avg_spearman_clip"] = float(pair_idx.loc["top8_densest_subjects_clip", "avg_spearman_rho"])
    values["top8_sd_spearman_clip"] = float(pair_idx.loc["top8_densest_subjects_clip", "std_spearman_rho"])
    values["top8_avg_spearman_dinov3"] = float(pair_idx.loc["top8_densest_subjects_dinov3", "avg_spearman_rho"])
    values["top8_sd_spearman_dinov3"] = float(pair_idx.loc["top8_densest_subjects_dinov3", "std_spearman_rho"])

    xx_count = len(re.findall(r"\bXX\b", TEX_PATH.read_text())) if TEX_PATH.is_file() else None

    lines = [
        "Autofill values for results_preprint.tex",
        "====================================",
        f"Using results directory: {RESULTS_DIR}",
        f"XX placeholders found: {xx_count if xx_count is not None else 'N/A (no tex file)'}",
        "",
        "Long-tailed distributions (frame prevalence, valid129 full infant-view pool):",
        f"- total unique frames in pool: {values['frame_prevalence_total_frames']:,}",
        f"- overall power-law alpha: {values['long_tail_alpha_overall']:.2f}",
        f"- alpha (semantic mean across listed CDI groups): {values['long_tail_alpha_semantic_mean']:.2f}",
        "- alpha by CDI semantic group:",
    ]
    for k in semantic_order:
        if k in values["long_tail_alpha_by_semantic"]:
            lines.append(f"  - {k}: {values['long_tail_alpha_by_semantic'][k]:.2f}")
    lines += [
        "- top-5 frame prevalences:",
    ]
    for i in range(1, 6):
        lines.append(
            f"  - {values[f'frame_prevalence_top{i}_category']}: "
            f"{values[f'frame_prevalence_top{i}_proportion']:.4f}"
        )
    lines += [
        "",
        "Precision vs frame prevalence (valid129):",
        f"- Pearson r: {values['precision_vs_frame_prevalence_pearson_r']:.3f}",
        f"- Spearman rho: {values['precision_vs_frame_prevalence_spearman_rho']:.3f}",
        "",
        "Category-wise BabyView vs THINGS similarity (CLIP):",
        f"- n categories: {values['cos_clip_n']}",
        f"- cosine min/max/mean: {values['cos_clip_min']:.2f} / {values['cos_clip_max']:.2f} / {values['cos_clip_mean']:.2f}",
        f"- cross-model category-wise Pearson r (CLIP vs DINOv3): {values['clip_vs_dino_categorywise_pearson_r']:.2f}",
        "",
        "Between-category RDM correlations (BabyView vs THINGS):",
        f"- CLIP Spearman rho: {values['rdm_bv_vs_things_spearman_clip']:.2f}",
        f"- DINOv3 Spearman rho: {values['rdm_bv_vs_things_spearman_dinov3']:.2f}",
        "",
        "Binary CDI template alignment (Spearman, dissimilarity coding):",
        f"- CLIP rho_BV={values['binary_template_rho_babyview_clip']:.3f}, rho_THINGS={values['binary_template_rho_things_clip']:.3f}",
        f"- DINOv3 rho_BV={values['binary_template_rho_babyview_dinov3']:.3f}, rho_THINGS={values['binary_template_rho_things_dinov3']:.3f}",
        f"- BabyDINOv3 rho_BV={values['binary_template_rho_babyview_babydinov3']:.3f}, rho_THINGS={values['binary_template_rho_things_babydinov3']:.3f}",
        "",
        "CDI cluster-strength difference (BabyView minus THINGS):",
        f"- CLIP delta={values['cluster_delta_diff_clip']:.3f}, 95% CI [{values['cluster_delta_ci_low_clip']:.3f}, {values['cluster_delta_ci_high_clip']:.3f}]",
        f"- DINOv3 delta={values['cluster_delta_diff_dinov3']:.3f}, 95% CI [{values['cluster_delta_ci_low_dinov3']:.3f}, {values['cluster_delta_ci_high_dinov3']:.3f}]",
        "",
        "Across-cluster BV-vs-THINGS agreement:",
        f"- CLIP Spearman rho={values['cluster_rank_spearman_clip']:.3f}, p={values['cluster_rank_spearman_p_clip']:.3g}, n_clusters={values['cluster_rank_n_clusters_clip']}",
        f"- DINOv3 Spearman rho={values['cluster_rank_spearman_dinov3']:.3f}, p={values['cluster_rank_spearman_p_dinov3']:.3g}, n_clusters={values['cluster_rank_n_clusters_dinov3']}",
        "",
        "Top-8 family pairwise RDM consistency (Spearman):",
        f"- CLIP mean={values['top8_avg_spearman_clip']:.3f}, SD={values['top8_sd_spearman_clip']:.3f}",
        f"- DINOv3 mean={values['top8_avg_spearman_dinov3']:.3f}, SD={values['top8_sd_spearman_dinov3']:.3f}",
    ]

    paragraphs = [
        (
            "Category frequencies were summarized as Clerkin-style frame prevalence: the proportion of "
            "unique infant-view frames containing at least one detection of each category, relative to "
            f"{values['frame_prevalence_total_frames']:,} unique frames in the 0.27-filtered pool."
        ),
        (
            "To quantify distributional shape, we fit power-law functions to frame-prevalence rank "
            f"distributions. Across the 129 categories, the estimated exponent was "
            f"$\\alpha={values['long_tail_alpha_overall']:.2f}$. Long-tailed structure was also present "
            "within CDI semantic groups, with "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['clothing']:.2f}$ (clothing), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['furniture_rooms']:.2f}$ (furniture/rooms), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['household']:.2f}$ (household), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['toys']:.2f}$ (toys), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['body_parts']:.2f}$ (body parts), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['food_drink']:.2f}$ (food/drink), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['outside']:.2f}$ (outside), "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['vehicles']:.2f}$ (vehicles), and "
            f"$\\alpha={values['long_tail_alpha_by_semantic']['animals']:.2f}$ (animals)."
        ),
        (
            f"Manual validation precision was modestly associated with frame prevalence "
            f"(Pearson $r={values['precision_vs_frame_prevalence_pearson_r']:.2f}$, "
            f"Spearman $\\rho={values['precision_vs_frame_prevalence_spearman_rho']:.2f}$)."
        ),
        (
            "Category-level BabyView-versus-THINGS similarity varied widely in CLIP space, from "
            f"{values['cos_clip_min']:.2f} to {values['cos_clip_max']:.2f} "
            f"(mean $\\cos(\\theta)={values['cos_clip_mean']:.2f}$; $n={values['cos_clip_n']}$ categories). "
            "Category rankings were moderately aligned between embedding spaces, with "
            f"CLIP-versus-DINOv3 category-wise correlation of $r={values['clip_vs_dino_categorywise_pearson_r']:.2f}$."
        ),
        (
            "Between-category representational geometry was correlated between BabyView and THINGS in both feature spaces "
            f"(CLIP Spearman $\\rho={values['rdm_bv_vs_things_spearman_clip']:.2f}$; "
            f"DINOv3 Spearman $\\rho={values['rdm_bv_vs_things_spearman_dinov3']:.2f}$). "
            "Superordinate CDI clustering was stronger in BabyView than in THINGS: "
            f"CLIP $\\Delta={values['cluster_delta_diff_clip']:.3f}$, 95\\% CI "
            f"[{values['cluster_delta_ci_low_clip']:.3f}, {values['cluster_delta_ci_high_clip']:.3f}]; "
            f"DINOv3 $\\Delta={values['cluster_delta_diff_dinov3']:.3f}$, 95\\% CI "
            f"[{values['cluster_delta_ci_low_dinov3']:.3f}, {values['cluster_delta_ci_high_dinov3']:.3f}]."
        ),
        (
            "Template-based checks supported this pattern: binary CDI template alignment was "
            f"CLIP ($\\rho_{{BV}}={values['binary_template_rho_babyview_clip']:.3f}$, "
            f"$\\rho_{{THINGS}}={values['binary_template_rho_things_clip']:.3f}$) and "
            f"DINOv3 ($\\rho_{{BV}}={values['binary_template_rho_babyview_dinov3']:.3f}$, "
            f"$\\rho_{{THINGS}}={values['binary_template_rho_things_dinov3']:.3f}$). "
            "BV-vs-THINGS cluster-strength ranks were also correlated across CDI groups "
            f"(CLIP $\\rho={values['cluster_rank_spearman_clip']:.3f}$, "
            f"$p={values['cluster_rank_spearman_p_clip']:.3g}$; "
            f"DINOv3 $\\rho={values['cluster_rank_spearman_dinov3']:.3f}$, "
            f"$p={values['cluster_rank_spearman_p_dinov3']:.3g}$)."
        ),
        (
            "Across the top eight densest families, pairwise RDM similarity remained high in both spaces "
            f"(CLIP mean Spearman $\\rho={values['top8_avg_spearman_clip']:.3f}$, SD={values['top8_sd_spearman_clip']:.3f}; "
            f"DINOv3 mean Spearman $\\rho={values['top8_avg_spearman_dinov3']:.3f}$, SD={values['top8_sd_spearman_dinov3']:.3f}), "
            "indicating consistent category structure across families despite idiosyncratic environments."
        ),
    ]

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_PARAGRAPHS_PATH.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")

    flat_rows = []
    for key, val in values.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_rows.append({"metric": f"{key}.{subkey}", "value": float(subval)})
        else:
            flat_rows.append({"metric": key, "value": val})
    pd.DataFrame(flat_rows).sort_values("metric").reset_index(drop=True).to_csv(OUT_TABLE_PATH, index=False)

    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {OUT_TABLE_PATH}")
    print(f"Wrote: {OUT_PARAGRAPHS_PATH}")
    print(f"Overall alpha: {values['long_tail_alpha_overall']:.3f}")
    print(f"Semantic mean alpha: {values['long_tail_alpha_semantic_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
