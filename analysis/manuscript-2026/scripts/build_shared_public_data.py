#!/usr/bin/env python3
"""Build anonymized, public-shareable intermediate data for the preprint pipeline.

Run from the repository root or from analysis/manuscript-2026:

    python analysis/manuscript-2026/scripts/build_shared_public_data.py

Outputs land in data/shared_data_manuscript_2026/. Real BabyView participant
IDs are mapped to participant_01 … participant_08 (densest-first ordering used
in the manuscript top-8 analyses). The mapping is applied in memory only; real
IDs are not written to shared_data_manuscript_2026/.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
DATA_DIR = PROJECT_ROOT / "data"
SHARED_DIR = DATA_DIR / "shared_data_manuscript_2026"

MAIN_RESULTS = MANUSCRIPT_DIR / "main_results_valid129s_04302026" / "results"
SUPP_RESULTS = (
    MANUSCRIPT_DIR / "supplemental_results_valid85cats_04302026" / "results"
)
EXEMPLAR_DIR = MANUSCRIPT_DIR / "exemplar_set_embeddings"

# Densest-first top-8 order (trajectory ranking; matches notebook 05 / top8 script).
_TOP8_REAL_IDS: tuple[str, ...] = (
    "00400001",
    "00370002",
    "00320001",
    "00400002",
    "00500001",
    "00510002",
    "00430001",
    "00240001",
)

_PARTICIPANT_MAP: dict[str, str] = {
    real: f"participant_{i:02d}" for i, real in enumerate(_TOP8_REAL_IDS, start=1)
}

_SUBJECT_COLS = frozenset(
    {"subject_id", "subject_a", "subject_b", "subject", "sid"}
)


def _normalize_subject_token(val: object) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.startswith("subject_"):
        s = s.split("_", 1)[1]
    if s.isdigit():
        s = s.zfill(8)
    return s


def _anonymize_value(val: object) -> object:
    token = _normalize_subject_token(val)
    if token is not None and token in _PARTICIPANT_MAP:
        return _PARTICIPANT_MAP[token]
    return val


def anonymize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in _SUBJECT_COLS or col.lower().startswith("subject"):
            out[col] = out[col].map(_anonymize_value)
    return out


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sanitize_run_json(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text())
    redact_keys = (
        "included_categories_txt",
        "clip_filter_list_path",
        "babydinov3_embeddings_dir",
        "clip_embeddings_dir",
        "dinov3_embeddings_dir",
    )
    for key in redact_keys:
        if key in data:
            data[key] = "<redacted_local_path>"
    if "clip_filter_list_stats" in data and isinstance(data["clip_filter_list_stats"], dict):
        data["clip_filter_list_stats"] = {
            k: v
            for k, v in data["clip_filter_list_stats"].items()
            if k not in {"n_lines_missing_file_or_dir"}
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2) + "\n")


def anonymize_filename(name: str) -> str:
    for real, anon in _PARTICIPANT_MAP.items():
        name = name.replace(f"subject_{real}", anon)
    return name


def export_top8_supplemental() -> list[str]:
    written: list[str] = []
    for model in ("clip", "dinov3"):
        src_dir = SUPP_RESULTS / f"top8_vs_things_within_between_{model}_valid85"
        if not src_dir.is_dir():
            continue
        dst_dir = SHARED_DIR / "top8_valid85" / model
        for src in sorted(src_dir.glob("*.csv")):
            df = pd.read_csv(src)
            df = anonymize_dataframe(df)
            out_name = anonymize_filename(src.name)
            if out_name.startswith("subject_"):
                out_name = out_name.replace("subject_", "participant_", 1)
            rel = f"top8_valid85/{model}/{out_name}"
            write_csv(df, SHARED_DIR / rel)
            written.append(rel)
    return written


def export_participant_registry() -> str:
    rows = []
    coverage_path = (
        SUPP_RESULTS
        / "top8_vs_things_within_between_clip_valid85"
        / "top8_category_coverage_clip_valid85.csv"
    )
    coverage: dict[str, float] = {}
    if coverage_path.exists():
        cov_df = pd.read_csv(coverage_path, dtype={"subject_id": str})
        if "subject_id" in cov_df.columns and "pct_coverage" in cov_df.columns:
            for _, r in cov_df.iterrows():
                sid = _normalize_subject_token(r["subject_id"])
                if sid:
                    coverage[sid] = float(r["pct_coverage"])

    for rank, real_id in enumerate(_TOP8_REAL_IDS, start=1):
        rows.append(
            {
                "participant_id": _PARTICIPANT_MAP[real_id],
                "top8_rank_densest_first": rank,
                "pct_category_coverage_valid85": coverage.get(real_id),
            }
        )
    rel = "metadata/participant_registry_top8.csv"
    write_csv(pd.DataFrame(rows), SHARED_DIR / rel)
    return rel


# Category-level tables shipped for offline reproduction (no per-image paths).
MAIN_RESULT_GLOBS = (
    "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
    "long_tailed_powerlaw_fits_filtered-0.27_valid129.csv",
    "category_wise_cosine_similarity_*_filtered-0.27_valid129.csv",
    "bv_things_rdm_comparison_summary_filtered-0.27_valid129.csv",
    "bv_things_rdm_order_*_filtered-0.27_valid129.csv",
    "bv_vs_things_cluster_strength_valid129.csv",
    "bv_vs_things_cluster_strength_rankcorr_valid129.csv",
    "bv_vs_things_cluster_strength_by_cdi_cluster_*_valid129.csv",
    "bv_vs_things_cdi_bin_significance_compact_*_valid129.csv",
    "main_cdi_bin_bv_vs_things_shuffle_*_valid129.csv",
    "within_between_rank_correlations_valid129.csv",
    "category_within_between_*_valid129.csv",
    "cluster_within_between_*_valid129.csv",
    "correlation_summary_*_valid129.csv",
    "binary_template_vs_real_rdm_correlations_valid129.csv",
    "binary_template_within_between_means_valid129.csv",
    "dinov3_vs_babydinov3_*_valid129.csv",
    "things_dinov3_vs_babydinov3_umap_cdi_valid129_coords.csv",
    "clip_threshold_sensitivity_valid129.csv",
    "precision_vs_detection_proportion_*_valid129.csv",
    "precision_sensitivity_*_valid129.csv",
)

TOP8_MAIN_GLOBS = (
    "individual_rdm_pairwise_correlations_top8_densest_*_valid129.csv",
    "individual_rdm_top8_subject_level_mean_correlations_*_valid129.csv",
    "individual_rdm_top8_upper_lower_bounds_*_valid129.csv",
    "individual_rdm_pairwise_correlation_summary_clip_dinov3_filtered-0.27_valid129.csv",
)


def _collect_globs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(sorted(root.glob(pat)))
    return found


def export_main_results() -> list[str]:
    written: list[str] = []
    for src in _collect_globs(MAIN_RESULTS, MAIN_RESULT_GLOBS):
        rel = f"results_valid129/{src.name}"
        copy_file(src, SHARED_DIR / rel)
        written.append(rel)

    for src in _collect_globs(MAIN_RESULTS, TOP8_MAIN_GLOBS):
        df = anonymize_dataframe(pd.read_csv(src))
        rel = f"results_valid129/{anonymize_filename(src.name)}"
        write_csv(df, SHARED_DIR / rel)
        written.append(rel)
    return written


def export_category_inputs() -> list[str]:
    written: list[str] = []
    copies = [
        (DATA_DIR / "included_categories_valid129.txt", "category_lists/included_categories_valid129.txt"),
        (DATA_DIR / "included_categories_valid85.txt", "category_lists/included_categories_valid85.txt"),
        (
            DATA_DIR / "long_tailed_dist_prop_included_categories_valid129.csv",
            "inputs/long_tailed_dist_prop_included_categories_valid129.csv",
        ),
        (
            DATA_DIR / "long_tailed_dist_prop_included_categories_valid85.csv",
            "inputs/long_tailed_dist_prop_included_categories_valid85.csv",
        ),
        (
            DATA_DIR / "average_precision_for_included_categories.csv",
            "inputs/average_precision_for_included_categories.csv",
        ),
    ]
    for src, rel in copies:
        if src.exists():
            copy_file(src, SHARED_DIR / rel)
            written.append(rel)

    filtered = MAIN_RESULTS / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv"
    if filtered.exists():
        rel = "inputs/long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv"
        copy_file(filtered, SHARED_DIR / rel)
        written.append(rel)
    return written


def export_embeddings() -> list[str]:
    written: list[str] = []
    for category_set in ("valid129", "valid85"):
        src_root = EXEMPLAR_DIR / category_set
        if not src_root.is_dir():
            continue
        for src in sorted(src_root.glob("*_exemplar_avg_zscore_within_*.csv")):
            rel = f"embeddings/{category_set}/{src.name}"
            copy_file(src, SHARED_DIR / rel)
            written.append(rel)
        for name in ("exemplar_embedding_run.json", "things_exemplar_embedding_run.json"):
            src = src_root / name
            if src.exists():
                rel = f"embeddings/{category_set}/{name}"
                sanitize_run_json(src, SHARED_DIR / rel)
                written.append(rel)
    return written


def export_animal_depiction() -> list[str]:
    """Category-level animal depiction proportions only (no image paths / IDs)."""
    src = (
        PROJECT_ROOT
        / "annotation"
        / "annotation_data"
        / "7k_animal_depiction_annotation_04302026.csv"
    )
    if not src.exists():
        return []
    df = pd.read_csv(src)
    if "category" not in df.columns or "label" not in df.columns:
        return []
    label = df["label"].astype(str).str.strip().str.lower()
    rows = []
    for category, g in df.groupby("category"):
        n = len(g)
        n_skip = (label.loc[g.index] == "skip").sum()
        n_real = (label.loc[g.index] == "real-life").sum()
        n_not_real = (label.loc[g.index] == "not real-life").sum()
        n_non_skip = n - n_skip
        rows.append(
            {
                "category": category,
                "n_annotated": n,
                "prop_skip": n_skip / n if n else 0.0,
                "prop_real_life": n_real / n if n else 0.0,
                "prop_not_real_life": n_not_real / n if n else 0.0,
                "prop_not_real_life_given_non_skip": (
                    n_not_real / n_non_skip if n_non_skip else float("nan")
                ),
            }
        )
    rel = "inputs/animal_depiction_label_proportions_by_category.csv"
    write_csv(pd.DataFrame(rows).sort_values("category"), SHARED_DIR / rel)
    return [rel]


def build_manifest(files: list[str]) -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Anonymized intermediate tables for the BabyView Objects preprint "
            "(June 2026 submission). participant_01–08 are pseudonyms for the "
            "eight densest-recording children; real IDs are not included."
        ),
        "participant_note": (
            "top8_rank_densest_first=1 is the densest child in the manuscript "
            "top-8 set (see metadata/participant_registry_top8.csv)."
        ),
        "files": sorted(set(files)),
    }
    (SHARED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    if SHARED_DIR.exists():
        shutil.rmtree(SHARED_DIR)
    SHARED_DIR.mkdir(parents=True)

    all_written: list[str] = []
    all_written.extend(export_category_inputs())
    all_written.extend(export_embeddings())
    all_written.extend(export_main_results())
    all_written.extend(export_top8_supplemental())
    all_written.append(export_participant_registry())
    all_written.extend(export_animal_depiction())

    build_manifest(all_written)
    print(f"Wrote {len(set(all_written))} files under {SHARED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {(SHARED_DIR / 'MANIFEST.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
