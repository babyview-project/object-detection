#!/usr/bin/env python3
"""Smoke-test that --from-zip metrics match committed valid7018 reference tables.

Writes to a temporary directory (does not overwrite analysis/ccn-2026/valid7018/).

Examples::

  python analysis/ccn-2026/scripts/verify_valid7018_reproduction.py
  python analysis/ccn-2026/scripts/verify_valid7018_reproduction.py --with-figures
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent
SCRIPTS_DIR = CCN_DIR / "scripts"
REFERENCE_DIR = CCN_DIR / "valid7018"
DEFAULT_ZIP = REPO_ROOT / "data" / "shared_data_ccn_2026" / "embeddings" / "valid7018_bv_embeddings.zip"

RHO_TOLERANCE = 1e-5
COUNT_FIELDS = (
    ("n_exemplars_loaded_clip", 7018),
    ("n_exemplars_loaded_dinov3", 7018),
    ("n_categories_merged", 85),
)

ABSTRACT_FIGURE_STEMS = (
    "fig1A_valid7018_montages_low_to_high_global",
    "fig1B_valid7018_tsne_dinov3",
    "fig1B_valid7018_tsne_dinov3_semantic_diverse",
    "fig1C_valid7018_cross_model_k5",
    "fig_explore_frequency_vs_global_robustness_2x2",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="Committed metrics directory (default: analysis/ccn-2026/valid7018/)",
    )
    p.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help="Embedding zip (default: data/shared_data_ccn_2026/embeddings/...)",
    )
    p.add_argument(
        "--with-figures",
        action="store_true",
        help="Also regenerate figures to scratch dirs and check abstract panel stems exist",
    )
    p.add_argument(
        "--rho-tolerance",
        type=float,
        default=RHO_TOLERANCE,
        help=f"Max |Δρ| allowed vs reference (default: {RHO_TOLERANCE})",
    )
    return p.parse_args()


def _run_compute(out_dir: Path, zip_path: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "compute_valid7018_local_global.py"),
        "--from-zip",
        "--out-dir",
        str(out_dir),
        "--zip-path",
        str(zip_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _compare_correlations(candidate_dir: Path, reference_dir: Path, tol: float) -> list[str]:
    errors: list[str] = []
    ref = pd.read_csv(reference_dir / "bv_valid7018_correlations_k5.csv")
    cand = pd.read_csv(candidate_dir / "bv_valid7018_correlations_k5.csv")
    merged = ref.merge(cand, on="comparison", suffixes=("_ref", "_cand"))
    if len(merged) != len(ref):
        errors.append(f"correlation row count mismatch: ref={len(ref)} cand={len(cand)}")
        return errors
    for _, row in merged.iterrows():
        delta = abs(float(row["spearman_rho_cand"]) - float(row["spearman_rho_ref"]))
        if delta > tol:
            errors.append(
                f"{row['comparison']}: |Δρ|={delta:.2e} > {tol} "
                f"(ref={row['spearman_rho_ref']:.6f}, cand={row['spearman_rho_cand']:.6f})"
            )
        if int(row["n_categories_cand"]) != int(row["n_categories_ref"]):
            errors.append(f"{row['comparison']}: n_categories mismatch")
    return errors


def _compare_run_json(candidate_dir: Path, reference_dir: Path) -> list[str]:
    errors: list[str] = []
    ref = json.loads((reference_dir / "valid7018_run.json").read_text())
    cand = json.loads((candidate_dir / "valid7018_run.json").read_text())
    for field, expected in COUNT_FIELDS:
        value = cand.get(field)
        if value != expected:
            errors.append(f"valid7018_run.json {field}={value!r}, expected {expected}")
        ref_value = ref.get(field)
        if field.startswith("n_") and ref_value is not None and value != ref_value:
            errors.append(f"valid7018_run.json {field} differs from reference ({value} vs {ref_value})")
    return errors


def _compare_category_metrics(candidate_dir: Path, reference_dir: Path, tol: float = 1e-4) -> list[str]:
    errors: list[str] = []
    for name in ("clip", "dinov3"):
        ref = pd.read_csv(reference_dir / f"bv_valid7018_{name}_local_global_k5.csv")
        cand = pd.read_csv(candidate_dir / f"bv_valid7018_{name}_local_global_k5.csv")
        if list(ref.columns) != list(cand.columns):
            errors.append(f"{name}: CSV column mismatch")
            continue
        ref = ref.sort_values("category").reset_index(drop=True)
        cand = cand.sort_values("category").reset_index(drop=True)
        if len(ref) != len(cand):
            errors.append(f"{name}: category count {len(cand)} vs ref {len(ref)}")
            continue
        if not np.all(ref["category"].values == cand["category"].values):
            errors.append(f"{name}: category labels differ")
        for col in ("global_dispersion", "mean_knn_dist"):
            delta = np.abs(ref[col].to_numpy() - cand[col].to_numpy()).max()
            if delta > tol:
                errors.append(f"{name}: max |Δ{col}|={delta:.2e} > {tol}")
    return errors


def _run_figures(abstract_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_valid7018_paper_figures.py"),
        "--from-zip",
        "--abstract-dir",
        str(abstract_dir),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _check_figure_outputs(abstract_dir: Path) -> list[str]:
    errors: list[str] = []
    for stem in ABSTRACT_FIGURE_STEMS:
        for ext in ("png", "pdf"):
            path = abstract_dir / f"{stem}.{ext}"
            if not path.is_file():
                errors.append(f"missing abstract panel: {path.relative_to(REPO_ROOT)}")
    selection = abstract_dir / "valid7018_figure_category_selection.csv"
    if not selection.is_file():
        errors.append(f"missing {selection.relative_to(REPO_ROOT)}")
    return errors


def main() -> int:
    args = parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    zip_path = args.zip_path.expanduser().resolve()

    if not zip_path.is_file():
        print(f"ERROR: embedding zip not found: {zip_path}", file=sys.stderr)
        return 1
    for required in (
        "bv_valid7018_correlations_k5.csv",
        "bv_valid7018_clip_local_global_k5.csv",
        "bv_valid7018_dinov3_local_global_k5.csv",
        "valid7018_run.json",
    ):
        if not (reference_dir / required).is_file():
            print(f"ERROR: reference missing {reference_dir / required}", file=sys.stderr)
            return 1

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="valid7018_verify_") as tmp:
        tmp_dir = Path(tmp)
        metrics_dir = tmp_dir / "metrics"
        print(f"Recomputing metrics in {metrics_dir} ...")
        _run_compute(metrics_dir, zip_path)

        errors.extend(_compare_correlations(metrics_dir, reference_dir, args.rho_tolerance))
        errors.extend(_compare_run_json(metrics_dir, reference_dir))
        errors.extend(_compare_category_metrics(metrics_dir, reference_dir))

        if args.with_figures:
            abstract_dir = tmp_dir / "abstract_figures"
            abstract_dir.mkdir()
            print("Regenerating figures (publish to temp abstract dir) ...")
            _run_figures(abstract_dir)
            errors.extend(_check_figure_outputs(abstract_dir))

    if errors:
        print("VERIFY FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("OK: valid7018 --from-zip reproduction matches committed reference.")
    if args.with_figures:
        print("OK: abstract figure panels regenerated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
