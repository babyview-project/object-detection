#!/usr/bin/env python3
"""Write binary CDI-template vs real-RDM correlations for all exemplar models.

Matches the commented block in ``05_within_between_cdi_cluster_correlation.ipynb``
(upper triangle, template_dis: within=0, between=1, cosine-distance RDM).

Example:
  CATEGORY_SET=valid129 python compute_binary_template_correlations.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
CATEGORY_SET = os.environ.get("CATEGORY_SET", "valid129").strip()
THRESHOLD_TOKEN = os.environ.get("BV_THINGS_EMBED_THRESHOLD", "0.27").strip()

if CATEGORY_SET == "valid85":
    RESULTS_DIR = PREPRINT_DIR / "supplemental_results_valid85cats_04302026" / "results"
else:
    RESULTS_DIR = PREPRINT_DIR / "main_results_valid129s_04302026" / "results"

EXEMPLAR_EMBED_DIR = PREPRINT_DIR / "exemplar_set_embeddings" / CATEGORY_SET
INCLUDED_CATEGORIES_TXT = PROJECT_ROOT / "data" / f"included_categories_{CATEGORY_SET}.txt"
CDI_SEMANTIC_CSV = PROJECT_ROOT / "data" / f"long_tailed_dist_prop_included_categories_{CATEGORY_SET}.csv"

MODELS = ("clip", "dinov3", "babydinov3")


def load_included_categories(path: Path) -> list[str]:
    return [ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()]


def load_cdi_semantic_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    col = "category" if "category" in df.columns else df.columns[0]
    sem_col = "cdi_semantic" if "cdi_semantic" in df.columns else "semantic"
    return {
        str(r[col]).strip().lower(): str(r[sem_col]).strip().lower()
        for _, r in df.iterrows()
    }


def load_embedding_csv(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    cats = [str(c).strip().lower() for c in df.index.tolist()]
    return cats, df.values.astype(np.float64)


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings.astype(np.float64, copy=False), metric="cosine"))


def safe_corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[mask], b[mask]
    pear_r, pear_p = pearsonr(a2, b2)
    spr_rho, spr_p = spearmanr(a2, b2)
    return {
        "n_pairs": int(mask.sum()),
        "pearson_r_template_dissim_vs_rdm": float(pear_r),
        "pearson_p_template_dissim_vs_rdm": float(pear_p),
        "spearman_rho_template_dissim_vs_rdm": float(spr_rho),
        "spearman_p_template_dissim_vs_rdm": float(spr_p),
    }


def build_binary_cdi_template_rdm(semantics: list[str]) -> np.ndarray:
    sem = np.asarray(semantics)
    template = (sem[:, None] == sem[None, :]).astype(float)
    np.fill_diagonal(template, np.nan)
    return template


def correlate_template_with_rdm(template_dis: np.ndarray, rdm: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices_from(template_dis, k=1)
    t_dis = template_dis[iu]
    r = rdm[iu]
    mask = np.isfinite(t_dis) & np.isfinite(r)
    return safe_corr(t_dis[mask], r[mask])


def main() -> None:
    included = set(load_included_categories(INCLUDED_CATEGORIES_TXT))
    cdi_map = load_cdi_semantic_map(CDI_SEMANTIC_CSV)
    cat_order = load_included_categories(INCLUDED_CATEGORIES_TXT)

    rows: list[dict] = []
    for model in MODELS:
        bv_path = EXEMPLAR_EMBED_DIR / f"bv_{model}_exemplar_avg_zscore_within_{CATEGORY_SET}.csv"
        th_path = EXEMPLAR_EMBED_DIR / f"things_{model}_exemplar_avg_zscore_within_{CATEGORY_SET}.csv"
        bv_cats, bv_emb_raw = load_embedding_csv(bv_path)
        th_cats, th_emb_raw = load_embedding_csv(th_path)
        bv_idx = {c: i for i, c in enumerate(bv_cats)}
        th_idx = {c: i for i, c in enumerate(th_cats)}
        common_set = included & set(bv_cats) & set(th_cats)
        cats = [c for c in cat_order if c in common_set]
        bv_emb = np.stack([bv_emb_raw[bv_idx[c]] for c in cats], axis=0)
        th_emb = np.stack([th_emb_raw[th_idx[c]] for c in cats], axis=0)
        sems = [cdi_map.get(c, "other") for c in cats]
        template_dis = 1.0 - build_binary_cdi_template_rdm(sems)
        for source, rdm in [("babyview", compute_rdm(bv_emb)), ("things", compute_rdm(th_emb))]:
            rows.append({"model": model, "source": source, **correlate_template_with_rdm(template_dis, rdm)})

    out = RESULTS_DIR / f"binary_template_vs_real_rdm_correlations_{CATEGORY_SET}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["model", "source"]).reset_index(drop=True).to_csv(out, index=False)
    print(f"Saved: {out} ({len(rows)} rows, threshold={THRESHOLD_TOKEN})")


if __name__ == "__main__":
    main()
