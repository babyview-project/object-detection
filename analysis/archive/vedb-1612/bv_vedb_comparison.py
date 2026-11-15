#!/usr/bin/env python3
"""BV-vs-VEDB RDM comparison: same methodology as run_analyses.py's
cosine_and_rdm() (which compares VEDB against THINGS), but compares VEDB
against the real BabyView category embeddings instead -- both datasets went
through the identical CDI-vocabulary YOLOE-detect -> crop -> CLIP-filter ->
embed pipeline (run_cdi_crop_pipeline.py), so no category-mapping step is
needed (unlike ImageNet/LVIS): both already use the literal valid129/valid85
CDI category lists.

Usage:
    python bv_vedb_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

VEDB_DIR = Path(__file__).resolve().parent
REPO = VEDB_DIR.parents[1]
MANUSCRIPT = REPO / "analysis" / "manuscript-2026"
VEDB_EXEMPLAR = VEDB_DIR / "exemplar_set_embeddings"
BV_EXEMPLAR = MANUSCRIPT / "exemplar_set_embeddings"
ANALYSES = VEDB_DIR / "analyses"
N_PERM = 2000
MODELS = ("clip", "dinov3")
SETS = ("valid129", "valid85")

sys.path.insert(0, str(MANUSCRIPT))
sys.path.insert(0, str(MANUSCRIPT / "scripts"))
sys.path.insert(0, str(VEDB_DIR))

from manuscript_config import load_cdi_semantic_map  # noqa: E402
from run_analyses import cluster_within_between  # noqa: E402
from bv_things_cdi_shuffle_inference import run_bv_things_shuffle_inference  # noqa: E402


def load_matrix(path: Path):
    df = pd.read_csv(path)
    cats = df["category"].astype(str).str.strip().str.lower().tolist()
    X = df.drop(columns=["category"]).to_numpy(dtype="float64")
    return cats, X


def common_embeddings(category_set: str, model: str):
    inc = [
        ln.strip().lower()
        for ln in (REPO / "data" / f"included_categories_{category_set}.txt").read_text().splitlines()
        if ln.strip()
    ]
    vedb_cats, vedb_X = load_matrix(
        VEDB_EXEMPLAR / category_set / f"bv_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    bv_cats, bv_X = load_matrix(
        BV_EXEMPLAR / category_set / f"bv_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    vedb_i = {c: i for i, c in enumerate(vedb_cats)}
    bv_i = {c: i for i, c in enumerate(bv_cats)}
    cats = [c for c in inc if c in vedb_i and c in bv_i]
    V = np.stack([vedb_X[vedb_i[c]] for c in cats])
    B = np.stack([bv_X[bv_i[c]] for c in cats])
    return cats, V, B


def bv_vedb_cosine_and_rdm(category_set: str) -> None:
    results = ANALYSES / category_set / "results"
    results.mkdir(parents=True, exist_ok=True)
    sem_map = load_cdi_semantic_map()
    rdm_rows = []
    for model in MODELS:
        cats, V, B = common_embeddings(category_set, model)
        rows = []
        for i, cat in enumerate(cats):
            sim = float(cosine_similarity(V[i : i + 1], B[i : i + 1])[0, 0])
            rows.append(
                {
                    "category": cat,
                    "cdi_semantic": sem_map.get(cat, "other"),
                    "cosine_similarity": sim,
                    "model": model,
                }
            )
        cos = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
        cos.to_csv(
            results / f"bv_vedb_category_wise_cosine_similarity_{model}_filtered-0.27_{category_set}.csv",
            index=False,
        )
        print(
            f"cosine {category_set} {model}: n={len(cos)} mean={cos['cosine_similarity'].mean():.3f}",
            flush=True,
        )

        vedb_rdm = squareform(pdist(V, metric="cosine"))
        bv_rdm = squareform(pdist(B, metric="cosine"))
        iu = np.triu_indices(len(cats), k=1)
        pr, pp = pearsonr(vedb_rdm[iu], bv_rdm[iu])
        sr, sp = spearmanr(vedb_rdm[iu], bv_rdm[iu])
        kr, kp = kendalltau(vedb_rdm[iu], bv_rdm[iu])
        rdm_rows.append(
            {
                "category_set": category_set,
                "model": model,
                "n_categories": len(cats),
                "n_pairs": int(iu[0].size),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
                "kendall_tau": float(kr),
                "kendall_p": float(kp),
            }
        )
        print(
            f"RDM {category_set} {model}: n_cat={len(cats)} pearson_r={pr:.3f} (p={pp:.2e}) "
            f"spearman_rho={sr:.3f} (p={sp:.2e})",
            flush=True,
        )

        sems = [sem_map.get(c, "other") for c in cats]
        within_vedb = cluster_within_between(cats, sems, vedb_rdm)
        within_bv = cluster_within_between(cats, sems, bv_rdm)
        # Suffixes/column names below ("_things", "delta_..._bv") are the literal
        # names run_bv_things_shuffle_inference requires internally -- it was
        # written for BV-vs-THINGS, so its "bv" slot takes our vedb_rdm and its
        # "th"/"things" slot takes our real bv_rdm. These get relabeled to their
        # true identity (vedb_*/bv_*) on the output further down, after the call.
        merged = within_vedb.merge(within_bv, on="cdi_semantic", suffixes=("_vedb", "_things"))
        merged["delta_between_minus_within_vedb"] = merged["between_mean_vedb"] - merged["within_mean_vedb"]
        merged["delta_between_minus_within_things"] = (
            merged["between_mean_things"] - merged["within_mean_things"]
        )
        present = set(merged["cdi_semantic"].tolist())
        order = [
            d
            for d in [
                "animals",
                "body_parts",
                "clothing",
                "food_drink",
                "furniture_rooms",
                "household",
                "outside",
                "people",
                "toys",
                "vehicles",
                "other",
            ]
            if d in present
        ]
        # run_bv_things_shuffle_inference's "bv_rdm" parameter slot gets vedb_rdm
        # here (its internal generic column names -- within_mean, between_mean,
        # delta_between_minus_within_bv -- must exist in `wide` regardless of
        # which dataset actually fills that slot), and "th_rdm" gets the real
        # BV RDM. Its output columns are prefixed bv_*/th_* and suffixed
        # _bv/_things accordingly, matching those slots -- NOT our BV/VEDB
        # identities. Explicit (non-overlapping) rename below fixes that.
        wide = merged.rename(
            columns={
                "delta_between_minus_within_vedb": "delta_between_minus_within_bv",
                "within_mean_vedb": "within_mean",
                "between_mean_vedb": "between_mean",
            }
        )
        infer = run_bv_things_shuffle_inference(
            cats,
            sems,
            vedb_rdm,
            bv_rdm,
            order,
            wide,
            n_perm=N_PERM,
            seed=0,
            cluster_within_between=cluster_within_between,
        )
        raw = infer["main_wide"]
        rename_map = {
            # input-side columns (suffix _bv / _things here means "whatever
            # filled the bv_rdm / th_rdm slot", i.e. vedb / real-bv)
            "n_categories_vedb": "n_categories_vedb",
            "within_median_vedb": "within_median_vedb",
            "within_mean": "within_mean_vedb",
            "between_mean": "between_mean_vedb",
            "between_median_vedb": "between_median_vedb",
            "n_within_pairs_vedb": "n_within_pairs_vedb",
            "n_between_pairs_vedb": "n_between_pairs_vedb",
            "delta_between_minus_within_bv": "delta_between_minus_within_vedb",
            "n_categories_things": "n_categories_bv",
            "within_mean_things": "within_mean_bv",
            "within_median_things": "within_median_bv",
            "between_mean_things": "between_mean_bv",
            "between_median_things": "between_median_bv",
            "n_within_pairs_things": "n_within_pairs_bv",
            "n_between_pairs_things": "n_between_pairs_bv",
            "delta_between_minus_within_things": "delta_between_minus_within_bv_real",
            # shuffle-inference output columns
            "bv_shuffle_null_mean_delta": "vedb_shuffle_null_mean_delta",
            "bv_shuffle_obs_minus_null_mean": "vedb_shuffle_obs_minus_null_mean",
            "bv_shuffle_null_std_delta": "vedb_shuffle_null_std_delta",
            "bv_shuffle_z_obs_minus_null_mean": "vedb_shuffle_z_obs_minus_null_mean",
            "bv_shuffle_p_perm_two_sided": "vedb_shuffle_p_perm_two_sided",
            "bv_shuffle_n_perm_valid": "vedb_shuffle_n_perm_valid",
            "th_shuffle_null_mean_delta": "bv_shuffle_null_mean_delta",
            "th_shuffle_obs_minus_null_mean": "bv_shuffle_obs_minus_null_mean",
            "th_shuffle_null_std_delta": "bv_shuffle_null_std_delta",
            "th_shuffle_z_obs_minus_null_mean": "bv_shuffle_z_obs_minus_null_mean",
            "th_shuffle_p_perm_two_sided": "bv_shuffle_p_perm_two_sided",
            "th_shuffle_n_perm_valid": "bv_shuffle_n_perm_valid",
            "obs_delta_diff_bv_minus_th": "obs_delta_diff_vedb_minus_bv",
            "bv_vs_th_shuffle_p_one_sided_gt": "vedb_vs_bv_shuffle_p_one_sided_gt",
            "bv_vs_th_shuffle_p_two_sided": "vedb_vs_bv_shuffle_p_two_sided",
            "bv_vs_th_shuffle_q_one_sided_gt_fdr_bh": "vedb_vs_bv_shuffle_q_one_sided_gt_fdr_bh",
        }
        # delta_between_minus_within_bv_real is the real-BV delta under the
        # function's internal "_things" slot name; drop the temporary "_real"
        # once renamed so the final column is just delta_between_minus_within_bv.
        out_df = raw.rename(columns=rename_map)
        out_df = out_df.rename(columns={"delta_between_minus_within_bv_real": "delta_between_minus_within_bv"})
        out_df.to_csv(
            results / f"bv_vedb_cdi_cluster_within_between_{model}_filtered-0.27_{category_set}.csv",
            index=False,
        )

        # Shuffle-null quantile bands (2.5/50/97.5%) per domain, for the ribbon
        # plot -- infer['null_bv']/['null_th'] are (n_domain, 3) arrays in
        # `order`'s row order; 'bv' slot = vedb_rdm, 'th' slot = real bv_rdm
        # (see note above), so columns are swapped to their true identity here.
        null_df = pd.DataFrame(
            {
                "cdi_semantic": order,
                "vedb_null_q025": infer["null_bv"][:, 0],
                "vedb_null_q500": infer["null_bv"][:, 1],
                "vedb_null_q975": infer["null_bv"][:, 2],
                "bv_null_q025": infer["null_th"][:, 0],
                "bv_null_q500": infer["null_th"][:, 1],
                "bv_null_q975": infer["null_th"][:, 2],
            }
        )
        null_df.to_csv(
            results / f"bv_vedb_cdi_cluster_shuffle_null_bands_{model}_filtered-0.27_{category_set}.csv",
            index=False,
        )
        print(f"within/between {category_set} {model} n_perm={N_PERM}", flush=True)

    out = results / f"bv_vedb_rdm_correlations_filtered-0.27_{category_set}.csv"
    pd.DataFrame(rdm_rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


def main() -> None:
    for category_set in SETS:
        bv_vedb_cosine_and_rdm(category_set)
    print("done", ANALYSES, flush=True)


if __name__ == "__main__":
    main()
