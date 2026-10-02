#!/usr/bin/env python3
"""Manuscript-style analyses on the VEDB packed shards.

Stage 0 category means (CLIP + DINOv3, valid129 and valid85), then the pooled
comparisons from notebooks 01, 02, 03, and 05. THINGS tables are the existing
manuscript ones. Per-child notebooks and BabyView rater-precision checks are
not run: these are adult headcam sessions.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vedb")
os.environ["BV_CLIP_FILTER_LIST_THRESHOLD"] = "0.27"
os.environ["BV_ANALYSIS_MODELS"] = "clip,dinov3"

VEDB_DIR = Path(__file__).resolve().parent
REPO = VEDB_DIR.parents[1]
MANUSCRIPT = REPO / "analysis" / "manuscript-2026"
EMB_ROOT = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full")
EXEMPLAR = VEDB_DIR / "exemplar_set_embeddings"
ANALYSES = VEDB_DIR / "analyses"
THINGS_SRC = MANUSCRIPT / "exemplar_set_embeddings"
PY = os.environ.get("PY", sys.executable)
N_PERM = 2000
MODELS = ("clip", "dinov3")
SETS = ("valid129", "valid85")

os.environ["BV_EXEMPLAR_EMBED_ROOT"] = str(EXEMPLAR)
os.environ["BV_EMBEDDINGS_BASE"] = str(EMB_ROOT)

sys.path.insert(0, str(MANUSCRIPT))
sys.path.insert(0, str(MANUSCRIPT / "scripts"))


def stage0() -> None:
    cmd = [
        PY,
        str(MANUSCRIPT / "scripts" / "exemplar_set_zscore_embeddings.py"),
        "--source",
        "packed",
        "--all-category-sets",
        "--emb-root",
        str(EMB_ROOT),
        "--models",
        "clip",
        "dinov3",
    ]
    env = os.environ.copy()
    env["BV_EXEMPLAR_EMBED_ROOT"] = str(EXEMPLAR)
    env["BV_EMBEDDINGS_BASE"] = str(EMB_ROOT)
    print("stage 0", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(MANUSCRIPT), env=env)


def link_things() -> None:
    for category_set in SETS:
        dest_dir = EXEMPLAR / category_set
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_dir = THINGS_SRC / category_set
        for model in MODELS:
            name = f"things_{model}_exemplar_avg_zscore_within_{category_set}.csv"
            dest = dest_dir / name
            src = src_dir / name
            if not src.is_file():
                raise FileNotFoundError(src)
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(src)
            print("linked", dest, flush=True)


def cluster_within_between(categories, semantics, rdm):
    import numpy as np
    import pandas as pd

    sem_to_idx = defaultdict(list)
    for i, sem in enumerate(semantics):
        sem_to_idx[sem].append(i)
    rows = []
    n = len(categories)
    all_idx = np.arange(n)
    for sem, idx_list in sorted(sem_to_idx.items()):
        idx = np.array(idx_list, dtype=int)
        other_idx = all_idx[~np.isin(all_idx, idx)]
        if len(idx) >= 2:
            sub = rdm[np.ix_(idx, idx)]
            iu = np.triu_indices_from(sub, k=1)
            within_vals = sub[iu]
        else:
            within_vals = np.array([], dtype=float)
        if len(idx) and len(other_idx):
            between_vals = rdm[np.ix_(idx, other_idx)].reshape(-1)
        else:
            between_vals = np.array([], dtype=float)
        rows.append(
            {
                "cdi_semantic": sem,
                "n_categories": int(len(idx)),
                "within_mean": float(np.mean(within_vals)) if within_vals.size else np.nan,
                "within_median": float(np.median(within_vals)) if within_vals.size else np.nan,
                "between_mean": float(np.mean(between_vals)) if between_vals.size else np.nan,
                "between_median": float(np.median(between_vals)) if between_vals.size else np.nan,
                "n_within_pairs": int(within_vals.size),
                "n_between_pairs": int(between_vals.size),
            }
        )
    return pd.DataFrame(rows)


def load_matrix(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    cats = df["category"].astype(str).str.strip().str.lower().tolist()
    X = df.drop(columns=["category"]).to_numpy(dtype="float64")
    return cats, X


def common_embeddings(category_set: str, model: str):
    import numpy as np

    inc = [
        ln.strip().lower()
        for ln in (REPO / "data" / f"included_categories_{category_set}.txt").read_text().splitlines()
        if ln.strip()
    ]
    vedb_cats, vedb_X = load_matrix(
        EXEMPLAR / category_set / f"bv_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    th_cats, th_X = load_matrix(
        EXEMPLAR / category_set / f"things_{model}_exemplar_avg_zscore_within_{category_set}.csv"
    )
    vedb_i = {c: i for i, c in enumerate(vedb_cats)}
    th_i = {c: i for i, c in enumerate(th_cats)}
    cats = [c for c in inc if c in vedb_i and c in th_i]
    V = np.stack([vedb_X[vedb_i[c]] for c in cats])
    T = np.stack([th_X[th_i[c]] for c in cats])
    return cats, V, T


def cosine_and_rdm(category_set: str) -> None:
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics.pairwise import cosine_similarity

    from manuscript_config import load_cdi_semantic_map

    results = ANALYSES / category_set / "results"
    results.mkdir(parents=True, exist_ok=True)
    sem_map = load_cdi_semantic_map()
    rdm_rows = []
    for model in MODELS:
        cats, V, T = common_embeddings(category_set, model)
        rows = []
        for i, cat in enumerate(cats):
            sim = float(cosine_similarity(V[i : i + 1], T[i : i + 1])[0, 0])
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
            results / f"category_wise_cosine_similarity_{model}_filtered-0.27_{category_set}.csv",
            index=False,
        )
        print(
            f"cosine {category_set} {model}: n={len(cos)} mean={cos['cosine_similarity'].mean():.3f}",
            flush=True,
        )
        vedb_rdm = squareform(pdist(V, metric="cosine"))
        th_rdm = squareform(pdist(T, metric="cosine"))
        iu = np.triu_indices(len(cats), k=1)
        pr, pp = pearsonr(vedb_rdm[iu], th_rdm[iu])
        sr, sp = spearmanr(vedb_rdm[iu], th_rdm[iu])
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
            }
        )
        sems = [sem_map.get(c, "other") for c in cats]
        within = cluster_within_between(cats, sems, vedb_rdm)
        within_th = cluster_within_between(cats, sems, th_rdm)
        merged = within.merge(within_th, on="cdi_semantic", suffixes=("_vedb", "_things"))
        merged["delta_between_minus_within_vedb"] = merged["between_mean_vedb"] - merged["within_mean_vedb"]
        merged["delta_between_minus_within_things"] = (
            merged["between_mean_things"] - merged["within_mean_things"]
        )
        from bv_things_cdi_shuffle_inference import run_bv_things_shuffle_inference

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
        wide = merged.rename(
            columns={
                "delta_between_minus_within_vedb": "delta_between_minus_within_bv",
                "within_mean_vedb": "within_mean",
                "between_mean_vedb": "between_mean",
            }
        )
        # run_bv_things_shuffle_inference rebuilds deltas from the RDMs.
        infer = run_bv_things_shuffle_inference(
            cats,
            sems,
            vedb_rdm,
            th_rdm,
            order,
            wide,
            n_perm=N_PERM,
            seed=0,
            cluster_within_between=cluster_within_between,
        )
        infer["main_wide"].to_csv(
            results / f"cdi_cluster_within_between_{model}_filtered-0.27_{category_set}.csv",
            index=False,
        )
        print(f"within/between {category_set} {model} n_perm={N_PERM}", flush=True)
    pd.DataFrame(rdm_rows).to_csv(
        results / f"vedb_things_rdm_correlations_filtered-0.27_{category_set}.csv",
        index=False,
    )


def long_tail() -> None:
    import run_long_tail_frame_prevalence as lt
    from frame_prevalence import resolve_detections

    for category_set in SETS:
        lt.RUN_ROOTS[category_set] = ANALYSES / category_set
    det, packed = resolve_detections()
    if not packed:
        raise RuntimeError("expected packed VEDB index")
    semantic_map = lt.semantic_map_from_cdi()
    for category_set in SETS:
        lt.run_category_set(category_set, semantic_map, det, packed=True)


def main() -> None:
    stage0()
    link_things()
    long_tail()
    for category_set in SETS:
        cosine_and_rdm(category_set)
    print("done", ANALYSES, flush=True)


if __name__ == "__main__":
    main()
