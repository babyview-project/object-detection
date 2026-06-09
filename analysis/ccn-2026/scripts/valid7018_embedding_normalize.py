"""Feature-wise z-score normalization for the valid7018 per-crop cohort.

Methodology (manuscript-aligned z-score, cohort-internal stats):
  - Stack all per-crop vectors for one model across valid85 categories (N=7,018)
  - ``mu = mean(axis=0)``, ``sigma = std(axis=0) + eps`` per embedding dimension
  - Apply ``(x - mu) / sigma`` to each exemplar

Same z-score recipe as ``zscore_rows()`` in ``exemplar_set_zscore_embeddings.py``,
but statistics are fit on the 7,018 validated crops (not grouped age-month files
and not z-scoring across category means).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STATS_EPS = 1e-10
NORMALIZATION_ID = "featurewise_zscore_within_valid7018_cohort"


def stack_category_embeddings(category_embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-category exemplar matrices into (n_exemplars, dim)."""
    if not category_embeddings:
        raise ValueError("category_embeddings is empty")
    blocks = [
        np.asarray(category_embeddings[cat], dtype=np.float64)
        for cat in sorted(category_embeddings.keys())
    ]
    return np.vstack(blocks)


def fit_featurewise_stats_from_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mu, sigma, meta) from exemplar matrix X (n, dim)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"Expected 2-D matrix with >=1 row, got {X.shape}")
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0) + STATS_EPS
    meta = {
        "n_exemplars": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "std_ddof": 0,
        "stats_eps": STATS_EPS,
    }
    return mu, sigma, meta


def fit_featurewise_stats_from_cohort(
    category_embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit mu/sigma on all exemplars pooled across valid85 categories."""
    X = stack_category_embeddings(category_embeddings)
    mu, sigma, meta = fit_featurewise_stats_from_matrix(X)
    meta["n_categories"] = int(len(category_embeddings))
    return mu, sigma, meta


def normalize_category_embeddings(
    category_embeddings: dict[str, np.ndarray],
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cat, x in category_embeddings.items():
        x = np.asarray(x, dtype=np.float64)
        out[cat] = (x - mu) / sigma
    return out


def fit_and_save_cohort_norm_stats(
    clip_emb: dict[str, np.ndarray],
    dino_emb: dict[str, np.ndarray],
    out_path: Path,
) -> dict:
    payload: dict = {
        "normalization": NORMALIZATION_ID,
        "note": (
            "Per-model feature-wise z-score: mu/sigma fit on all 7,018 per-crop "
            "vectors pooled across valid85 categories (cohort-internal)."
        ),
        "models": {},
    }
    for model, emb in [("clip", clip_emb), ("dinov3", dino_emb)]:
        mu, sigma, meta = fit_featurewise_stats_from_cohort(emb)
        payload["models"][model] = {
            **meta,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
        }
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


# Backward-compatible alias used by build/compute scripts.
fit_and_save_norm_stats = fit_and_save_cohort_norm_stats


def load_norm_stats(stats_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = json.loads(stats_path.read_text())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, block in data["models"].items():
        out[model] = (
            np.asarray(block["mu"], dtype=np.float64),
            np.asarray(block["sigma"], dtype=np.float64),
        )
    return out


def apply_norm_stats(
    clip_emb: dict[str, np.ndarray],
    dino_emb: dict[str, np.ndarray],
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mu_c, sig_c = stats["clip"]
    mu_d, sig_d = stats["dinov3"]
    return (
        normalize_category_embeddings(clip_emb, mu_c, sig_c),
        normalize_category_embeddings(dino_emb, mu_d, sig_d),
    )
