"""Feature-wise global normalization for valid7018 per-crop embeddings.

Matches ``05_normalize_grouped_embeddings.ipynb``: compute per-dimension mean/std
from grouped age-month ``*_month_level_avg.npy`` files (0.27-filtered, excluding
subject 00270001), then apply ``(x - mu) / sigma`` to each valid7018 crop vector.

The committed ``*_filtered-0.27_normalized`` trees store already-normalized
age-month vectors; valid7018 uses the same statistics on per-crop ``.npy``
from ``clip_embeddings_new`` / DINO flat dirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

EXCLUDED_SUBJECT = "00270001"
DEFAULT_THRESHOLD = "0.27"


def grouped_embedding_dirs(emb_base: Path, threshold: str = DEFAULT_THRESHOLD) -> dict[str, Path]:
    emb_base = emb_base.expanduser()
    return {
        "clip_stats_source": emb_base / f"clip_embeddings_grouped_by_age-mo_filtered-{threshold}",
        "dinov3_stats_source": emb_base / f"dinov3_embeddings_grouped_by_age-mo_filtered-{threshold}",
        "clip_normalized_reference": emb_base
        / f"clip_embeddings_grouped_by_age-mo_filtered-{threshold}_normalized",
        "dinov3_normalized_reference": emb_base
        / f"dinov3_embeddings_grouped_by_age-mo_filtered-{threshold}_normalized",
    }


def _is_month_level_avg(path: Path) -> bool:
    return path.suffix.lower() == ".npy" and path.stem.endswith("_month_level_avg")


def _subject_from_month_level_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[0] if parts else ""


def fit_featurewise_stats_from_grouped_dir(
    grouped_dir: Path,
    exclude_subject: str = EXCLUDED_SUBJECT,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mu, sigma, meta) from all month-level avg .npy under grouped_dir."""
    grouped_dir = grouped_dir.expanduser()
    if not grouped_dir.is_dir():
        raise FileNotFoundError(f"Grouped embedding dir not found: {grouped_dir}")

    files: list[Path] = []
    for cat_dir in sorted(grouped_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for npy in sorted(cat_dir.glob("*.npy")):
            if not _is_month_level_avg(npy):
                continue
            if exclude_subject and _subject_from_month_level_stem(npy.stem) == exclude_subject:
                continue
            files.append(npy)
            if max_files is not None and len(files) >= max_files:
                break
        if max_files is not None and len(files) >= max_files:
            break

    if not files:
        raise RuntimeError(f"No month_level_avg .npy files under {grouped_dir}")

    dim: int | None = None
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    for path in tqdm(files, desc=f"Fit norm stats {grouped_dir.name}", unit="file"):
        v = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64).ravel()
        if dim is None:
            dim = v.shape[0]
            mean = np.zeros(dim, dtype=np.float64)
            m2 = np.zeros(dim, dtype=np.float64)
        count += 1
        delta = v - mean
        mean += delta / count
        delta2 = v - mean
        m2 += delta * delta2

    assert mean is not None and m2 is not None and dim is not None
    sigma = np.sqrt(m2 / max(count - 1, 1))
    sigma = np.where(sigma > 1e-10, sigma, 1.0)
    meta = {
        "stats_source_dir": str(grouped_dir),
        "n_files": count,
        "embedding_dim": int(dim),
        "exclude_subject": exclude_subject,
        "file_pattern": "*_month_level_avg.npy",
    }
    return mean, sigma, meta


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


def fit_and_save_norm_stats(
    emb_base: Path,
    out_path: Path,
    threshold: str = DEFAULT_THRESHOLD,
    exclude_subject: str = EXCLUDED_SUBJECT,
) -> dict:
    dirs = grouped_embedding_dirs(emb_base, threshold=threshold)
    payload: dict = {
        "normalization": "featurewise_global_from_grouped_age_month",
        "threshold": threshold,
        "exclude_subject": exclude_subject,
        "note": (
            "mu/sigma fit on grouped age-month avg .npy (notebook 05); applied to valid7018 per-crop vectors."
        ),
        "models": {},
    }
    for model, key in [("clip", "clip_stats_source"), ("dinov3", "dinov3_stats_source")]:
        mu, sigma, meta = fit_featurewise_stats_from_grouped_dir(
            dirs[key], exclude_subject=exclude_subject
        )
        payload["models"][model] = {
            **meta,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


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
