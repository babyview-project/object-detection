"""Shared paths for analysis/manuscript-2026 (BabyView Objects manuscript)."""
from __future__ import annotations

import os
from pathlib import Path

# This file lives at analysis/manuscript-2026/manuscript_config.py
MANUSCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MANUSCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
EXEMPLAR_EMBED_ROOT = Path(
    os.environ.get("BV_EXEMPLAR_EMBED_ROOT", str(MANUSCRIPT_DIR / "exemplar_set_embeddings"))
).expanduser()

# Back-compat alias used in older notebooks/scripts
PREPRINT_DIR = MANUSCRIPT_DIR

CATEGORY_SET_FILES: dict[str, Path] = {
    "valid85": DATA_DIR / "included_categories_valid85.txt",
    "valid129": DATA_DIR / "included_categories_valid129.txt",
}

# Tables expected by notebooks 02–05 (main text uses valid129 + three backbones).
REQUIRED_EMBEDDING_MODELS: tuple[str, ...] = ("clip", "dinov3", "babydinov3")

# Result-folder suffix. Default keeps the 868-hour manuscript trees.
# Example: BV_OUTPUT_TAG=2025.2 → main_results_valid129s_2025.2/
DEFAULT_OUTPUT_TAG = "04302026"


def resolve_manuscript_dir(cwd: Path | None = None) -> Path:
    """Find manuscript-2026 from repo root or when cwd is the analysis folder."""
    base = cwd or Path.cwd()
    candidates = [
        base,
        base / "analysis" / "manuscript-2026",
        base.parent / "manuscript-2026",
    ]
    for c in candidates:
        if (c / "01_long_tailed_distribution.ipynb").exists():
            return c.resolve()
    return MANUSCRIPT_DIR


def output_tag() -> str:
    return os.environ.get("BV_OUTPUT_TAG", DEFAULT_OUTPUT_TAG).strip() or DEFAULT_OUTPUT_TAG


def analysis_models(default: tuple[str, ...] = REQUIRED_EMBEDDING_MODELS) -> tuple[str, ...]:
    """Backbones to run. 2025.2 packed store has CLIP + DINOv3 only."""
    raw = os.environ.get("BV_ANALYSIS_MODELS", "").strip()
    if not raw:
        return default
    return tuple(m.strip().lower() for m in raw.split(",") if m.strip())


def output_run_root(category_set: str, manuscript_dir: Path | None = None) -> Path:
    """valid129 → main_results_valid129s_{tag}/; valid85 → supplemental_results_valid85cats_{tag}/."""
    if category_set not in CATEGORY_SET_FILES:
        raise ValueError(f"Unknown category_set: {category_set!r}")
    md = manuscript_dir or MANUSCRIPT_DIR
    tag = output_tag()
    if category_set == "valid85":
        return md / f"supplemental_results_valid85cats_{tag}"
    return md / f"main_results_valid129s_{tag}"


def exemplar_embed_dir(category_set: str) -> Path:
    if category_set not in CATEGORY_SET_FILES:
        raise ValueError(f"Unknown category_set: {category_set!r}")
    return EXEMPLAR_EMBED_ROOT / category_set


def bv_embedding_csv(model: str, category_set: str, *, zscore: bool = True) -> Path:
    suffix = "zscore" if zscore else "raw"
    return exemplar_embed_dir(category_set) / (
        f"bv_{model}_exemplar_avg_{suffix}_within_{category_set}.csv"
    )


def things_embedding_csv(model: str, category_set: str, *, zscore: bool = True) -> Path:
    suffix = "zscore" if zscore else "raw"
    return exemplar_embed_dir(category_set) / (
        f"things_{model}_exemplar_avg_{suffix}_within_{category_set}.csv"
    )


def required_embedding_tables(
    category_set: str,
    models: tuple[str, ...] = REQUIRED_EMBEDDING_MODELS,
) -> list[Path]:
    """BV + THINGS z-scored category tables used downstream (notebooks 02–05)."""
    paths: list[Path] = []
    for model in models:
        paths.append(bv_embedding_csv(model, category_set))
        paths.append(things_embedding_csv(model, category_set))
    return paths


def missing_embedding_tables(
    category_set: str,
    models: tuple[str, ...] = REQUIRED_EMBEDDING_MODELS,
) -> list[Path]:
    return [p for p in required_embedding_tables(category_set, models) if not p.is_file()]


def load_cdi_semantic_map() -> dict[str, str]:
    """``uni_lemma`` → CDI semantic domain from ``data/cdi_words.csv``.

    Frequency tables under ``data/long_tailed_dist_prop_included_categories_*.csv``
    are not a semantic source: the unfiltered valid129 file currently copies
    valid85 labels and marks the extra 44 categories as ``other``.
    """
    import pandas as pd

    path = DATA_DIR / "cdi_words.csv"
    df = pd.read_csv(path, usecols=["uni_lemma", "category"])
    df["uni_lemma"] = df["uni_lemma"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    return (
        df.drop_duplicates(subset=["uni_lemma"], keep="first")
        .set_index("uni_lemma")["category"]
        .to_dict()
    )
