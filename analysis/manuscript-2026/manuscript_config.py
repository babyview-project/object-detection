"""Shared paths for analysis/manuscript-2026 (BabyView Objects manuscript)."""
from __future__ import annotations

from pathlib import Path

# This file lives at analysis/manuscript-2026/manuscript_config.py
MANUSCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MANUSCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
EXEMPLAR_EMBED_ROOT = MANUSCRIPT_DIR / "exemplar_set_embeddings"

# Back-compat alias used in older notebooks/scripts
PREPRINT_DIR = MANUSCRIPT_DIR

CATEGORY_SET_FILES: dict[str, Path] = {
    "valid85": DATA_DIR / "included_categories_valid85.txt",
    "valid129": DATA_DIR / "included_categories_valid129.txt",
}

# Tables expected by notebooks 02–05 (main text uses valid129 + three backbones).
REQUIRED_EMBEDDING_MODELS: tuple[str, ...] = ("clip", "dinov3", "babydinov3")


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
