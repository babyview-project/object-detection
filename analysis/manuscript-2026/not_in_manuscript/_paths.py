"""Path helpers for analyses outside the submitted manuscript."""
from __future__ import annotations

from pathlib import Path

NOT_IN_MANUSCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = NOT_IN_MANUSCRIPT_DIR.parent
PROJECT_ROOT = MANUSCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Back-compat alias used in older notebooks
PREPRINT_DIR = MANUSCRIPT_DIR
