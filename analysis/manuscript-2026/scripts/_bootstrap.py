"""Path setup for scripts in analysis/manuscript-2026/scripts/.

Import at the top of each script (after __future__ imports):

    from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = MANUSCRIPT_DIR.parent.parent
PREPRINT_DIR = MANUSCRIPT_DIR

if str(MANUSCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MANUSCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
