# Manuscript helper scripts

Runnable `.py` helpers for Stage 0 (embedding tables), supplement statistics, and
public data export. **`manuscript_config.py` stays in the parent folder** so notebooks
can `from manuscript_config import ...` with cwd = `analysis/manuscript-2026/`.

Run commands from **`analysis/manuscript-2026/`** (e.g. `python scripts/check_exemplar_stage.py`).

Full catalog: **[../SCRIPTS.md](../SCRIPTS.md)**.

Imports between scripts use [`_bootstrap.py`](_bootstrap.py) (`MANUSCRIPT_DIR`, `SCRIPTS_DIR`, `PROJECT_ROOT`).
