#!/usr/bin/env python3
"""Execute a notebook's code cells in-process (no jupyter kernel / nbclient)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-j7yang")

import matplotlib

matplotlib.use("Agg")


def _cell_source(cell: dict) -> str:
    raw = cell.get("source", "")
    if isinstance(raw, str):
        return raw
    return "".join(raw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebook", type=Path)
    p.add_argument("--cwd", type=Path, default=None)
    args = p.parse_args()
    nb_path = args.notebook.resolve()
    cwd = (args.cwd or nb_path.parent).resolve()
    os.chdir(cwd)
    sys.path.insert(0, str(cwd))
    sys.path.insert(0, str(cwd / "scripts"))

    from IPython.core.interactiveshell import InteractiveShell

    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    shell = InteractiveShell.instance()
    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    print(f"Executing {nb_path.name}  ({n_code} code cells, cwd={cwd})", flush=True)
    done = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        if not src.strip():
            continue
        done += 1
        print(f"\n===== {nb_path.name} cell {i} ({done}/{n_code}) =====", flush=True)
        result = shell.run_cell(src)
        if result.error_before_exec is not None or result.error_in_exec is not None:
            err = result.error_before_exec or result.error_in_exec
            print(f"FAILED {nb_path.name} cell {i}: {err!r}", flush=True)
            traceback.print_exception(type(err), err, err.__traceback__)
            return 1
    print(f"OK {nb_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
