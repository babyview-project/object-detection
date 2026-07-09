#!/usr/bin/env python3
"""Verify Stage 0 outputs in exemplar_set_embeddings/ before running notebooks 02–05.

  python scripts/check_exemplar_stage.py
  python scripts/check_exemplar_stage.py --category-set valid85
  python scripts/check_exemplar_stage.py --models clip dinov3
"""
from __future__ import annotations

import argparse
import sys

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR

from manuscript_config import (
    CATEGORY_SET_FILES,
    REQUIRED_EMBEDDING_MODELS,
    missing_embedding_tables,
    required_embedding_tables,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--category-set",
        choices=tuple(CATEGORY_SET_FILES),
        default="valid129",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=("clip", "dinov3", "babydinov3"),
        default=list(REQUIRED_EMBEDDING_MODELS),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = tuple(args.models)
    missing = missing_embedding_tables(args.category_set, models)
    expected = required_embedding_tables(args.category_set, models)

    print(f"Category set: {args.category_set}")
    print(f"Models: {', '.join(models)}")
    print(f"Expected {len(expected)} tables under exemplar_set_embeddings/{args.category_set}/")

    if not missing:
        print("OK — all required embedding CSVs exist.")
        return 0

    print(f"MISSING {len(missing)} file(s):")
    for p in missing:
        print(f"  - {p.name}")
    print("\nRun Stage 0: see 00_build_exemplar_embeddings.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
