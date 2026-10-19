#!/usr/bin/env python3
"""Build example-page practice CSV + copied images for one CDI category (same pattern as table)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
from sample_object_crops_variability import (  # noqa: E402
    DEFAULT_CROPPED_DIR,
    DEFAULT_FILTER_LIST,
    write_example_page_correct_detections,
)

DEFAULT_TRIAL_BUCKET_CSV = (
    PROJECT_ROOT / "annotation" / "sampled_object_crops_100_bucket_assignments_100ex_8subj.csv"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--category",
        type=str,
        required=True,
        help="CDI category name (e.g. book, bottle, table).",
    )
    ap.add_argument(
        "--trial-bucket-csv",
        type=Path,
        default=DEFAULT_TRIAL_BUCKET_CSV,
        help="Bucket assignment CSV; regular-trial stems are excluded from practice picks.",
    )
    ap.add_argument(
        "--cropped-dir",
        type=Path,
        default=DEFAULT_CROPPED_DIR,
        help="Babyview crops root (category subfolders).",
    )
    ap.add_argument(
        "--filter-list",
        type=Path,
        default=DEFAULT_FILTER_LIST,
        help="CLIP-filtered embedding list (one path per line).",
    )
    ap.add_argument(
        "--n-examples",
        type=int,
        default=12,
        help="How many highest-confidence crops to export (default 12).",
    )
    ap.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Only write the CSV; do not copy JPEGs into the _images folder.",
    )
    args = ap.parse_args()

    cat = args.category.strip().lower()
    out_csv = PROJECT_ROOT / "annotation" / f"example_page_correct_detections_{cat}.csv"
    out_dir = PROJECT_ROOT / "annotation" / f"example_page_correct_detections_{cat}_images"

    trial_df = pd.read_csv(args.trial_bucket_csv)
    excluded = set(trial_df.loc[trial_df["trial_type"] == "regular", "stem"].astype(str))

    n = write_example_page_correct_detections(
        sampled_df=None,
        category=cat,
        n_examples=args.n_examples,
        output_csv=out_csv,
        copy_images=not args.no_copy_images,
        output_dir=out_dir,
        excluded_stems=excluded,
        cropped_dir=args.cropped_dir,
        filter_list_path=args.filter_list,
    )
    print(f"Wrote {n} rows -> {out_csv}")
    if not args.no_copy_images:
        print(f"Copied images -> {out_dir}")


if __name__ == "__main__":
    main()
