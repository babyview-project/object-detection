#!/usr/bin/env python3
"""
Drop OOD attention-check trials from a bucket-assignment CSV (keep BabyView crop rows only).

Rows with trial_type starting with 'ood_' (e.g. ood_attention_check_real_photo) are omitted.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Remove OOD attention-check rows from bucket CSV")
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=here / "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=here / "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv",
    )
    args = ap.parse_args()

    if not args.in_csv.is_file():
        print(f"Input not found: {args.in_csv}")
        return 1

    kept = dropped = 0
    with open(args.in_csv, newline="", encoding="utf-8") as fin, open(
        args.out_csv, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames or "trial_type" not in fieldnames:
            print("CSV missing trial_type column")
            return 1
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            tt = (row.get("trial_type") or "").strip()
            if tt.startswith("ood_"):
                dropped += 1
                continue
            writer.writerow(row)
            kept += 1

    print(f"Wrote {args.out_csv} ({kept} rows kept, {dropped} ood_* rows dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
