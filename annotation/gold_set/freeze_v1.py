#!/usr/bin/env python3
"""Freeze gold set v1 unchanged, and write its BV-main subset for v2.

v1 is the CCN 2026 gold set: valid5921_public (5,921 rater-validated YOLOE crops,
72 categories, opaque crop ids) plus the frame manifest that maps each crop back to
its 868-hour frame and YOLOE box (gemini_pilot/out/private/gold_frames.csv).

Which recordings are BV-Bing (preschool) and which BV-main (home)? The raw-video
storage keeps them apart (/ccn2/dataset/babyview/unzip*/babyview_{bing,main}_storage)
and no subject appears in both, so a crop's dataset is its subject's store. Videos
named 2023_<n>_<hash> carry no subject in the name; their GoPro sidecar files name
00270001, which is excluded from v2 with the rest of S00270001.

Outputs (out/private/, git-ignored: subject ids and dates)
  v1_original/valid5921_public/   byte-for-byte copy of the public release
  v1_original/gold_frames.csv     copy of the frame manifest
  v1_original/CHECKSUMS.txt       sha256 of every copied file
  v1_bvmain.csv                   v1 rows kept for v2 (BV-main, not S00270001)
  v1_split_summary.json

Usage
  python freeze_v1.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
V1_PUBLIC = Path("/home/j7yang/babyview-projects/ccn-2026-object-variability/data/"
                 "shared_data_ccn_2026/valid5921_public")
V1_FRAMES = HERE.parent / "gemini_pilot" / "out" / "private" / "gold_frames.csv"
STORES = [Path("/ccn2/dataset/babyview/unzip"), Path("/ccn2/dataset/babyview/unzip_2025")]
EXCLUDED_SUBJECTS = {"00270001"}
UNNAMED_VIDEO_SUBJECT = "00270001"   # 2023_<n>_<hash> videos, from their GoPro sidecar files


def subject_store() -> dict[str, str]:
    """subject id (8 digits) -> 'main' or 'bing', from the raw-video storage folders."""
    out: dict[str, set[str]] = {}
    for root in STORES:
        for kind in ("main", "bing"):
            d = root / f"babyview_{kind}_storage"
            if d.is_dir():
                for v in os.listdir(d):
                    out.setdefault(v.split("_")[0], set()).add(kind)
    both = sorted(s for s, k in out.items() if len(k) > 1)
    assert not both, f"subjects in both stores: {both}"
    return {s: k.pop() for s, k in out.items()}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    dst = OUT / "v1_original"
    if dst.exists():
        print(f"{dst} exists; leaving it untouched")
        return
    dst.mkdir(parents=True)
    shutil.copytree(V1_PUBLIC, dst / "valid5921_public")
    shutil.copy2(V1_FRAMES, dst / "gold_frames.csv")
    lines = [f"{sha256(p)}  {p.relative_to(dst)}" for p in sorted(dst.rglob("*")) if p.is_file()]
    (dst / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")
    for p in sorted(dst.rglob("*")):           # read-only, so nothing edits the frozen copy
        p.chmod(0o555 if p.is_dir() else 0o444)
    dst.chmod(0o555)
    print(f"froze v1 -> {dst} ({len(lines)} files)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze()
    store = subject_store()
    d = pd.read_csv(V1_FRAMES, dtype=str)
    d["subject_resolved"] = d.subject.where(~d.video.str.startswith("2023_"), UNNAMED_VIDEO_SUBJECT)
    d["dataset"] = d.subject_resolved.map(store).fillna("unknown")
    keep = (d.dataset == "main") & ~d.subject_resolved.isin(EXCLUDED_SUBJECTS)
    d[keep].to_csv(OUT / "v1_bvmain.csv", index=False)
    summary = {
        "v1_crops": len(d),
        "by_dataset": d.dataset.value_counts().to_dict(),
        "dropped_excluded_subject": int(((d.dataset == "main")
                                         & d.subject_resolved.isin(EXCLUDED_SUBJECTS)).sum()),
        "v1_bvmain_crops": int(keep.sum()),
        "v1_bvmain_subjects": int(d[keep].subject_resolved.nunique()),
        "v1_bvmain_frames": int(d[keep].frame_path.nunique()),
        "v1_bvmain_categories": int(d[keep].category.nunique()),
        "bing_subjects": sorted(d[d.dataset == "bing"].subject_resolved.unique().tolist()),
    }
    (OUT / "v1_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "bing_subjects"}, indent=2))


if __name__ == "__main__":
    main()
