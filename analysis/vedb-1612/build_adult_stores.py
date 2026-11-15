#!/usr/bin/env python3
"""Adults-only VEDB: the session list and adult-only views of the VEDB stores.

VEDB volume 1612 includes some child observers. Our analyses use adults only (>= 18 years at recording), so every
VEDB input is read from an "_adults" view of its store: the same files, shards symlinked for adult sessions only and
index tables filtered to those sessions. Nothing is recomputed; the original stores stay untouched.

1. Ages: each Databrary session has one participant record with the age at recording. `--fetch` pulls them (needs
   `python -m databrary login`) into AGES (on /ccn2a, not in git: the repo is public and ages are authorized-user
   data). Without --fetch the saved table is used.
2. Session list: vedb_adult_sessions.txt (session ids only; in git).
3. Views (ADULT_ROOTS below): shards / session folders symlinked, *.csv.gz / *.txt index tables filtered by video id.

  python build_adult_stores.py [--fetch] [--min-age 18]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AGES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/participant_ages.csv")
SESSION_RE = re.compile(r"session_\d+_worldPrivate")
SESSIONS_TXT = HERE / "vedb_adult_sessions.txt"
STORES = {  # original -> adults-only view
    Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full"):
        Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full_adults"),
    Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv"):
        Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults"),
}
SESSION_DIRS = {  # one sub-folder per session
    Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi"):
        Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi_adults"),
    Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"):
        Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_adults"),
}


def fetch_ages() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(REPO))
    from databrary.client import DatabraryClient
    rows = []
    for s in DatabraryClient().list_sessions(1612):
        ps = [r for r in s["fileRecords"] if r.get("categoryId") == 1 and r.get("age")]
        assert len(ps) == 1, (s["id"], len(ps))
        a = ps[0]["age"]
        rows.append(dict(session_id=s["id"], participant_id=ps[0]["measures"].get("1"), age_days=a["totalDays"],
                         age_years=round(a["totalDays"] / 365.25, 2), age_estimated=a["isEstimated"],
                         age_blurred=a["isBlurred"]))
    return pd.DataFrame(rows).sort_values("session_id")


def video_id(sid: int) -> str:
    return f"session_{sid}_worldPrivate"


def is_adult(name: str, keep: set[str]) -> bool:
    m = SESSION_RE.search(name)
    return m is not None and m.group(0) in keep


def filter_table(src: Path, dst: Path, keep: set[str]):
    """Keep rows whose first session_<id>_worldPrivate (video id, image id or crop path) is an adult session; a first
    line without one is a header and is kept. Row order is preserved (index order = shard order within a video)."""
    op = gzip.open if src.suffix == ".gz" else open
    n_in = n_out = 0
    with op(src, "rt") as fi, op(dst, "wt") as fo:
        for k, line in enumerate(fi):
            m = SESSION_RE.search(line)
            if m is None:
                assert k == 0, (src, k, line[:200])
                fo.write(line)
                continue
            n_in += 1
            if m.group(0) in keep:
                fo.write(line)
                n_out += 1
    print(f"  {dst.name}: {n_out:,} of {n_in:,} rows")


def build_view(src: Path, dst: Path, keep: set[str]):
    print(src, "->", dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        q = dst / p.name
        if p.name in ("shards", "_status"):
            q.mkdir(exist_ok=True)
            for f in p.iterdir():
                link = q / f.name
                if is_adult(f.name, keep) and not link.exists():
                    link.symlink_to(f)
            print(f"  {p.name}/: {sum(1 for _ in q.iterdir())} of {sum(1 for _ in p.iterdir())} entries")
        elif p.name.endswith((".csv.gz", ".csv", ".txt")) and p.name != "README.txt":
            filter_table(p, q, keep)
        elif p.is_file():
            shutil.copy2(p, q)
    (dst / "ADULTS_ONLY.txt").write_text(
        f"Adults-only view of {src} ({len(keep)} sessions, vedb-1612/vedb_adult_sessions.txt), "
        "built by analysis/vedb-1612/build_adult_stores.py. Shards are symlinks; index tables are filtered copies.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--min-age", type=float, default=18)
    a = ap.parse_args()
    if a.fetch:
        fetch_ages().to_csv(AGES, index=False)
    ages = pd.read_csv(AGES)
    adult = ages[ages.age_days >= a.min_age * 365.25]
    print(f"{len(ages)} sessions / {ages.participant_id.nunique()} participants; adults: {len(adult)} sessions / "
          f"{adult.participant_id.nunique()} participants (age {adult.age_years.min():.1f}-{adult.age_years.max():.1f}); "
          f"excluded: {len(ages) - len(adult)} sessions / {ages.participant_id.nunique() - adult.participant_id.nunique()} participants")
    SESSIONS_TXT.write_text("".join(f"{video_id(s)}\n" for s in sorted(adult.session_id)))
    keep = {video_id(s) for s in adult.session_id}
    for src, dst in STORES.items():
        build_view(src, dst, keep)
    for src, dst in SESSION_DIRS.items():
        dst.mkdir(parents=True, exist_ok=True)
        for d in src.iterdir():                        # names can carry a suffix, e.g. session_68220_worldPrivate_inverted
            if is_adult(d.name, keep) and not (dst / d.name).exists():
                (dst / d.name).symlink_to(d)
        print(src, "->", dst, f"{sum(1 for _ in dst.iterdir())} session folders")


if __name__ == "__main__":
    main()
