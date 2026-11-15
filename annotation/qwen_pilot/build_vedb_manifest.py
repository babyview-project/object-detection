#!/usr/bin/env python3
"""VEDB (volume 1612) frame list for the Qwen run, built like the BabyView full-run manifest.

Frames: every 5th 1 fps frame (frame number % 5 == 0, i.e. one frame per 5 s) of every session in
/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps (717 world-camera sessions, frames named
00000000.jpg, ...). Same rule as gemini_full_run/out/private/manifest.csv for BabyView.

Outputs (out/private/):
  vedb_manifest.csv        unit_id (<session>/<8-digit frame>), video_id, frame, frame_path
  vedb_pilot500.csv        500 frames for the pilot: 500 random sessions, one random grid frame each (seed 2026)

  python build_vedb_manifest.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FRAMES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps")
OUT = HERE / "out" / "private"
STRIDE, N_PILOT, SEED = 5, 500, 2026


def main():
    rows = []
    for d in sorted(os.listdir(FRAMES)):
        for f in sorted(os.listdir(FRAMES / d)):
            if not f.endswith(".jpg"):
                continue
            n = int(f[:-4])
            if n % STRIDE == 0:
                rows.append((f"{d}/{f[:-4]}", d, n, str(FRAMES / d / f)))
    m = pd.DataFrame(rows, columns=["unit_id", "video_id", "frame", "frame_path"])
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "vedb_manifest.csv", index=False)
    rng = np.random.default_rng(SEED)
    sessions = rng.choice(m.video_id.unique(), N_PILOT, replace=False)
    pilot = (m[m.video_id.isin(sessions)].groupby("video_id", group_keys=False)
             .apply(lambda g: g.sample(1, random_state=int(rng.integers(1 << 31)))))
    pilot.sort_values("unit_id").to_csv(OUT / "vedb_pilot500.csv", index=False)
    print(f"{m.video_id.nunique()} sessions, {len(m):,} frames (every {STRIDE}th 1 fps frame); "
          f"pilot {len(pilot)} frames from {pilot.video_id.nunique()} sessions")


if __name__ == "__main__":
    main()
