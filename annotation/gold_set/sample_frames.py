#!/usr/bin/env python3
"""Sample BabyView 2025.2 frames (BV-main) for gold set v2: full-frame box correction.

RAs correct pre-filled boxes (YOLOE+CLIP and Gemini, merged, source hidden), add missed
objects and give every box a format. Frames, not crops, so the set measures recall as
well as precision, and compares detection methods on the same objects.

Population
  Every 1 fps frame in /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps whose video
  is BV-main (all 2025.2 subjects are S00xx = main storage), excluding subject S00270001.
  Frames within V1_GAP_S seconds of a gold set v1 frame in the same recording session
  (same subject, date and session number) are excluded, so v1 and v2 never hold near-
  duplicate frames; v1 used one frame per video, so excluding whole videos would drop 42%
  of them. First and last EDGE_S seconds of a video are skipped (camera being put on or
  taken off).

Two strata, one frame per video, no video used twice
  random (N_RANDOM): the headline sample. Child quotas are proportional to the square
    root of the child's video count (so the heaviest children do not dominate and small
    ones still appear), at least MIN_PER_CHILD, at most one frame per video. Within a
    child, videos are chosen by stratified sampling over the date-sorted video list, so
    frames spread across the child's recording period (age). The frame is uniform
    within the usable part of the video. Use this stratum alone for prevalence and recall.
  depiction (N_DEPICTION): frames from 10 s clips whose context label is an activity
    where pictures and screens are common (DEPICTION_ACTIVITIES; merged_contexts.csv,
    labelled per clip, independent of any detector). Same child-quota and date-spread
    rules, among videos not used by the random stratum. Adds books, drawings and screens
    for checking format labels; not representative of the input.

Every frame records its inclusion probability under its stratum's design (child quota /
child's eligible videos x 1 / usable frames in the video), so estimates can be weighted.

Assignment
  N_SHARED random-stratum frames (spread over children) go to both RAs (agreement, and the
  detection-method comparison on double-coded frames). The rest alternate between RAs
  within each child, so each RA sees every child. Batches: batch 1 is the shared frames
  plus BATCH1_EACH frames per RA; later batches follow in a fixed random order.

Outputs (out/private/, git-ignored: subject ids, dates, frame paths)
  frames.csv            one row per sampled frame (opaque frame_uid for the RA bundles)
  sample_summary.json

Usage
  python sample_frames.py --raters Mira RA2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
RELEASE = Path("/ccn2a/dataset/babyview/2025.2")
FRAMES_ROOT = RELEASE / "extracted_frames_1fps"
CONTEXTS = RELEASE / "outputs" / "merged_contexts.csv"
DEMOGS = Path("/ccn2a/dataset/babyview/demogs.csv")
V1_BVMAIN = OUT / "v1_bvmain.csv"            # from freeze_v1.py

EXCLUDED_SUBJECTS = {"S00270001"}
DEPICTION_ACTIVITIES = {"reading", "watching tv", "looking at device", "drawing"}
CLIP_S = 10                                   # merged_contexts clips are 10 s (_000, _001, ...)
EDGE_S = 5
V1_GAP_S = 10
MIN_FRAMES = 30                               # skip videos shorter than 30 s
N_RANDOM, N_DEPICTION = 800, 200
MIN_PER_CHILD = 2
N_SHARED, BATCH1_EACH, BATCH_EACH = 100, 100, 100
SEED = 2026


def base_video(dirname: str) -> str:
    return re.sub(r"_(rotated|processed)$", "", dirname)


def session_key(subject8: str, date: str, idx: str) -> tuple[str, str, str]:
    return subject8, date, idx


def eligible_videos() -> pd.DataFrame:
    rows = []
    for d in sorted(os.listdir(FRAMES_ROOT)):
        v = base_video(d)
        subj, date, idx = v.split("_")[:3]
        if subj in EXCLUDED_SUBJECTS or not subj.startswith("S00"):
            continue
        n = sum(1 for f in os.scandir(FRAMES_ROOT / d) if f.name.endswith(".jpg"))
        rows.append(dict(frame_dir=d, video_id=v, subject_id=subj, date=date, session=idx,
                         n_frames=n))
    vids = pd.DataFrame(rows)
    dup = vids.video_id.duplicated(keep=False)
    assert not dup.any(), f"video ids with two frame folders: {vids[dup].video_id.tolist()[:5]}"

    v1 = pd.read_csv(V1_BVMAIN, dtype=str)
    v1_frames: dict[tuple, list[int]] = {}
    for s, v, f in zip(v1.subject_resolved, v1.video, v1.frame_id):
        v1_frames.setdefault(session_key(s, *v.split("_")[1:3]), []).append(int(f))
    vids["v1_frames"] = [v1_frames.get(session_key(s[1:], d, i), [])
                         for s, d, i in zip(vids.subject_id, vids.date, vids.session)]
    keep = vids.n_frames >= MIN_FRAMES
    print(f"{len(vids)} BV-main videos (S00270001 excluded); {int((vids.v1_frames.str.len() > 0).sum())} "
          f"share a session with v1 (frames within {V1_GAP_S} s of a v1 frame excluded); "
          f"{int((~keep).sum())} shorter than {MIN_FRAMES} s; {int(keep.sum())} eligible")
    return vids[keep].reset_index(drop=True)


def usable_frames(lo: int, hi: int, v1_frames: list[int]) -> np.ndarray:
    f = np.arange(lo, hi)
    for g in v1_frames:
        f = f[np.abs(f - g) > V1_GAP_S]
    return f


def quotas(counts: pd.Series, total: int) -> pd.Series:
    """sqrt-proportional, floor MIN_PER_CHILD, capped at each child's video count."""
    q = pd.Series(0, index=counts.index)
    remaining, open_ = total, counts.index.tolist()
    while remaining > 0 and open_:
        w = np.sqrt(counts[open_].astype(float))
        share = w / w.sum() * remaining
        add = np.floor(share).astype(int)
        add = np.maximum(add, np.where(q[open_] == 0, MIN_PER_CHILD, 0))
        # largest remainder for what floor() left over
        left = remaining - add.sum()
        if left > 0:
            order = (share - np.floor(share)).sort_values(ascending=False).index[:left]
            add[order] += 1
        add = np.minimum(add, counts[open_] - q[open_])
        if add.sum() == 0:
            break
        if add.sum() > remaining:             # floors can overshoot on the first pass
            excess = add.sum() - remaining
            for c in add.sort_values(ascending=False).index:
                take = min(excess, add[c] - (MIN_PER_CHILD if q[c] == 0 else 0))
                if take > 0:
                    add[c] -= take
                    excess -= take
                if excess == 0:
                    break
        q[open_] += add
        remaining = total - q.sum()
        open_ = [c for c in open_ if q[c] < counts[c]]
    return q


def spread_pick(n_items: int, k: int, rng) -> np.ndarray:
    """k indices out of range(n_items), one uniformly inside each of k equal date bins."""
    edges = np.linspace(0, n_items, k + 1)
    return np.array([int(rng.integers(int(np.floor(a)), max(int(np.floor(a)) + 1, int(np.floor(b)))))
                     for a, b in zip(edges[:-1], edges[1:])])


def draw(pool: pd.DataFrame, total: int, rng, frame_range) -> pd.DataFrame:
    counts = pool.groupby("subject_id").size()
    q = quotas(counts, total)
    out = []
    for subj, k in q.items():
        if k == 0:
            continue
        v = pool[pool.subject_id == subj].sort_values(["date", "session", "video_id"]).reset_index(drop=True)
        pick = v.iloc[spread_pick(len(v), int(k), rng)].copy()
        cands = [frame_range(r) for r in pick.itertuples()]
        pick["frame_number"] = [int(rng.choice(c)) for c in cands]
        pick["p_frame"] = [1.0 / len(c) for c in cands]
        pick["p_video"] = k / len(v)
        out.append(pick)
    s = pd.concat(out, ignore_index=True)
    s["p_incl"] = s.p_video * s.p_frame
    s["weight"] = 1.0 / s.p_incl
    return s


def depiction_pool(vids: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_csv(CONTEXTS, usecols=["video_id", "Activity"])
    c = c[c.Activity.str.strip().str.lower().isin(DEPICTION_ACTIVITIES)]
    m = c.video_id.str.extract(r"^(?P<video_id>.+)_(?P<clip>\d{3})\.mp4$")
    c = c.assign(video_id=m["video_id"], clip=m["clip"].astype(int))
    clips = c.groupby("video_id").clip.apply(sorted).rename("dep_clips")
    pool = vids.merge(clips, left_on="video_id", right_index=True)
    # keep clips whose seconds fall inside the usable part of the video
    pool["dep_clips"] = [[k for k in cl if k * CLIP_S >= EDGE_S and (k + 1) * CLIP_S <= n - EDGE_S]
                         for cl, n in zip(pool.dep_clips, pool.n_frames)]
    pool = pool[pool.dep_clips.str.len() > 0]
    # and that still have a frame after the v1 gap rule
    ok = [np.isin(usable_frames(0, n, v1f) // CLIP_S, cl).any()
          for n, v1f, cl in zip(pool.n_frames, pool.v1_frames, pool.dep_clips)]
    return pool[ok].reset_index(drop=True)


def age_months(s: pd.DataFrame) -> pd.Series:
    d = pd.read_csv(DEMOGS, encoding="utf-8-sig", usecols=["subject_id", "date_birth_rounded"])
    dob = pd.to_datetime(d.set_index("subject_id").date_birth_rounded, format="%m/%d/%Y")
    rec = pd.to_datetime(s.date, format="%Y-%m-%d")
    return ((rec - s.subject_id.map(dob)).dt.days / 30.4375).round(1)


def assign(s: pd.DataFrame, raters: list[str], rng) -> pd.DataFrame:
    s = s.copy()
    s["rater"] = ""
    rand = s[s.stratum == "random"]
    # shared: systematic sample over the random stratum sorted by child, so children
    # contribute in proportion to their random-stratum frames
    srt = rand.assign(_k=rng.random(len(rand))).sort_values(["subject_id", "_k"])
    step = len(srt) / N_SHARED
    shared_idx = srt.index[(rng.random() * step + step * np.arange(N_SHARED)).astype(int)]
    s.loc[shared_idx, "rater"] = "shared"
    for subj, g in s[s.rater == ""].groupby("subject_id"):
        idx = rng.permutation(g.index.to_numpy())
        start = int(rng.integers(len(raters)))
        for j, i in enumerate(idx):
            s.loc[i, "rater"] = raters[(start + j) % len(raters)]
    # batches: batch 1 = all shared + BATCH1_EACH per rater (random stratum first)
    s["order"] = rng.permutation(len(s))
    s["batch"] = 0
    s.loc[s.rater == "shared", "batch"] = 1
    for r in raters:
        g = s[s.rater == r].sort_values(["stratum", "order"], key=lambda c: c if c.name == "order"
                                        else c.map({"random": 0, "depiction": 1}))
        b = np.empty(len(g), dtype=int)
        b[:BATCH1_EACH] = 1
        b[BATCH1_EACH:] = 2 + np.arange(len(g) - BATCH1_EACH) // BATCH_EACH
        s.loc[g.index, "batch"] = b
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", default=["Mira", "RA2"])
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    vids = eligible_videos()
    rand = draw(vids, N_RANDOM, rng,
                lambda r: usable_frames(EDGE_S, r.n_frames - EDGE_S, r.v1_frames))
    rand["stratum"] = "random"

    pool = depiction_pool(vids[~vids.video_id.isin(rand.video_id)])
    dep = draw(pool, N_DEPICTION, rng,
               lambda r: usable_frames(0, r.n_frames, r.v1_frames)[
                   np.isin(usable_frames(0, r.n_frames, r.v1_frames) // CLIP_S, r.dep_clips)])
    dep["stratum"] = "depiction"
    dep = dep.drop(columns="dep_clips")

    s = pd.concat([rand, dep], ignore_index=True)
    assert s.video_id.is_unique
    s["frame_path"] = [str(FRAMES_ROOT / d / f"{n:05d}.jpg") for d, n in zip(s.frame_dir, s.frame_number)]
    missing = [p for p in s.frame_path if not Path(p).exists()]
    assert not missing, f"{len(missing)} missing frames, e.g. {missing[:3]}"
    s["age_months"] = age_months(s)
    s["frame_uid"] = [hashlib.sha1(f"{SEED}:{p}".encode()).hexdigest()[:12] for p in s.frame_path]
    s = assign(s, args.raters, rng)
    cols = ["frame_uid", "stratum", "rater", "batch", "subject_id", "video_id", "date", "session",
            "age_months", "frame_number", "n_frames", "frame_path", "p_video", "p_frame", "p_incl",
            "weight"]
    s = s[cols].sort_values(["batch", "rater", "frame_uid"]).reset_index(drop=True)
    s.to_csv(OUT / "frames.csv", index=False)

    per_child = s.groupby("subject_id").size()
    summary = {
        "n_frames": len(s), "by_stratum": s.stratum.value_counts().to_dict(),
        "n_subjects": int(s.subject_id.nunique()), "n_videos": int(s.video_id.nunique()),
        "eligible_subjects": int(vids.subject_id.nunique()), "eligible_videos": len(vids),
        "frames_per_child_min_median_max": [int(per_child.min()), float(per_child.median()),
                                            int(per_child.max())],
        "age_months_min_median_max": [float(s.age_months.min()), float(s.age_months.median()),
                                      float(s.age_months.max())],
        "by_rater": s.rater.value_counts().to_dict(),
        "batch1": s[s.batch == 1].rater.value_counts().to_dict(),
        "frames_per_rater_incl_shared": {r: int(((s.rater == r) | (s.rater == "shared")).sum())
                                         for r in args.raters},
        "depiction_activities": sorted(DEPICTION_ACTIVITIES), "seed": SEED,
    }
    (OUT / "sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(per_child.rename("frames").to_frame().join(
        vids.groupby("subject_id").size().rename("eligible_videos")).to_string())


if __name__ == "__main__":
    main()
