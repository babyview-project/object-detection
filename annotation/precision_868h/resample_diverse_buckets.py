#!/usr/bin/env python3
"""Rebalance bucket assignments to reduce subject/video concentration.

This script:
1) Reads `sampled_object_crops_100_bucket_assignments.csv`.
2) Reassigns *regular* rows across existing buckets per category while keeping:
   - per-(bucket, category) regular counts unchanged
   - each regular stem used exactly once
3) Preserves non-regular rows (attention checks) and their positions.
4) Writes a new assignment CSV + before/after concentration reports.

Speed vs preprocessing/sample_object_crops_variability.py:
- The sampler only does diverse round-robin + shuffled bucket labels (fast).
- This script can optionally run many random-restart swap steps per category
  (`--restarts`, `--swap-steps`), which is much slower. Use `--fast` to match
  the sampler-style diversity assignment without local search, or `--workers`
  to parallelize local search across categories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


@lru_cache(maxsize=None)
def parse_subject_video(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    subject = parts[2] if len(parts) > 2 else ""
    video = "_".join(parts[2:6]) if len(parts) > 5 else ""
    return subject, video


def n_choose_2(n: int) -> int:
    return n * (n - 1) // 2


def print_progress(
    prefix: str,
    current: int,
    total: int,
    *,
    width: int = 28,
    done: bool = False,
) -> None:
    """Render a one-line terminal progress bar."""
    total = max(1, total)
    current = min(max(current, 0), total)
    frac = current / total
    filled = int(width * frac)
    bar = "=" * filled + "." * (width - filled)
    end = "\n" if done else "\r"
    sys.stdout.write(f"{prefix} [{bar}] {current}/{total}")
    sys.stdout.write(end)
    sys.stdout.flush()


def parse_detection_score(row: dict) -> float:
    """Extract detection score embedded in stem/path."""
    for key in ("stem", "path"):
        value = row.get(key, "")
        match = re.search(r"_(\d+\.\d+)_", value)
        if match:
            return float(match.group(1))
    raise ValueError(
        f"Could not parse detection score from row stem/path: {row.get('stem', '')}"
    )


def load_allowed_stems(filter_list_path: Path) -> set[str]:
    """Load allowed stems from CLIP filtered list (one path per line)."""
    allowed: set[str] = set()
    with filter_list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.suffix.lower() == ".npy":
                allowed.add(p.stem)
            else:
                allowed.add(p.stem if p.suffix else p.name)
    return allowed


def load_allowed_stems_by_category(filter_list_path: Path) -> Dict[str, set[str]]:
    """Load allowed stems grouped by category from CLIP filtered list."""
    out: Dict[str, set[str]] = defaultdict(set)
    with filter_list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            category = p.parent.name.strip().lower()
            stem = p.stem if p.suffix else p.name
            out[category].add(stem)
    return out


def get_category_crop_dir(cropped_dir: Path, category: str) -> Path | None:
    """Case-insensitive category dir lookup under cropped root."""
    direct = cropped_dir / category
    if direct.exists() and direct.is_dir():
        return direct
    category_lower = category.strip().lower()
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.strip().lower() == category_lower:
            return p
    return None


def even_bucket_targets(n_items: int, n_buckets: int) -> Dict[str, int]:
    """Evenly split n_items into buckets 1..n_buckets."""
    base = n_items // n_buckets
    rem = n_items % n_buckets
    return {
        str(b): base + (1 if b <= rem else 0)
        for b in range(1, n_buckets + 1)
    }


def cap_paths_by_subject(
    paths: List[Path], max_per_subject: int, rng: random.Random
) -> List[Path]:
    """Keep at most max_per_subject images per subject (0 = no cap)."""
    if max_per_subject <= 0:
        return paths
    by_subj: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        subj, _ = parse_subject_video(p.stem)
        by_subj[subj].append(p)
    out: List[Path] = []
    for subj in by_subj:
        pool = by_subj[subj]
        rng.shuffle(pool)
        out.extend(pool[:max_per_subject])
    return out


def cap_paths_by_video(
    paths: List[Path], max_per_video: int, rng: random.Random
) -> List[Path]:
    """Keep at most max_per_video images per source video (0 = no cap)."""
    if max_per_video <= 0:
        return paths
    by_video: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        _subj, video = parse_subject_video(p.stem)
        by_video[video].append(p)
    out: List[Path] = []
    for video in by_video:
        pool = by_video[video]
        rng.shuffle(pool)
        out.extend(pool[:max_per_video])
    return out


def sample_rows_diverse_subject_video(
    rows: List[dict],
    rng: random.Random,
    take_n: int | None = None,
    max_per_video: int = 0,
    enforce_strict_video_cap: bool = False,
) -> List[dict]:
    """
    Same strategy as preprocessing/sample_object_crops_variability.sample_paths_diverse_subject_video:
    round-robin across subjects and videos to spread diversity, then fill remainder.
    Operates on row dicts using parse_subject_video(stem).
    If take_n is set, return at most that many rows (<= len(rows)).
    If max_per_video > 0 and enforce_strict_video_cap, only picks rows whose video is
    still below the cap (water-fill on counts); requires a feasible pool for take_n.
    If max_per_video > 0 and not enforce_strict_video_cap, uses a soft greedy that may
    exceed the cap when necessary to reach take_n.
    """
    pool_n = len(rows)
    if pool_n == 0:
        return []
    n = pool_n if take_n is None else min(take_n, pool_n)
    if n <= 1:
        return rows[:n]

    if max_per_video > 0:
        remaining = list(rows)
        rng.shuffle(remaining)
        chosen: List[dict] = []
        per_video: Dict[str, int] = {}
        if enforce_strict_video_cap:
            while len(chosen) < n and remaining:
                best_r: dict | None = None
                best_key: Tuple[int, float] | None = None
                for r in remaining:
                    _, v = parse_subject_video(r["stem"])
                    c = per_video.get(v, 0)
                    if c >= max_per_video:
                        continue
                    key = (c, rng.random())
                    if best_key is None or key < best_key:
                        best_key = key
                        best_r = r
                if best_r is None:
                    raise ValueError(
                        f"strict video cap infeasible at {len(chosen)}/{n} with "
                        f"{len(remaining)} rows remaining"
                    )
                remaining.remove(best_r)
                _, v = parse_subject_video(best_r["stem"])
                per_video[v] = per_video.get(v, 0) + 1
                chosen.append(best_r)
            return chosen[:n]

        while len(chosen) < n and remaining:
            best_i = -1
            best_key: Tuple[int, int, float, int] | None = None
            for i, r in enumerate(remaining):
                _, v = parse_subject_video(r["stem"])
                c = per_video.get(v, 0)
                key = (
                    0 if c < max_per_video else 1,
                    c,
                    rng.random(),
                    i,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            r = remaining.pop(best_i)
            _, v = parse_subject_video(r["stem"])
            per_video[v] = per_video.get(v, 0) + 1
            chosen.append(r)
        return chosen[:n]

    subject_groups: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        subject_id, gcp_name = parse_subject_video(r["stem"])
        subject_groups.setdefault(subject_id, {}).setdefault(gcp_name, []).append(r)

    subject_order = list(subject_groups.keys())
    rng.shuffle(subject_order)

    subject_video_order: Dict[str, List[str]] = {}
    for subject_id in subject_order:
        videos = list(subject_groups[subject_id].keys())
        rng.shuffle(videos)
        subject_video_order[subject_id] = videos
        for v in videos:
            rng.shuffle(subject_groups[subject_id][v])

    chosen_rr: List[dict] = []

    progressed = True
    while len(chosen_rr) < n and progressed:
        progressed = False
        for subject_id in subject_order:
            if len(chosen_rr) >= n:
                break
            videos = subject_video_order[subject_id]
            if not videos:
                continue
            picked_row = None
            for _ in range(len(videos)):
                video = videos.pop(0)
                pool = subject_groups[subject_id][video]
                if pool:
                    picked_row = pool.pop()
                    videos.append(video)
                    break
                videos.append(video)
            if picked_row is not None:
                chosen_rr.append(picked_row)
                progressed = True

    if len(chosen_rr) >= n:
        return chosen_rr[:n]

    chosen_ids = {id(r) for r in chosen_rr}
    leftovers = [r for r in rows if id(r) not in chosen_ids]
    rng.shuffle(leftovers)
    chosen_rr.extend(leftovers[: n - len(chosen_rr)])
    return chosen_rr[:n]


def assign_diverse_to_bucket_targets(
    ordered_rows: List[dict], bucket_targets: Dict[str, int]
) -> Dict[str, List[dict]]:
    """Fill buckets 1..K in order with consecutive slices of ordered_rows."""
    buckets = sorted(bucket_targets.keys(), key=int)
    out: Dict[str, List[dict]] = {b: [] for b in bucket_targets}
    pos = 0
    for b in buckets:
        take = int(bucket_targets[b])
        out[b] = ordered_rows[pos : pos + take]
        pos += take
    if pos != len(ordered_rows):
        raise ValueError("bucket_targets do not sum to len(ordered_rows)")
    return out


def fast_diverse_bucket_assignment(
    candidates: List[dict],
    bucket_targets: Dict[str, int],
    rng: random.Random,
    max_per_video: int = 0,
    enforce_strict_video_cap: bool = False,
) -> Dict[str, List[dict]]:
    """Preprocessing-style diversity + deterministic bucket fill (no local search)."""
    ordered = sample_rows_diverse_subject_video(
        candidates,
        rng,
        max_per_video=max_per_video,
        enforce_strict_video_cap=enforce_strict_video_cap,
    )
    return assign_diverse_to_bucket_targets(ordered, bucket_targets)


def build_regular_rows_from_filter(
    filter_list_path: Path,
    cropped_dir: Path,
    categories: List[str],
    n_per_category: int,
    allow_fewer: bool,
    rng: random.Random,
    max_per_subject: int = 0,
    max_per_video: int = 0,
) -> List[dict]:
    """Build regular rows directly from filtered list + cropped image folders.

    Optionally cap how many exemplars may come from each subject (``max_per_subject``)
    before subsampling. Per-video limit ``max_per_video`` is enforced during
    ``sample_rows_diverse_subject_video`` (round-robin phase); the remainder fill can
    exceed it so ``n_per_category`` is still reachable.
    """
    allowed_by_category = load_allowed_stems_by_category(filter_list_path)
    categories_to_use = [c.strip().lower() for c in categories if c.strip()]

    rows: List[dict] = []
    shortages: List[Tuple[str, int]] = []
    for cat in categories_to_use:
        cat_dir = get_category_crop_dir(cropped_dir, cat)
        if cat_dir is None:
            continue

        path_by_stem = {}
        for ext in (".jpg", ".jpeg", ".png"):
            for p in cat_dir.glob(f"*{ext}"):
                path_by_stem[p.stem] = p

        allowed_stems = allowed_by_category.get(cat, set())
        matched_paths = [path_by_stem[s] for s in allowed_stems if s in path_by_stem]
        if not matched_paths:
            continue

        matched_paths = cap_paths_by_subject(matched_paths, max_per_subject, rng)
        if len(matched_paths) < n_per_category and not allow_fewer:
            shortages.append((cat, len(matched_paths)))
            continue

        take_n = min(n_per_category, len(matched_paths)) if allow_fewer else n_per_category
        temp_rows = [{"stem": p.stem, "_path": p} for p in matched_paths]
        ordered = sample_rows_diverse_subject_video(
            temp_rows, rng, take_n=take_n, max_per_video=max_per_video
        )
        for r in ordered:
            p = r["_path"]
            rows.append(
                {
                    "bucket_index": "",
                    "category": cat,
                    "path": str(p.resolve()),
                    "stem": p.stem,
                    "trial_type": "regular",
                    "attention_check_source_category": "",
                    "bucket_trial_index": "",
                }
            )

    if shortages and not allow_fewer:
        detail = ", ".join([f"{c}={n}" for c, n in shortages[:10]])
        raise RuntimeError(
            f"Not enough filtered crops for strict n-per-category sampling: {detail}"
        )
    if not rows:
        raise RuntimeError("No regular rows constructed from filter list and cropped dir.")
    return rows


def bucket_penalty(rows: List[dict]) -> float:
    """Penalty for one category within one bucket."""
    if len(rows) <= 1:
        return 0.0

    subj = []
    vids = []
    for r in rows:
        s, v = parse_subject_video(r["stem"])
        subj.append(s)
        vids.append(v)

    subj_counts = Counter(subj)
    vid_counts = Counter(vids)
    top_subj = max(subj_counts.values())
    top_vid = max(vid_counts.values())

    # Same-subject/video pairs are sum over groups of nC2.
    ss_pairs = sum(n_choose_2(c) for c in subj_counts.values())
    sv_pairs = sum(n_choose_2(c) for c in vid_counts.values())

    # Heavier weight on same-video concentration.
    return (
        3.0 * ss_pairs
        + 6.0 * sv_pairs
        + 10.0 * max(0, top_subj - 3)
        + 12.0 * max(0, top_vid - 2)
    )


def initial_assignment(
    candidates: List[dict], bucket_targets: Dict[str, int], rng: random.Random
) -> Dict[str, List[dict]]:
    out = {b: [] for b in bucket_targets}
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    i = 0
    for bucket, target in sorted(bucket_targets.items(), key=lambda x: int(x[0])):
        out[bucket] = shuffled[i : i + target]
        i += target
    return out


def optimize_category_assignment(
    candidates: List[dict],
    bucket_targets: Dict[str, int],
    rng: random.Random,
    restarts: int = 30,
    swap_steps: int = 4000,
    show_progress: bool = False,
    progress_prefix: str = "",
    idle_patience: int = 0,
) -> Dict[str, List[dict]]:
    """Random-restart local search minimizing concentration penalty."""
    buckets = sorted(bucket_targets.keys(), key=int)
    best = None
    best_score = float("inf")

    for restart_i in range(restarts):
        assign = initial_assignment(candidates, bucket_targets, rng)
        score = sum(bucket_penalty(assign[b]) for b in buckets)

        # Pairwise cross-bucket swaps.
        idle = 0
        for _step in range(swap_steps):
            b1, b2 = rng.sample(buckets, 2)
            if not assign[b1] or not assign[b2]:
                continue
            i = rng.randrange(len(assign[b1]))
            j = rng.randrange(len(assign[b2]))

            old_local = bucket_penalty(assign[b1]) + bucket_penalty(assign[b2])
            assign[b1][i], assign[b2][j] = assign[b2][j], assign[b1][i]
            new_local = bucket_penalty(assign[b1]) + bucket_penalty(assign[b2])

            if new_local <= old_local:
                score += new_local - old_local
                idle = 0
            else:
                # Reject non-improving move.
                assign[b1][i], assign[b2][j] = assign[b2][j], assign[b1][i]
                if idle_patience > 0:
                    idle += 1
                    if idle >= idle_patience:
                        break

        if score < best_score:
            best_score = score
            best = {b: v[:] for b, v in assign.items()}
        if show_progress:
            print_progress(
                prefix=progress_prefix,
                current=restart_i + 1,
                total=restarts,
                done=(restart_i + 1 == restarts),
            )

    assert best is not None
    return best


def _optimize_category_worker(payload: Tuple) -> Tuple[str, Dict[str, List[dict]]]:
    """Picklable worker for ProcessPoolExecutor (must be top-level)."""
    (
        category,
        candidates,
        bucket_targets,
        seed,
        restarts,
        swap_steps,
        idle_patience,
    ) = payload
    rng = random.Random(seed)
    best = optimize_category_assignment(
        candidates=candidates,
        bucket_targets=bucket_targets,
        rng=rng,
        restarts=restarts,
        swap_steps=swap_steps,
        show_progress=False,
        progress_prefix="",
        idle_patience=idle_patience,
    )
    return category, best


def build_concentration_rows(rows: List[dict], bucket_col: str) -> List[dict]:
    by_bucket = defaultdict(list)
    for row in rows:
        if row["trial_type"] != "regular":
            continue
        by_bucket[row[bucket_col]].append(row)

    out = []
    for bucket in sorted(by_bucket, key=lambda x: int(x)):
        by_cat = defaultdict(list)
        for row in by_bucket[bucket]:
            by_cat[row["category"]].append(row)

        for cat, items in by_cat.items():
            n = len(items)
            subj = []
            vids = []
            for r in items:
                s, v = parse_subject_video(r["stem"])
                subj.append(s)
                vids.append(v)
            subj_counts = Counter(subj)
            vid_counts = Counter(vids)

            pair_total = n_choose_2(n)
            ss_pairs = sum(n_choose_2(c) for c in subj_counts.values())
            sv_pairs = sum(n_choose_2(c) for c in vid_counts.values())

            top_subj, top_subj_count = subj_counts.most_common(1)[0]
            top_vid, top_vid_count = vid_counts.most_common(1)[0]
            out.append(
                {
                    "bucket": bucket,
                    "category": cat,
                    "n": n,
                    "uniq_subjects": len(subj_counts),
                    "uniq_videos": len(vid_counts),
                    "top_subject": top_subj,
                    "top_subject_count": top_subj_count,
                    "top_subject_fraction": round(top_subj_count / n, 4),
                    "top_video": top_vid,
                    "top_video_count": top_vid_count,
                    "top_video_fraction": round(top_vid_count / n, 4),
                    "same_subject_pair_rate": round(
                        ss_pairs / pair_total, 4
                    )
                    if pair_total
                    else 0.0,
                    "same_video_pair_rate": round(sv_pairs / pair_total, 4)
                    if pair_total
                    else 0.0,
                }
            )
    return out


def build_global_summary(concentration_rows: List[dict], label: str) -> List[dict]:
    by_bucket = defaultdict(list)
    for row in concentration_rows:
        by_bucket[row["bucket"]].append(row)

    out = []
    for bucket in sorted(by_bucket, key=lambda x: int(x)):
        rows = by_bucket[bucket]
        total_pairs = 0
        total_ss = 0.0
        total_sv = 0.0
        severe = 0
        total_trials = 0
        for r in rows:
            n = int(r["n"])
            total_trials += n
            pairs = n_choose_2(n)
            total_pairs += pairs
            total_ss += float(r["same_subject_pair_rate"]) * pairs
            total_sv += float(r["same_video_pair_rate"]) * pairs
            if float(r["top_video_fraction"]) >= 0.5 or float(
                r["top_subject_fraction"]
            ) >= 0.5:
                severe += 1

        out.append(
            {
                "version": label,
                "bucket": bucket,
                "regular_trials": total_trials,
                "categories": len(rows),
                "same_subject_pair_rate_global": round(total_ss / total_pairs, 4),
                "same_video_pair_rate_global": round(total_sv / total_pairs, 4),
                "severe_categories_top_frac_ge_0_5": severe,
            }
        )
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional existing assignment CSV. If omitted, build from filter list.",
    )
    parser.add_argument(
        "--output-csv",
        default="annotation/precision_868h/sampled_object_crops_100_bucket_assignments_diverse_rebalanced.csv",
    )
    parser.add_argument(
        "--concentration-report",
        default="annotation/precision_868h/sampled_object_crops_100_diverse_rebalanced_concentration.csv",
    )
    parser.add_argument(
        "--global-report",
        default="annotation/precision_868h/sampled_object_crops_100_diverse_rebalanced_global_summary.csv",
    )
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="Random restarts per category for local search (ignored with --fast).",
    )
    parser.add_argument(
        "--swap-steps",
        type=int,
        default=600,
        help="Swap attempts per restart (ignored with --fast).",
    )
    parser.add_argument(
        "--idle-patience",
        type=int,
        default=500,
        help="Stop inner swap loop after this many rejected moves in a row (0=disabled).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use preprocessing-style diverse sampling + bucket fill only (no local search).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel processes for local search (0=auto, 1=serial). Ignored with --fast.",
    )
    parser.add_argument(
        "--filter-list",
        default=(
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/"
            "clip_image_embeddings_filtered-by-clip-0.27_exclude-people_"
            "exclude-subject-00270001.txt"
        ),
        help="CLIP-filtered embedding list; regular rows must be in this list.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.27,
        help="Only keep regular rows with detection score >= this threshold.",
    )
    parser.add_argument(
        "--cropped-dir",
        default="/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class",
        help="Required when --input-csv is omitted; source cropped images root.",
    )
    parser.add_argument(
        "--categories-file",
        default="data/things_bv_overlap_categories_exclude_zero_precisions.txt",
        help="Category allowlist (one per line). Defaults to 163-category overlap list.",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=100,
        help="Per-category sample count in build-from-filter mode.",
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=0,
        help=(
            "In build-from-filter mode, keep at most this many exemplars per subject "
            "per category before diverse subsampling (0 = no cap)."
        ),
    )
    parser.add_argument(
        "--max-per-video",
        type=int,
        default=0,
        help=(
            "Per source video (stem parts 2–6): at most this many rows in the diverse "
            "round-robin phase of sampling; remainder fill can exceed it (0 = no cap). "
            "Applies to build-from-filter and --fast reassignment."
        ),
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Allow categories with fewer than --n-per-category in build-from-filter mode.",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=4,
        help="Bucket count in build-from-filter mode.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable optimization progress bars.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_path = Path(args.input_csv) if args.input_csv else None
    output_path = Path(args.output_csv)
    concentration_path = Path(args.concentration_report)
    global_path = Path(args.global_report)
    filter_list_path = Path(args.filter_list)

    if not filter_list_path.exists():
        raise FileNotFoundError(f"Filter list not found: {filter_list_path}")
    allowed_stems = load_allowed_stems(filter_list_path)
    if not allowed_stems:
        raise ValueError(f"No stems loaded from filter list: {filter_list_path}")

    categories_path = Path(args.categories_file)
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")
    categories: List[str] = []
    with categories_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                categories.append(line)
    allowed_categories = {c.strip().lower() for c in categories}
    if not allowed_categories:
        raise ValueError(f"No categories loaded from {categories_path}")

    if input_path is not None:
        with input_path.open(newline="") as f:
            all_rows = list(csv.DictReader(f))

        regular_rows_all = [r for r in all_rows if r["trial_type"] == "regular"]
        regular_rows_all = [
            r
            for r in regular_rows_all
            if r["category"].strip().lower() in allowed_categories
        ]
        regular_rows_in_filter = [r for r in regular_rows_all if r["stem"] in allowed_stems]
        regular_rows = [
            r for r in regular_rows_in_filter if parse_detection_score(r) >= args.min_score
        ]
        non_regular_rows = [r for r in all_rows if r["trial_type"] != "regular"]
        if not regular_rows:
            raise ValueError("No regular rows found.")

        # Keep original row order while dropping disallowed regular rows.
        rows = []
        for row in all_rows:
            if row["trial_type"] != "regular":
                rows.append(row)
            elif (
                row["category"].strip().lower() in allowed_categories
                and row["stem"] in allowed_stems
                and parse_detection_score(row) >= args.min_score
            ):
                rows.append(row)
        # Explicitly confirm filtered-list constraint.
        if any(r["stem"] not in allowed_stems for r in regular_rows):
            raise RuntimeError("Detected regular rows outside filter list after filtering.")
        print(
            "Filtering summary:",
            f"regular_total={len(regular_rows_all)}",
            f"in_filter={len(regular_rows_in_filter)}",
            f"score_ge_{args.min_score}={len(regular_rows)}",
            f"allowed_categories={len(allowed_categories)}",
        )

        # Preserve existing per-(bucket, category) regular counts.
        targets = defaultdict(lambda: defaultdict(int))
        for r in regular_rows:
            targets[r["category"]][r["bucket_index"]] += 1
    else:
        cropped_dir = Path(args.cropped_dir)
        if not cropped_dir.exists():
            raise FileNotFoundError(f"Cropped dir not found: {cropped_dir}")

        regular_rows = build_regular_rows_from_filter(
            filter_list_path=filter_list_path,
            cropped_dir=cropped_dir,
            categories=categories,
            n_per_category=args.n_per_category,
            allow_fewer=args.allow_fewer,
            rng=rng,
            max_per_subject=args.max_per_subject,
            max_per_video=args.max_per_video,
        )
        non_regular_rows = []
        rows = regular_rows[:]
        if any(r["stem"] not in allowed_stems for r in regular_rows):
            raise RuntimeError("Build-from-filter mode produced rows outside filter list.")
        print(
            "Filtering summary:",
            f"regular_built={len(regular_rows)}",
            f"allowed_stems={len(allowed_stems)}",
            f"allowed_categories={len(allowed_categories)}",
        )

        # Create fresh bucket targets with even split by category.
        targets = defaultdict(lambda: defaultdict(int))
        by_cat_tmp = defaultdict(list)
        for r in regular_rows:
            by_cat_tmp[r["category"]].append(r)
        for cat, cat_rows in by_cat_tmp.items():
            for b, n in even_bucket_targets(len(cat_rows), args.n_buckets).items():
                targets[cat][b] = n

    # Candidate pools by category.
    by_category = defaultdict(list)
    for r in regular_rows:
        by_category[r["category"]].append(r)

    # Optimize each category independently (or fast diverse path).
    reassigned_regular_by_bucket = defaultdict(list)
    category_items = sorted(by_category.items())
    total_categories = len(category_items)

    def _append_best(best: Dict[str, List[dict]]) -> None:
        for bucket, bucket_rows in best.items():
            for row in bucket_rows:
                new_row = copy.deepcopy(row)
                new_row["bucket_index"] = bucket
                reassigned_regular_by_bucket[bucket].append(new_row)

    if args.fast:
        print(
            "Mode: --fast (preprocessing-style diversity + bucket fill; no local search)"
        )
        for cat_i, (category, candidates) in enumerate(category_items, start=1):
            bucket_targets = {k: int(v) for k, v in dict(targets[category]).items()}
            if sum(bucket_targets.values()) != len(candidates):
                raise ValueError(f"Count mismatch in category {category}")
            if not args.no_progress:
                print(
                    f"Diverse assign {cat_i}/{total_categories}: {category} "
                    f"(n={len(candidates)})"
                )
            cat_rng = random.Random(rng.randrange(2**31))
            best = fast_diverse_bucket_assignment(
                candidates,
                bucket_targets,
                cat_rng,
                max_per_video=args.max_per_video,
            )
            _append_best(best)
    else:
        max_workers = args.workers
        if max_workers == 0:
            max_workers = min(32, (os.cpu_count() or 8))
        max_workers = max(1, max_workers)

        if max_workers == 1:
            for cat_i, (category, candidates) in enumerate(category_items, start=1):
                bucket_targets = dict(targets[category])
                if sum(int(v) for v in bucket_targets.values()) != len(candidates):
                    raise ValueError(f"Count mismatch in category {category}")
                if not args.no_progress:
                    print(
                        f"Optimizing category {cat_i}/{total_categories}: {category} "
                        f"(n={len(candidates)}, restarts={args.restarts})"
                    )
                best = optimize_category_assignment(
                    candidates=candidates,
                    bucket_targets=bucket_targets,
                    rng=rng,
                    restarts=args.restarts,
                    swap_steps=args.swap_steps,
                    show_progress=(not args.no_progress),
                    progress_prefix=f"  restarts {cat_i}/{total_categories} {category}",
                    idle_patience=args.idle_patience,
                )
                _append_best(best)
        else:
            print(
                f"Parallel local search: workers={max_workers} "
                f"(restarts={args.restarts}, swap_steps={args.swap_steps}, "
                f"idle_patience={args.idle_patience})"
            )
            payloads = []
            for category, candidates in category_items:
                bucket_targets = dict(targets[category])
                if sum(int(v) for v in bucket_targets.values()) != len(candidates):
                    raise ValueError(f"Count mismatch in category {category}")
                payloads.append(
                    (
                        category,
                        candidates,
                        bucket_targets,
                        rng.randrange(2**31),
                        args.restarts,
                        args.swap_steps,
                        args.idle_patience,
                    )
                )
            results: Dict[str, Dict[str, List[dict]]] = {}
            done = 0
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                future_map = {
                    ex.submit(_optimize_category_worker, p): p[0] for p in payloads
                }
                for fut in as_completed(future_map):
                    category, best = fut.result()
                    results[category] = best
                    done += 1
                    if not args.no_progress:
                        print(f"  finished {done}/{total_categories}: {category}")
            for category, _candidates in category_items:
                _append_best(results[category])

    # Shuffle regular rows within each bucket for output ordering.
    for b in list(reassigned_regular_by_bucket.keys()):
        rng.shuffle(reassigned_regular_by_bucket[b])

    # Reconstruct full table. If input csv exists, preserve non-regular positions.
    original_by_bucket = defaultdict(list)
    if input_path is not None:
        for row in rows:
            original_by_bucket[row["bucket_index"]].append(row)
    else:
        # Build regular-only bucket layout in fresh randomized order.
        for bucket in reassigned_regular_by_bucket:
            for row in reassigned_regular_by_bucket[bucket]:
                original_by_bucket[bucket].append(row)

    rebuilt = []
    for bucket in sorted(original_by_bucket, key=lambda x: int(x)):
        reg_iter = iter(reassigned_regular_by_bucket[bucket])
        trial_idx = 0
        for orig in original_by_bucket[bucket]:
            trial_idx += 1
            if orig["trial_type"] == "regular":
                row = copy.deepcopy(next(reg_iter))
            else:
                row = copy.deepcopy(orig)
            row["bucket_trial_index"] = str(trial_idx)
            row["bucket_index"] = bucket
            rebuilt.append(row)

        # Ensure no leftover regular rows for this bucket.
        try:
            leftover = next(reg_iter)
            raise RuntimeError(
                f"Leftover regular rows in bucket {bucket}: {leftover['stem']}"
            )
        except StopIteration:
            pass

    # Write output assignment.
    write_csv(output_path, rebuilt)

    # Write before/after reports.
    # In build-from-filter mode, `rows` has blank bucket_index values; use the
    # pre-reassignment bucket layout instead so the "before" report is valid.
    if input_path is None:
        before_source_rows = []
        for bucket in sorted(original_by_bucket, key=lambda x: int(x)):
            for row in original_by_bucket[bucket]:
                rr = copy.deepcopy(row)
                rr["bucket_index"] = bucket
                before_source_rows.append(rr)
    else:
        before_source_rows = rows
    before_rows = build_concentration_rows(before_source_rows, bucket_col="bucket_index")
    after_rows = build_concentration_rows(rebuilt, bucket_col="bucket_index")

    combined = []
    for r in before_rows:
        rr = dict(r)
        rr["version"] = "before"
        combined.append(rr)
    for r in after_rows:
        rr = dict(r)
        rr["version"] = "after"
        combined.append(rr)
    write_csv(concentration_path, combined)

    global_combined = build_global_summary(before_rows, "before") + build_global_summary(
        after_rows, "after"
    )
    write_csv(global_path, global_combined)

    # Console summary.
    print(f"Wrote {output_path}")
    print(f"Wrote {concentration_path}")
    print(f"Wrote {global_path}")
    print(f"Regular rows: {len(regular_rows)}")
    print(f"Non-regular rows preserved: {len(non_regular_rows)}")
    print(f"Filter list source: {filter_list_path}")
    print(f"Allowed stems loaded: {len(allowed_stems)}")
    print(f"Minimum regular score threshold: {args.min_score}")
    if input_path is None:
        print(
            f"Mode: build_from_filter (n_per_category={args.n_per_category}, "
            f"n_buckets={args.n_buckets}, max_per_subject={args.max_per_subject}, "
            f"max_per_video={args.max_per_video})"
        )
    else:
        print(f"Mode: rebalance_existing ({input_path})")
    print(f"Category allowlist source: {categories_path}")
    print(f"Category allowlist size: {len(allowed_categories)}")

    by_version_bucket = defaultdict(dict)
    for row in global_combined:
        by_version_bucket[row["bucket"]][row["version"]] = row

    print("\nGlobal concentration change by bucket:")
    for bucket in sorted(by_version_bucket, key=lambda x: int(x)):
        b = by_version_bucket[bucket]["before"]
        a = by_version_bucket[bucket]["after"]
        print(
            "bucket",
            bucket,
            "same_video_pair_rate",
            f"{b['same_video_pair_rate_global']} -> {a['same_video_pair_rate_global']}",
            "same_subject_pair_rate",
            f"{b['same_subject_pair_rate_global']} -> {a['same_subject_pair_rate_global']}",
            "severe_cats",
            f"{b['severe_categories_top_frac_ge_0_5']} -> {a['severe_categories_top_frac_ge_0_5']}",
        )


if __name__ == "__main__":
    main()
