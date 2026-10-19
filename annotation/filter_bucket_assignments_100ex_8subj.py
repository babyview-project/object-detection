#!/usr/bin/env python3
"""Filter bucket assignment CSVs to categories with >=100 CLIP-filter exemplars and >=8 subjects.

1. Loads the category allowlist (default: 163 overlap categories), then removes names in
   ``--exclude-categories-file`` (default: writing tools, horse, CDI body_parts overlap).
   Also see ``data/things_bv_overlap_categories_annotation_use.txt`` for the same subset
   as a ready-made allowlist.
2. Scans the CLIP filter list and excludes any remaining category with fewer than
   ``--n-per-category`` stems or fewer than ``--min-unique-subjects`` distinct subject IDs,
   and (when resampling) fewer than ``n_per_category`` on-disk crops passing ``--min-score``.
3. Drops all regular rows for excluded categories (non-regular trials are unchanged).
4. ``--max-per-video`` (default 2): categories must have enough spread across videos that
   ``sum_v min(count_v, max_per_video) >= n_per_category`` on the matched crop pool;
   otherwise they are dropped (cannot get 100 exemplars without exceeding the cap).
   Sampling then stays within the cap for kept categories. ``--min-unique-subjects``
   (default 8) applies to the filter list at eligibility and to the resampled set.
   When ``--max-per-video`` > 0, **every** remaining eligible category is resampled.
   When it is 0, only categories with too few subjects in the input CSV are resampled.

Writes the filtered CSV, an eligible-categories list, and a short report.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import shared helpers from sibling module.
_ANN = Path(__file__).resolve().parent
if str(_ANN) not in sys.path:
    sys.path.insert(0, str(_ANN))

import resample_diverse_buckets as rdb  # noqa: E402


def load_exclude_category_names(path: Path) -> Set[str]:
    """Lowercase category names to drop from the allowlist (lines starting with # skipped)."""
    if not path.exists():
        return set()
    out: Set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.lower())
    return out


def parse_subject_id(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) > 2 else ""


def count_unique_subjects(rows: List[dict]) -> int:
    s: Set[str] = set()
    for r in rows:
        sid = parse_subject_id(r["stem"])
        if sid:
            s.add(sid)
    return len(s)


def max_selectable_under_video_cap(paths: List[Path], max_per_video: int) -> int:
    """
    Max number of distinct crops one can take if each source video (stem parts 2–6)
    contributes at most ``max_per_video`` exemplars. If this is < n_per_category, it is
    impossible to reach n_per_category without breaking the cap.
    """
    if max_per_video <= 0 or not paths:
        return len(paths)
    vc: Counter[str] = Counter()
    for p in paths:
        _, vid = rdb.parse_subject_video(p.stem)
        vc[vid] += 1
    return sum(min(c, max_per_video) for c in vc.values())


def matched_paths_for_category(
    filter_list_path: Path,
    cropped_dir: Path,
    category: str,
    min_score: float,
    allowed_by_category: Optional[Dict[str, Set[str]]] = None,
) -> List[Path]:
    if allowed_by_category is None:
        allowed_by_category = rdb.load_allowed_stems_by_category(filter_list_path)
    cat = category.strip().lower()
    cat_dir = rdb.get_category_crop_dir(cropped_dir, cat)
    if cat_dir is None:
        return []
    path_by_stem: Dict[str, Path] = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for p in cat_dir.glob(f"*{ext}"):
            path_by_stem[p.stem] = p
    allowed_stems = allowed_by_category.get(cat, set())
    out: List[Path] = []
    for s in allowed_stems:
        if s not in path_by_stem:
            continue
        p = path_by_stem[s]
        try:
            if rdb.parse_detection_score({"stem": p.stem}) < min_score:
                continue
        except ValueError:
            continue
        out.append(p)
    return out


def resample_one_category(
    category: str,
    bucket_targets: Dict[str, int],
    filter_list_path: Path,
    cropped_dir: Path,
    n_take: int,
    min_unique_subjects: int,
    min_score: float,
    rng: rdb.random.Random,
    max_per_video: int = 0,
    max_attempts: int = 40,
    allowed_by_category: Optional[Dict[str, Set[str]]] = None,
    matched_paths: Optional[List[Path]] = None,
    enforce_strict_video_cap: bool = False,
) -> List[dict]:
    """Return full assignment rows (with bucket_index set) for one category."""
    if matched_paths is not None:
        matched = matched_paths
    else:
        matched = matched_paths_for_category(
            filter_list_path,
            cropped_dir,
            category,
            min_score,
            allowed_by_category=allowed_by_category,
        )
    if len(matched) < n_take:
        raise RuntimeError(
            f"{category}: only {len(matched)} paths pass score>={min_score}, need {n_take}"
        )

    cat_lower = category.strip().lower()
    for attempt in range(max_attempts):
        crng = rdb.random.Random(rng.randrange(2**31))
        temp_rows = [{"stem": p.stem, "_path": p} for p in matched]
        ordered = rdb.sample_rows_diverse_subject_video(
            temp_rows,
            crng,
            take_n=n_take,
            max_per_video=max_per_video,
            enforce_strict_video_cap=enforce_strict_video_cap,
        )
        subs: Set[str] = set()
        for r in ordered:
            sid = parse_subject_id(r["stem"])
            if sid:
                subs.add(sid)
        if len(subs) >= min_unique_subjects:
            best = rdb.assign_diverse_to_bucket_targets(ordered, bucket_targets)
            flat: List[dict] = []
            for bucket, bucket_rows in sorted(best.items(), key=lambda x: int(x[0])):
                for r in bucket_rows:
                    p = r["_path"]
                    flat.append(
                        {
                            "bucket_index": bucket,
                            "category": cat_lower,
                            "path": str(p.resolve()),
                            "stem": p.stem,
                            "trial_type": "regular",
                            "attention_check_source_category": "",
                            "bucket_trial_index": "",
                        }
                    )
            return flat
    raise RuntimeError(
        f"{category}: could not reach {min_unique_subjects} unique subjects in "
        f"{max_attempts} resample attempts (pool may be too concentrated)."
    )


def rebuild_output_rows(
    all_rows: List[dict],
    eligible: Set[str],
    excluded: Set[str],
    replaced: Dict[str, List[dict]],
) -> List[dict]:
    """Drop excluded regular rows; substitute resampled categories; renumber trials per bucket."""
    original_by_bucket: Dict[int, List[dict]] = defaultdict(list)
    for r in all_rows:
        original_by_bucket[int(r["bucket_index"])].append(r)
    for b in original_by_bucket:
        original_by_bucket[b].sort(key=lambda x: int(x["bucket_trial_index"]))

    pools: Dict[Tuple[str, str], deque] = {}
    for cat, rows in replaced.items():
        for r in rows:
            key = (str(r["bucket_index"]), cat)
            pools.setdefault(key, deque()).append(r)

    rebuilt: List[dict] = []
    for bucket in sorted(original_by_bucket.keys()):
        trial_idx = 0
        for orig in original_by_bucket[bucket]:
            if orig["trial_type"] != "regular":
                trial_idx += 1
                row = copy.deepcopy(orig)
                row["bucket_trial_index"] = str(trial_idx)
                row["bucket_index"] = str(bucket)
                rebuilt.append(row)
                continue

            cat = orig["category"].strip().lower()
            if cat in excluded:
                continue
            if cat not in eligible:
                continue

            trial_idx += 1
            key = (str(bucket), cat)
            if cat in replaced and key in pools and pools[key]:
                row = copy.deepcopy(pools[key].popleft())
                row["bucket_trial_index"] = str(trial_idx)
                row["bucket_index"] = str(bucket)
                rebuilt.append(row)
            elif cat in replaced:
                raise RuntimeError(
                    f"Missing resampled row for bucket={bucket} category={cat}"
                )
            else:
                row = copy.deepcopy(orig)
                row["bucket_trial_index"] = str(trial_idx)
                row["bucket_index"] = str(bucket)
                rebuilt.append(row)
    return rebuilt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        default="annotation/sampled_object_crops_100_bucket_assignments_diverse_rebalanced.csv",
    )
    p.add_argument(
        "--output-csv",
        default=(
            "annotation/"
            "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap.csv"
        ),
    )
    p.add_argument(
        "--eligible-categories-out",
        default="data/included_categories.txt",
        help="Writes one category per line (eligible after filter-list constraints).",
    )
    p.add_argument(
        "--report",
        default=(
            "annotation/"
            "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_report.txt"
        ),
    )
    p.add_argument(
        "--filter-list",
        default=(
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/"
            "clip_image_embeddings_filtered-by-clip-0.27_exclude-people_"
            "exclude-subject-00270001.txt"
        ),
    )
    p.add_argument(
        "--categories-file",
        default="data/things_bv_overlap_categories_exclude_zero_precisions.txt",
    )
    p.add_argument(
        "--exclude-categories-file",
        default="data/annotation_excluded_categories.txt",
        help=(
            "Categories to remove from the allowlist before eligibility (one per line; "
            "use empty path or --no-exclude-categories-file to disable)."
        ),
    )
    p.add_argument(
        "--no-exclude-categories-file",
        action="store_true",
        help="Do not load --exclude-categories-file; use every name from --categories-file.",
    )
    p.add_argument(
        "--allow-exceed-per-video-cap-to-reach-n",
        action="store_true",
        help=(
            "When set with --max-per-video > 0, do not drop categories whose matched pool "
            "cannot supply n_per_category exemplars without exceeding the per-video cap; "
            "allow the sampler to go over the cap when needed (legacy behavior)."
        ),
    )
    p.add_argument(
        "--cropped-dir",
        default="/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class",
    )
    p.add_argument("--n-per-category", type=int, default=100)
    p.add_argument(
        "--min-unique-subjects",
        type=int,
        default=8,
        help="Minimum distinct subject IDs in filter list and in each resampled category.",
    )
    p.add_argument("--min-score", type=float, default=0.27)
    p.add_argument("--seed", type=int, default=20260327)
    p.add_argument(
        "--max-per-video",
        type=int,
        default=2,
        help=(
            "Target max exemplars per source video during greedy sampling (0 = no cap). "
            "Under-cap picks are exhausted first; going over is only used to reach "
            "n-per-category if the pool requires it. When > 0, all eligible categories "
            "are resampled."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-category resampling progress.",
    )
    args = p.parse_args()

    categories_path = Path(args.categories_file)
    with categories_path.open() as f:
        categories_raw = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    exclude_path = Path(args.exclude_categories_file)
    exclude_names: Set[str] = set()
    if not args.no_exclude_categories_file and str(args.exclude_categories_file).strip():
        if exclude_path.exists():
            exclude_names = load_exclude_category_names(exclude_path)
        elif not args.quiet:
            print(
                f"Warning: --exclude-categories-file missing ({exclude_path}); "
                "continuing without policy exclusions.",
                flush=True,
            )
    excluded_policy = sorted(
        {c for c in categories_raw if c.strip().lower() in exclude_names},
        key=str.lower,
    )
    categories = [c for c in categories_raw if c.strip().lower() not in exclude_names]

    filter_list_path = Path(args.filter_list)
    if not filter_list_path.exists():
        raise FileNotFoundError(filter_list_path)

    if not args.quiet:
        print("Loading CLIP filter list into memory (one pass)…", flush=True)
    allowed_by_category = rdb.load_allowed_stems_by_category(filter_list_path)

    rng = rdb.random.Random(args.seed)
    cropped_dir = Path(args.cropped_dir)

    eligible: Set[str] = set()
    excluded_too_few: List[str] = []
    excluded_subjects: List[str] = []
    for c in categories:
        cl = c.strip().lower()
        stems = allowed_by_category.get(cl, set())
        if len(stems) < args.n_per_category:
            excluded_too_few.append(cl)
            continue
        subj_set: Set[str] = set()
        for stem in stems:
            sid = parse_subject_id(stem)
            if sid:
                subj_set.add(sid)
        if len(subj_set) < args.min_unique_subjects:
            excluded_subjects.append(cl)
            continue
        eligible.add(cl)

    excluded: Set[str] = set(excluded_too_few) | set(excluded_subjects)

    input_path = Path(args.input_csv)
    with input_path.open(newline="") as f:
        all_rows = list(csv.DictReader(f))

    regular_by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in all_rows:
        if r["trial_type"] == "regular":
            regular_by_cat[r["category"].strip().lower()].append(r)

    resample_plan: List[Tuple[str, Dict[str, int]]] = []
    for cat in sorted(eligible):
        rows = regular_by_cat.get(cat, [])
        n = len(rows)
        if n != args.n_per_category:
            raise RuntimeError(
                f"{cat}: expected {args.n_per_category} regular rows in input, got {n}"
            )
        bucket_targets: Dict[str, int] = defaultdict(int)
        for r in rows:
            bucket_targets[r["bucket_index"]] += 1
        bucket_targets = {k: int(v) for k, v in bucket_targets.items()}
        if sum(bucket_targets.values()) != args.n_per_category:
            raise RuntimeError(f"{cat}: bucket target sum mismatch")

        u = count_unique_subjects(rows)
        need_resample = args.max_per_video > 0 or u < args.min_unique_subjects
        if need_resample:
            resample_plan.append((cat, bucket_targets))

    excluded_insufficient_crops: List[Tuple[str, int]] = []
    excluded_insufficient_video_spread: List[Tuple[str, int]] = []
    matched_pool: Dict[str, List[Path]] = {}
    filtered_plan: List[Tuple[str, Dict[str, int]]] = []
    if resample_plan:
        if not args.quiet:
            print(
                f"Checking on-disk crops + score>={args.min_score} for "
                f"{len(resample_plan)} categories to resample…",
                flush=True,
            )
        for cat, bucket_targets in resample_plan:
            matched = matched_paths_for_category(
                filter_list_path,
                cropped_dir,
                cat,
                args.min_score,
                allowed_by_category=allowed_by_category,
            )
            if len(matched) < args.n_per_category:
                excluded_insufficient_crops.append((cat, len(matched)))
                eligible.discard(cat)
                continue
            if (
                args.max_per_video > 0
                and not args.allow_exceed_per_video_cap_to_reach_n
            ):
                cap_ok = max_selectable_under_video_cap(
                    matched, args.max_per_video
                )
                if cap_ok < args.n_per_category:
                    excluded_insufficient_video_spread.append((cat, cap_ok))
                    eligible.discard(cat)
                    continue
            matched_pool[cat] = matched
            filtered_plan.append((cat, bucket_targets))
        resample_plan = filtered_plan

    excluded = excluded | {c for c, _ in excluded_insufficient_crops}
    excluded = excluded | {c for c, _ in excluded_insufficient_video_spread}

    replaced: Dict[str, List[dict]] = {}
    if not args.quiet and resample_plan:
        print(
            f"Resampling {len(resample_plan)} categories (sampling)…",
            flush=True,
        )
    for i, (cat, bucket_targets) in enumerate(resample_plan, start=1):
        if not args.quiet:
            print(f"  [{i}/{len(resample_plan)}] {cat}", flush=True)
        replaced[cat] = resample_one_category(
            cat,
            bucket_targets,
            filter_list_path,
            cropped_dir,
            args.n_per_category,
            args.min_unique_subjects,
            args.min_score,
            rng,
            max_per_video=args.max_per_video,
            allowed_by_category=allowed_by_category,
            matched_paths=matched_pool[cat],
            enforce_strict_video_cap=(
                args.max_per_video > 0
                and not args.allow_exceed_per_video_cap_to_reach_n
            ),
        )

    out_rows = rebuild_output_rows(all_rows, eligible, excluded, replaced)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdb.write_csv(out_path, out_rows)

    eligible_path = Path(args.eligible_categories_out)
    eligible_path.parent.mkdir(parents=True, exist_ok=True)
    with eligible_path.open("w") as f:
        for c in sorted(eligible):
            f.write(f"{c}\n")

    report_path = Path(args.report)
    lines = [
        f"Input: {input_path}",
        f"Filter list: {filter_list_path}",
        f"Categories file: {categories_path}",
        f"Excluded by policy file ({exclude_path}): {len(excluded_policy)}",
        ", ".join(excluded_policy) if excluded_policy else "(none — use --no-exclude-categories-file?)",
        "",
        f"max_per_video (target cap per source video): {args.max_per_video}",
        "Per-video cap policy: "
        + (
            "off (max_per_video=0)"
            if args.max_per_video <= 0
            else (
                "soft (may exceed K to reach n; no feasibility drop)"
                if args.allow_exceed_per_video_cap_to_reach_n
                else "strict (drop if sum_v min(count_v,K) < n; sampling never exceeds K)"
            )
        ),
        f"n_per_category (min stems in filter list): {args.n_per_category}",
        f"min_unique_subjects: {args.min_unique_subjects}",
        f"min_score: {args.min_score}",
        "",
        f"Eligible categories: {len(eligible)}",
        f"Excluded (< {args.n_per_category} stems in filter list): {len(excluded_too_few)}",
        ", ".join(sorted(excluded_too_few)),
        "",
        f"Excluded (< {args.min_unique_subjects} subjects in filter list): {len(excluded_subjects)}",
        ", ".join(sorted(excluded_subjects)),
        "",
        "Excluded (fewer than n_per_category on-disk crops passing score filter): "
        f"{len(excluded_insufficient_crops)}",
        ", ".join(f"{c} (n={n})" for c, n in sorted(excluded_insufficient_crops))
        if excluded_insufficient_crops
        else "(none)",
        "",
        "Excluded (cannot reach n_per_category without exceeding max_per_video; "
        f"sum_v min(count_v, K) < n): {len(excluded_insufficient_video_spread)}",
        ", ".join(
            f"{c} (max_under_cap={m})"
            for c, m in sorted(excluded_insufficient_video_spread)
        )
        if excluded_insufficient_video_spread
        else "(none)",
        "",
        f"Categories resampled: {len(replaced)}",
        ", ".join(sorted(replaced.keys())) if replaced else "(none)",
        "",
        f"Output rows: {len(out_rows)} (was {len(all_rows)})",
        f"Wrote: {out_path}",
        f"Wrote: {eligible_path}",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
