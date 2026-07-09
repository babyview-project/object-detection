#!/usr/bin/env python3
"""Generate per-image THINGS embeddings from BabyDinov3 training checkpoints.

Walks THINGS category image folders (same layout as DINOv3 THINGS exports) and writes
one L2-normalized vector per image:

  {out_dir}/step_{iteration}/{category}/{stem}.npy

where ``stem`` matches the image basename (e.g. ``apple_01b`` for ``apple_01b.jpg``),
consistent with ``facebook_dinov3-vitb16-pretrain-lvd1689m/{category}/apple_01b.npy``.

Examples:
  conda activate vislearnlabpy

  CUDA_VISIBLE_DEVICES=6 python analysis/manuscript-2026/scripts/create_babydinov3_things_embeddings.py \\
    --checkpoint-step 119999 --skip-existing

  CUDA_VISIBLE_DEVICES=6 python ... --load-mode hf \\
    --hf-model /data/khaiaw/baby_dinov3_backup/grad_accum_1/ckpt/119999/huggingface
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
from create_babydinov3_crop_embeddings import (  # noqa: E402
    CATEGORY_SET_FILES,
    CropItem,
    IMAGE_EXTENSIONS,
    apply_skip_existing,
    build_existing_embedding_index,
    discover_checkpoint_steps,
    embed_crops_dinov3_native,
    embed_crops_hf,
    load_allowed_categories,
    resolve_checkpoint_steps,
    write_embedding_list,
    _ensure_dinov3_on_path,
)

DEFAULT_CHECKPOINT_ROOT = Path("/data/khaiaw/baby_dinov3_backup/grad_accum_1/ckpt")
DEFAULT_CONFIG_FILE = Path("/data/khaiaw/baby_dinov3_backup/grad_accum_1/config.yaml")
DEFAULT_DINOV3_REPO = Path("/ccn2/u/khaiaw/Code/baselines/dinov3")

DEFAULT_THINGS_IMAGES_DIR = Path(
    "/ccn2/dataset/babyview/outputs_20250312/things_bv_overlapping_categories_corrected"
)
# Writable on lab nodes (ccn2 THINGS tree is read-only for most users).
DEFAULT_THINGS_EMB_OUT = Path(
    "/data2/dataset/babyview/868_hours/outputs/things_babydinov3_grad_accum_1"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BabyDinov3 embeddings for THINGS exemplar images.")
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-step", type=int, default=None)
    ck.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)
    ck.add_argument("--all-checkpoints", action="store_true")
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE)
    p.add_argument(
        "--load-mode",
        choices=("dcp", "pth", "hf"),
        default=os.environ.get("BV_BABYDINOV3_LOAD_MODE", "dcp"),
    )
    p.add_argument(
        "--hf-model",
        type=str,
        default=os.environ.get("BV_BABYDINOV3_HF_MODEL", "awwkl/dinov3-vitl-babyview"),
    )
    p.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    p.add_argument(
        "--dcp-source",
        choices=("teacher", "student"),
        default=os.environ.get("BV_BABYDINOV3_DCP_SOURCE", "teacher"),
    )
    p.add_argument("--category-set", choices=tuple(CATEGORY_SET_FILES), default="valid129")
    p.add_argument(
        "--things-images-dir",
        type=Path,
        default=Path(os.environ.get("THINGS_IMAGES_DIR", str(DEFAULT_THINGS_IMAGES_DIR))),
        help="THINGS images: {dir}/{category}/{stem}.jpg",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "THINGS_BABYDINOV3_EMBED_OUT",
                str(DEFAULT_THINGS_EMB_OUT),
            )
        ),
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", type=str, default=os.environ.get("BV_DEVICE", "cuda"))
    p.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-images", type=int, default=0, help="Debug cap (0 = all).")
    p.add_argument("--write-embedding-list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile-model", action="store_true", help="torch.compile (hf mode only).")
    return p.parse_args()


def resolve_category_subdir(images_root: Path, cat: str) -> Path | None:
    direct = images_root / cat
    if direct.is_dir():
        return direct
    for p in images_root.iterdir():
        if p.is_dir() and p.name.lower() == cat.lower():
            return p
    return None


def collect_things_images(
    images_dir: Path,
    allowed_categories: set[str],
    max_images: int,
) -> tuple[list[CropItem], dict[str, int]]:
    ext_ok = {e.lower() for e in IMAGE_EXTENSIONS}
    items: list[CropItem] = []
    missing_dir = 0

    for cat in tqdm(sorted(allowed_categories), desc="scan THINGS categories"):
        cat_dir = resolve_category_subdir(images_dir, cat)
        if cat_dir is None:
            missing_dir += 1
            continue
        for img_path in sorted(cat_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in ext_ok:
                continue
            items.append(
                CropItem(category=cat, stem=img_path.stem.lower(), image_path=img_path)
            )
            if max_images > 0 and len(items) >= max_images:
                break
        if max_images > 0 and len(items) >= max_images:
            break

    stats = {
        "n_categories_requested": len(allowed_categories),
        "n_categories_missing_dir": missing_dir,
        "n_images": len(items),
    }
    return items, stats


def main() -> None:
    args = parse_args()
    if not args.things_images_dir.is_dir():
        raise FileNotFoundError(f"THINGS images dir not found: {args.things_images_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp

    if args.load_mode in ("dcp", "pth"):
        if not args.config_file.is_file():
            raise FileNotFoundError(f"Config not found: {args.config_file}")
        _ensure_dinov3_on_path(args.dinov3_repo)

    steps = resolve_checkpoint_steps(args)
    print(f"Checkpoints to run ({len(steps)}): {steps}")

    allowed = load_allowed_categories(CATEGORY_SET_FILES[args.category_set])
    print(
        f"Collecting THINGS images (category_set={args.category_set}, "
        f"{len(allowed)} categories)..."
    )
    items, stats = collect_things_images(args.things_images_dir, allowed, args.max_images)
    print(
        f"THINGS images: {stats['n_images']:,} | "
        f"categories missing dir: {stats['n_categories_missing_dir']}"
    )

    run_records = []
    for step in tqdm(steps, desc="checkpoints"):
        if args.load_mode == "hf":
            out_step = args.out_dir / "hf" / args.hf_model.replace("/", "_")
        else:
            out_step = args.out_dir / f"step_{step}"
        out_step.mkdir(parents=True, exist_ok=True)

        to_embed = items
        skipped = 0
        if args.skip_existing:
            existing = build_existing_embedding_index(out_step)
            to_embed, skipped = apply_skip_existing(items, out_step, existing)
        print(f"step {step}: embed {len(to_embed):,} | skipped existing: {skipped:,}")

        if args.dry_run:
            print(f"[dry-run] step {step}: would embed {len(to_embed)} -> {out_step}")
            continue
        if not to_embed:
            print(f"step {step}: nothing to do.")
            continue

        common = dict(
            items=to_embed,
            out_step_dir=out_step,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_dtype=args.save_dtype,
            use_amp=use_amp,
        )

        if args.load_mode == "hf":
            embed_stats = embed_crops_hf(
                **common,
                hf_model=args.hf_model,
                compile_model=args.compile_model,
            )
            weights = args.hf_model
        elif args.load_mode == "pth":
            weights = args.checkpoint_root / str(step) / "teacher_checkpoint.pth"
            if not weights.is_file():
                raise FileNotFoundError(f"Missing teacher checkpoint: {weights}")
            embed_stats = embed_crops_dinov3_native(
                **common,
                config_file=args.config_file,
                weights_path=weights,
                dinov3_repo=args.dinov3_repo,
                dcp_source=args.dcp_source,
            )
        else:
            weights = args.checkpoint_root / str(step)
            if not weights.is_dir():
                raise FileNotFoundError(f"Missing DCP checkpoint dir: {weights}")
            embed_stats = embed_crops_dinov3_native(
                **common,
                config_file=args.config_file,
                weights_path=weights,
                dinov3_repo=args.dinov3_repo,
                dcp_source=args.dcp_source,
            )

        manifest_path = None
        if args.write_embedding_list:
            manifest_path = write_embedding_list(out_step, to_embed)

        rec = {
            "step": step,
            "out_dir": str(out_step),
            "weights": str(weights),
            "load_mode": args.load_mode,
            "n_items": len(to_embed),
            **embed_stats,
        }
        if manifest_path:
            rec["embedding_list"] = str(manifest_path)
        run_records.append(rec)
        print(f"step {step}: saved {embed_stats['n_ok']:,} embeddings under {out_step}")

    meta = {
        "dataset": "things",
        "checkpoint_root": str(args.checkpoint_root),
        "config_file": str(args.config_file),
        "out_dir": str(args.out_dir),
        "load_mode": args.load_mode,
        "category_set": args.category_set,
        "things_images_dir": str(args.things_images_dir),
        "steps": steps,
        "collection_stats": stats,
        "records": run_records,
    }
    meta_path = args.out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
