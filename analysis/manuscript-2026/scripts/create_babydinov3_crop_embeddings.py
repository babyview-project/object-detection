#!/usr/bin/env python3
"""Generate per-crop image embeddings from BabyDinov3 training checkpoints.

Reads CLIP-filtered crop JPGs, runs each through a checkpoint backbone, and writes
one L2-normalized vector per crop (same layout as existing facebook_dinov3 embeddings):

  {out_dir}/step_{iteration}/{category}/{stem}.npy

Inputs (defaults):
  - Crops: yoloe_cdi_all_cropped_by_class + CLIP filter list (threshold 0.27)
  - Checkpoints: /data/khaiaw/baby_dinov3_backup/grad_accum_1/ckpt/{step}/

The dinov3 repo on disk is only used to *load* those .distcp checkpoints (not to train).
Override with --dinov3-repo if your clone lives elsewhere.

Examples:
  conda activate vislearnlabpy

  # One checkpoint, full filtered valid129 crops:
  CUDA_VISIBLE_DEVICES=6 python analysis/manuscript-2026/scripts/create_babydinov3_crop_embeddings.py \\
    --checkpoint-step 119999 --skip-existing

  # Already on Hugging Face (single model, no DCP folders):
  CUDA_VISIBLE_DEVICES=6 python ... --load-mode hf

  # Training DCP folders store weights under model.teacher.backbone.* (not model.backbone.*):
  CUDA_VISIBLE_DEVICES=6 python ... --checkpoint-step 119999 --dcp-source teacher

  # Or use the HF export bundled next to the DCP shards:
  CUDA_VISIBLE_DEVICES=6 python ... --load-mode hf \\
    --hf-model /data/khaiaw/baby_dinov3_backup/grad_accum_1/ckpt/119999/huggingface
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_CHECKPOINT_ROOT = Path("/data/khaiaw/baby_dinov3_backup/grad_accum_1/ckpt")
DEFAULT_CONFIG_FILE = Path("/data/khaiaw/baby_dinov3_backup/grad_accum_1/config.yaml")
DEFAULT_DINOV3_REPO = Path("/ccn2/u/khaiaw/Code/baselines/dinov3")

DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")
DEFAULT_EMB_BASE = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
DEFAULT_FILTER_LIST = (
    DEFAULT_EMB_BASE
    / "clip_image_embeddings_filtered-by-clip-0.27_exclude-people_exclude-subject-00270001.txt"
)

CATEGORY_SET_FILES = {
    "valid129": DATA_DIR / "included_categories_valid129.txt",
    "valid85": DATA_DIR / "included_categories_valid85.txt",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class CropItem:
    category: str
    stem: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BabyDinov3 embeddings for filtered object crops.")
    ck = p.add_mutually_exclusive_group()
    ck.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Single checkpoint iteration folder to run (recommended for testing).",
    )
    ck.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help="Multiple checkpoint iterations.",
    )
    ck.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Run every integer-named folder under --checkpoint-root.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("BV_BABYDINOV3_CKPT_ROOT", str(DEFAULT_CHECKPOINT_ROOT))),
        help="Parent directory with integer-named DCP checkpoint folders.",
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=Path(os.environ.get("BV_BABYDINOV3_CONFIG", str(DEFAULT_CONFIG_FILE))),
        help="Training config.yaml for the run (vit_large, crops, etc.).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "BV_BABYDINOV3_EMBED_OUT",
                str(DEFAULT_EMB_BASE / "babydinov3_grad_accum_1"),
            )
        ),
        help="Root output directory; writes step_{iter}/{category}/{stem}.npy",
    )
    p.add_argument(
        "--load-mode",
        choices=("dcp", "pth", "hf"),
        default=os.environ.get("BV_BABYDINOV3_LOAD_MODE", "dcp"),
        help="dcp: FSDP shard dir; pth: consolidated teacher .pth; hf: HuggingFace AutoModel.",
    )
    p.add_argument(
        "--hf-model",
        type=str,
        default=os.environ.get("BV_BABYDINOV3_HF_MODEL", "awwkl/dinov3-vitl-babyview"),
        help="HuggingFace model id (load-mode=hf).",
    )
    p.add_argument(
        "--dinov3-repo",
        type=Path,
        default=Path(os.environ.get("BV_DINOV3_REPO", str(DEFAULT_DINOV3_REPO))),
        help="Path to facebookresearch/dinov3 clone (for load-mode dcp/pth).",
    )
    p.add_argument(
        "--dcp-source",
        choices=("teacher", "student"),
        default=os.environ.get("BV_BABYDINOV3_DCP_SOURCE", "teacher"),
        help="SSL role to load from a training DCP folder (teacher=EMA; default for eval).",
    )
    p.add_argument("--category-set", choices=tuple(CATEGORY_SET_FILES), default="valid129")
    p.add_argument(
        "--filter-list",
        type=Path,
        default=Path(os.environ.get("BV_CLIP_FILTER_LIST", str(DEFAULT_FILTER_LIST))),
        help="CLIP-pass list: one embedding path per line.",
    )
    p.add_argument(
        "--cropped-dir",
        type=Path,
        default=Path(os.environ.get("BV_CROPPED_DIR", str(DEFAULT_CROPPED_DIR))),
        help="YOLOE cropped images: {cropped_dir}/{category}/{stem}.jpg",
    )
    p.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch (workers>0).")
    p.add_argument("--device", type=str, default=os.environ.get("BV_DEVICE", "cuda"))
    p.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--skip-existing", action="store_true", help="Skip crops with existing .npy.")
    p.add_argument("--max-crops", type=int, default=0, help="Debug cap (0 = all).")
    p.add_argument("--write-embedding-list", action="store_true", help="Write manifest .txt per step.")
    p.add_argument("--dry-run", action="store_true", help="List crops only; no model load.")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable autocast (bf16) during forward pass.",
    )
    p.add_argument(
        "--compile-model",
        action="store_true",
        help="torch.compile the backbone after load (experimental; hf mode only).",
    )
    p.add_argument(
        "--count-filter-lines",
        action="store_true",
        help="Pre-count filter list lines for an accurate tqdm %% (extra full-file read).",
    )
    return p.parse_args()


def load_allowed_categories(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_category_dir_index(cropped_dir: Path) -> dict[str, Path]:
    """Map lowercase category name -> crop subdirectory (one scan)."""
    index: dict[str, Path] = {}
    if not cropped_dir.is_dir():
        return index
    entries = [p for p in cropped_dir.iterdir() if p.is_dir()]
    for p in tqdm(entries, desc="index crop categories", leave=False):
        index[p.name.lower()] = p
    return index


def build_stem_path_index(cat_dir: Path, *, desc: str | None = None) -> dict[str, Path]:
    """Map lowercase stem (no ext) -> image path for one category folder."""
    index: dict[str, Path] = {}
    ext_ok = {e.lower() for e in IMAGE_EXTENSIONS}
    entries = [p for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_ok]
    it = tqdm(entries, desc=desc or f"index {cat_dir.name}", leave=False) if desc else entries
    for p in it:
        index[p.stem.lower()] = p
    return index


def build_existing_embedding_index(out_step_dir: Path) -> set[tuple[str, str]]:
    """One walk of output tree: (category, stem) pairs that already have .npy."""
    existing: set[tuple[str, str]] = set()
    if not out_step_dir.is_dir():
        return existing
    cat_dirs = [p for p in out_step_dir.iterdir() if p.is_dir()]
    for cat_dir in tqdm(cat_dirs, desc="index existing embeddings", leave=False):
        cat = cat_dir.name.lower()
        for npy in cat_dir.glob("*.npy"):
            existing.add((cat, npy.stem.lower()))
    return existing


def _iter_filter_pairs(filter_list: Path, allowed_categories: set[str]):
    with filter_list.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            stem = p.stem if p.suffix.lower() == ".npy" else p.name
            category = p.parent.name.strip().lower()
            if category in allowed_categories:
                yield category, stem.lower()


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_crops_from_filter(
    filter_list: Path,
    cropped_dir: Path,
    allowed_categories: set[str],
    max_crops: int,
    *,
    filter_line_count: int | None = None,
) -> tuple[list[CropItem], dict[str, int]]:
    """Scan filter list once; map each unique (category, stem) to a crop JPG."""
    cat_dirs = build_category_dir_index(cropped_dir)
    stem_cache: dict[str, dict[str, Path]] = {}
    seen: set[tuple[str, str]] = set()

    items: list[CropItem] = []
    n_lines = 0
    missing_image = 0

    total = filter_line_count if filter_line_count and filter_line_count > 0 else None
    pair_iter = _iter_filter_pairs(filter_list, allowed_categories)
    for category, stem in tqdm(pair_iter, total=total, desc="scan filter list", unit=" lines"):
        n_lines += 1
        key = (category, stem)
        if key in seen:
            continue
        seen.add(key)

        cat_dir = cat_dirs.get(category)
        if cat_dir is None:
            missing_image += 1
            continue
        if category not in stem_cache:
            stem_cache[category] = build_stem_path_index(
                cat_dir, desc=f"index crops/{category}"
            )
        img_path = stem_cache[category].get(stem)
        if img_path is None:
            missing_image += 1
            continue

        items.append(CropItem(category=category, stem=stem, image_path=img_path))
        if max_crops > 0 and len(items) >= max_crops:
            break

    stats = {
        "n_lines": n_lines,
        "n_unique": len(seen),
        "missing_image": missing_image,
        "n_resolved": len(items),
    }
    return items, stats


def apply_skip_existing(
    items: list[CropItem],
    out_step_dir: Path,
    existing: set[tuple[str, str]] | None = None,
) -> tuple[list[CropItem], int]:
    if existing is None:
        existing = build_existing_embedding_index(out_step_dir)
    kept = [it for it in items if (it.category, it.stem) not in existing]
    return kept, len(items) - len(kept)


def collect_crops(
    filter_list: Path,
    cropped_dir: Path,
    allowed_categories: set[str],
    out_step_dir: Path | None,
    skip_existing: bool,
    max_crops: int,
    *,
    resolved_cache: list[CropItem] | None = None,
    resolved_stats: dict[str, int] | None = None,
    filter_line_count: int | None = None,
) -> list[CropItem]:
    if resolved_cache is None:
        items, stats = resolve_crops_from_filter(
            filter_list,
            cropped_dir,
            allowed_categories,
            max_crops,
            filter_line_count=filter_line_count,
        )
    else:
        items = resolved_cache
        stats = resolved_stats or {"n_resolved": len(items)}

    skipped = 0
    if skip_existing and out_step_dir is not None:
        items, skipped = apply_skip_existing(items, out_step_dir)

    print(
        f"Crops: {len(items):,} to embed | filter lines scanned: {stats.get('n_lines', '?'):,} | "
        f"unique pairs: {stats.get('n_unique', '?'):,} | missing image: {stats.get('missing_image', 0):,} | "
        f"skipped existing: {skipped:,}"
    )
    return items


def discover_checkpoint_steps(ckpt_root: Path) -> list[int]:
    found = []
    entries = list(ckpt_root.iterdir())
    for p in tqdm(entries, desc="discover checkpoints", leave=False):
        if p.is_dir() and p.name.isdigit():
            found.append(int(p.name))
    if not found:
        raise FileNotFoundError(f"No integer checkpoint folders under {ckpt_root}")
    return sorted(found)


def resolve_checkpoint_steps(args: argparse.Namespace) -> list[int]:
    if args.load_mode == "hf":
        if args.checkpoint_step is not None:
            return [args.checkpoint_step]
        if args.checkpoint_steps:
            return sorted(args.checkpoint_steps)
        return [0]

    if args.all_checkpoints:
        return discover_checkpoint_steps(args.checkpoint_root)

    if args.checkpoint_steps:
        steps = sorted(args.checkpoint_steps)
    elif args.checkpoint_step is not None:
        steps = [args.checkpoint_step]
    else:
        # Default: latest checkpoint only (safe for first test runs).
        steps = [discover_checkpoint_steps(args.checkpoint_root)[-1]]
        print(f"No --checkpoint-step given; using latest checkpoint: {steps[0]}")

    for s in steps:
        p = args.checkpoint_root / str(s)
        if args.load_mode == "pth":
            if not (p / "teacher_checkpoint.pth").is_file() and not p.is_dir():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
        elif not p.is_dir():
            raise FileNotFoundError(f"Checkpoint folder not found: {p}")
    return steps


def _ensure_dinov3_on_path(dinov3_repo: Path) -> None:
    repo = dinov3_repo.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _init_single_gpu_distributed() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def _enable_inference_optimizations() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _make_loader(dataset, batch_size: int, num_workers: int, prefetch_factor: int):
    from torch.utils.data import DataLoader

    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _save_embeddings_batch(
    out_step_dir: Path,
    batch_items: list[CropItem],
    feats: np.ndarray,
    save_dtype: str,
) -> int:
    """Write a batch of vectors; feats already L2-normalized float32."""
    n = 0
    for it, vec in zip(batch_items, feats):
        out_path = out_step_dir / it.category / f"{it.stem}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        v = np.asarray(vec, dtype=np.float32).ravel()
        if save_dtype == "float16":
            v = v.astype(np.float16)
        np.save(out_path, v)
        n += 1
    return n


def embed_crops_hf(
    items: list[CropItem],
    out_step_dir: Path,
    hf_model: str,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    save_dtype: str,
    use_amp: bool,
    compile_model: bool,
) -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoImageProcessor, AutoModel

    _enable_inference_optimizations()

    processor = AutoImageProcessor.from_pretrained(hf_model)
    model = AutoModel.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    if compile_model:
        model = torch.compile(model)

    class _HFDataset(Dataset):
        def __init__(self, crop_items: list[CropItem]):
            self.items = crop_items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            it = self.items[idx]
            img = Image.open(it.image_path).convert("RGB")
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0), idx

    loader = _make_loader(_HFDataset(items), batch_size, num_workers, prefetch_factor)
    autocast_dtype = torch.bfloat16 if use_amp else None

    n_ok, n_fail = 0, 0
    for pixel_values, indices in tqdm(loader, desc="embed hf"):
        try:
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = model(pixel_values=pixel_values)
                else:
                    outputs = model(pixel_values=pixel_values)
            pooled = outputs.pooler_output.float()
            pooled = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            batch_items = [items[i] for i in indices.tolist()]
            n_ok += _save_embeddings_batch(out_step_dir, batch_items, pooled, save_dtype)
        except Exception:
            n_fail += len(indices)
    return {"n_ok": n_ok, "n_fail": n_fail}


def _build_and_load_dinov3_backbone(
    config_file: Path,
    weights_path: Path,
    dinov3_repo: Path,
    *,
    load_from_dcp: bool,
    dcp_source: str,
    output_dir: str,
):
    """Load a teacher-only ViT for eval.

    Training DCP checkpoints use keys like model.{teacher,student}.backbone.*.
    dinov3's default eval loader expects model.backbone.* and fails on raw ckpt dirs.
    """
    import torch.nn as nn

    _ensure_dinov3_on_path(dinov3_repo)
    from dinov3.checkpointer import init_model_from_checkpoint_for_evals, load_checkpoint
    from dinov3.configs import DinoV3SetupArgs, setup_config
    from dinov3.models import build_model_from_cfg
    setup_args = DinoV3SetupArgs(
        config_file=str(config_file),
        pretrained_weights=str(weights_path) if load_from_dcp else None,
        output_dir=output_dir,
        opts=[],
    )
    config = setup_config(setup_args, strict_cfg=False)
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if load_from_dcp:
        # Training ckpts use model.{teacher,student}.backbone.*; ac_compile_parallelize
        # requires a top-level "backbone" key (eval-only sharded ckpts). Skip FSDP for
        # single-GPU crop embedding — DCP load still works on the SSL-shaped wrapper.
        moduledict = nn.ModuleDict({dcp_source: nn.ModuleDict({"backbone": model})})
        model.to_empty(device="cuda")
        load_checkpoint(weights_path, model=moduledict, strict_loading=True)
    else:
        model.to_empty(device="cuda")
        init_model_from_checkpoint_for_evals(model, weights_path, "teacher")
    model.eval()
    return model


def embed_crops_dinov3_native(
    items: list[CropItem],
    out_step_dir: Path,
    config_file: Path,
    weights_path: Path,
    dinov3_repo: Path,
    dcp_source: str,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    save_dtype: str,
    use_amp: bool,
) -> dict:
    import torch
    from torch.utils.data import Dataset

    _ensure_dinov3_on_path(dinov3_repo)
    _init_single_gpu_distributed()
    _enable_inference_optimizations()

    from dinov3.data.transforms import make_classification_eval_transform
    from dinov3.eval.utils import ModelWithNormalize
    from dinov3.run.init import job_context

    class _CropDataset(Dataset):
        def __init__(self, crop_items: list[CropItem], transform):
            self.items = crop_items
            self.transform = transform

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            it = self.items[idx]
            img = Image.open(it.image_path).convert("RGB")
            return self.transform(img), idx

    transform = make_classification_eval_transform()
    loader = _make_loader(_CropDataset(items, transform), batch_size, num_workers, prefetch_factor)
    autocast_dtype = torch.bfloat16 if use_amp else None

    n_ok, n_fail = 0, 0
    with job_context(output_dir=str(out_step_dir), distributed_enabled=True):
        model = _build_and_load_dinov3_backbone(
            config_file,
            weights_path,
            dinov3_repo,
            load_from_dcp=weights_path.is_dir(),
            dcp_source=dcp_source,
            output_dir=str(out_step_dir),
        )
        model = ModelWithNormalize(model)
        if device.startswith("cuda") and not next(model.parameters()).is_cuda:
            model = model.to(device)

        for batch_tensors, indices in tqdm(loader, desc="embed dino"):
            try:
                batch_tensors = batch_tensors.to(device, non_blocking=True)
                with torch.inference_mode():
                    if autocast_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            feats = model(batch_tensors)
                    else:
                        feats = model(batch_tensors)
                feats = feats.float().cpu().numpy()
                batch_items = [items[i] for i in indices.tolist()]
                n_ok += _save_embeddings_batch(out_step_dir, batch_items, feats, save_dtype)
            except Exception:
                n_fail += len(indices)
    return {"n_ok": n_ok, "n_fail": n_fail}


def write_embedding_list(out_step_dir: Path, items: list[CropItem]) -> Path:
    manifest = out_step_dir / "embedding_paths.txt"
    lines = []
    for it in tqdm(items, desc="write embedding manifest", unit=" crops"):
        p = out_step_dir / it.category / f"{it.stem}.npy"
        if p.is_file():
            lines.append(str(p.resolve()))
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    if not args.filter_list.is_file():
        raise FileNotFoundError(f"Filter list not found: {args.filter_list}")
    if not args.cropped_dir.is_dir():
        raise FileNotFoundError(f"Cropped image dir not found: {args.cropped_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp

    if args.load_mode in ("dcp", "pth"):
        if not args.config_file.is_file():
            raise FileNotFoundError(f"Config not found: {args.config_file}")
        _ensure_dinov3_on_path(args.dinov3_repo)

    steps = resolve_checkpoint_steps(args)
    print(f"Checkpoints to run ({len(steps)}): {steps}")

    filter_line_count: int | None = None
    if args.count_filter_lines:
        print("Counting filter list lines (for progress bar)...")
        filter_line_count = _count_nonempty_lines(args.filter_list)
        print(f"Filter list: {filter_line_count:,} non-empty lines")

    allowed = load_allowed_categories(CATEGORY_SET_FILES[args.category_set])
    print(f"Resolving crops from filter (category_set={args.category_set}, {len(allowed)} categories)...")
    resolved_cache, resolved_stats = resolve_crops_from_filter(
        args.filter_list,
        args.cropped_dir,
        allowed,
        args.max_crops,
        filter_line_count=filter_line_count,
    )
    print(
        f"Resolved {resolved_stats['n_resolved']:,} crops from "
        f"{resolved_stats['n_unique']:,} unique (category, stem) pairs"
    )

    run_records = []
    for step in tqdm(steps, desc="checkpoints"):
        if args.load_mode == "hf":
            out_step = args.out_dir / "hf" / args.hf_model.replace("/", "_")
        else:
            out_step = args.out_dir / f"step_{step}"
        out_step.mkdir(parents=True, exist_ok=True)

        items = collect_crops(
            args.filter_list,
            args.cropped_dir,
            allowed,
            out_step,
            args.skip_existing,
            args.max_crops,
            resolved_cache=resolved_cache,
            resolved_stats=resolved_stats,
            filter_line_count=filter_line_count,
        )
        if args.dry_run:
            print(f"[dry-run] step {step}: would embed {len(items)} crops -> {out_step}")
            continue
        if not items:
            print(f"step {step}: nothing to do.")
            continue

        common = dict(
            items=items,
            out_step_dir=out_step,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_dtype=args.save_dtype,
            use_amp=use_amp,
        )

        if args.load_mode == "hf":
            stats = embed_crops_hf(
                **common,
                hf_model=args.hf_model,
                compile_model=args.compile_model,
            )
            weights = args.hf_model
        elif args.load_mode == "pth":
            weights = args.checkpoint_root / str(step) / "teacher_checkpoint.pth"
            if not weights.is_file():
                raise FileNotFoundError(f"Missing teacher checkpoint: {weights}")
            stats = embed_crops_dinov3_native(
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
            hf_export = weights / "huggingface"
            if hf_export.is_dir():
                print(
                    f"Note: HF weights at {hf_export} — use "
                    f"--load-mode hf --hf-model {hf_export} to skip DCP loading."
                )
            stats = embed_crops_dinov3_native(
                **common,
                config_file=args.config_file,
                weights_path=weights,
                dinov3_repo=args.dinov3_repo,
                dcp_source=args.dcp_source,
            )

        manifest_path = None
        if args.write_embedding_list:
            manifest_path = write_embedding_list(out_step, items)

        rec = {
            "step": step,
            "out_dir": str(out_step),
            "weights": str(weights),
            "load_mode": args.load_mode,
            "n_items": len(items),
            **stats,
        }
        if manifest_path:
            rec["embedding_list"] = str(manifest_path)
        run_records.append(rec)
        print(f"step {step}: saved {stats['n_ok']:,} embeddings under {out_step}")

    meta = {
        "checkpoint_root": str(args.checkpoint_root),
        "config_file": str(args.config_file),
        "out_dir": str(args.out_dir),
        "load_mode": args.load_mode,
        "category_set": args.category_set,
        "filter_list": str(args.filter_list),
        "cropped_dir": str(args.cropped_dir),
        "steps": steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "records": run_records,
    }
    meta_path = args.out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
