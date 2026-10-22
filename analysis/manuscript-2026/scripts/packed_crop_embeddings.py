"""Read 2025.2 packed crop shards for manuscript analyses.

The 868-hour layout was ``{model_dir}/{category}/{stem}.npy``. 2025.2 stores CLIP
vectors (and DINOv3 CLS for CLIP-passers) in one ``{emb_root}/shards/{video_id}.npz``
plus ``crops_index.csv.gz``. Analysis code should go through this module rather than
opening shards per crop or per category.

Scan cost: one sequential pass over ~3k npz files.

Detect a packed store::

    from packed_crop_embeddings import is_packed_store, packed_root_from_path
    root = packed_root_from_path(path)  # None → use the legacy .npy tree
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CLIP_DIM = 512
DINO_DIM = 768
DEFAULT_PACKED_ROOT = Path(
    "/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full"
)
DEFAULT_EXCLUDE_SUBJECTS = ("00270001",)


def frame_sampling() -> tuple[int, int]:
    """(stride, offset) from BV_FRAME_STRIDE / BV_FRAME_OFFSET; default (1, 0) = every frame.

    Frames are 1 fps, so stride 5 keeps one frame every 5 s: frame_number % stride == offset.
    """
    stride = int(os.environ.get("BV_FRAME_STRIDE", "1") or 1)
    offset = int(os.environ.get("BV_FRAME_OFFSET", "0") or 0)
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError(f"bad frame sampling: stride={stride} offset={offset}")
    return stride, offset


def frame_keep(frame_numbers) -> np.ndarray:
    """Boolean mask of rows whose frame is kept under frame_sampling()."""
    stride, offset = frame_sampling()
    f = np.asarray(frame_numbers)
    if stride == 1:
        return np.ones(f.shape[0], dtype=bool)
    return (f.astype(np.int64) % stride) == offset


def is_packed_store(root: Path | str | None) -> bool:
    if root is None:
        return False
    shards = Path(root) / "shards"
    if not shards.is_dir():
        return False
    return next(shards.glob("*.npz"), None) is not None


def packed_root_from_path(path: Path | str | None) -> Path | None:
    """If ``path`` is a packed store (or a model subdir of one), return the store root."""
    if path is None:
        return None
    p = Path(path)
    if is_packed_store(p):
        return p
    if is_packed_store(p.parent):
        return p.parent
    return None


def infer_model_from_path(path: Path | str, default: str | None = None) -> str:
    name = Path(path).name.lower()
    env = (os.environ.get("BV_EMBED_MODEL") or default or "clip").strip().lower()
    if "babydinov3" in name:
        return "babydinov3"
    if "dinov3" in name or "dino" in name:
        return "dinov3"
    if "clip" in name:
        return "clip"
    return env if env in {"clip", "dinov3", "babydinov3"} else "clip"


def subject_id_from_video_id(video_id: str) -> str | None:
    head = str(video_id).split("_", 1)[0]
    if head.startswith("S") and head[1:].isdigit():
        return head[1:].zfill(8)
    if head.isdigit():
        return head.zfill(8)
    return None


def iter_shard_paths(emb_root: Path):
    shards = Path(emb_root) / "shards"
    for p in sorted(shards.glob("*.npz")):
        if ".tmp" in p.name:
            continue
        yield p


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        n = np.linalg.norm(X)
        return X / n if n > 0 else X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def zscore_rows(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    return (X - mu) / (sig + eps)


def zscore_subject_embedding_dict(cat_to_vec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(cat_to_vec) < 2:
        return dict(cat_to_vec)
    cats = sorted(cat_to_vec.keys())
    X = np.stack([np.asarray(cat_to_vec[c], dtype=np.float64).ravel() for c in cats], axis=0)
    Xz = zscore_rows(X)
    return {c: Xz[i] for i, c in enumerate(cats)}


def _row_mask(
    shard: dict,
    *,
    categories: set[str] | None,
    min_sim: float | None,
    passed_only: bool,
    model: str,
) -> np.ndarray:
    n = len(shard["class_name"])
    mask = frame_keep(shard["frame_number"]) if "frame_number" in shard else np.ones(n, dtype=bool)
    if categories is not None:
        names = np.array([str(x).strip().lower() for x in shard["class_name"]])
        mask &= np.isin(names, list(categories))
    if passed_only and "passed" in shard:
        mask &= shard["passed"].astype(bool)
    if min_sim is not None:
        mask &= shard["clip_sim"] >= float(min_sim)
    if model == "dinov3":
        mask &= shard["dino_idx"] >= 0
    return mask


def _vectors_for_mask(shard: dict, mask: np.ndarray, model: str) -> np.ndarray:
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        dim = CLIP_DIM if model == "clip" else DINO_DIM
        return np.zeros((0, dim), dtype=np.float64)
    if model == "clip":
        return np.asarray(shard["clip_emb"][sel], dtype=np.float64)
    if model == "dinov3":
        return np.asarray(shard["dino_emb"][shard["dino_idx"][sel]], dtype=np.float64)
    raise ValueError(model)


def category_means(
    emb_root: Path,
    categories: set[str],
    *,
    model: str = "clip",
    min_sim: float | None = 0.27,
    passed_only: bool = False,
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS,
    l2_normalize: bool = True,
) -> tuple[list[str], np.ndarray, dict[str, int], dict]:
    """Mean of (optionally L2-normalized) crop vectors per category. One shard pass."""
    model = model.strip().lower()
    cats = {c.strip().lower() for c in categories}
    exclude = {s.strip() for s in exclude_subjects if s}
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    n_videos = 0
    n_dropped_subj = 0
    shard_paths = list(iter_shard_paths(emb_root))
    for path in shard_paths:
        sid = subject_id_from_video_id(path.stem)
        if sid is not None and sid in exclude:
            n_dropped_subj += 1
            continue
        n_videos += 1
        with np.load(path, allow_pickle=True) as z:
            shard = {k: z[k] for k in z.files}
        mask = _row_mask(
            shard, categories=cats, min_sim=min_sim, passed_only=passed_only, model=model
        )
        if not mask.any():
            continue
        names = np.array([str(x).strip().lower() for x in shard["class_name"]])
        vecs = _vectors_for_mask(shard, mask, model)
        if l2_normalize:
            vecs = l2_normalize_rows(vecs)
        sel_names = names[mask]
        for name in np.unique(sel_names):
            sub = vecs[sel_names == name]
            if name not in sums:
                sums[name] = np.zeros(sub.shape[1], dtype=np.float64)
            sums[name] += sub.sum(axis=0)
            counts[name] += int(sub.shape[0])

    ordered = sorted(c for c in cats if counts[c] > 0)
    if not ordered:
        dim = CLIP_DIM if model == "clip" else DINO_DIM
        X = np.zeros((0, dim), dtype=np.float64)
    else:
        X = np.stack([sums[c] / counts[c] for c in ordered], axis=0)
    stats = {
        "model": model,
        "n_shards": len(shard_paths),
        "n_videos_used": n_videos,
        "n_dropped_subject": n_dropped_subj,
        "n_crops": dict(counts),
        "min_sim": min_sim,
        "frame_stride": frame_sampling()[0],
        "frame_offset": frame_sampling()[1],
        "passed_only": passed_only,
        "l2_normalize": l2_normalize,
    }
    return ordered, X, dict(counts), stats


def category_means_by_threshold(
    emb_root: Path,
    categories: set[str],
    thresholds: list[float],
    *,
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS,
    l2_normalize: bool = True,
) -> dict[float, dict[str, object]]:
    """One shard pass; CLIP + DINOv3 category means at each CLIP cosine threshold.

    DINOv3 only exists for crops that passed the *write-time* filter (0.27), so
    thresholds below that have CLIP for extra boxes but no extra DINO rows.
    """
    cats = {c.strip().lower() for c in categories}
    exclude = {s.strip() for s in exclude_subjects if s}
    thresholds = [float(t) for t in thresholds]
    clip_sum: dict[float, dict[str, np.ndarray]] = {t: {} for t in thresholds}
    dino_sum: dict[float, dict[str, np.ndarray]] = {t: {} for t in thresholds}
    clip_n: dict[float, dict[str, int]] = {t: defaultdict(int) for t in thresholds}
    dino_n: dict[float, dict[str, int]] = {t: defaultdict(int) for t in thresholds}

    n_videos = 0
    for path in iter_shard_paths(emb_root):
        sid = subject_id_from_video_id(path.stem)
        if sid is not None and sid in exclude:
            continue
        n_videos += 1
        with np.load(path, allow_pickle=True) as z:
            names = np.array([str(x).strip().lower() for x in z["class_name"]])
            sim = np.asarray(z["clip_sim"], dtype=np.float64)
            clip = np.asarray(z["clip_emb"], dtype=np.float64)
            dino_idx = np.asarray(z["dino_idx"], dtype=np.int32)
            dino = np.asarray(z["dino_emb"], dtype=np.float64)
            frames = np.asarray(z["frame_number"])
        keep_cat = np.isin(names, list(cats)) & frame_keep(frames)
        clip_norm = l2_normalize_rows(clip) if l2_normalize else clip
        dino_norm = None
        if dino.size:
            dino_norm = l2_normalize_rows(dino) if l2_normalize else dino

        for t in thresholds:
            m = keep_cat & (sim >= t)
            if not m.any():
                continue
            sel_names = names[m]
            sel_clip = clip_norm[m]
            for name in np.unique(sel_names):
                sub = sel_clip[sel_names == name]
                if name not in clip_sum[t]:
                    clip_sum[t][name] = np.zeros(sub.shape[1], dtype=np.float64)
                clip_sum[t][name] += sub.sum(axis=0)
                clip_n[t][name] += int(sub.shape[0])
            if dino_norm is None:
                continue
            md = m & (dino_idx >= 0)
            if not md.any():
                continue
            d_names = names[md]
            d_vecs = dino_norm[dino_idx[md]]
            for name in np.unique(d_names):
                sub = d_vecs[d_names == name]
                if name not in dino_sum[t]:
                    dino_sum[t][name] = np.zeros(sub.shape[1], dtype=np.float64)
                dino_sum[t][name] += sub.sum(axis=0)
                dino_n[t][name] += int(sub.shape[0])

    out: dict[float, dict[str, object]] = {}
    for t in thresholds:
        c_cats = sorted(c for c in cats if clip_n[t][c] > 0)
        d_cats = sorted(c for c in cats if dino_n[t][c] > 0)
        Xc = (
            np.stack([clip_sum[t][c] / clip_n[t][c] for c in c_cats], axis=0)
            if c_cats
            else np.zeros((0, CLIP_DIM))
        )
        Xd = (
            np.stack([dino_sum[t][c] / dino_n[t][c] for c in d_cats], axis=0)
            if d_cats
            else np.zeros((0, DINO_DIM))
        )
        out[t] = {
            "clip_categories": c_cats,
            "clip_X": Xc,
            "clip_n": dict(clip_n[t]),
            "dino_categories": d_cats,
            "dino_X": Xd,
            "dino_n": dict(dino_n[t]),
            "n_videos": n_videos,
        }
    return out


def subject_category_means(
    emb_root: Path,
    categories: set[str],
    *,
    model: str = "clip",
    subjects: set[str] | None = None,
    min_sim: float | None = 0.27,
    passed_only: bool = False,
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS,
    l2_normalize: bool = True,
    zscore_across_categories: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """subject_id → {category: mean vector}. One shard pass."""
    model = model.strip().lower()
    cats = {c.strip().lower() for c in categories}
    want = {s.zfill(8) for s in subjects} if subjects is not None else None
    exclude = {s.strip() for s in exclude_subjects if s}
    sums: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in iter_shard_paths(emb_root):
        sid = subject_id_from_video_id(path.stem)
        if sid is None:
            continue
        if sid in exclude:
            continue
        if want is not None and sid not in want:
            continue
        with np.load(path, allow_pickle=True) as z:
            shard = {k: z[k] for k in z.files}
        mask = _row_mask(
            shard, categories=cats, min_sim=min_sim, passed_only=passed_only, model=model
        )
        if not mask.any():
            continue
        names = np.array([str(x).strip().lower() for x in shard["class_name"]])
        vecs = _vectors_for_mask(shard, mask, model)
        if l2_normalize:
            vecs = l2_normalize_rows(vecs)
        sel_names = names[mask]
        bucket = sums[sid]
        nmap = counts[sid]
        for name in np.unique(sel_names):
            sub = vecs[sel_names == name]
            if name not in bucket:
                bucket[name] = np.zeros(sub.shape[1], dtype=np.float64)
            bucket[name] += sub.sum(axis=0)
            nmap[name] += int(sub.shape[0])

    out: dict[str, dict[str, np.ndarray]] = {}
    for sid, cmap in sums.items():
        means = {c: cmap[c] / counts[sid][c] for c in cmap if counts[sid][c] > 0}
        out[sid] = zscore_subject_embedding_dict(means) if zscore_across_categories else means
    if want is not None:
        return {s: out.get(s, {}) for s in sorted(want)}
    return out


def try_packed_category_subject_vectors(
    embeddings_dir: Path | str,
    categories: list[str] | set[str],
    subjects: list[str] | set[str],
    *,
    model: str | None = None,
    min_sim: float | None = 0.27,
    l2_normalize: bool = True,
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS,
) -> dict[str, dict[str, list[np.ndarray]]] | None:
    """category → subject → list of crop vectors. For local/global (notebook 09)."""
    root = packed_root_from_path(embeddings_dir)
    if root is None:
        return None
    mdl = model or infer_model_from_path(embeddings_dir)
    cats = {c.strip().lower() for c in categories}
    want = {str(s).zfill(8) for s in subjects}
    exclude = {s.strip() for s in exclude_subjects if s}
    out: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for path in iter_shard_paths(root):
        sid = subject_id_from_video_id(path.stem)
        if sid is None or sid in exclude or sid not in want:
            continue
        with np.load(path, allow_pickle=True) as z:
            shard = {k: z[k] for k in z.files}
        mask = _row_mask(
            shard, categories=cats, min_sim=min_sim, passed_only=False, model=mdl
        )
        if not mask.any():
            continue
        names = np.array([str(x).strip().lower() for x in shard["class_name"]])
        vecs = _vectors_for_mask(shard, mask, mdl)
        if l2_normalize:
            vecs = l2_normalize_rows(vecs)
        for name, vec in zip(names[mask], vecs):
            out[name][sid].append(np.asarray(vec, dtype=np.float64).ravel())
    return {c: dict(out[c]) for c in out}


def try_packed_subject_means(
    embeddings_dir: Path | str,
    categories: list[str] | set[str],
    subjects: list[str] | set[str] | None = None,
    *,
    model: str | None = None,
    min_sim: float | None = 0.27,
    l2_normalize: bool = True,
    zscore_across_categories: bool = False,
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS,
) -> dict[str, dict[str, np.ndarray]] | None:
    """Drop-in for notebooks: returns subject centroids if ``embeddings_dir`` is packed, else None."""
    root = packed_root_from_path(embeddings_dir)
    if root is None:
        return None
    mdl = model or infer_model_from_path(embeddings_dir)
    subj = {str(s).zfill(8) for s in subjects} if subjects is not None else None
    return subject_category_means(
        root,
        {c.strip().lower() for c in categories},
        model=mdl,
        subjects=subj,
        min_sim=min_sim,
        passed_only=False,
        exclude_subjects=exclude_subjects,
        l2_normalize=l2_normalize,
        zscore_across_categories=zscore_across_categories,
    )
