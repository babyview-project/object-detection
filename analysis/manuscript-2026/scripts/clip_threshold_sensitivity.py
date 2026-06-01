#!/usr/bin/env python3
"""
Small-scale CLIP detection-filter threshold sensitivity (0.26 / 0.27 / 0.28).

Recomputes BabyView category centroids (CLIP + DINOv3) by varying only the CLIP filter list
threshold, intersecting each list with either the high-precision valid85 sampled manifest
or the full 129-category annotation manifest (same ``.npy`` pairing logic as notebook 06).

THINGS exemplar centroids are fixed from on-disk CSVs (THINGS exports do not vary with the
BabyView CLIP list). In parallel, streams per-category CLIP-filtered detection counts within the
same inclusion list for proportion-stability summaries.

Writes (default ``--category-set both``):
  - ``clip_threshold_sensitivity_{valid85|valid129}.csv`` under the matching results folder
  - ``figures/si/clip_threshold_sensitivity_{valid85|valid129}.{pdf,png}`` (2×2 line plots)

Run from repo root or analysis/manuscript-2026/ (paths resolve to project root).

Performance (CLIP filter lists are huge):
  - One combined binary stream per threshold tallies (i) manifest hits for valid85 / valid129 and
    (ii) per-category CLIP-filtered detection counts for the inclusion list, so each large file is read once.
  - Large read chunks (default 8 MiB; ``--read-chunk-mb``) and no ``Path`` per line.
  - Parallel scans across thresholds (--filter-workers; default up to 8).
  - Each ``.npy`` is loaded at most once for the union of manifest hits at the loosest threshold;
    optional parallel reads (``--embedding-workers``; default 4).

Env (optional, same as notebook 06):
  BV_EMBEDDINGS_BASE, BV_CLIP_EMBEDDINGS_DIR, BV_DINOV3_EMBEDDINGS_DIR,
  BV_CROP_PATH_PREFIX, BV_CROP_PATH_PREFIX_NEW
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def load_included_categories(txt_path: Path) -> list[str]:
    return [line.strip().lower() for line in txt_path.read_text().splitlines() if line.strip()]


def load_valid_classes(per_class_csv: Path, threshold: float) -> set[str]:
    df = pd.read_csv(per_class_csv, usecols=["class", "precision"])
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    return set(df.loc[df["precision"] > threshold, "class"])


def load_valid_pairs(per_file_csv: Path, threshold: float) -> set[tuple[str, str]]:
    df = pd.read_csv(per_file_csv, usecols=["filename", "class", "precision"])
    df = df[df["precision"] > threshold].copy()
    df["class_norm"] = df["class"].astype(str).str.strip().str.lower()
    df["stem"] = (
        df["filename"]
        .astype(str)
        .str.strip()
        .str.rsplit("/", n=1)
        .str[-1]
        .str.rsplit(".", n=1)
        .str[0]
        .str.lower()
    )
    return set(zip(df["class_norm"], df["stem"]))


def build_valid85_sampled_exemplar_table(
    included_txt: Path,
    per_class_csv: Path,
    per_file_csv: Path,
    sampled_csv: Path,
    precision_threshold: float,
) -> pd.DataFrame:
    included = set(load_included_categories(included_txt))
    valid_classes = load_valid_classes(per_class_csv, precision_threshold)
    eligible_cats = included & valid_classes
    valid_pairs = load_valid_pairs(per_file_csv, precision_threshold)

    sampled = pd.read_csv(sampled_csv)
    sampled = sampled[sampled["trial_type"] == "regular"].copy()
    sampled["category"] = sampled["category"].astype(str).str.strip().str.lower()
    sampled["stem"] = sampled["stem"].astype(str).str.strip().str.lower()

    vp = pd.DataFrame(list(valid_pairs), columns=["category", "stem"])
    sampled = sampled.merge(vp, on=["category", "stem"], how="inner")
    sampled = sampled[sampled["category"].isin(eligible_cats)].copy()
    return sampled[["category", "path", "stem"]].reset_index(drop=True)


def build_annotation_manifest_129(sampled_csv: Path, included129_txt: Path) -> pd.DataFrame:
    """129 × ~100 regular exemplars used for crowdsourced annotation (no precision gate)."""
    included = set(load_included_categories(included129_txt))
    sampled = pd.read_csv(sampled_csv)
    sampled = sampled[sampled["trial_type"] == "regular"].copy()
    sampled["category"] = sampled["category"].astype(str).str.strip().str.lower()
    sampled["stem"] = sampled["stem"].astype(str).str.strip().str.lower()
    sampled = sampled[sampled["category"].isin(included)].copy()
    return sampled[["category", "path", "stem"]].reset_index(drop=True)


def remap_absolute_path(p: Path, crop_prefix: str, crop_prefix_new: str) -> Path:
    s = str(p)
    if crop_prefix and crop_prefix_new and s.startswith(crop_prefix):
        return Path(crop_prefix_new + s[len(crop_prefix) :])
    return p


# CLIP filter lists are multi-million lines; avoid `Path` per line and use large reads.
_DEFAULT_FILTER_READ_CHUNK = 8 * 1024 * 1024


@dataclass
class ClipFilterScanResult:
    """One pass over a CLIP filter list file (paths of CLIP-filtered crops)."""

    manifest_hits_85: set[tuple[str, str]]
    manifest_hits_129: set[tuple[str, str]]
    counts_by_category: dict[str, int]


def scan_clip_filter_combined(
    filter_list_path: Path,
    manifest_keys_85: frozenset[tuple[str, str]],
    manifest_keys_129: frozenset[tuple[str, str]],
    count_categories: frozenset[str],
    *,
    read_chunk_bytes: int = _DEFAULT_FILTER_READ_CHUNK,
) -> ClipFilterScanResult:
    """Single stream: manifest hits (85 / 129) + per-category CLIP-filtered counts (included set only)."""
    if not filter_list_path.is_file():
        raise FileNotFoundError(f"CLIP filter list not found: {filter_list_path}")

    h85: set[tuple[str, str]] = set()
    h129: set[tuple[str, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    suffix = b".npy"
    pending = b""

    def _handle_line(raw: bytes) -> None:
        n = len(raw)
        if n < 5 or raw[n - 4 : n].lower() != suffix:
            return
        i = raw.rfind(b"/")
        if i <= 0:
            return
        k = raw.rfind(b"/", 0, i)
        if k < 0:
            return
        try:
            cat = raw[k + 1 : i].decode("utf-8").strip().lower()
            stem = raw[i + 1 : n - 4].decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            return
        if cat in count_categories:
            counts[cat] += 1
        key = (cat, stem)
        if key in manifest_keys_85:
            h85.add(key)
        if key in manifest_keys_129:
            h129.add(key)

    with filter_list_path.open("rb") as f:
        while True:
            chunk = f.read(read_chunk_bytes)
            if not chunk:
                break
            data = pending + chunk
            *lines, pending = data.split(b"\n")
            for raw in lines:
                _handle_line(raw)

    if pending:
        _handle_line(pending.strip())

    return ClipFilterScanResult(h85, h129, dict(counts))


def scan_filter_lists_parallel_combined(
    filter_paths: list[Path],
    manifest_keys_85: frozenset[tuple[str, str]],
    manifest_keys_129: frozenset[tuple[str, str]],
    count_categories: frozenset[str],
    *,
    read_chunk_bytes: int = _DEFAULT_FILTER_READ_CHUNK,
    max_workers: int | None = None,
) -> list[ClipFilterScanResult]:
    def _one(p: Path) -> ClipFilterScanResult:
        return scan_clip_filter_combined(
            p, manifest_keys_85, manifest_keys_129, count_categories, read_chunk_bytes=read_chunk_bytes
        )

    if not filter_paths:
        return []

    workers = max_workers or min(8, len(filter_paths))
    workers = max(1, workers)
    if len(filter_paths) == 1 or workers == 1:
        return [_one(p) for p in filter_paths]

    out: list[ClipFilterScanResult | None] = [None] * len(filter_paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_one, p): i for i, p in enumerate(filter_paths)}
        for fut in as_completed(fut_map):
            out[fut_map[fut]] = fut.result()
    return [r if r is not None else ClipFilterScanResult(set(), set(), {}) for r in out]


def load_normalized_npy_vector(path: Path, crop_prefix: str, crop_prefix_new: str) -> np.ndarray | None:
    p = remap_absolute_path(path, crop_prefix, crop_prefix_new)
    if not p.is_file():
        return None
    v = np.load(p, mmap_mode="r")
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def zscore_rows(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    return (X - mu) / (sig + eps)


def _load_one_manifest_pair(
    key: tuple[str, str],
    clip_root: Path,
    dino_root: Path,
    crop_prefix: str,
    crop_prefix_new: str,
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray] | None]:
    cat, stem = key
    pc = remap_absolute_path(clip_root / cat / f"{stem}.npy", crop_prefix, crop_prefix_new)
    pd = remap_absolute_path(dino_root / cat / f"{stem}.npy", crop_prefix, crop_prefix_new)
    vc = load_normalized_npy_vector(pc, crop_prefix, crop_prefix_new)
    vd = load_normalized_npy_vector(pd, crop_prefix, crop_prefix_new)
    if vc is None or vd is None:
        return key, None
    return key, (vc, vd)


def build_manifest_embedding_cache(
    exemplar_df: pd.DataFrame,
    keys_to_load: set[tuple[str, str]],
    clip_root: Path,
    dino_root: Path,
    crop_prefix: str,
    crop_prefix_new: str,
    *,
    embedding_workers: int = 1,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Load each (category, stem) once (CLIP + DINO L2-normalized vectors)."""
    # Unique keys in manifest row order (stable) intersect keys_to_load.
    seen: set[tuple[str, str]] = set()
    ordered_keys: list[tuple[str, str]] = []
    for _, row in exemplar_df.iterrows():
        cat = str(row["category"]).strip().lower()
        stem = str(row["stem"]).strip().lower()
        key = (cat, stem)
        if key not in keys_to_load or key in seen:
            continue
        seen.add(key)
        ordered_keys.append(key)

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if embedding_workers <= 1:
        for key in ordered_keys:
            k, tup = _load_one_manifest_pair(key, clip_root, dino_root, crop_prefix, crop_prefix_new)
            if tup is not None:
                cache[k] = tup
        return cache

    workers = max(1, embedding_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_load_one_manifest_pair, key, clip_root, dino_root, crop_prefix, crop_prefix_new)
            for key in ordered_keys
        ]
        for fut in as_completed(futs):
            k, tup = fut.result()
            if tup is not None:
                cache[k] = tup
    return cache


def category_mean_zscored(
    exemplar_df: pd.DataFrame,
    pass_pairs: set[tuple[str, str]],
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean CLIP / DINO vectors per category (raw mean), then z-score each model's rows across categories."""
    sums_c: dict[str, np.ndarray] = {}
    sums_d: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)

    for _, row in exemplar_df.iterrows():
        cat = str(row["category"]).strip().lower()
        stem = str(row["stem"]).strip().lower()
        key = (cat, stem)
        if key not in pass_pairs or key not in cache:
            continue
        vc, vd = cache[key]
        if cat not in sums_c:
            sums_c[cat] = np.zeros_like(vc)
            sums_d[cat] = np.zeros_like(vd)
        sums_c[cat] += vc
        sums_d[cat] += vd
        counts[cat] += 1

    cats = sorted(c for c in sums_c if counts[c] > 0)
    if not cats:
        return [], np.zeros((0, 0)), np.zeros((0, 0))
    Xc = np.stack([sums_c[c] / counts[c] for c in cats])
    Xd = np.stack([sums_d[c] / counts[c] for c in cats])
    return cats, zscore_rows(Xc), zscore_rows(Xd)


def load_embedding_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if "category" in df.columns:
        categories = df["category"].astype(str).str.strip().str.lower().tolist()
        embedding_df = df.drop(columns=["category"])
    else:
        raise ValueError(f"Unexpected columns in {csv_path}")
    embedding_df = embedding_df.select_dtypes(include=[np.number])
    embeddings = embedding_df.to_numpy(dtype=np.float64)
    if len(categories) != embeddings.shape[0]:
        raise ValueError(f"Row mismatch in {csv_path}")
    return categories, embeddings


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings, metric="cosine"))


def vectorize_upper_triangle(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def align_rows(
    ref_order: list[str], cats: list[str], X: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Return X rows in ref_order; skip ref categories missing from (cats, X)."""
    idx = {c: i for i, c in enumerate(cats)}
    rows: list[np.ndarray] = []
    present: list[str] = []
    for c in ref_order:
        if c not in idx:
            continue
        rows.append(X[idx[c]])
        present.append(c)
    if not rows:
        return np.zeros((0, X.shape[1])), []
    return np.stack(rows, axis=0), present


def ref_threshold_index(thresholds: list[float], ref: float = 0.27) -> int:
    return int(min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - ref)))


def proportion_vector(counts: dict[str, int], cat_order: list[str]) -> np.ndarray:
    v = np.array([float(counts.get(c, 0)) for c in cat_order], dtype=np.float64)
    s = float(v.sum())
    if s <= 0:
        return v
    return v / s


def spearman_pearson_log_vs_ref(
    counts_per_t: list[dict[str, int]],
    cat_order: list[str],
    ref_idx: int,
) -> tuple[list[float], list[float]]:
    """Compare category-wise *shares* of CLIP-filtered detections to the reference threshold.

    For each threshold *t*, let c_i be the count of CLIP-filtered detections in category *i*
    (only categories in ``cat_order`` are tallied). The **proportion vector** is
    p_i = c_i / sum_j c_j (same fixed category ordering for all *t*).

    - **Spearman vs. 0.27:** Spearman correlation between p at *t* and p at the reference index
      (rank agreement of which categories dominate the mix). Defined as 1.0 when *t* is the reference.
    - **Pearson on log:** Pearson correlation between log(p_i + 1e-12) at *t* and at the reference
      (emphasizes multiplicative changes in rare vs. frequent categories). Defined as 1.0 at reference.
    """
    props = [proportion_vector(c, cat_order) for c in counts_per_t]
    ref = props[ref_idx]
    log_ref = np.log(ref + 1e-12)
    spears: list[float] = []
    pears: list[float] = []
    for i, p in enumerate(props):
        if i == ref_idx:
            spears.append(1.0)
            pears.append(1.0)
            continue
        sr, _ = spearmanr(p, ref)
        pr, _ = pearsonr(np.log(p + 1e-12), log_ref)
        spears.append(float(sr))
        pears.append(float(pr))
    return spears, pears


def plot_sensitivity_figure(df: pd.DataFrame, suptitle: str, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt

    det_col = (
        "clip_filtered_detections_in_included_categories"
        if "clip_filtered_detections_in_included_categories" in df.columns
        else "clip_pass_detections_in_count_scope"
    )

    xv = df["clip_threshold"].to_numpy(dtype=float)
    xticklabs = [f"{v:.2f}" for v in xv]

    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 7.0), sharex=True, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.35, h_pad=0.45, hspace=0.08, wspace=0.08)

    def _style_x(ax, *, with_xlabel: bool) -> None:
        ax.set_xticks(xv)
        ax.set_xticklabels(xticklabs)
        ax.tick_params(axis="x", labelsize=9, pad=4)
        if with_xlabel:
            ax.set_xlabel("CLIP image–text cosine threshold (detection filter)", fontsize=9)

    def _style_y_corr(ax) -> None:
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None))

    def _style_y_int(ax) -> None:
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(round(x)):,}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True, prune="both"))

    ax = axes[0, 0]
    ax.plot(xv, df["bv_pearson_clip_vs_dinov3_rdm"], "o-", lw=1.5, label="BV: CLIP vs DINO RDM (Pearson)")
    ax.plot(xv, df["bv_things_spearman_rdm_clip"], "s-", lw=1.5, label="BV vs THINGS RDM, CLIP (Spearman)")
    ax.plot(xv, df["bv_things_spearman_rdm_dinov3"], "^-", lw=1.5, label="BV vs THINGS RDM, DINO (Spearman)")
    ax.plot(xv, df["things_pearson_clip_vs_dinov3_rdm_same_cat_subset"], "D-", lw=1.5, label="THINGS: CLIP vs DINO RDM (Pearson)")
    _style_x(ax, with_xlabel=False)
    ax.set_ylabel("Correlation (off-diagonal RDM entries)", fontsize=9)
    ax.set_title("Representational geometry", fontsize=10, pad=6)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    _style_y_corr(ax)

    ax = axes[0, 1]
    ax.plot(xv, df["n_manifest_exemplar_paths_after_clip_gate"], "o-", color="steelblue", lw=1.5)
    _style_x(ax, with_xlabel=False)
    ax.set_ylabel("Exemplar crops retained (annotation manifest)", fontsize=9)
    ax.set_title("Human-validated manifest vs. filter", fontsize=10, pad=6)
    ax.grid(True, alpha=0.3)
    _style_y_int(ax)

    ax = axes[1, 0]
    ax.plot(xv, df[det_col], "o-", color="darkgreen", lw=1.5)
    _style_x(ax, with_xlabel=False)
    ax.set_ylabel("CLIP-filtered detections (included categories)", fontsize=9)
    ax.set_title("Total retained detections by category scope", fontsize=10, pad=6)
    ax.grid(True, alpha=0.3)
    _style_y_int(ax)

    ax = axes[1, 1]
    ax.plot(
        xv,
        df["spearman_category_prop_vs_ref"],
        "o-",
        lw=1.5,
        label=r"Spearman $\rho$ (proportion ranks vs. 0.27)",
    )
    ax.plot(
        xv,
        df["pearson_log_category_prop_vs_ref"],
        "s-",
        lw=1.5,
        label=r"Pearson $r$ (log proportions vs. 0.27)",
    )
    _style_x(ax, with_xlabel=False)
    ax.set_ylabel("Agreement with category mix at 0.27", fontsize=9)
    ax.set_title("CLIP-filtered category mixture vs. reference", fontsize=10, pad=6)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    _style_y_corr(ax)
    mix_note = (
        "Each category's share = its CLIP-filtered count divided by the sum over included "
        "categories (same fixed category list and order for every threshold). Spearman "
        r"$\rho$ compares that share vector to the shares at 0.27 (rank agreement). Pearson "
        r"$r$ compares $\log(\mathrm{share}+10^{-12})$ to the same at 0.27 (multiplicative "
        "shape). Both equal 1 at the reference threshold."
    )
    ax.text(
        0.02,
        0.98,
        mix_note,
        transform=ax.transAxes,
        fontsize=6.2,
        va="top",
        ha="left",
        linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92),
        zorder=5,
    )

    fig.suptitle(suptitle, fontsize=11)
    fig.supxlabel("CLIP image–text cosine threshold (detection filter)", fontsize=9)
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.26, 0.27, 0.28],
        help="CLIP cosine thresholds for filter list filenames",
    )
    parser.add_argument(
        "--category-set",
        choices=("valid85", "valid129", "both"),
        default="both",
        help="Which category inventory / manifest to report (default: both)",
    )
    parser.add_argument(
        "--read-chunk-mb",
        type=float,
        default=8.0,
        help="Binary read chunk size when streaming filter lists (default: 8 MiB)",
    )
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=0,
        help="Parallel workers for scanning filter lists (0 = min(8, number of thresholds); 1 = sequential)",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=4,
        help="Parallel workers for loading .npy embeddings (1 = fully sequential)",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Only write CSV; skip matplotlib figure",
    )
    args = parser.parse_args()

    from _bootstrap import MANUSCRIPT_DIR, PROJECT_ROOT

    root = PROJECT_ROOT
    preprint = MANUSCRIPT_DIR
    data_dir = root / "data"
    ann_dir = root / "annotation"

    included85_txt = data_dir / "included_categories_valid85.txt"
    included129_txt = data_dir / "included_categories_valid129.txt"
    per_class = ann_dir / "per_class_validation_data.csv"
    per_file = ann_dir / "per_file_precision_data.csv"
    sampled = ann_dir / "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv"
    precision_t = 0.6

    run85 = args.category_set in ("valid85", "both")
    run129 = args.category_set in ("valid129", "both")

    included85 = frozenset(load_included_categories(included85_txt))
    included129 = frozenset(load_included_categories(included129_txt))
    count_scope = included129 if run129 else included85

    emb_base = Path(
        os.environ.get(
            "BV_EMBEDDINGS_BASE",
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings",
        )
    ).expanduser()
    clip_dir = Path(os.environ.get("BV_CLIP_EMBEDDINGS_DIR", str(emb_base / "clip_embeddings_new"))).expanduser()
    dino_dir = Path(
        os.environ.get(
            "BV_DINOV3_EMBEDDINGS_DIR",
            str(emb_base / "facebook_dinov3-vitb16-pretrain-lvd1689m"),
        )
    ).expanduser()
    crop_prefix = os.environ.get("BV_CROP_PATH_PREFIX", "").strip()
    crop_prefix_new = os.environ.get("BV_CROP_PATH_PREFIX_NEW", "").strip()

    exemplar_df_85: pd.DataFrame | None = None
    manifest_keys_85: frozenset[tuple[str, str]] = frozenset()
    if run85:
        exemplar_df_85 = build_valid85_sampled_exemplar_table(
            included85_txt, per_class, per_file, sampled, precision_t
        )
        manifest_keys_85 = frozenset(
            zip(
                exemplar_df_85["category"].astype(str).str.strip().str.lower(),
                exemplar_df_85["stem"].astype(str).str.strip().str.lower(),
            )
        )

    exemplar_df_129: pd.DataFrame | None = None
    manifest_keys_129: frozenset[tuple[str, str]] = frozenset()
    if run129:
        exemplar_df_129 = build_annotation_manifest_129(sampled, included129_txt)
        manifest_keys_129 = frozenset(
            zip(
                exemplar_df_129["category"].astype(str).str.strip().str.lower(),
                exemplar_df_129["stem"].astype(str).str.strip().str.lower(),
            )
        )

    out_fig = preprint / "figures" / "si"
    out_fig.mkdir(parents=True, exist_ok=True)
    out85 = preprint / "supplemental_results_valid85cats_04302026" / "results"
    out129 = preprint / "main_results_valid129s_04302026" / "results"
    out85.mkdir(parents=True, exist_ok=True)
    out129.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds]
    read_chunk_bytes = max(64 * 1024, int(args.read_chunk_mb * 1024 * 1024))
    filter_paths = [
        emb_base
        / f"clip_image_embeddings_filtered-by-clip-{t:.2f}_exclude-people_exclude-subject-00270001.txt"
        for t in thresholds
    ]
    fw = args.filter_workers
    if fw <= 0:
        fw = min(8, len(filter_paths))
    print(
        f"Scanning {len(filter_paths)} CLIP filter list(s) "
        f"(chunk={read_chunk_bytes // (1024 * 1024)} MiB, workers={fw}; "
        f"category-set={args.category_set})…",
        flush=True,
    )
    scans = scan_filter_lists_parallel_combined(
        filter_paths,
        manifest_keys_85,
        manifest_keys_129,
        count_scope,
        read_chunk_bytes=read_chunk_bytes,
        max_workers=fw,
    )

    loose_idx = min(range(len(thresholds)), key=lambda i: thresholds[i])
    loose_t = thresholds[loose_idx]
    loose_token = f"{loose_t:.2f}"
    keys_for_cache = set(scans[loose_idx].manifest_hits_85) | set(scans[loose_idx].manifest_hits_129)
    exemplar_for_cache = exemplar_df_129 if exemplar_df_129 is not None else exemplar_df_85
    if exemplar_for_cache is None:
        print("No manifest selected.", file=sys.stderr)
        return 1

    print(
        f"Loading embeddings once for manifest ∩ CLIP≥{loose_token} ({len(keys_for_cache)} exemplar crops)…",
        flush=True,
    )
    vec_cache = build_manifest_embedding_cache(
        exemplar_for_cache,
        keys_for_cache,
        clip_dir,
        dino_dir,
        crop_prefix,
        crop_prefix_new,
        embedding_workers=args.embedding_workers,
    )
    print(f"Cached {len(vec_cache)} (category, stem) vector pairs.", flush=True)

    counts_per_t = [s.counts_by_category for s in scans]
    ref_idx = ref_threshold_index(thresholds, 0.27)

    def _rows_for_run(
        label: str,
        exemplar_df: pd.DataFrame,
        manifest_hits_per_t: list[set[tuple[str, str]]],
        things_clip_csv: Path,
        things_dino_csv: Path,
        detection_counts_per_t: list[dict[str, int]],
        cat_order_detection: list[str],
        prop_ref_idx: int,
    ) -> pd.DataFrame:
        if not things_clip_csv.is_file() or not things_dino_csv.is_file():
            raise FileNotFoundError(f"Missing THINGS CSV: {things_clip_csv} / {things_dino_csv}")
        th_cats_c, th_Xc = load_embedding_csv(things_clip_csv)
        th_cats_d, th_Xd = load_embedding_csv(things_dino_csv)
        if th_cats_c != th_cats_d:
            raise RuntimeError("THINGS CLIP/DINO category rows differ")
        ref_order = sorted(th_cats_c)
        th_Xc_full, ord_full = align_rows(ref_order, th_cats_c, th_Xc)
        th_Xd_full, ord_d = align_rows(ref_order, th_cats_d, th_Xd)
        if ord_full != ord_d:
            raise RuntimeError("Alignment mismatch THINGS clip/dino")

        sp_mix, pr_mix = spearman_pearson_log_vs_ref(
            detection_counts_per_t, cat_order_detection, prop_ref_idx
        )

        rows: list[dict] = []
        for i, t in enumerate(thresholds):
            token = f"{t:.2f}"
            filt = filter_paths[i]
            clip_filter_pairs = manifest_hits_per_t[i]
            n_manifest_pass = len(clip_filter_pairs)

            cats_c, Xc_z, Xd_z = category_mean_zscored(exemplar_df, clip_filter_pairs, vec_cache)
            if len(cats_c) == 0:
                raise RuntimeError(f"[{label}] threshold {token}: no categories with exemplars")

            Xc_a, present = align_rows(ref_order, cats_c, Xc_z)
            Xd_a, present_d = align_rows(ref_order, cats_c, Xd_z)
            if present != present_d:
                raise RuntimeError("CLIP/DINO alignment lists differ")
            if set(present) != set(ord_full):
                missing = set(ord_full) - set(present)
                extra = set(present) - set(ord_full)
                print(
                    f"[{label} {token}] WARNING: BV category set differs from THINGS reference "
                    f"(missing {len(missing)}, extra {len(extra)})",
                    file=sys.stderr,
                )

            th_Xc_sub, ord_c_sub = align_rows(present, ord_full, th_Xc_full)
            th_Xd_sub, ord_d_sub = align_rows(present, ord_full, th_Xd_full)
            if ord_c_sub != present or ord_d_sub != present:
                raise RuntimeError("THINGS subset alignment failed")

            th_rdm_clip = compute_rdm(th_Xc_sub)
            th_rdm_dino = compute_rdm(th_Xd_sub)
            th_vec_clip = vectorize_upper_triangle(th_rdm_clip)
            th_vec_dino = vectorize_upper_triangle(th_rdm_dino)
            th_cd_p, _ = pearsonr(th_vec_clip, th_vec_dino)
            th_cd_s, _ = spearmanr(th_vec_clip, th_vec_dino)

            bv_rdm_c = compute_rdm(Xc_a)
            bv_rdm_d = compute_rdm(Xd_a)
            bv_vc = vectorize_upper_triangle(bv_rdm_c)
            bv_vd = vectorize_upper_triangle(bv_rdm_d)

            pr_cd, pp_cd = pearsonr(bv_vc, bv_vd)
            sp_cd, sp_p_cd = spearmanr(bv_vc, bv_vd)

            pr_cx, _ = pearsonr(bv_vc, th_vec_clip)
            sp_cx, _ = spearmanr(bv_vc, th_vec_clip)
            pr_dx, _ = pearsonr(bv_vd, th_vec_dino)
            sp_dx, _ = spearmanr(bv_vd, th_vec_dino)

            det = detection_counts_per_t[i]
            clip_pass_total = int(sum(det.get(c, 0) for c in cat_order_detection))

            rows.append(
                {
                    "category_set": label,
                    "clip_threshold": float(t),
                    "filter_list": str(filt),
                    "n_manifest_exemplar_paths_after_clip_gate": int(n_manifest_pass),
                    "n_categories_bv": len(present),
                    "n_pairs_upper_triangle": len(bv_vc),
                    "clip_filtered_detections_in_included_categories": clip_pass_total,
                    "spearman_category_prop_vs_ref": float(sp_mix[i]),
                    "pearson_log_category_prop_vs_ref": float(pr_mix[i]),
                    "bv_pearson_clip_vs_dinov3_rdm": float(pr_cd),
                    "bv_pearson_clip_vs_dinov3_p": float(pp_cd),
                    "bv_spearman_clip_vs_dinov3_rdm": float(sp_cd),
                    "bv_spearman_clip_vs_dinov3_p": float(sp_p_cd),
                    "bv_things_pearson_rdm_clip": float(pr_cx),
                    "bv_things_spearman_rdm_clip": float(sp_cx),
                    "bv_things_pearson_rdm_dinov3": float(pr_dx),
                    "bv_things_spearman_rdm_dinov3": float(sp_dx),
                    "things_pearson_clip_vs_dinov3_rdm_same_cat_subset": float(th_cd_p),
                    "things_spearman_clip_vs_dinov3_rdm_same_cat_subset": float(th_cd_s),
                }
            )
        return pd.DataFrame(rows)

    things85_clip = preprint / "exemplar_set_embeddings" / "valid85" / "things_clip_exemplar_avg_zscore_within_valid85.csv"
    things85_dino = preprint / "exemplar_set_embeddings" / "valid85" / "things_dinov3_exemplar_avg_zscore_within_valid85.csv"
    things129_clip = preprint / "exemplar_set_embeddings" / "valid129" / "things_clip_exemplar_avg_zscore_within_valid129.csv"
    things129_dino = preprint / "exemplar_set_embeddings" / "valid129" / "things_dinov3_exemplar_avg_zscore_within_valid129.csv"

    cat_order85 = sorted(included85)
    cat_order129 = sorted(included129)
    counts85_per_t = [{c: int(ct.get(c, 0)) for c in cat_order85} for ct in counts_per_t]

    if run85:
        if exemplar_df_85 is None:
            raise RuntimeError("internal: exemplar_df_85")
        hits85_per_t = [set(s.manifest_hits_85) for s in scans]
        df85 = _rows_for_run(
            "valid85",
            exemplar_df_85,
            hits85_per_t,
            things85_clip,
            things85_dino,
            counts85_per_t,
            cat_order85,
            ref_idx,
        )
        p85 = out85 / "clip_threshold_sensitivity_valid85.csv"
        df85.to_csv(p85, index=False)
        print(f"Wrote {p85}")
        if not args.skip_figure:
            plot_sensitivity_figure(
                df85,
                "CLIP threshold sensitivity (valid85)",
                out_fig / "clip_threshold_sensitivity_valid85.png",
                out_fig / "clip_threshold_sensitivity_valid85.pdf",
            )
            print(f"Wrote {out_fig / 'clip_threshold_sensitivity_valid85.png'}")

    if run129:
        if exemplar_df_129 is None:
            raise RuntimeError("internal: exemplar_df_129")
        hits129_per_t = [set(s.manifest_hits_129) for s in scans]
        df129 = _rows_for_run(
            "valid129",
            exemplar_df_129,
            hits129_per_t,
            things129_clip,
            things129_dino,
            counts_per_t,
            cat_order129,
            ref_idx,
        )
        p129 = out129 / "clip_threshold_sensitivity_valid129.csv"
        df129.to_csv(p129, index=False)
        print(f"Wrote {p129}")
        if not args.skip_figure:
            plot_sensitivity_figure(
                df129,
                "CLIP threshold sensitivity (valid129)",
                out_fig / "clip_threshold_sensitivity_valid129.png",
                out_fig / "clip_threshold_sensitivity_valid129.pdf",
            )
            print(f"Wrote {out_fig / 'clip_threshold_sensitivity_valid129.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
