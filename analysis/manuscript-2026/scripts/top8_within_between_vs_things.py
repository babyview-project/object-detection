"""Top-8 densest kids: category-level within/between vs THINGS (valid85).

Optimizations vs naive full-tree scan:
  - Manifest filter: only (category, stem) pairs from sampled regular exemplars (nb 09).
  - Per-subject centroid .npz cache under results/.../cache/ (skip re-read on rerun).
  - Online exemplar mean (no growing Python lists of arrays).
  - mmap np.load; parallel category folders + parallel file reads.

Standalone:
  python analysis/manuscript-2026/scripts/top8_within_between_vs_things.py
  python ... --rebuild-cache
  BV_EMBED_MODEL=dinov3 MAX_EXEMPLARS_PER_CATEGORY=32 python ...
  TOP8_CATEGORY_SET=valid129 BV_EMBED_MODEL=dinov3 python ... --recompute-shuffle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
from bv_things_cdi_shuffle_inference import (  # noqa: E402
    draw_horizontal_paired_cluster_delta_bars,
    run_bv_things_shuffle_inference,
)
DATA_DIR = PROJECT_ROOT / 'data'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

TOP8_CATEGORY_SET = os.environ.get('TOP8_CATEGORY_SET', 'valid85').strip()
EMBED_MODEL = os.environ.get('BV_EMBED_MODEL', 'clip').strip().lower()
THRESHOLD_TOKEN = os.environ.get('BV_THINGS_EMBED_THRESHOLD', '0.27').strip()
if THRESHOLD_TOKEN in {'0', '0.0', '0.00'}:
    THRESHOLD_TOKEN = '0.27'

# valid129 top-8 kids: rank by manifest category coverage in the 129-cat set (not trajectory CSV).
TOP8_RANK_MODE = os.environ.get(
    'TOP8_RANK_MODE',
    'category_density' if TOP8_CATEGORY_SET == 'valid129' else 'trajectory',
).strip().lower()


def output_run_root_for_category_set(category_set: str) -> Path:
    if category_set == 'valid129':
        return PREPRINT_DIR / 'main_results_valid129s_04302026'
    return PREPRINT_DIR / 'supplemental_results_valid85cats_04302026'


OUTPUT_RUN_ROOT = output_run_root_for_category_set(TOP8_CATEGORY_SET)
RESULTS_DIR = OUTPUT_RUN_ROOT / 'results'
FIGURES_DIR = OUTPUT_RUN_ROOT / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXEMPLAR_EMBED_DIR = PREPRINT_DIR / 'exemplar_set_embeddings' / TOP8_CATEGORY_SET
THINGS_EMBED_CSV = EXEMPLAR_EMBED_DIR / f'things_{EMBED_MODEL}_exemplar_avg_zscore_within_{TOP8_CATEGORY_SET}.csv'
ORDER_CSV = RESULTS_DIR / f'bv_things_rdm_order_bv_semantic_{EMBED_MODEL}_filtered-{THRESHOLD_TOKEN}_{TOP8_CATEGORY_SET}.csv'
CDI_SEMANTIC_CSV = DATA_DIR / f'long_tailed_dist_prop_included_categories_{TOP8_CATEGORY_SET}.csv'
INCLUDED_TXT = DATA_DIR / f'included_categories_{TOP8_CATEGORY_SET}.txt'
SAMPLED_EXEMPLAR_CSV = (
    PROJECT_ROOT
    / 'annotation'
    / 'sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv'
)
TRAJECTORY_CSV = ANALYSIS_DIR / 'individual_analyses' / 'developmental_trajectory_rdms_clip' / 'trajectory_correlations.csv'

YOLOE_ROOT = Path('/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings')
EMBEDDINGS_DIR = YOLOE_ROOT / (
    'clip_embeddings_new' if EMBED_MODEL == 'clip' else 'facebook_dinov3-vitb16-pretrain-lvd1689m'
)

N_TOP_DENSEST = 8
USE_MANIFEST_FILTER = os.environ.get('TOP8_USE_MANIFEST_FILTER', '1').strip().lower() not in {'0', 'false', 'no'}
MAX_EXEMPLARS_PER_CATEGORY: int | None = int(os.environ['MAX_EXEMPLARS_PER_CATEGORY']) if os.environ.get('MAX_EXEMPLARS_PER_CATEGORY') else 32
EXEMPLAR_SUBSAMPLE_SEED = 42
ZSCORE_EMBEDDINGS_ACROSS_CATEGORIES = True
EMBED_LOAD_MAX_WORKERS = max(4, min(16, (os.cpu_count() or 8)))
EMBED_PARALLEL_CATEGORIES = True
EMBED_CATEGORY_MAX_WORKERS = min(
    12,
    max(4, int(os.environ.get('TOP8_CATEGORY_MAX_WORKERS', '0')) or (os.cpu_count() or 8) // 2),
)
NPY_MMAP_MODE = os.environ.get('TOP8_NPY_MMAP', 'r').strip() or 'r'

CDI_SEMANTIC_ORDER = [
    'animals', 'body_parts', 'clothing', 'food_drink', 'furniture_rooms', 'household',
    'outside', 'people', 'toys', 'vehicles', 'other',
]
CDI_SEMANTIC_COLORS = {
    'animals': '#4DB8A8', 'body_parts': '#E87A5F', 'clothing': '#9B7EC8', 'food_drink': '#E8A54C',
    'furniture_rooms': '#6BAB7A', 'household': '#D97B9E', 'outside': '#5B9BD5', 'people': '#E8C44C',
    'toys': '#B07CC8', 'vehicles': '#6BA3D5', 'other': '#8B9A9E',
}
KID_BAR_COLOR = '#0d47a1'
THINGS_BAR_COLOR = '#90caf9'
N_PERM = int(os.environ.get('BV_CLUSTER_PERMUTATIONS', '5000'))
SHUFFLE_SEED = int(os.environ.get('BV_CLUSTER_PERMUTATIONS_SEED', os.environ.get('BV_CLUSTER_PERM_SEED', '42')))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Top-8 vs THINGS category within/between (valid85).')
    p.add_argument('--rebuild-cache', action='store_true', help='Ignore cached subject centroids.')
    p.add_argument('--skip-plots', action='store_true', help='Only write CSV tables.')
    p.add_argument(
        '--plots-only',
        action='store_true',
        help='Regenerate figures from existing CSVs (no embedding I/O).',
    )
    p.add_argument(
        '--recompute-shuffle',
        action='store_true',
        help='Re-run label-shuffle nulls for vertical Δ plots (uses centroid cache).',
    )
    p.add_argument(
        '--skip-delta-vertical',
        action='store_true',
        help='Skip vertical Δ plots with shuffle null whiskers.',
    )
    return p.parse_args()


def load_included_categories(txt_path: Path) -> list[str]:
    out = []
    for line in txt_path.read_text(encoding='utf-8').splitlines():
        v = line.strip().lower()
        if v:
            out.append(v)
    return sorted(set(out))


def load_cdi_semantic_map(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df[['category', 'cdi_semantic']].dropna().iterrows():
        cat = str(row['category']).strip().lower()
        sem = str(row['cdi_semantic']).strip().lower()
        if cat:
            mapping[cat] = sem if sem else 'other'
    return mapping


def load_embedding_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if 'category' in df.columns:
        categories = df['category'].astype(str).str.strip().str.lower().tolist()
        emb_df = df.drop(columns=['category'])
    else:
        first_col = df.columns[0]
        categories = df[first_col].astype(str).str.strip().str.lower().tolist()
        emb_df = df.drop(columns=[first_col])
    emb_df = emb_df.apply(pd.to_numeric, errors='coerce')
    valid_rows = ~emb_df.isna().all(axis=1)
    emb_df = emb_df.loc[valid_rows]
    categories = [c for c, keep in zip(categories, valid_rows.tolist()) if keep]
    return categories, emb_df.to_numpy(dtype=np.float32)


def load_order(order_csv: Path) -> list[str]:
    order_df = pd.read_csv(order_csv).sort_values('position').reset_index(drop=True)
    return order_df['category'].astype(str).str.strip().str.lower().tolist()


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    return squareform(pdist(embeddings.astype(np.float64, copy=False), metric='cosine'))


def category_within_between_detailed(
    categories: list[str],
    semantics: list[str],
    rdm: np.ndarray,
) -> pd.DataFrame:
    sem_arr = np.array(semantics)
    rows = []
    for i, cat in enumerate(categories):
        same = sem_arr == sem_arr[i]
        same[i] = False
        diff = ~same
        diff[i] = False
        within_vals = rdm[i, same]
        between_vals = rdm[i, diff]
        rows.append(
            {
                'category': cat,
                'cdi_semantic': semantics[i],
                'within_mean': float(np.mean(within_vals)) if within_vals.size else np.nan,
                'within_std': float(np.std(within_vals, ddof=1)) if within_vals.size > 1 else np.nan,
                'between_mean': float(np.mean(between_vals)) if between_vals.size else np.nan,
                'between_std': float(np.std(between_vals, ddof=1)) if between_vals.size > 1 else np.nan,
                'delta_between_minus_within': (
                    float(np.mean(between_vals) - np.mean(within_vals))
                    if within_vals.size and between_vals.size
                    else np.nan
                ),
                'n_within_pairs': int(within_vals.size),
                'n_between_pairs': int(between_vals.size),
            }
        )
    return pd.DataFrame(rows)


def cluster_within_between(
    categories: list[str],
    semantics: list[str],
    rdm: np.ndarray,
) -> pd.DataFrame:
    sem_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, sem in enumerate(semantics):
        sem_to_idx[sem].append(i)

    rows = []
    n = len(categories)
    all_idx = np.arange(n)

    for sem, idx_list in sorted(sem_to_idx.items()):
        idx = np.array(idx_list, dtype=int)
        other_idx = all_idx[~np.isin(all_idx, idx)]

        if len(idx) >= 2:
            sub = rdm[np.ix_(idx, idx)]
            iu = np.triu_indices_from(sub, k=1)
            within_vals = sub[iu]
        else:
            within_vals = np.array([], dtype=float)

        if len(idx) and len(other_idx):
            between_vals = rdm[np.ix_(idx, other_idx)].reshape(-1)
        else:
            between_vals = np.array([], dtype=float)

        rows.append(
            {
                'cdi_semantic': sem,
                'n_categories': int(len(idx)),
                'within_mean': float(np.mean(within_vals)) if within_vals.size else np.nan,
                'between_mean': float(np.mean(between_vals)) if between_vals.size else np.nan,
                'delta_between_minus_within': (
                    float(np.mean(between_vals) - np.mean(within_vals))
                    if within_vals.size and between_vals.size
                    else np.nan
                ),
                'n_within_pairs': int(within_vals.size),
                'n_between_pairs': int(between_vals.size),
            }
        )

    return pd.DataFrame(rows)


def _cdi_semantic_rank(sem: str) -> int:
    return CDI_SEMANTIC_ORDER.index(sem) if sem in CDI_SEMANTIC_ORDER else len(CDI_SEMANTIC_ORDER)


def sort_categories_by_cdi_semantic(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-03 / 05 order: CDI domain, then original position within domain."""
    out = df.copy()
    out['_cdi_rank'] = out['cdi_semantic'].map(lambda s: _cdi_semantic_rank(str(s)))
    out = out.sort_values(['_cdi_rank', 'position', 'category'], na_position='last').drop(columns=['_cdi_rank'])
    return out.reset_index(drop=True)


def cdi_domain_boundaries(ordered_semantics: list[str]) -> list[int]:
    """Bar indices where CDI semantic label changes (for vertical separators)."""
    boundaries = []
    prev = None
    for i, sem in enumerate(ordered_semantics):
        if i > 0 and sem != prev:
            boundaries.append(i)
        prev = sem
    return boundaries


def stripe_domain_order(present_domains: set[str]) -> list[str]:
    order = [d for d in CDI_SEMANTIC_ORDER if d in present_domains]
    for d in sorted(present_domains):
        if d not in order:
            order.append(d)
    return order


def add_cdi_semantic_legend(fig: plt.Figure, present_domains: set[str]) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=CDI_SEMANTIC_COLORS.get(d, CDI_SEMANTIC_COLORS['other']), label=d.replace('_', ' '))
        for d in CDI_SEMANTIC_ORDER
        if d in present_domains
    ]
    if handles:
        fig.legend(handles=handles, title='CDI semantic', loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)


def safe_corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[mask], b[mask]
    if len(a2) < 3:
        return {'n': int(len(a2)), 'pearson_r': np.nan, 'pearson_p': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan}
    pr, pp = pearsonr(a2, b2)
    sr, sp = spearmanr(a2, b2)
    return {
        'n': int(len(a2)),
        'pearson_r': float(pr),
        'pearson_p': float(pp),
        'spearman_rho': float(sr),
        'spearman_p': float(sp),
    }


def normalize_subject_id(sid) -> str:
    return str(sid).strip().zfill(8)


def get_top8_subjects(trajectory_csv: Path, n_top: int = 8) -> list[str]:
    traj_df = pd.read_csv(trajectory_csv)
    traj_df['subject_id'] = traj_df['subject_id'].astype(str).str.zfill(8)
    if 'density' not in traj_df.columns:
        traj_df['density'] = traj_df['n_categories_younger'] + traj_df['n_categories_older']
    ranked = (
        traj_df[['subject_id', 'density']]
        .drop_duplicates('subject_id')
        .sort_values('density', ascending=False)
    )
    return ranked.head(n_top)['subject_id'].tolist()


def get_top8_by_manifest_category_density(
    included_categories_txt: Path,
    sampled_exemplar_csv: Path,
    n_top: int = 8,
) -> tuple[list[str], pd.DataFrame]:
    """Rank subjects by # distinct included categories with ≥1 regular manifest exemplar."""
    valid_categories = set(load_included_categories(included_categories_txt))
    n_total = len(valid_categories)
    sampled = pd.read_csv(sampled_exemplar_csv)
    sampled = sampled[sampled['trial_type'] == 'regular'].copy()
    sampled['category'] = sampled['category'].astype(str).str.strip().str.lower()
    sampled['stem'] = sampled['stem'].astype(str).str.strip().str.lower()
    sampled = sampled[sampled['category'].isin(valid_categories)]
    sampled['subject_id'] = sampled['stem'].map(parse_subject_id_from_stem)
    sampled = sampled.dropna(subset=['subject_id'])
    sampled['subject_id'] = sampled['subject_id'].map(normalize_subject_id)

    density = (
        sampled.groupby('subject_id')['category']
        .nunique()
        .reset_index(name='n_categories_manifest')
    )
    density['n_categories_in_set'] = n_total
    density['density'] = density['n_categories_manifest']
    density['pct_coverage'] = 100.0 * density['n_categories_manifest'] / max(n_total, 1)
    density = density.sort_values(['density', 'subject_id'], ascending=[False, True]).reset_index(drop=True)
    density['rank'] = np.arange(1, len(density) + 1)
    top = density.head(n_top)['subject_id'].tolist()
    return top, density


def resolve_top8_subjects() -> tuple[list[str], pd.DataFrame | None]:
    if TOP8_RANK_MODE in {'category_density', 'category', 'manifest', 'valid129'}:
        top8, rank_df = get_top8_by_manifest_category_density(INCLUDED_TXT, SAMPLED_EXEMPLAR_CSV, N_TOP_DENSEST)
        return top8, rank_df
    return get_top8_subjects(TRAJECTORY_CSV, N_TOP_DENSEST), None


def parse_subject_id_from_stem(stem: str) -> str | None:
    parts = stem.split('_')
    if len(parts) >= 6:
        sid = parts[2].strip()
        if sid.isdigit():
            return sid.zfill(8)
    if parts and parts[0].strip().isdigit():
        return parts[0].strip().zfill(8)
    return None


def load_manifest_regular_pairs(
    included_categories_txt: Path,
    sampled_exemplar_csv: Path,
    top_subjects: set[str],
) -> dict[str, set[str]]:
    """category -> set of exemplar stems (lowercase) for top-8 subjects only."""
    valid_categories = {
        x.strip().lower()
        for x in included_categories_txt.read_text(encoding='utf-8').splitlines()
        if x.strip()
    }
    sampled = pd.read_csv(sampled_exemplar_csv)
    sampled = sampled[sampled['trial_type'] == 'regular'].copy()
    sampled['category'] = sampled['category'].astype(str).str.strip().str.lower()
    sampled['stem'] = sampled['stem'].astype(str).str.strip().str.lower()
    sampled = sampled[sampled['category'].isin(valid_categories)]

    stems_by_cat: dict[str, set[str]] = defaultdict(set)
    n_skipped = 0
    for cat, stem in zip(sampled['category'], sampled['stem']):
        sid = parse_subject_id_from_stem(stem)
        if sid is None or sid not in top_subjects:
            n_skipped += 1
            continue
        stems_by_cat[cat].add(stem)
    print(
        f'Manifest filter: {sum(len(v) for v in stems_by_cat.values())} exemplar stems '
        f'across {len(stems_by_cat)} categories (skipped {n_skipped} non-top8 rows)'
    )
    return dict(stems_by_cat)


class _OnlineMean:
    """Capped reservoir mean over exemplar vectors (max_n); exact mean if max_n is None."""

    __slots__ = ('max_n', 'sum_vec', 'n_seen', 'rng', '_reservoir')

    def __init__(self, max_n: int | None, seed: int):
        self.max_n = max_n
        self.sum_vec: np.ndarray | None = None
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)
        self._reservoir: list[np.ndarray] = []

    def add(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float64).ravel()
        self.n_seen += 1
        if self.max_n is None:
            if self.sum_vec is None:
                self.sum_vec = np.zeros_like(v)
            self.sum_vec += v
            return
        if len(self._reservoir) < self.max_n:
            self._reservoir.append(v)
            return
        j = int(self.rng.integers(0, self.n_seen))
        if j < self.max_n:
            self._reservoir[j] = v

    def mean(self) -> np.ndarray | None:
        if self.max_n is None:
            if self.sum_vec is None or self.n_seen == 0:
                return None
            return self.sum_vec / self.n_seen
        if not self._reservoir:
            return None
        return np.mean(np.stack(self._reservoir, axis=0), axis=0)


def _load_one_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path, mmap_mode=NPY_MMAP_MODE)
    except Exception:
        return None


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def zscore_rows_across_categories(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    return (X - mu) / (sig + eps)


def zscore_subject_embedding_dict(cat_to_vec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(cat_to_vec) < 2:
        return dict(cat_to_vec)
    cats = sorted(cat_to_vec.keys())
    X = np.stack([cat_to_vec[c].ravel() for c in cats], axis=0)
    Xz = zscore_rows_across_categories(X)
    return {c: Xz[i] for i, c in enumerate(cats)}


def _load_one_category_manifest(
    cat_dir: Path,
    allowed_stems: set[str],
    inner_workers: int,
) -> tuple[str, dict[str, _OnlineMean]]:
    category = cat_dir.name.strip().lower()
    if not allowed_stems:
        return category, {}

    # Build load list: only stems present on disk
    tasks: list[tuple[Path, str]] = []
    with os.scandir(cat_dir) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not name.endswith('.npy'):
                continue
            stem = name[:-4].lower()
            if stem not in allowed_stems:
                continue
            sid = parse_subject_id_from_stem(stem)
            if sid is None:
                continue
            tasks.append((Path(entry.path), sid))

    by_sid: dict[str, _OnlineMean] = defaultdict(
        lambda: _OnlineMean(MAX_EXEMPLARS_PER_CATEGORY, EXEMPLAR_SUBSAMPLE_SEED)
    )
    if not tasks:
        return category, {}

    iw = max(1, inner_workers)

    def _consume(path: Path, sid: str) -> None:
        arr = _load_one_npy(path)
        if arr is None:
            return
        vec = _l2_normalize_vec(arr) if ZSCORE_EMBEDDINGS_ACROSS_CATEGORIES else arr.ravel()
        by_sid[sid].add(vec)

    if iw <= 1 or len(tasks) < 8:
        for path, sid in tasks:
            _consume(path, sid)
    else:
        chunksize = max(1, min(128, len(tasks) // (iw * 4)))

        def _work(item: tuple[Path, str]) -> None:
            path, sid = item
            _consume(path, sid)

        with ThreadPoolExecutor(max_workers=iw) as pool:
            list(pool.map(_work, tasks, chunksize=chunksize))

    return category, dict(by_sid)


def _centroid_cache_path(cache_dir: Path, subject_id: str) -> Path:
    return cache_dir / f'subject_{subject_id}_{EMBED_MODEL}_{TOP8_CATEGORY_SET}_centroids.npz'


def _save_subject_cache(path: Path, cat_to_vec: dict[str, np.ndarray]) -> None:
    cats = sorted(cat_to_vec.keys())
    X = np.stack([cat_to_vec[c].astype(np.float32) for c in cats], axis=0)
    np.savez_compressed(path, categories=np.array(cats, dtype=object), embeddings=X)


def _load_subject_cache(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    cats = [str(c) for c in z['categories'].tolist()]
    X = z['embeddings']
    return {c: X[i] for i, c in enumerate(cats)}


def load_top8_subject_centroids(
    embeddings_dir: Path,
    allowed_categories: set[str],
    top_subjects: list[str],
    stems_by_cat: dict[str, set[str]],
    cache_dir: Path,
    rebuild_cache: bool,
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    top_set = set(top_subjects)
    out: dict[str, dict[str, np.ndarray]] = {}

    missing = []
    if not rebuild_cache:
        for sid in top_subjects:
            cached = _load_subject_cache(_centroid_cache_path(cache_dir, sid))
            if cached is not None:
                out[sid] = cached
            else:
                missing.append(sid)
        if not missing:
            print(f'Loaded all {len(top_subjects)} subjects from centroid cache ({cache_dir})')
            return out
        print(f'Cache hit: {len(out)}/{len(top_subjects)}; loading embeddings for: {missing}')
    else:
        missing = list(top_subjects)

    if USE_MANIFEST_FILTER and stems_by_cat:
        scan_categories = set(stems_by_cat.keys()) & allowed_categories
    else:
        scan_categories = allowed_categories
    category_dirs = sorted(
        [p for p in embeddings_dir.iterdir() if p.is_dir() and p.name.strip().lower() in scan_categories],
        key=lambda p: p.name.lower(),
    )

    # subject -> category -> raw mean (before z-score)
    raw: dict[str, dict[str, np.ndarray]] = {sid: {} for sid in missing}

    def _process_category(cat_dir: Path) -> tuple[str, dict[str, np.ndarray]]:
        cat = cat_dir.name.strip().lower()
        allowed_stems = stems_by_cat.get(cat, set())
        if not allowed_stems:
            return cat, {}
        _, online_by_sid = _load_one_category_manifest(
            cat_dir,
            allowed_stems,
            EMBED_LOAD_MAX_WORKERS if not EMBED_PARALLEL_CATEGORIES else 1,
        )
        means = {}
        for sid, om in online_by_sid.items():
            if sid not in top_set:
                continue
            m = om.mean()
            if m is not None:
                means[sid] = m
        return cat, means

    if EMBED_PARALLEL_CATEGORIES and len(category_dirs) >= 4:
        with ThreadPoolExecutor(max_workers=EMBED_CATEGORY_MAX_WORKERS) as pool:
            futs = {pool.submit(_process_category, cd): cd for cd in category_dirs}
            for fut in tqdm(as_completed(futs), total=len(category_dirs), desc='Categories'):
                cat, means = fut.result()
                for sid, vec in means.items():
                    if sid in raw:
                        raw[sid][cat] = vec
    else:
        for cd in tqdm(category_dirs, desc='Categories'):
            cat, means = _process_category(cd)
            for sid, vec in means.items():
                if sid in raw:
                    raw[sid][cat] = vec

    for sid in missing:
        cat_dict = raw.get(sid, {})
        if ZSCORE_EMBEDDINGS_ACROSS_CATEGORIES:
            out[sid] = zscore_subject_embedding_dict(cat_dict)
        else:
            out[sid] = {}
            for cat, vec in cat_dict.items():
                n = np.linalg.norm(vec)
                out[sid][cat] = vec / n if n > 0 else vec
        cache_path = _centroid_cache_path(cache_dir, sid)
        _save_subject_cache(cache_path, out[sid])
        print(f'  cached {sid}: {len(out[sid])} categories -> {cache_path.name}')

    meta = {
        'embed_model': EMBED_MODEL,
        'category_set': TOP8_CATEGORY_SET,
        'use_manifest_filter': USE_MANIFEST_FILTER,
        'max_exemplars_per_category': MAX_EXEMPLARS_PER_CATEGORY,
        'n_categories_dirs': len(category_dirs),
    }
    (cache_dir / 'cache_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out


def subject_within_between_df(
    subject_embeds: dict[str, np.ndarray],
    ordered_categories: list[str],
    semantic_map: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in ordered_categories if c in subject_embeds]
    if len(available) < 3:
        return pd.DataFrame(), available
    semantics = [semantic_map.get(c, 'other') for c in available]
    X = np.stack([subject_embeds[c] for c in available], axis=0)
    return category_within_between_detailed(available, semantics, compute_rdm(X)), available


def save_figure_png_pdf(fig, out_path: Path, *, dpi: int = 200) -> None:
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')


def plot_subject_vs_things_scatter(merged: pd.DataFrame, subject_id: str, out_prefix: Path) -> None:
    sub = merged.dropna(subset=['within_mean_subject', 'within_mean_things'])
    if sub.empty:
        return
    corr_w = safe_corr(sub['within_mean_subject'].to_numpy(), sub['within_mean_things'].to_numpy())
    corr_b = safe_corr(sub['between_mean_subject'].to_numpy(), sub['between_mean_things'].to_numpy())
    corr_d = safe_corr(
        sub['delta_between_minus_within_subject'].to_numpy(),
        sub['delta_between_minus_within_things'].to_numpy(),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, x, y, title, corr in [
        (axes[0], 'within_mean_subject', 'within_mean_things', 'Within-cluster mean', corr_w),
        (axes[1], 'between_mean_subject', 'between_mean_things', 'Between-cluster mean', corr_b),
        (axes[2], 'delta_between_minus_within_subject', 'delta_between_minus_within_things', 'Δ (between − within)', corr_d),
    ]:
        plot_df = sub[[x, y, 'cdi_semantic']].dropna()
        colors = [CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS['other']) for s in plot_df['cdi_semantic']]
        ax.scatter(plot_df[x], plot_df[y], c=colors, s=42, alpha=0.85, edgecolors='0.35', linewidths=0.35)
        lims = [np.nanmin([plot_df[x].min(), plot_df[y].min()]), np.nanmax([plot_df[x].max(), plot_df[y].max()])]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlabel(f'Kid {subject_id}')
        ax.set_ylabel('THINGS')
        ax.set_title(title)
        ax.text(
            0.03, 0.97,
            f"n={corr['n']}\nPearson r={corr['pearson_r']:.3f}\nSpearman rho={corr['spearman_rho']:.3f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )
    fig.suptitle(f'Top-8 kid vs THINGS within/between ({EMBED_MODEL}, {TOP8_CATEGORY_SET})', y=1.02)
    save_figure_png_pdf(fig, out_prefix, dpi=200)
    plt.close(fig)


def plot_delta_comparison_bars_cdi_grouped(merged: pd.DataFrame, subject_id: str, out_prefix: Path) -> None:
    """Category Δ bars in CDI-semantic block order (notebook 03/05), with domain separators."""
    plot_df = sort_categories_by_cdi_semantic(merged)
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.36
    gap = 0.02
    kid_x = x - width / 2 - gap / 2
    things_x = x + width / 2 + gap / 2
    fig_w = max(14, len(plot_df) * 0.15)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    sem_list = plot_df['cdi_semantic'].astype(str).tolist()
    for i, sem in enumerate(sem_list):
        color = CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS['other'])
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.07, zorder=0)

    for bb in cdi_domain_boundaries(sem_list):
        ax.axvline(bb - 0.5, color='0.35', linewidth=1.2, alpha=0.65, zorder=1)

    ax.bar(
        kid_x,
        plot_df['delta_between_minus_within_subject'],
        width=width,
        label='Kid',
        color=KID_BAR_COLOR,
        edgecolor='white',
        linewidth=0.6,
        alpha=0.92,
        zorder=2,
    )
    ax.bar(
        things_x,
        plot_df['delta_between_minus_within_things'],
        width=width,
        label='THINGS',
        color=THINGS_BAR_COLOR,
        edgecolor='white',
        linewidth=0.6,
        alpha=0.92,
        zorder=2,
    )
    ax.axhline(0, color='0.35', linewidth=0.85, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['category'], rotation=90, ha='center', fontsize=6)
    for tick, sem in zip(ax.get_xticklabels(), sem_list):
        tick.set_color(CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS['other']))

    ax.set_ylabel(r'$\overline{d}_{\mathrm{between}} - \overline{d}_{\mathrm{within}}$ (cosine)')
    ax.set_xlabel('Category (grouped by CDI semantic domain)')
    ax.set_title(f'{subject_id}: category cluster strength vs THINGS (n={len(plot_df)})')
    ax.legend(loc='upper right', frameon=False, ncol=2)
    ax.grid(axis='y', color='0.9', linewidth=0.7)
    ax.set_axisbelow(True)
    add_cdi_semantic_legend(fig, set(sem_list))
    fig.tight_layout()
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def plot_cluster_averaged_kid_vs_things(
    cluster_merged: pd.DataFrame,
    subject_id: str,
    out_prefix: Path,
) -> None:
    """Barplots at CDI semantic-domain level only (means pooled over categories in each domain)."""
    sid = normalize_subject_id(subject_id)
    present = set(cluster_merged['cdi_semantic'].astype(str))
    bar_order = stripe_domain_order(present)
    plot_df = cluster_merged.set_index('cdi_semantic').loc[bar_order].reset_index()
    if plot_df.empty:
        return

    x = np.arange(len(bar_order))
    w = 0.36
    gap = 0.02
    kid_x = x - w / 2 - gap / 2
    things_x = x + w / 2 + gap / 2

    fig, axes = plt.subplots(1, 3, figsize=(max(14, 1.35 * len(bar_order)), 4.8), constrained_layout=True)

    panels = [
        (axes[0], 'within_mean_subject', 'within_mean_things', r'Within-cluster $\overline{d}$'),
        (axes[1], 'between_mean_subject', 'between_mean_things', r'Between-cluster $\overline{d}$'),
        (axes[2], 'delta_between_minus_within_subject', 'delta_between_minus_within_things', r'$\Delta=\overline{d}_{\mathrm{between}}-\overline{d}_{\mathrm{within}}$'),
    ]

    for ax, kid_col, th_col, ylab in panels:
        ax.bar(
            kid_x,
            plot_df[kid_col],
            width=w,
            label='Kid',
            color=KID_BAR_COLOR,
            edgecolor='white',
            linewidth=0.7,
            alpha=0.92,
        )
        ax.bar(
            things_x,
            plot_df[th_col],
            width=w,
            label='THINGS',
            color=THINGS_BAR_COLOR,
            edgecolor='white',
            linewidth=0.7,
            alpha=0.92,
        )
        ax.axhline(0, color='0.35', linewidth=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ') for s in bar_order], rotation=45, ha='right', fontsize=9)
        for tick, sem in zip(ax.get_xticklabels(), bar_order):
            tick.set_color(CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS['other']))
        ax.set_ylabel(ylab)
        ax.grid(axis='y', color='0.9', linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle(
        f'{sid}: CDI-domain averages vs THINGS ({EMBED_MODEL}, {TOP8_CATEGORY_SET})\n'
        'Means pooled over categories within each CDI semantic domain',
        fontsize=11,
        y=1.04,
    )
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def merged_for_shuffle(cmerged: pd.DataFrame) -> pd.DataFrame:
    """Rename kid/subject columns to bv/* for shared shuffle inference."""
    return cmerged.rename(
        columns={
            'within_mean_subject': 'within_mean_bv',
            'between_mean_subject': 'between_mean_bv',
            'delta_between_minus_within_subject': 'delta_between_minus_within_bv',
            'n_categories_subject': 'n_categories_bv',
            'n_within_pairs_subject': 'n_within_pairs_bv',
            'n_between_pairs_subject': 'n_between_pairs_bv',
        }
    )


def kid_things_rdms(
    subject_embeds: dict[str, np.ndarray],
    ordered_categories: list[str],
    semantic_map: dict[str, str],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    available = [c for c in ordered_categories if c in subject_embeds]
    if len(available) < 3:
        raise ValueError(f'Fewer than 3 categories for RDM ({len(available)})')
    semantics = [semantic_map.get(c, 'other') for c in available]
    kid_X = np.stack([subject_embeds[c] for c in available], axis=0)
    th_cats, th_emb = load_embedding_csv(THINGS_EMBED_CSV)
    th_idx = {c: i for i, c in enumerate(th_cats)}
    th_X = np.stack([th_emb[th_idx[c]] for c in available], axis=0)
    return available, semantics, compute_rdm(kid_X), compute_rdm(th_X)


def subject_shuffle_paths(out_subdir: Path, sid: str) -> tuple[Path, Path]:
    sid = normalize_subject_id(sid)
    return (
        out_subdir / f'subject_{sid}_shuffle_vs_things_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv',
        out_subdir / f'subject_{sid}_null_bands_vs_things_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.npz',
    )


def save_subject_shuffle_artifacts(out_subdir: Path, sid: str, shuffle_meta: dict) -> None:
    csv_path, npz_path = subject_shuffle_paths(out_subdir, sid)
    shuffle_meta['main_wide'].to_csv(csv_path, index=False)
    np.savez(
        npz_path,
        null_kid=shuffle_meta['null_bv'],
        null_th=shuffle_meta['null_th'],
        bar_order=np.array(shuffle_meta['bar_order'], dtype=object),
        n_perm=int(shuffle_meta['n_perm']),
    )


def load_subject_shuffle_artifacts(out_subdir: Path, sid: str) -> dict | None:
    csv_path, npz_path = subject_shuffle_paths(out_subdir, sid)
    if not csv_path.is_file() or not npz_path.is_file():
        return None
    z = np.load(npz_path, allow_pickle=True)
    main_wide = pd.read_csv(csv_path)
    return {
        'main_wide': main_wide,
        'null_bv': z['null_kid'],
        'null_th': z['null_th'],
        'bar_order': [str(x) for x in z['bar_order'].tolist()],
        'sig_qvals': main_wide.set_index('cdi_semantic')['bv_vs_th_shuffle_q_one_sided_gt_fdr_bh'].to_dict(),
        'n_perm': int(z['n_perm']),
    }


def compute_kid_shuffle_meta(
    sid: str,
    cmerged: pd.DataFrame,
    subject_embeds: dict[str, np.ndarray],
    ordered_categories: list[str],
    semantic_map: dict[str, str],
) -> dict:
    available, semantics, kid_rdm, th_rdm = kid_things_rdms(subject_embeds, ordered_categories, semantic_map)
    present = set(cmerged['cdi_semantic'].astype(str))
    bar_order = stripe_domain_order(present)
    merged = merged_for_shuffle(cmerged)
    seed = SHUFFLE_SEED + (int(sid) % 10_000)
    return run_bv_things_shuffle_inference(
        available,
        semantics,
        kid_rdm,
        th_rdm,
        bar_order,
        merged,
        n_perm=N_PERM,
        seed=seed,
        cluster_within_between=cluster_within_between,
    )


def ensure_kid_shuffle_meta(
    sid: str,
    cmerged: pd.DataFrame,
    subject_embeds: dict[str, np.ndarray],
    ordered_categories: list[str],
    semantic_map: dict[str, str],
    out_subdir: Path,
    *,
    recompute: bool = False,
) -> dict:
    if not recompute:
        loaded = load_subject_shuffle_artifacts(out_subdir, sid)
        if loaded is not None:
            return loaded
    meta = compute_kid_shuffle_meta(sid, cmerged, subject_embeds, ordered_categories, semantic_map)
    save_subject_shuffle_artifacts(out_subdir, sid, meta)
    return meta


def _shuffle_x_limits(meta: dict) -> tuple[float, float]:
    xs: list[float] = []
    for arr in (meta['null_bv'], meta['null_th']):
        xs.extend(arr[:, 0].tolist())
        xs.extend(arr[:, 2].tolist())
    mw = meta['main_wide']
    xs.extend(mw['delta_between_minus_within_bv'].tolist())
    xs.extend(mw['delta_between_minus_within_things'].tolist())
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return (-0.05, 0.45)
    x_lo, x_hi = min(xs), max(xs)
    pad = max(0.02, 0.1 * (x_hi - x_lo + 1e-9))
    return (x_lo - pad, x_hi + pad)


def plot_kid_delta_vertical_shuffle(
    shuffle_meta: dict,
    sid: str,
    out_prefix: Path,
    *,
    n_categories: int | None = None,
) -> None:
    sid = normalize_subject_id(sid)
    bar_order = shuffle_meta['bar_order']
    fig_h = max(5.8, 0.58 * len(bar_order))
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.22)

    draw_horizontal_paired_cluster_delta_bars(
        ax,
        bar_order,
        shuffle_meta['main_wide'],
        shuffle_meta['null_bv'],
        shuffle_meta['null_th'],
        cdi_semantic_colors=CDI_SEMANTIC_COLORS,
        title=f'Kid {sid}',
        show_ylabel=True,
        show_legend=False,
        x_limits=_shuffle_x_limits(shuffle_meta),
        sig_qvals=shuffle_meta['sig_qvals'],
        n_perm=shuffle_meta['n_perm'],
        bar_label_a='Kid Δ',
        bar_label_b='THINGS Δ',
        null_whisk_label_a=f'Null kid whisker (2.5–97.5%; $n_{{perm}}$={shuffle_meta["n_perm"]})',
        sig_note='',
    )

    merged = shuffle_meta['main_wide']
    if 'n_categories_bv' in merged.columns:
        ix = merged.set_index('cdi_semantic')
        x_hi = ax.get_xlim()[1]
        for kk, sem in enumerate(bar_order):
            if sem not in ix.index:
                continue
            nc = int(ix.loc[sem, 'n_categories_bv'])
            if nc <= 2:
                ax.text(
                    x_hi * 0.98,
                    kk,
                    f'n={nc} cat.',
                    ha='right',
                    va='center',
                    fontsize=7.5,
                    color='0.45',
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )

    n_cat_note = f' · {n_categories} categories' if n_categories else ''
    leg_handles = [
        Line2D([0], [0], color='#37474f', linewidth=2.2, label=f'Null kid whisker (2.5–97.5%; $n_{{perm}}$={shuffle_meta["n_perm"]})'),
        Line2D([0], [0], color='#78909c', linewidth=2.2, label='Null THINGS whisker'),
        Patch(facecolor=KID_BAR_COLOR, edgecolor='white', label='Kid Δ'),
        Patch(facecolor=THINGS_BAR_COLOR, edgecolor='white', label='THINGS Δ'),
    ]
    fig.legend(handles=leg_handles, frameon=False, fontsize=8.0, loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0.02))
    fig.suptitle(
        f'Top-8 kid vs THINGS · CDI-domain Δ ({EMBED_MODEL}, {TOP8_CATEGORY_SET}{n_cat_note})\n'
        'Parallel label shuffle null · stars: BH-FDR q<0.05, kid>TH',
        fontsize=10.5,
        y=0.98,
    )
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def cluster_merged_kid_vs_things(
    subject_embeds: dict[str, np.ndarray],
    things_cluster_df: pd.DataFrame,
    ordered_categories: list[str],
    semantic_map: dict[str, str],
) -> pd.DataFrame:
    available = [c for c in ordered_categories if c in subject_embeds]
    if len(available) < 3:
        return pd.DataFrame()
    semantics = [semantic_map.get(c, 'other') for c in available]
    X = np.stack([subject_embeds[c] for c in available], axis=0)
    kid_cluster = cluster_within_between(available, semantics, compute_rdm(X))
    kid_cluster = kid_cluster.rename(
        columns={
            c: f'{c}_subject'
            for c in kid_cluster.columns
            if c != 'cdi_semantic'
        }
    )
    th_cluster = things_cluster_df.rename(
        columns={
            c: f'{c}_things'
            for c in things_cluster_df.columns
            if c != 'cdi_semantic'
        }
    )
    return kid_cluster.merge(th_cluster, on='cdi_semantic', how='inner')


def make_all_subject_plots(
    top8: list[str],
    subject_embeddings: dict[str, dict[str, np.ndarray]],
    things_cluster_df: pd.DataFrame,
    ordered_categories: list[str],
    semantic_map: dict[str, str],
    fig_subdir: Path,
    out_subdir: Path,
    *,
    merged_by_subject: dict[str, pd.DataFrame] | None = None,
    recompute_shuffle: bool = False,
    skip_delta_vertical: bool = False,
) -> list[pd.DataFrame]:
    """CDI-domain barplots only (no per-category bars). Optional category scatter if merged data passed."""
    include_scatter = (
        merged_by_subject is not None
        and os.environ.get('TOP8_INCLUDE_CATEGORY_SCATTER', '').strip().lower() in {'1', 'true', 'yes'}
    )
    cluster_long = []
    for sid in top8:
        sid = normalize_subject_id(sid)
        if include_scatter and merged_by_subject and sid in merged_by_subject:
            plot_subject_vs_things_scatter(
                merged_by_subject[sid],
                sid,
                fig_subdir / f'top8_{sid}_vs_things_scatter_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.png',
            )
        if sid not in subject_embeddings:
            print(f'Warning: no embeddings for {sid}, skipping CDI-domain bars')
            continue
        cmerged = cluster_merged_kid_vs_things(
            subject_embeddings[sid], things_cluster_df, ordered_categories, semantic_map
        )
        if cmerged.empty:
            continue
        cmerged = cmerged.copy()
        cmerged['subject_id'] = sid
        cluster_long.append(cmerged)
        cmerged.to_csv(
            out_subdir / f'subject_{sid}_cluster_within_between_vs_things_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv',
            index=False,
        )
        plot_cluster_averaged_kid_vs_things(
            cmerged,
            sid,
            fig_subdir / f'top8_{sid}_vs_things_cdi_domain_bars_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.png',
        )
        if not skip_delta_vertical:
            shuffle_meta = ensure_kid_shuffle_meta(
                sid,
                cmerged,
                subject_embeddings[sid],
                ordered_categories,
                semantic_map,
                out_subdir,
                recompute=recompute_shuffle,
            )
            plot_kid_delta_vertical_shuffle(
                shuffle_meta,
                sid,
                fig_subdir / f'top8_{sid}_vs_things_cdi_domain_delta_vertical_{EMBED_MODEL}_{TOP8_CATEGORY_SET}',
                n_categories=len([c for c in ordered_categories if c in subject_embeddings[sid]]),
            )
    return cluster_long


def _things_cluster_and_category_tables(
    ordered_categories: list[str],
    semantic_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    th_cats, th_emb = load_embedding_csv(THINGS_EMBED_CSV)
    th_idx = {c: i for i, c in enumerate(th_cats)}
    th_ordered = [c for c in ordered_categories if c in th_idx]
    th_X = np.stack([th_emb[th_idx[c]] for c in th_ordered], axis=0)
    th_sem = [semantic_map.get(c, 'other') for c in th_ordered]
    th_rdm = compute_rdm(th_X)
    things_cat = category_within_between_detailed(th_ordered, th_sem, th_rdm)
    things_cluster = cluster_within_between(th_ordered, th_sem, th_rdm)
    things_cat = things_cat.rename(
        columns={c: f'{c}_things' for c in things_cat.columns if c not in {'category', 'cdi_semantic'}}
    )
    return things_cat, things_cluster


def main() -> None:
    args = parse_args()
    sns.set_context('talk')
    sns.set_style('whitegrid')

    semantic_map = load_cdi_semantic_map(CDI_SEMANTIC_CSV)
    ordered_categories = load_order(ORDER_CSV)
    allowed = set(load_included_categories(INCLUDED_TXT))
    ordered_categories = [c for c in ordered_categories if c in allowed]
    pos_map = {c: i for i, c in enumerate(ordered_categories)}

    top8, rank_df = resolve_top8_subjects()
    print(f'Top-8 subjects ({TOP8_RANK_MODE}): {top8}')
    print(f'EMBED_MODEL={EMBED_MODEL}, category_set={TOP8_CATEGORY_SET}, manifest_filter={USE_MANIFEST_FILTER}, max_exemplars={MAX_EXEMPLARS_PER_CATEGORY}')

    out_subdir = RESULTS_DIR / f'top8_vs_things_within_between_{EMBED_MODEL}_{TOP8_CATEGORY_SET}'
    fig_subdir = FIGURES_DIR / f'top8_vs_things_within_between_{EMBED_MODEL}_{TOP8_CATEGORY_SET}'
    cache_dir = out_subdir / 'cache_subject_centroids'
    out_subdir.mkdir(parents=True, exist_ok=True)
    fig_subdir.mkdir(parents=True, exist_ok=True)

    if rank_df is not None:
        rank_path = out_subdir / f'top8_subject_density_rank_{TOP8_CATEGORY_SET}.csv'
        rank_df.to_csv(rank_path, index=False)
        print(f'Wrote density rank: {rank_path}')
        print(rank_df.head(N_TOP_DENSEST).to_string(index=False))

    things_df, things_cluster_df = _things_cluster_and_category_tables(ordered_categories, semantic_map)
    things_df.to_csv(out_subdir / f'things_category_within_between_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)
    things_cluster_df.to_csv(out_subdir / f'things_cluster_within_between_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)

    if args.plots_only:
        long_path = out_subdir / f'top8_vs_things_category_merged_long_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv'
        if not long_path.exists():
            raise FileNotFoundError(f'--plots-only requires {long_path}')
        long_df = pd.read_csv(long_path)
        long_df['subject_id'] = long_df['subject_id'].map(normalize_subject_id)
        subject_embeddings = {}
        for sid in top8:
            cached = _load_subject_cache(_centroid_cache_path(cache_dir, sid))
            if cached is not None:
                subject_embeddings[sid] = cached
        merged_by_subject = {
            normalize_subject_id(sid): sort_categories_by_cdi_semantic(g.drop(columns=['subject_id'], errors='ignore'))
            for sid, g in long_df.groupby('subject_id')
        }
        if not args.skip_plots:
            cluster_long = make_all_subject_plots(
                top8,
                subject_embeddings,
                things_cluster_df,
                ordered_categories,
                semantic_map,
                fig_subdir,
                out_subdir,
                recompute_shuffle=args.recompute_shuffle,
                skip_delta_vertical=args.skip_delta_vertical,
            )
            if cluster_long:
                pd.concat(cluster_long, ignore_index=True).to_csv(
                    out_subdir / f'top8_vs_things_cluster_merged_long_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv',
                    index=False,
                )
            print(f'Regenerated figures: {fig_subdir}')
            if not args.skip_delta_vertical:
                print(f'  vertical Δ + shuffle: top8_{{subject}}_vs_things_cdi_domain_delta_vertical_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.png')
        return

    stems_by_cat = (
        load_manifest_regular_pairs(INCLUDED_TXT, SAMPLED_EXEMPLAR_CSV, set(top8))
        if USE_MANIFEST_FILTER
        else {}
    )

    subject_embeddings = load_top8_subject_centroids(
        EMBEDDINGS_DIR,
        allowed,
        top8,
        stems_by_cat,
        cache_dir,
        rebuild_cache=args.rebuild_cache,
    )

    long_rows = []
    summary_rows = []
    coverage_rows = []
    merged_by_subject: dict[str, pd.DataFrame] = {}

    for sid in top8:
        if sid not in subject_embeddings:
            print(f'Warning: no embeddings for {sid}')
            continue
        sub_df, available = subject_within_between_df(subject_embeddings[sid], ordered_categories, semantic_map)
        coverage_rows.append({
            'subject_id': sid,
            'n_categories': len(available),
            'n_categories_in_set': len(ordered_categories),
            'pct_coverage': 100.0 * len(available) / len(ordered_categories),
        })
        sub_df = sub_df.rename(
            columns={c: f'{c}_subject' for c in sub_df.columns if c not in {'category', 'cdi_semantic'}}
        )
        sub_df.to_csv(out_subdir / f'subject_{sid}_category_within_between_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)

        merged = things_df.merge(sub_df, on=['category', 'cdi_semantic'], how='inner')
        merged['position'] = merged['category'].map(pos_map)
        merged['subject_id'] = normalize_subject_id(sid)
        merged = sort_categories_by_cdi_semantic(merged)
        long_rows.append(merged)
        merged_by_subject[sid] = merged

        for metric in ('within_mean', 'between_mean', 'delta_between_minus_within'):
            c = safe_corr(merged[f'{metric}_subject'].to_numpy(), merged[f'{metric}_things'].to_numpy())
            summary_rows.append({'subject_id': sid, 'metric': metric, **c})

    coverage_df = pd.DataFrame(coverage_rows)
    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    coverage_df.to_csv(out_subdir / f'top8_category_coverage_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)
    summary_df.to_csv(out_subdir / f'top8_vs_things_category_corr_summary_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)
    long_df.to_csv(out_subdir / f'top8_vs_things_category_merged_long_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv', index=False)

    if not args.skip_plots:
        cluster_long = make_all_subject_plots(
            top8,
            subject_embeddings,
            things_cluster_df,
            ordered_categories,
            semantic_map,
            fig_subdir,
            out_subdir,
            merged_by_subject=merged_by_subject if merged_by_subject else None,
            recompute_shuffle=args.recompute_shuffle,
            skip_delta_vertical=args.skip_delta_vertical,
        )
        if cluster_long:
            pd.concat(cluster_long, ignore_index=True).to_csv(
                out_subdir / f'top8_vs_things_cluster_merged_long_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.csv',
                index=False,
            )

    print(f'\nCategory coverage ({TOP8_CATEGORY_SET}):')
    print(coverage_df.to_string(index=False))
    print('\nKid vs THINGS correlations:')
    print(summary_df.to_string(index=False))
    print(f'\nTables: {out_subdir}')
    if not args.skip_plots:
        print(f'Figures: {fig_subdir}')
        print('  CDI-domain bars: top8_{subject}_vs_things_cdi_domain_bars_*.png')
        if not args.skip_delta_vertical:
            print(f'  vertical Δ + shuffle: top8_{{subject}}_vs_things_cdi_domain_delta_vertical_{EMBED_MODEL}_{TOP8_CATEGORY_SET}.png')


if __name__ == '__main__':
    main()
