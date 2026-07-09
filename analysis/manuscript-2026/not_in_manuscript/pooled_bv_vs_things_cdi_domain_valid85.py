"""Pooled BabyView exemplar set vs THINGS: CDI-domain within/between bars (valid85).

Uses the same exemplar-set z-scored CSVs as notebooks 05–07 (06 BV, 07 THINGS) and the
same cluster metrics as top8_within_between_vs_things.py. Output matches top-8 kid figure
layout (within / between / Δ by CDI domain) for direct supplemental comparison.

  python analysis/manuscript-2026/not_in_manuscript/pooled_bv_vs_things_cdi_domain_valid85.py
  BV_EMBED_MODEL=dinov3 python analysis/manuscript-2026/not_in_manuscript/pooled_bv_vs_things_cdi_domain_valid85.py
  python ... --models clip dinov3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = MANUSCRIPT_DIR / "scripts"
for _p in (MANUSCRIPT_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bv_things_cdi_shuffle_inference import (  # noqa: E402
    draw_horizontal_paired_cluster_delta_bars,
    run_bv_things_shuffle_inference,
)
from top8_within_between_vs_things import (  # noqa: E402
    CDI_SEMANTIC_COLORS,
    category_within_between_detailed,
    cluster_within_between,
    compute_rdm,
    load_cdi_semantic_map,
    load_embedding_csv,
    load_order,
    safe_corr,
    save_figure_png_pdf,
    stripe_domain_order,
)

PREPRINT_DIR = MANUSCRIPT_DIR
PROJECT_ROOT = MANUSCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

CATEGORY_SET = os.environ.get('BV_CATEGORY_SET', 'valid85').strip()
THRESHOLD_TOKEN = os.environ.get('BV_THINGS_EMBED_THRESHOLD', '0.27').strip()
if THRESHOLD_TOKEN in {'0', '0.0', '0.00'}:
    THRESHOLD_TOKEN = '0.27'

OUTPUT_RUN_ROOT = PREPRINT_DIR / 'supplemental_results_valid85cats_04302026'
RESULTS_DIR = OUTPUT_RUN_ROOT / 'results'
FIGURES_DIR = OUTPUT_RUN_ROOT / 'figures'
EXEMPLAR_EMBED_DIR = PREPRINT_DIR / 'exemplar_set_embeddings' / CATEGORY_SET
CDI_SEMANTIC_CSV = DATA_DIR / f'long_tailed_dist_prop_included_categories_{CATEGORY_SET}.csv'
ORDER_CSV = RESULTS_DIR / f'bv_things_rdm_order_bv_semantic_clip_filtered-{THRESHOLD_TOKEN}_{CATEGORY_SET}.csv'

BV_BAR_COLOR = '#0d47a1'
THINGS_BAR_COLOR = '#90caf9'
N_PERM = int(os.environ.get('BV_CLUSTER_PERMUTATIONS', '5000'))
SHUFFLE_SEED = int(os.environ.get('BV_CLUSTER_PERM_SEED', '42'))


def embed_paths(model: str) -> tuple[Path, Path]:
    return (
        EXEMPLAR_EMBED_DIR / f'bv_{model}_exemplar_avg_zscore_within_{CATEGORY_SET}.csv',
        EXEMPLAR_EMBED_DIR / f'things_{model}_exemplar_avg_zscore_within_{CATEGORY_SET}.csv',
    )


def aligned_embeddings(
    bv_csv: Path,
    things_csv: Path,
    ordered_categories: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    bv_cats, bv_emb = load_embedding_csv(bv_csv)
    th_cats, th_emb = load_embedding_csv(things_csv)
    bv_idx = {c: i for i, c in enumerate(bv_cats)}
    th_idx = {c: i for i, c in enumerate(th_cats)}
    shared = [c for c in ordered_categories if c in bv_idx and c in th_idx]
    if len(shared) < 3:
        raise ValueError(f'Fewer than 3 shared categories ({len(shared)})')
    bv_X = np.stack([bv_emb[bv_idx[c]] for c in shared], axis=0)
    th_X = np.stack([th_emb[th_idx[c]] for c in shared], axis=0)
    return shared, bv_X, th_X


def cluster_merged_bv_vs_things(
    bv_cluster: pd.DataFrame,
    things_cluster: pd.DataFrame,
) -> pd.DataFrame:
    bv = bv_cluster.rename(
        columns={c: f'{c}_bv' for c in bv_cluster.columns if c != 'cdi_semantic'}
    )
    th = things_cluster.rename(
        columns={c: f'{c}_things' for c in things_cluster.columns if c != 'cdi_semantic'}
    )
    return bv.merge(th, on='cdi_semantic', how='inner')


def global_cluster_strength(cluster_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    w = cluster_df['within_mean'].to_numpy(dtype=float)
    b = cluster_df['between_mean'].to_numpy(dtype=float)
    mask = np.isfinite(w) & np.isfinite(b)
    if not mask.any():
        return {}
    return {
        f'{prefix}_within_mean': float(np.nanmean(w[mask])),
        f'{prefix}_between_mean': float(np.nanmean(b[mask])),
        f'{prefix}_delta_between_minus_within': float(np.nanmean(b[mask] - w[mask])),
    }


def plot_cdi_domain_bars(merged: pd.DataFrame, model: str, out_prefix: Path) -> None:
    present = set(merged['cdi_semantic'].astype(str))
    bar_order = stripe_domain_order(present)
    plot_df = merged.set_index('cdi_semantic').loc[bar_order].reset_index()
    if plot_df.empty:
        return

    x = np.arange(len(bar_order))
    w = 0.36
    gap = 0.02
    bv_x = x - w / 2 - gap / 2
    th_x = x + w / 2 + gap / 2

    fig, axes = plt.subplots(1, 3, figsize=(max(14, 1.35 * len(bar_order)), 4.8), constrained_layout=True)
    panels = [
        (axes[0], 'within_mean_bv', 'within_mean_things', r'Within-cluster $\overline{d}$'),
        (axes[1], 'between_mean_bv', 'between_mean_things', r'Between-cluster $\overline{d}$'),
        (axes[2], 'delta_between_minus_within_bv', 'delta_between_minus_within_things', r'$\Delta=\overline{d}_{\mathrm{between}}-\overline{d}_{\mathrm{within}}$'),
    ]

    for ax, bv_col, th_col, ylab in panels:
        ax.bar(
            bv_x,
            plot_df[bv_col],
            width=w,
            label='BabyView (pooled)',
            color=BV_BAR_COLOR,
            edgecolor='white',
            linewidth=0.7,
            alpha=0.92,
        )
        ax.bar(
            th_x,
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

    corr_d = safe_corr(
        plot_df['delta_between_minus_within_bv'].to_numpy(),
        plot_df['delta_between_minus_within_things'].to_numpy(),
    )
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle(
        f'Pooled BabyView vs THINGS: CDI-domain averages ({model}, {CATEGORY_SET})\n'
        f'Means pooled over categories within each domain · domain Δ Pearson r={corr_d["pearson_r"]:.3f}',
        fontsize=11,
        y=1.05,
    )
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def shuffle_paths(model: str) -> tuple[Path, Path]:
    out_results = RESULTS_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}'
    return (
        out_results / f'pooled_bv_vs_things_shuffle_{model}_{CATEGORY_SET}.csv',
        out_results / f'pooled_bv_vs_things_null_bands_{model}_{CATEGORY_SET}.npz',
    )


def save_shuffle_artifacts(model: str, shuffle_meta: dict) -> None:
    csv_path, npz_path = shuffle_paths(model)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    shuffle_meta['main_wide'].to_csv(csv_path, index=False)
    np.savez(
        npz_path,
        null_bv=shuffle_meta['null_bv'],
        null_th=shuffle_meta['null_th'],
        bar_order=np.array(shuffle_meta['bar_order'], dtype=object),
        n_perm=int(shuffle_meta['n_perm']),
    )


def load_shuffle_artifacts(model: str) -> dict | None:
    csv_path, npz_path = shuffle_paths(model)
    if not csv_path.is_file() or not npz_path.is_file():
        return None
    z = np.load(npz_path, allow_pickle=True)
    main_wide = pd.read_csv(csv_path)
    return {
        'main_wide': main_wide,
        'null_bv': z['null_bv'],
        'null_th': z['null_th'],
        'bar_order': [str(x) for x in z['bar_order'].tolist()],
        'sig_qvals': main_wide.set_index('cdi_semantic')['bv_vs_th_shuffle_q_one_sided_gt_fdr_bh'].to_dict(),
        'n_perm': int(z['n_perm']),
    }


def compute_shuffle_meta(
    model: str,
    categories: list[str],
    semantics: list[str],
    bv_rdm: np.ndarray,
    th_rdm: np.ndarray,
    merged: pd.DataFrame,
) -> dict:
    present = set(merged['cdi_semantic'].astype(str))
    bar_order = stripe_domain_order(present)
    meta = run_bv_things_shuffle_inference(
        categories,
        semantics,
        bv_rdm,
        th_rdm,
        bar_order,
        merged,
        n_perm=N_PERM,
        seed=SHUFFLE_SEED,
        cluster_within_between=cluster_within_between,
    )
    save_shuffle_artifacts(model, meta)
    return meta


def _pooled_x_limits(meta_list: list[dict]) -> tuple[float, float]:
    xs: list[float] = []
    for meta in meta_list:
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


def plot_cdi_domain_delta_vertical(
    shuffle_meta: dict,
    model: str,
    out_prefix: Path,
) -> None:
    """Horizontal paired Δ bars with label-shuffle null whiskers (notebook 05 style)."""
    bar_order = shuffle_meta['bar_order']
    merged = shuffle_meta['main_wide']
    fig_h = max(5.8, 0.58 * len(bar_order))
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.22)

    draw_horizontal_paired_cluster_delta_bars(
        ax,
        bar_order,
        merged,
        shuffle_meta['null_bv'],
        shuffle_meta['null_th'],
        cdi_semantic_colors=CDI_SEMANTIC_COLORS,
        title=f'{model.upper()} · pooled exemplar set ({CATEGORY_SET})',
        show_ylabel=True,
        show_legend=False,
        x_limits=_pooled_x_limits([shuffle_meta]),
        sig_qvals=shuffle_meta['sig_qvals'],
        n_perm=shuffle_meta['n_perm'],
    )

    if 'n_categories_bv' in merged.columns:
        ix = merged.set_index('cdi_semantic')
        x_hi = ax.get_xlim()[1]
        for kk, sem in enumerate(bar_order):
            nc = int(ix.loc[sem, 'n_categories_bv']) if sem in ix.index else 0
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

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    leg_handles = [
        Line2D([0], [0], color='#37474f', linewidth=2.2, label=f'Null BV whisker (2.5–97.5%; $n_{{perm}}$={shuffle_meta["n_perm"]})'),
        Line2D([0], [0], color='#78909c', linewidth=2.2, label='Null THINGS whisker'),
        Patch(facecolor=BV_BAR_COLOR, edgecolor='white', label='BabyView Δ'),
        Patch(facecolor=THINGS_BAR_COLOR, edgecolor='white', label='THINGS Δ'),
    ]
    fig.legend(handles=leg_handles, frameon=False, fontsize=8.2, loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0.02))

    corr_d = safe_corr(
        merged['delta_between_minus_within_bv'].to_numpy(),
        merged['delta_between_minus_within_things'].to_numpy(),
    )
    fig.suptitle(
        r'CDI-domain $\overline{d}_{\mathrm{between}}-\overline{d}_{\mathrm{within}}$ · parallel label shuffle null'
        f'\nPearson r={corr_d["pearson_r"]:.3f} across domains · * BH-FDR q<0.05 (BV>TH)',
        fontsize=10.5,
        y=0.98,
    )
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def plot_cdi_domain_delta_vertical_2panel(
    shuffle_by_model: dict[str, dict],
    out_prefix: Path,
) -> None:
    models = [m for m in ('clip', 'dinov3') if m in shuffle_by_model]
    if not models:
        return

    n_domains = max(len(m['bar_order']) for m in shuffle_by_model.values())
    fig_h = max(5.8, 0.58 * n_domains)
    fig, axes = plt.subplots(1, len(models), figsize=(7.4 * len(models), fig_h))
    if len(models) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.20, wspace=0.38)

    x_lim = _pooled_x_limits([shuffle_by_model[m] for m in models])
    for ax, model in zip(axes, models):
        meta = shuffle_by_model[model]
        draw_horizontal_paired_cluster_delta_bars(
            ax,
            meta['bar_order'],
            meta['main_wide'],
            meta['null_bv'],
            meta['null_th'],
            cdi_semantic_colors=CDI_SEMANTIC_COLORS,
            title=f'{model.upper()}',
            show_ylabel=(model == models[0]),
            show_legend=False,
            x_limits=x_lim,
            sig_qvals=meta['sig_qvals'],
            n_perm=meta['n_perm'],
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_perm = shuffle_by_model[models[0]]['n_perm']
    leg_handles = [
        Line2D([0], [0], color='#37474f', linewidth=2.2, label=f'Null BV whisker (2.5–97.5%; $n_{{perm}}$={n_perm})'),
        Line2D([0], [0], color='#78909c', linewidth=2.2, label='Null THINGS whisker'),
        Patch(facecolor=BV_BAR_COLOR, edgecolor='white', label='BabyView Δ'),
        Patch(facecolor=THINGS_BAR_COLOR, edgecolor='white', label='THINGS Δ'),
    ]
    fig.legend(handles=leg_handles, frameon=False, fontsize=8.5, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        r'Pooled BV vs THINGS: $\overline{d}_{\mathrm{between}}-\overline{d}_{\mathrm{within}}$ by CDI domain (valid85)'
        '\nParallel label shuffle null · stars: BH-FDR q<0.05, BV>TH (* q<0.05, ** q<0.01, *** q<0.001)',
        fontsize=10.5,
        y=0.97,
    )
    save_figure_png_pdf(fig, out_prefix, dpi=220)
    plt.close(fig)


def ensure_shuffle_meta(
    model: str,
    merged: pd.DataFrame,
    ordered_categories: list[str],
    semantic_map: dict[str, str],
    *,
    recompute: bool = False,
) -> dict:
    if not recompute:
        loaded = load_shuffle_artifacts(model)
        if loaded is not None:
            return loaded

    bv_csv, th_csv = embed_paths(model)
    shared, bv_X, th_X = aligned_embeddings(bv_csv, th_csv, ordered_categories)
    semantics = [semantic_map.get(c, 'other') for c in shared]
    return compute_shuffle_meta(model, shared, semantics, compute_rdm(bv_X), compute_rdm(th_X), merged)


def load_merged_from_results(model: str) -> pd.DataFrame:
    path = RESULTS_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}' / (
        f'pooled_bv_vs_things_cluster_merged_{model}_{CATEGORY_SET}.csv'
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run_model(model: str, ordered_categories: list[str], semantic_map: dict[str, str]) -> pd.DataFrame:
    bv_csv, th_csv = embed_paths(model)
    if not bv_csv.is_file():
        raise FileNotFoundError(bv_csv)
    if not th_csv.is_file():
        raise FileNotFoundError(th_csv)

    shared, bv_X, th_X = aligned_embeddings(bv_csv, th_csv, ordered_categories)
    semantics = [semantic_map.get(c, 'other') for c in shared]

    bv_rdm = compute_rdm(bv_X)
    th_rdm = compute_rdm(th_X)
    bv_cat = category_within_between_detailed(shared, semantics, bv_rdm)
    th_cat = category_within_between_detailed(shared, semantics, th_rdm)
    bv_cluster = cluster_within_between(shared, semantics, bv_rdm)
    th_cluster = cluster_within_between(shared, semantics, th_rdm)
    merged = cluster_merged_bv_vs_things(bv_cluster, th_cluster)

    out_results = RESULTS_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}'
    out_figures = FIGURES_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}'
    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    bv_cat.to_csv(out_results / f'bv_category_within_between_{model}_{CATEGORY_SET}.csv', index=False)
    th_cat.to_csv(out_results / f'things_category_within_between_{model}_{CATEGORY_SET}.csv', index=False)
    bv_cluster.to_csv(out_results / f'bv_cluster_within_between_{model}_{CATEGORY_SET}.csv', index=False)
    th_cluster.to_csv(out_results / f'things_cluster_within_between_{model}_{CATEGORY_SET}.csv', index=False)
    merged.to_csv(out_results / f'pooled_bv_vs_things_cluster_merged_{model}_{CATEGORY_SET}.csv', index=False)

    merged['delta_diff_bv_minus_things'] = (
        merged['delta_between_minus_within_bv'] - merged['delta_between_minus_within_things']
    )
    merged.to_csv(out_results / f'pooled_bv_vs_things_cluster_by_domain_{model}_{CATEGORY_SET}.csv', index=False)

    cat_merged = bv_cat.merge(
        th_cat.rename(columns={c: f'{c}_things' for c in th_cat.columns if c not in {'category', 'cdi_semantic'}}),
        on=['category', 'cdi_semantic'],
        how='inner',
    )
    cat_merged['delta_diff_bv_minus_things'] = (
        cat_merged['delta_between_minus_within'] - cat_merged['delta_between_minus_within_things']
    )
    cat_merged.to_csv(out_results / f'pooled_bv_vs_things_category_merged_{model}_{CATEGORY_SET}.csv', index=False)

    corr_rows = []
    for metric, bv_col, th_col in [
        ('within_mean', 'within_mean', 'within_mean_things'),
        ('between_mean', 'between_mean', 'between_mean_things'),
        ('delta_between_minus_within', 'delta_between_minus_within', 'delta_between_minus_within_things'),
    ]:
        c = safe_corr(cat_merged[bv_col].to_numpy(), cat_merged[th_col].to_numpy())
        c['metric'] = metric
        c['model'] = model
        c['n_categories'] = len(cat_merged)
        corr_rows.append(c)
    pd.DataFrame(corr_rows).to_csv(
        out_results / f'pooled_bv_vs_things_category_corr_{model}_{CATEGORY_SET}.csv',
        index=False,
    )

    global_row = {
        'model': model,
        'category_set': CATEGORY_SET,
        'n_categories': len(shared),
        'bv_embedding_path': str(bv_csv),
        'things_embedding_path': str(th_csv),
        **global_cluster_strength(bv_cluster, 'bv'),
        **global_cluster_strength(th_cluster, 'things'),
    }
    global_row['delta_diff_bv_minus_things'] = (
        global_row['bv_delta_between_minus_within'] - global_row['things_delta_between_minus_within']
    )
    pd.DataFrame([global_row]).to_csv(
        out_results / f'pooled_bv_vs_things_cluster_strength_global_{model}_{CATEGORY_SET}.csv',
        index=False,
    )

    plot_cdi_domain_bars(
        merged,
        model,
        out_figures / f'pooled_bv_vs_things_cdi_domain_bars_{model}_{CATEGORY_SET}',
    )
    shuffle_meta = compute_shuffle_meta(model, shared, semantics, bv_rdm, th_rdm, merged)
    plot_cdi_domain_delta_vertical(
        shuffle_meta,
        model,
        out_figures / f'pooled_bv_vs_things_cdi_domain_delta_vertical_{model}_{CATEGORY_SET}',
    )

    g = global_row
    print(
        f'[{model}] n={len(shared)} · BV Δ={g["bv_delta_between_minus_within"]:.4f} · '
        f'THINGS Δ={g["things_delta_between_minus_within"]:.4f} · '
        f'BV−THINGS={g["delta_diff_bv_minus_things"]:.4f}'
    )
    print(f'  figure: {out_figures / f"pooled_bv_vs_things_cdi_domain_bars_{model}_{CATEGORY_SET}.png"}')
    print(
        f'  vertical Δ: {out_figures / f"pooled_bv_vs_things_cdi_domain_delta_vertical_{model}_{CATEGORY_SET}.png"}'
    )
    return merged, shuffle_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Pooled BV vs THINGS CDI-domain bars (valid85).')
    p.add_argument('--models', nargs='+', default=None, help='clip dinov3 (default: env BV_EMBED_MODEL or clip)')
    p.add_argument(
        '--plots-only',
        action='store_true',
        help='Regenerate figures from existing pooled_bv_vs_things_cluster_merged_* CSVs.',
    )
    p.add_argument(
        '--recompute-shuffle',
        action='store_true',
        help='Re-run label-shuffle nulls even if cached npz/csv exist.',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models
    if not models:
        env = os.environ.get('BV_EMBED_MODEL', 'clip').strip().lower()
        models = [env] if env else ['clip']
    models = [m.strip().lower() for m in models]

    if not ORDER_CSV.is_file():
        raise FileNotFoundError(f'Missing order CSV: {ORDER_CSV}')
    ordered_categories = load_order(ORDER_CSV)
    semantic_map = load_cdi_semantic_map(CDI_SEMANTIC_CSV)

    sns.set_context('talk')
    sns.set_style('whitegrid')

    shuffle_by_model: dict[str, dict] = {}

    for model in models:
        if args.plots_only:
            merged = load_merged_from_results(model)
            shuffle_meta = ensure_shuffle_meta(
                model,
                merged,
                ordered_categories,
                semantic_map,
                recompute=args.recompute_shuffle,
            )
            out_figures = FIGURES_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}'
            out_figures.mkdir(parents=True, exist_ok=True)
            plot_cdi_domain_bars(
                merged,
                model,
                out_figures / f'pooled_bv_vs_things_cdi_domain_bars_{model}_{CATEGORY_SET}',
            )
            plot_cdi_domain_delta_vertical(
                shuffle_meta,
                model,
                out_figures / f'pooled_bv_vs_things_cdi_domain_delta_vertical_{model}_{CATEGORY_SET}',
            )
            print(f'[{model}] replotted (shuffle n_perm={shuffle_meta["n_perm"]})')
        else:
            _, shuffle_meta = run_model(model, ordered_categories, semantic_map)
            if args.recompute_shuffle:
                merged = load_merged_from_results(model)
                shuffle_meta = ensure_shuffle_meta(
                    model, merged, ordered_categories, semantic_map, recompute=True
                )
                out_figures = FIGURES_DIR / f'pooled_bv_vs_things_within_between_{model}_{CATEGORY_SET}'
                plot_cdi_domain_delta_vertical(
                    shuffle_meta,
                    model,
                    out_figures / f'pooled_bv_vs_things_cdi_domain_delta_vertical_{model}_{CATEGORY_SET}',
                )
        shuffle_by_model[model] = shuffle_meta

    if len(shuffle_by_model) >= 2 and {'clip', 'dinov3'}.issubset(shuffle_by_model):
        combo_dir = FIGURES_DIR / f'pooled_bv_vs_things_within_between_{CATEGORY_SET}'
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_path = combo_dir / f'pooled_bv_vs_things_cdi_domain_delta_vertical_2panel_{CATEGORY_SET}'
        plot_cdi_domain_delta_vertical_2panel(
            {k: shuffle_by_model[k] for k in ('clip', 'dinov3')},
            combo_path,
        )
        print(f'  2-panel vertical Δ: {combo_path}.png')


if __name__ == '__main__':
    main()
