"""CDI-domain label-shuffle nulls and BV vs THINGS inference (notebook 05)."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def cluster_delta_table(categories: list[str], rdm: np.ndarray, semantics: list[str], cluster_within_between) -> pd.DataFrame:
    df = cluster_within_between(categories, semantics, rdm).set_index('cdi_semantic')
    df['delta_between_minus_within'] = df['between_mean'] - df['within_mean']
    return df[['n_categories', 'within_mean', 'between_mean', 'delta_between_minus_within']]


def marginal_shuffle_quantile_bands(mat: np.ndarray) -> np.ndarray:
    """Shape (n_domains, 3): [q0.025, q0.5, q0.975] per domain column in mat."""
    n_dom = mat.shape[1]
    q = np.full((n_dom, 3), np.nan, dtype=float)
    for j in range(n_dom):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            q[j, 0] = float(np.quantile(col, 0.025))
            q[j, 1] = float(np.quantile(col, 0.5))
            q[j, 2] = float(np.quantile(col, 0.975))
    return q


def parallel_shuffle_domain_delta_mats(
    bv_rdm: np.ndarray,
    th_rdm: np.ndarray,
    categories: list[str],
    semantics: list[str],
    domain_order: list[str],
    *,
    n_perm: int,
    rng: np.random.Generator,
    cluster_within_between,
) -> tuple[np.ndarray, np.ndarray]:
    sem_arr = np.asarray(semantics)
    n_dom = len(domain_order)
    mat_bv = np.full((n_perm, n_dom), np.nan, dtype=float)
    mat_th = np.full((n_perm, n_dom), np.nan, dtype=float)
    for i in range(n_perm):
        perm_sem = rng.permutation(sem_arr).tolist()
        tbl_bv = cluster_delta_table(categories, bv_rdm, perm_sem, cluster_within_between)
        tbl_th = cluster_delta_table(categories, th_rdm, perm_sem, cluster_within_between)
        for j, d in enumerate(domain_order):
            if d in tbl_bv.index:
                v = float(tbl_bv.loc[d, 'delta_between_minus_within'])
                if np.isfinite(v):
                    mat_bv[i, j] = v
            if d in tbl_th.index:
                v = float(tbl_th.loc[d, 'delta_between_minus_within'])
                if np.isfinite(v):
                    mat_th[i, j] = v
    return mat_bv, mat_th


def shuffle_vs_observed_per_domain_summary(
    null_mat: np.ndarray,
    obs_tbl: pd.DataFrame,
    domain_order: list[str],
    *,
    col_prefix: str,
) -> pd.DataFrame:

    def _perm_p_two_sided(obs: float, col: np.ndarray) -> float:
        valid = col[np.isfinite(col)]
        if valid.size == 0 or not np.isfinite(obs):
            return np.nan
        return float((np.sum(np.abs(valid) >= abs(obs)) + 1.0) / (valid.size + 1.0))

    rows = []
    for j, d in enumerate(domain_order):
        col = null_mat[:, j]
        valid = col[np.isfinite(col)]
        obs = (
            float(obs_tbl.loc[d, 'delta_between_minus_within'])
            if d in obs_tbl.index and np.isfinite(obs_tbl.loc[d, 'delta_between_minus_within'])
            else np.nan
        )
        nm = float(np.nanmean(col)) if valid.size else np.nan
        ns = float(np.nanstd(col, ddof=1)) if valid.size > 1 else np.nan
        z = (obs - nm) / ns if (np.isfinite(obs) and np.isfinite(ns) and ns > 0) else np.nan
        rows.append(
            {
                'cdi_semantic': d,
                f'{col_prefix}null_mean_delta': nm,
                f'{col_prefix}obs_minus_null_mean': obs - nm if np.isfinite(obs) and np.isfinite(nm) else np.nan,
                f'{col_prefix}null_std_delta': ns,
                f'{col_prefix}z_obs_minus_null_mean': z,
                f'{col_prefix}p_perm_two_sided': _perm_p_two_sided(obs, col),
                f'{col_prefix}n_perm_valid': int(valid.size),
            }
        )
    return pd.DataFrame(rows)


def q_to_sig_star(qv: float, alpha: float = 0.05) -> str:
    if not np.isfinite(qv):
        return ''
    if qv <= 0.001:
        return '***'
    if qv <= 0.01:
        return '**'
    if qv <= alpha:
        return '*'
    return ''


def run_bv_things_shuffle_inference(
    categories: list[str],
    semantics: list[str],
    bv_rdm: np.ndarray,
    th_rdm: np.ndarray,
    bar_order: list[str],
    merged: pd.DataFrame,
    *,
    n_perm: int,
    seed: int,
    cluster_within_between,
) -> dict:
    rng = np.random.default_rng(seed)
    null_mat_bv, null_mat_th = parallel_shuffle_domain_delta_mats(
        bv_rdm,
        th_rdm,
        categories,
        semantics,
        bar_order,
        n_perm=n_perm,
        rng=rng,
        cluster_within_between=cluster_within_between,
    )
    null_bv = marginal_shuffle_quantile_bands(null_mat_bv)
    null_th = marginal_shuffle_quantile_bands(null_mat_th)

    bv_tbl = cluster_delta_table(categories, bv_rdm, semantics, cluster_within_between)
    th_tbl = cluster_delta_table(categories, th_rdm, semantics, cluster_within_between)

    shuffle_bv = shuffle_vs_observed_per_domain_summary(null_mat_bv, bv_tbl, bar_order, col_prefix='bv_shuffle_')
    shuffle_th = shuffle_vs_observed_per_domain_summary(null_mat_th, th_tbl, bar_order, col_prefix='th_shuffle_')

    main_wide = merged.merge(shuffle_bv, on='cdi_semantic', how='left').merge(shuffle_th, on='cdi_semantic', how='left')
    main_wide['obs_delta_diff_bv_minus_th'] = (
        main_wide['delta_between_minus_within_bv'] - main_wide['delta_between_minus_within_things']
    )

    ndiff = null_mat_bv - null_mat_th
    mix = main_wide.set_index('cdi_semantic')
    p_gt: list[float] = []
    p_two: list[float] = []
    for j, dn in enumerate(bar_order):
        d_obs = float(mix.loc[dn, 'obs_delta_diff_bv_minus_th'])
        col = ndiff[:, j]
        vf = col[np.isfinite(col)]
        if vf.size:
            p_gt.append(float((1.0 + np.sum(vf >= d_obs)) / (vf.size + 1.0)))
            p_two.append(float((1.0 + np.sum(np.abs(vf) >= abs(d_obs))) / (vf.size + 1.0)))
        else:
            p_gt.append(np.nan)
            p_two.append(np.nan)

    main_wide = main_wide.merge(
        pd.DataFrame({'cdi_semantic': bar_order, 'bv_vs_th_shuffle_p_one_sided_gt': p_gt, 'bv_vs_th_shuffle_p_two_sided': p_two}),
        on='cdi_semantic',
        how='left',
    )

    pq = main_wide['bv_vs_th_shuffle_p_one_sided_gt'].to_numpy(dtype=float)
    mask = np.isfinite(pq)
    q_out = np.full(pq.shape, np.nan, dtype=float)
    if mask.any():
        pq_valid = np.clip(pq[mask], 0.0, 1.0)
        q_out[mask] = false_discovery_control(pq_valid, method='bh')
    main_wide['bv_vs_th_shuffle_q_one_sided_gt_fdr_bh'] = q_out

    sig_qvals = main_wide.set_index('cdi_semantic')['bv_vs_th_shuffle_q_one_sided_gt_fdr_bh'].to_dict()

    return {
        'main_wide': main_wide,
        'null_bv': null_bv,
        'null_th': null_th,
        'null_mat_bv': null_mat_bv,
        'null_mat_th': null_mat_th,
        'sig_qvals': sig_qvals,
        'bar_order': bar_order,
        'n_perm': n_perm,
    }


def draw_horizontal_paired_cluster_delta_bars(
    ax: Axes,
    bar_order: list[str],
    merged: pd.DataFrame,
    null_bv: np.ndarray,
    null_th: np.ndarray,
    *,
    cdi_semantic_colors: dict[str, str],
    title: str = '',
    show_ylabel: bool = True,
    show_legend: bool = False,
    x_limits: tuple[float, float] | None = None,
    sig_qvals: dict[str, float] | None = None,
    sig_alpha: float = 0.05,
    n_perm: int | None = None,
    bar_label_a: str = 'BabyView Δ',
    bar_label_b: str = 'THINGS Δ',
    null_whisk_label_a: str | None = None,
    sig_note: str = '* q<0.05, ** q<0.01, *** q<0.001 (BH-FDR; A>TH)',
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def _draw_null_whisker_h(yc, q_lo, q_hi, whisk_col, *, lw=1.55, cap_h, zz):
        if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
            return
        ax.plot([q_lo, q_hi], [yc, yc], color=whisk_col, linewidth=lw, solid_capstyle='round', zorder=zz)
        ax.plot([q_lo, q_lo], [yc - cap_h, yc + cap_h], color=whisk_col, linewidth=lw, zorder=zz)
        ax.plot([q_hi, q_hi], [yc - cap_h, yc + cap_h], color=whisk_col, linewidth=lw, zorder=zz)

    bv_col = '#0d47a1'
    th_col = '#90caf9'
    bv_whisk = '#37474f'
    th_whisk = '#78909c'
    ixm = merged.set_index('cdi_semantic')

    yy = np.arange(len(bar_order), dtype=float)
    sep = 0.16
    bv_yy = yy - sep
    th_yy = yy + sep
    h_bt = 0.32
    cap_h = 0.26 * h_bt
    ax.grid(False)

    def _lab(sn: str) -> str:
        return str(sn).replace('_', ' ').title()

    xs_all: list[float] = []
    med_lw = 1.05
    z_bar, z_whisk, z_med = 4, 8, 9

    for kk, sem in enumerate(bar_order):
        row = ixm.loc[sem]
        lo_bv, q50_bv, hi_bv = null_bv[kk, 0], null_bv[kk, 1], null_bv[kk, 2]
        lo_th, q50_th, hi_th = null_th[kk, 0], null_th[kk, 1], null_th[kk, 2]
        db_v = float(row['delta_between_minus_within_bv'])
        db_t = float(row['delta_between_minus_within_things'])

        xs_all.extend([v for v in (lo_bv, hi_bv, lo_th, hi_th, db_v, db_t) if np.isfinite(v)])
        if np.isfinite(q50_bv):
            xs_all.append(float(q50_bv))
        if np.isfinite(q50_th):
            xs_all.append(float(q50_th))

        ax.barh(bv_yy[kk], db_v, height=h_bt, color=bv_col, edgecolor='white', linewidth=0.85, zorder=z_bar)
        ax.barh(th_yy[kk], db_t, height=h_bt, color=th_col, edgecolor='white', linewidth=0.85, zorder=z_bar)

        _draw_null_whisker_h(bv_yy[kk], lo_bv, hi_bv, bv_whisk, cap_h=cap_h, lw=1.65, zz=z_whisk)
        _draw_null_whisker_h(th_yy[kk], lo_th, hi_th, th_whisk, cap_h=cap_h, lw=1.65, zz=z_whisk)

        if np.isfinite(q50_bv):
            ax.plot(
                [q50_bv, q50_bv],
                [bv_yy[kk] - cap_h * 1.08, bv_yy[kk] + cap_h * 1.08],
                color=bv_whisk,
                linewidth=med_lw + 0.2,
                linestyle=(0, (2.8, 2.2)),
                zorder=z_med,
            )
        if np.isfinite(q50_th):
            ax.plot(
                [q50_th, q50_th],
                [th_yy[kk] - cap_h * 1.08, th_yy[kk] + cap_h * 1.08],
                color=th_whisk,
                linewidth=med_lw + 0.2,
                linestyle=(0, (2.8, 2.2)),
                zorder=z_med,
            )

        if sig_qvals is not None:
            star = q_to_sig_star(float(sig_qvals.get(sem, np.nan)), alpha=sig_alpha)
            if star:
                x_star = max(db_v, db_t, hi_bv if np.isfinite(hi_bv) else db_v, hi_th if np.isfinite(hi_th) else db_t)
                pad_star = 0.025 * max(1e-6, np.nanmax(np.abs(np.asarray(xs_all, dtype=float))) if xs_all else 0.1)
                ax.text(x_star + pad_star, yy[kk], star, ha='left', va='center', fontsize=10.5, fontweight='bold', color='black', zorder=z_med + 1)

    ax.axvline(0, color='0.35', linewidth=0.85, zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([_lab(s) for s in bar_order], fontsize=10)
    for tic, sem in zip(ax.get_yticklabels(), bar_order):
        tic.set_color(cdi_semantic_colors.get(sem, cdi_semantic_colors.get('other', '#8B9A9E')))
    ax.invert_yaxis()

    ax.set_xlabel(r'$\overline{d}_{\mathrm{between}}-\overline{d}_{\mathrm{within}}$ (cosine)')
    if show_ylabel:
        ax.set_ylabel('CDI domain')
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    try:
        x_lo, x_hi = min(xs_all), max(xs_all)
    except ValueError:
        x_lo, x_hi = (-0.05, 0.05)
    pad_u = max(0.02, 0.1 * (x_hi - x_lo + 1e-9))
    ax.set_xlim(*(x_limits if x_limits is not None else (x_lo - pad_u, x_hi + pad_u)))
    ax.grid(axis='x', color='0.9', linewidth=0.7)
    ax.set_axisbelow(True)

    if show_legend:
        n_note = f'; $n_{{\\mathrm{{perm}}}}={n_perm}$' if n_perm else ''
        whisk_a = null_whisk_label_a or f'Null whisker A (2.5–97.5%{n_note}; dashed = median)'
        leg_handles = [
            Line2D([0], [0], color=bv_whisk, linewidth=2.2, label=whisk_a),
            Line2D([0], [0], color=th_whisk, linewidth=2.2, label='Null THINGS whisker'),
            Patch(facecolor=bv_col, edgecolor='white', linewidth=0.6, label=bar_label_a),
            Patch(facecolor=th_col, edgecolor='white', linewidth=0.6, label=bar_label_b),
        ]
        ax.legend(handles=leg_handles, frameon=False, fontsize=8.0, loc='lower right', ncol=1)
        if sig_note:
            ax.text(0.0, 1.02, sig_note, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.8, color='0.25')
