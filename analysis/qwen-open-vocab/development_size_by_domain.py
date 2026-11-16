#!/usr/bin/env python3
"""Object size and age, by CDI domain: where the within-child fall in object size comes from (manuscript: "The layout
of the child's view changes with age"; Fig. 8; Supplementary Figure S14).

The manuscript reports that the median object in the child's view shrinks with age, mostly because of *which*
objects fill the view rather than the same objects appearing smaller. This script asks how that splits across the CDI
domains. Same data and unit as development_within_child.py: core-category boxes (183 categories, all formats or real
objects only), one row per child x month of age, child-months with >= MIN_BOXES boxes; age effects are within-child
(child-demeaned OLS, SEs clustered by child; the mixed model of the manuscript gives the same signs).

Per domain d, with s_d = share of the child-month's boxes in d and m_d = mean log box area of those boxes:
  size level        median box area (share of frame) of the domain's boxes, pooled
  share_slope       change in s_d per year (percentage points)
  size_slope        change in m_d per year (% box area): the domain's boxes shrink or grow
    within_cat      ... of which the same categories shrink (child x category demeaned, pooled over the domain)
    cat_mix         ... of which the domain's category mix shifts (size_slope - within_cat)
Decomposition of the change in overall mean log area L = sum_d s_d m_d per year:
  size term   sum_d  mean(s_d) * slope(m_d)                     (objects of a domain get smaller)
  mix term    sum_d (mean(m_d) - mean(L)) * slope(s_d)           (the view fills with smaller or larger domains)
Both terms are reported per domain; their sum approximates slope(L) (the remainder is the s x m interaction).

  QOV_LABELS=renamed python development_size_by_domain.py
Outputs: results/bv_vedb_renamed/dev_size_by_domain.csv, dev_size_decomposition.csv,
         figures/bv_vedb_renamed/dev_size_by_domain.{png,pdf}
"""
import os
from pathlib import Path

os.environ.setdefault("QOV_LABELS", "renamed")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import fig_style as FS  # noqa: E402
import variant as V  # noqa: E402

HERE = Path(__file__).resolve().parent
D = V.out_dir(HERE / "out" / "private" / "dev_within_child")
DOM = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
RES = V.out_dir(HERE / "results" / "bv_vedb"); FIG = V.out_dir(HERE / "figures" / "bv_vedb")
MIN_BOXES = 200          # child-months with fewer core boxes are dropped (as in development_within_child.py)
MIN_DOM = 10             # a domain's size in a child-month needs >= this many boxes
DOMAINS = ["furniture_rooms", "household", "clothing", "food_drink", "toys", "animals", "vehicles",  # no body parts in core
           "outside"]


ALL_DOMAINS = ["room_surfaces", "furniture_rooms", "household", "clothing", "body_parts", "people", "food_drink", "toys",
               "animals", "vehicles", "outside", "other", "generic_labels", "unlisted"]
# Boxes whose name is outside the 828-category vocabulary (20% of boxes; 93% are the names below) are split into
# people, room surfaces (very large boxes) and generic labels Qwen sometimes gives instead of a basic-level name.
SURFACES = {"wall", "floor", "ceiling", "carpet", "sky", "ground", "road", "staircase", "doorway"}
GENERIC = {"toy", "clothing", "animal", "food", "furniture", "character", "object", "figure", "shape",
           "electronic device"}
EXTRA_LABEL = {"room_surfaces": "room surfaces", "generic_labels": "generic labels", "unlisted": "other unlisted"}
EXTRA_COLOR = {"room_surfaces": "#A1887F", "generic_labels": "#BDBDBD", "unlisted": "#E0E0E0"}


def split_unlisted(b):
    d = b.domain.astype(str).copy(); c = b.category.astype(str); u = d == "not_in_vocabulary"
    d[u] = "unlisted"
    d[u & (c == "person")] = "people"; d[u & c.isin(SURFACES)] = "room_surfaces"; d[u & c.isin(GENERIC)] = "generic_labels"
    return b.assign(domain=d)
AGG = V.out_dir(HERE / "out" / "private" / "dev_size_by_domain")


def all_boxes():
    """Every BabyView box (open vocabulary, renamed labels): unit, month, age, category, domain, format, area.
    Categories outside the 828-category vocabulary are kept as domain "not_in_vocabulary"; cached as parquet."""
    f = AGG / "boxes_all_v2.parquet"            # v2: adds the recording day (date in the video id)
    if f.exists():
        return pd.read_parquet(f)
    import build_bv_vedb_data as B
    from build_open_means import basic_of
    ix = B.load_index("bv")
    ix["category"] = ix.label.map(basic_of(ix.label.unique())).fillna("")
    vid = ix.video_id.astype(str)
    ix["unit"] = vid.map(B.unit_of); ix["age_months"] = vid.map(B.age_table(ix.video_id.cat.categories))
    W, H = B.FRAME_WH["bv"]
    ix["area"] = ((ix.xmax - ix.xmin) * (ix.ymax - ix.ymin) / (W * H)).astype("float32")
    ix["domain"] = ix.category.map(DOM).fillna("not_in_vocabulary")
    ix = ix[ix.age_months.notna()]
    ix["day"] = vid.str.split("_").str[1]
    out = ix[["unit", "day", "age_months", "category", "domain", "format", "area"]].copy()
    out["month"] = np.floor(out.age_months).astype(int)
    for c in ("unit", "day", "category", "domain", "format"):
        out[c] = out[c].astype("category")
    AGG.mkdir(parents=True, exist_ok=True); out.to_parquet(f)
    return out


def fe_slope(df, y, unit="unit", x="age"):
    """Within-unit OLS slope of y on x (per year), SE clustered by child; returns (b, lo, hi, p)."""
    d = df.dropna(subset=[y])
    d = d[d.groupby(unit)[unit].transform("size") >= 2]
    xx = (d[x] - d.groupby(unit)[x].transform("mean")) / 12
    yy = d[y] - d.groupby(unit)[y].transform("mean")
    b = (xx * yy).sum() / (xx * xx).sum(); e = yy - b * xx
    g = (xx * e).groupby(d["unit"]).sum(); G = len(g)
    se = np.sqrt((g ** 2).sum() * G / (G - 1)) / (xx * xx).sum()
    return b, b - 1.96 * se, b + 1.96 * se, 2 * stats.t.sf(abs(b / se), G - 1)


def analyse(bx, fmt, domains=DOMAINS, boxset="core"):
    bx = bx[bx.area > 0].assign(la=lambda d: np.log(d.area))
    bx["dom"] = bx.domain.astype(str) if "domain" in bx else bx.category.map(DOM).fillna("other")
    bx["category"] = bx.category.astype(str); bx["unit"] = bx.unit.astype(str)
    bx = bx[bx.dom.isin(domains)]
    DOMAINS_ = domains
    cm = bx.groupby(["unit", "month"]).agg(n=("la", "size"), L=("la", "mean"), age=("age_months", "mean")).reset_index()
    cm = cm[cm.n >= MIN_BOXES]
    bx = bx.merge(cm[["unit", "month"]], on=["unit", "month"])
    dm = bx.groupby(["unit", "month", "dom"]).agg(k=("la", "size"), m=("la", "mean")).reset_index()
    dm = dm.merge(cm, on=["unit", "month"]); dm["s"] = dm.k / dm.n
    # every domain in every child-month (share 0 when absent); size only where the domain has enough boxes
    full = cm[["unit", "month"]].merge(pd.DataFrame({"dom": DOMAINS_}), how="cross")
    dm = full.merge(dm.drop(columns=["n", "L", "age"]), on=["unit", "month", "dom"], how="left").merge(cm, on=["unit", "month"])
    dm["s"] = dm.s.fillna(0); dm.loc[dm.k.fillna(0) < MIN_DOM, "m"] = np.nan

    bL = fe_slope(cm, "L")
    Lbar = cm.L.mean()
    rows = []
    for d in DOMAINS_:
        g = dm[dm.dom == d]
        bs = fe_slope(g, "s"); bm = fe_slope(g, "m")
        # within-category change, pooled over the domain's categories (child x category demeaned)
        c = bx[bx.dom == d].groupby(["unit", "month", "category"]).agg(la=("la", "median"), n=("la", "size"),
                                                                     age=("age_months", "mean")).reset_index()
        c = c[c.n >= 3]; c["cc"] = c.unit + "|" + c.category
        bw = fe_slope(c.assign(unit_cc=c.cc), "la", unit="cc") if len(c) else (np.nan,) * 4
        mbar = g.m.mean(); sbar = g.s.mean()
        rows.append(dict(boxes_set=boxset, format=fmt, domain=d, categories=bx[bx.dom == d].category.nunique(), boxes=int((bx.dom == d).sum()),
                         median_area=float(bx.loc[bx.dom == d, "area"].median()), share=sbar,
                         share_pp_per_year=bs[0] * 100, share_lo=bs[1] * 100, share_hi=bs[2] * 100, share_p=bs[3],
                         size_pct_per_year=np.expm1(bm[0]) * 100, size_lo=np.expm1(bm[1]) * 100,
                         size_hi=np.expm1(bm[2]) * 100, size_p=bm[3],
                         within_cat_pct_per_year=np.expm1(bw[0]) * 100, within_cat_lo=np.expm1(bw[1]) * 100,
                         within_cat_hi=np.expm1(bw[2]) * 100, within_cat_p=bw[3],
                         size_term=sbar * bm[0] * 100, mix_term=(mbar - Lbar) * bs[0] * 100))
    t = pd.DataFrame(rows)
    t["cat_mix_pct_per_year"] = t.size_pct_per_year - t.within_cat_pct_per_year
    q = stats.false_discovery_control(np.r_[t.share_p, t.size_p, t.within_cat_p].astype(float))
    n = len(t); t["share_q"], t["size_q"], t["within_cat_q"] = q[:n], q[n:2 * n], q[2 * n:]
    summ = dict(boxes_set=boxset, format=fmt, child_months=len(cm), children=cm.unit.nunique(),
                overall_pct_per_year=np.expm1(bL[0]) * 100, overall_lo=np.expm1(bL[1]) * 100,
                overall_hi=np.expm1(bL[2]) * 100, overall_p=bL[3],
                size_term_pct=t.size_term.sum(), mix_term_pct=t.mix_term.sum(),
                interaction_pct=bL[0] * 100 - t.size_term.sum() - t.mix_term.sum())
    return t, summ


def figure(t, s, domains):
    FS.apply(base=10)
    t = t[t.format == "all"].set_index("domain").loc[domains]
    lab = [FS.DOMAIN_LABEL.get(d, EXTRA_LABEL.get(d, d)) for d in domains]
    col = [FS.DOMAIN_COLOR.get(d, EXTRA_COLOR.get(d, "#B0B0B0")) for d in domains]
    y = np.arange(len(domains))[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True, gridspec_kw={"width_ratios": [0.8, 1, 1, 1.1]})
    a = ax[0]; a.barh(y, t.median_area * 100, color=col, height=0.65)
    for yi, v in zip(y, t.median_area * 100):
        a.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=8)
    a.set_yticks(y, lab); a.set_xlabel("median box area (% of frame)"); a.set_title("A  Size", loc="left")
    a.set_xlim(0, t.median_area.max() * 100 * 1.3)

    def ci(a, v, lo, hi, qv, xlabel, title):
        a.axvline(0, color=FS.BASELINE, lw=1)
        a.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=FS.MUTED, elinewidth=1.2)
        a.scatter(v, y, s=48, c=[c if q < .05 else "white" for c, q in zip(col, qv)], edgecolors=col, linewidths=1.6, zorder=3)
        a.set_xlabel(xlabel); a.set_title(title, loc="left")
    ci(ax[1], t.share_pp_per_year, t.share_lo, t.share_hi, t.share_q, "share of boxes, pp per year", "B  How often in view")
    ci(ax[2], t.within_cat_pct_per_year, t.within_cat_lo, t.within_cat_hi, t.within_cat_q,
       "box area of the same categories, % per year", "C  Same categories, smaller?")
    a = ax[3]; a.axvline(0, color=FS.BASELINE, lw=1)
    h = 0.36
    a.barh(y + h / 2, t.mix_term, height=h, color=FS.DS_COLOR["bv"], label="more or fewer of the domain")
    a.barh(y - h / 2, t.size_term, height=h, color="white", edgecolor=FS.DS_COLOR["bv"], hatch="///", lw=1, label="the domain's objects smaller")
    a.set_xlabel("contribution to change in size, % per year"); a.set_title("D  What drives the overall fall", loc="left")
    a.legend(loc="lower left", fontsize=8)
    r = s[s.format == "all"].iloc[0]
    fig.suptitle(f"Within-child change with age, by domain (all boxes, all formats; {int(r.children)} children, "
                 f"{int(r.child_months)} child-months). Overall: {r.overall_pct_per_year:+.1f}% per year; "
                 f"mix {r.mix_term_pct:+.1f}, size {r.size_term_pct:+.1f}. Filled: q < .05.",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"dev_size_by_domain.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 3-month age bins
BIN, MIN_CHILDREN, MIN_CHILDREN_LINES, N_BOOT = 3, 5, 9, 200
MIN_DAYS, MIN_DAY_BOXES, MIN_DOM_DAY = 3, 20, 5   # child-bin needs >= 3 recording days; a day >= 20 boxes
REFS = (12, 6)                 # reference bins: 12-14 mo (26 children, main), 6-8 mo (10 children)


def twoway(df, y, w="w"):
    """y ~ child + bin (weighted least squares). Returns (bin levels at the mean child, child intercepts centred so that
    a child's own trajectory is y - intercept + mean intercept)."""
    d = df.dropna(subset=[y])
    Ud = pd.get_dummies(d.unit); U = Ud.to_numpy(float); B_ = pd.get_dummies(d.bin)
    X = np.c_[U, B_.to_numpy(float)[:, 1:]]
    sw = np.sqrt(d[w].to_numpy(float))
    beta = np.linalg.lstsq(X * sw[:, None], d[y].to_numpy(float) * sw, rcond=None)[0]
    child = pd.Series(beta[:U.shape[1]], index=Ud.columns); eff = np.r_[0, beta[U.shape[1]:]]
    cbar = np.average(child, weights=d.groupby("unit")[w].sum().reindex(Ud.columns))
    return pd.Series(cbar + eff, index=B_.columns), child - cbar


def bin_tables(bx, domains):
    """Child x 3-month-bin tables, day-balanced. Boxes within a recording day are not independent (the same book or toy
    recurs across frames), so every measure is first computed per day and then averaged over the child-bin's days, and
    the child-bin is weighted by its number of days (w). cb: overall mean log area L; dm: per domain share s and mean log
    area m (m from days with >= MIN_DOM_DAY boxes of the domain, needs >= MIN_DAYS such days)."""
    bx = bx[bx.area > 0].assign(la=lambda d: np.log(d.area), dom=lambda d: d.domain.astype(str),
                                unit=lambda d: d.unit.astype(str), day=lambda d: d.day.astype(str))
    bx = bx[bx.dom.isin(domains)]
    bx["bin"] = (bx.month // BIN) * BIN
    keep = bx.groupby("bin").unit.nunique(); keep = keep[keep >= MIN_CHILDREN].index
    bx = bx[bx.bin.isin(keep)]
    dy = bx.groupby(["unit", "bin", "day"]).agg(n=("la", "size"), L=("la", "mean")).reset_index()
    dy = dy[dy.n >= MIN_DAY_BOXES]
    cb = dy.groupby(["unit", "bin"]).agg(L=("L", "mean"), w=("day", "size"), n=("n", "sum")).reset_index()
    cb = cb[cb.w >= MIN_DAYS]
    dy = dy.merge(cb[["unit", "bin"]], on=["unit", "bin"])
    dd = bx.groupby(["unit", "bin", "day", "dom"]).agg(k=("la", "size"), m=("la", "mean")).reset_index()
    full = dy[["unit", "bin", "day", "n"]].merge(pd.DataFrame({"dom": domains}), how="cross")
    dd = full.merge(dd, on=["unit", "bin", "day", "dom"], how="left"); dd["k"] = dd.k.fillna(0); dd["s"] = dd.k / dd.n
    dd.loc[dd.k < MIN_DOM_DAY, "m"] = np.nan
    dm = dd.groupby(["unit", "bin", "dom"]).agg(s=("s", "mean"), w=("day", "size"), m=("m", "mean"), wm=("m", "count"),
                                                k=("k", "sum")).reset_index()
    dm.loc[dm.wm < MIN_DAYS, "m"] = np.nan
    return cb, dm


def fit_levels(cb, dm, domains):
    out = {("overall", "L"): twoway(cb, "L")[0]}
    for d in domains:
        g = dm[dm.dom == d]
        out[(d, "s")] = twoway(g, "s")[0]; out[(d, "m")] = twoway(g.assign(w=g.wm), "m")[0]
    return pd.DataFrame(out)


def boot_levels(cb, dm, domains, est, tag):
    """Child-bootstrap draws of the bin levels (N_BOOT x bins x columns); cached per box set."""
    f = AGG / f"bins_boot_v2_{tag}.npy"
    if f.exists():
        Bt = np.load(f)
        if Bt.shape[1:] == est.shape:
            return Bt
    rng = np.random.default_rng(0); units = cb.unit.unique(); boots = []
    gc, gd = dict(tuple(cb.groupby("unit"))), dict(tuple(dm.groupby("unit")))
    for _ in range(N_BOOT):
        pick = rng.choice(units, len(units), replace=True)
        cbb = pd.concat([gc[u].assign(unit=f"{u}#{i}") for i, u in enumerate(pick)])
        dmb = pd.concat([gd[u].assign(unit=f"{u}#{i}") for i, u in enumerate(pick)])
        try:
            boots.append(fit_levels(cbb, dmb, domains).reindex(index=est.index, columns=est.columns).to_numpy())
        except Exception:
            continue
    Bt = np.stack(boots); AGG.mkdir(parents=True, exist_ok=True); np.save(f, Bt)
    return Bt


def relative(est, Bt, cb, ref):
    """Change from the reference bin, with CIs from the bootstrap distribution of that change: shares in percentage
    points, sizes in %. Also the mix / size decomposition of the overall change."""
    cols = list(est.columns); E = est.to_numpy(); r = list(est.index).index(ref)
    D = E - E[r]; DB = Bt - Bt[:, r:r + 1, :]
    lo, hi = np.nanpercentile(DB, 2.5, axis=0), np.nanpercentile(DB, 97.5, axis=0)
    long = []
    for c, (d, kind) in enumerate(cols):
        conv = (lambda v: v * 100) if kind == "s" else (lambda v: np.expm1(v) * 100)
        for k, b in enumerate(est.index):
            long.append(dict(domain=d, measure="share_pp" if kind == "s" else "area_pct", bin=b, ref=ref,
                             value=conv(D[k, c]), lo=conv(lo[k, c]), hi=conv(hi[k, c])))
    S_ = est.xs("s", axis=1, level=1); M_ = est.xs("m", axis=1, level=1); mbar, sbar = M_.mean(), S_.mean()
    iL = cols.index(("overall", "L")); rows = []
    for k, b in enumerate(est.index):
        mix = ((S_.loc[b] - S_.loc[ref]) * mbar).sum(); size = (sbar * (M_.loc[b] - M_.loc[ref])).sum(min_count=1)
        rows.append(dict(bin=b, ref=ref, children=int(cb[cb.bin == b].unit.nunique()),
                         overall_pct=np.expm1(D[k, iL]) * 100, overall_lo=np.expm1(lo[k, iL]) * 100,
                         overall_hi=np.expm1(hi[k, iL]) * 100, mix_pct=np.expm1(mix) * 100, size_pct=np.expm1(size) * 100))
    return pd.DataFrame(rows), pd.DataFrame(long)


def child_lines(cb, dm, est, ref, panels):
    """Each child's own trajectory on the group scale: value - child intercept (from the two-way fit), as change from
    the group level at the reference bin."""
    out = []
    for d, kind in panels:
        g = cb.rename(columns={"L": "y"}) if d == "overall" else dm[dm.dom == d].rename(columns={kind: "y"})
        if kind == "m":
            g = g.assign(w=g.wm)
        g = g.dropna(subset=["y"])
        _, ci = twoway(g, "y")
        v = g.y - g.unit.map(ci) - est.loc[ref, (d, kind)]
        v = v * 100 if kind == "s" else np.expm1(v) * 100
        out.append(pd.DataFrame({"domain": d, "measure": "share_pp" if kind == "s" else "area_pct", "unit": g.unit,
                                 "bin": g.bin, "value": v, "days": g.w}))
    return pd.concat(out, ignore_index=True)


def _lab(d): return "overall" if d == "overall" else FS.DOMAIN_LABEL.get(d, EXTRA_LABEL.get(d, d))
def _col(d): return FS.DS_COLOR["bv"] if d == "overall" else FS.DOMAIN_COLOR.get(d, EXTRA_COLOR.get(d, "#B0B0B0"))


def bin_figure(dec, lg, fmt_label, ref):
    FS.apply(base=10)
    rt = f"{ref}–{ref + BIN - 1} mo"
    xs = dec.bin.to_numpy(); xt = [f"{b}–{b + BIN - 1}" for b in xs]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    a = ax[0]
    a.axhline(0, color=FS.BASELINE, lw=1); a.axvline(ref, color=FS.FAINT, lw=6, zorder=0)
    a.fill_between(xs, dec.overall_lo, dec.overall_hi, color=FS.DS_COLOR["bv"], alpha=.15, lw=0)
    a.plot(xs, dec.overall_pct, "-o", color=FS.DS_COLOR["bv"], lw=2, ms=5, label="observed")
    a.plot(xs, dec.mix_pct, "--", color=FS.INK, lw=1.6, label="mix of domains only")
    a.plot(xs, dec.size_pct, ":", color=FS.MUTED, lw=1.8, label="size within domains only")
    for x_, n_ in zip(xs, dec.children):
        a.annotate(f"n={n_}", (x_, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -10),
                   textcoords="offset points", ha="center", fontsize=7, color=FS.MUTED)
    a.set_xticks(xs, xt, rotation=45, ha="right"); a.set_xlabel("age (months)")
    a.set_ylabel(f"mean box area vs {rt} (%)"); a.legend(loc="lower left", fontsize=8)
    a.set_title("A  Object size and its two sources", loc="left", pad=14)
    xb = dec.loc[dec.children >= MIN_CHILDREN_LINES, "bin"].to_numpy(); xbt = [f"{b}–{b + BIN - 1}" for b in xb]
    show_s = ["household", "room_surfaces", "clothing", "furniture_rooms"]
    show_m = ["household", "furniture_rooms", "clothing", "toys", "animals", "outside", "people"]
    for a, kind, show, ylab, ttl in ((ax[1], "share_pp", show_s, f"share of boxes vs {rt} (pp)", "B  How often in view"),
                                     (ax[2], "area_pct", show_m, f"box area vs {rt} (%)", "C  How large in view")):
        a.axhline(0, color=FS.BASELINE, lw=1); a.axvline(ref, color=FS.FAINT, lw=6, zorder=0)
        ends = []
        for d in show:
            g = lg[(lg.domain == d) & (lg.measure == kind)].set_index("bin").loc[xb]
            a.fill_between(xb, g.lo, g.hi, color=_col(d), alpha=.10, lw=0)
            a.plot(xb, g.value, "-o", color=_col(d), lw=2, ms=4)
            ends.append([g.value.iloc[-1], d])
        ends.sort(); lo_, hi_ = a.get_ylim(); gap = (hi_ - lo_) * 0.055
        for _ in range(200):                                # spread end labels apart, both directions
            moved = False
            for k in range(1, len(ends)):
                over = gap - (ends[k][0] - ends[k - 1][0])
                if over > 1e-9:
                    ends[k][0] += over / 2; ends[k - 1][0] -= over / 2; moved = True
            if not moved:
                break
        for yv, d in ends:
            a.text(xb[-1] + 0.5, yv, _lab(d), color=_col(d), va="center", fontsize=8.5, fontweight="bold")
        a.set_xticks(xb, xbt, rotation=45, ha="right"); a.set_xlabel("age (months)"); a.set_ylabel(ylab)
        a.set_xlim(xb[0] - 0.8, xb[-1] + 7); a.set_title(ttl, loc="left", pad=14)
    fig.suptitle(f"Object size by 3-month age bin, within children (day-balanced, weighted by recording days): change from "
                 f"{rt} (grey band), 95% child-bootstrap CIs of the change; {fmt_label}", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


CHILD_PANELS = [("overall", "L"), ("household", "s"), ("furniture_rooms", "s"), ("room_surfaces", "s"),
                ("clothing", "s"), ("toys", "m"), ("people", "m"), ("animals", "m"), ("outside", "m")]


def children_figure(cl, lg, dec, fmt_label, ref):
    """One thin line per child (its own trajectory, child level removed) behind the group curve."""
    FS.apply(base=10)
    rt = f"{ref}–{ref + BIN - 1} mo"; xs = dec.bin.to_numpy(); xt = [f"{b}–{b + BIN - 1}" for b in xs]
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 11), sharex=True); axs = axs.ravel()
    for a, (d, kind) in zip(axs, CHILD_PANELS):
        meas = "share_pp" if kind == "s" else "area_pct"
        c = cl[(cl.domain == d) & (cl.measure == meas)]
        for u, g in c.groupby("unit"):
            g = g.sort_values("bin")
            if len(g) >= 2:
                a.plot(g.bin, g.value, "-", color=FS.MUTED, lw=0.6, alpha=.35)
            a.scatter(g.bin, g.value, s=np.clip(g.days, 3, 60) * 1.2, color=FS.MUTED, alpha=.35, lw=0)
        if d == "overall":
            gg = dec.set_index("bin").rename(columns={"overall_pct": "value", "overall_lo": "lo", "overall_hi": "hi"})
        else:
            gg = lg[(lg.domain == d) & (lg.measure == meas)].set_index("bin")
        gg = gg.loc[xs]
        a.fill_between(xs, gg.lo, gg.hi, color=_col(d), alpha=.22, lw=0)
        a.plot(xs, gg.value, "-o", color=_col(d), lw=2.4, ms=4, zorder=3)
        a.axhline(0, color=FS.BASELINE, lw=1); a.axvline(ref, color=FS.FAINT, lw=6, zorder=0)
        q = np.nanpercentile(c.value, [2, 98]); pad = (q[1] - q[0]) * 0.1
        a.set_ylim(min(q[0], gg.lo.min()) - pad, max(q[1], gg.hi.max()) + pad)
        what = "mean box area, all objects" if d == "overall" else ("share of boxes" if kind == "s" else "box area")
        a.set_title(f"{_lab(d)}: {what}", loc="left", fontsize=10)
        a.set_ylabel(f"vs {rt} ({'pp' if kind == 's' else '%'})", fontsize=9)
    for a in axs[-3:]:
        a.set_xticks(xs, xt, rotation=45, ha="right"); a.set_xlabel("age (months)")
    n_multi = cl[cl.domain == "overall"].groupby("unit").bin.nunique().ge(2).sum()
    fig.suptitle(f"Each child's own trajectory (grey; child's level removed, {n_multi} children with >= 2 bins) and the "
                 f"group estimate (colour, 95% CI); dot area = recording days; change from {rt}; {fmt_label}", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    core = pd.read_parquet(D / "boxes.parquet")
    allb = split_unlisted(all_boxes())
    out, summ = [], []
    for boxset, bx, doms in (("all boxes", allb, ALL_DOMAINS), ("core", core, DOMAINS)):
        for fmt in ("all", "real"):
            t, s = analyse(bx if fmt == "all" else bx[bx.format == "real"], fmt, doms, boxset)
            out.append(t); summ.append(s)
    T, S = pd.concat(out, ignore_index=True), pd.DataFrame(summ)
    T.to_csv(RES / "dev_size_by_domain.csv", index=False); S.to_csv(RES / "dev_size_decomposition.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.round(2).to_string())
    print(T[["boxes_set", "format", "domain", "categories", "median_area", "share", "share_pp_per_year", "share_q",
             "size_pct_per_year", "size_q", "within_cat_pct_per_year", "within_cat_q", "cat_mix_pct_per_year",
             "size_term", "mix_term"]].round(3).to_string())
    figure(T[T.boxes_set == "all boxes"], S[S.boxes_set == "all boxes"], ALL_DOMAINS)
    decs, lgs, cls = [], [], []
    for fmt in ("all", "real"):
        cb, dm = bin_tables(allb if fmt == "all" else allb[allb.format == "real"], ALL_DOMAINS)
        est = fit_levels(cb, dm, ALL_DOMAINS); Bt = boot_levels(cb, dm, ALL_DOMAINS, est, fmt)
        flab = "all boxes, all formats" if fmt == "all" else "all boxes, real objects only"
        sfx = "" if fmt == "all" else "_real"
        for ref in REFS:
            dec, lg = relative(est, Bt, cb, ref)
            decs.append(dec.assign(format=fmt)); lgs.append(lg.assign(format=fmt))
            fig = bin_figure(dec, lg, flab, ref)
            for ext in ("png", "pdf"):
                fig.savefig(FIG / f"dev_size_by_domain_bins_ref{ref}{sfx}.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)
            cl = child_lines(cb, dm, est, ref, CHILD_PANELS)
            cls.append(cl.assign(format=fmt, ref=ref))
            fig = children_figure(cl, lg, dec, flab, ref)
            for ext in ("png", "pdf"):
                fig.savefig(FIG / f"dev_size_by_domain_bins_children_ref{ref}{sfx}.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)
    pd.concat(decs).to_csv(RES / "dev_size_bins_decomposition.csv", index=False)
    pd.concat(lgs).to_csv(RES / "dev_size_bins_by_domain.csv", index=False)
    pd.concat(cls).to_csv(V.out_dir(HERE / "out" / "private" / "dev_size_by_domain") / "child_lines.csv", index=False)  # per child: private
    print(pd.concat(decs).round(1).to_string())


if __name__ == "__main__":
    main()
