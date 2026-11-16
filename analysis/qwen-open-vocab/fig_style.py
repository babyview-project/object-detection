"""Manuscript figure style for the revision figures: Lato, bold axis labels and legends, one colour per dataset and
the manuscript's CDI domain colours (shared with analysis/talk-figures/unr-2026/talk_style.py).

  import fig_style as FS; FS.apply()          # before plotting
  FS.DS_COLOR["bv"], FS.DOMAIN_COLOR["animals"], FS.label("vedb")
"""
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.axis
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONTS = Path(__file__).resolve().parents[1] / "talk-figures" / "unr-2026" / "fonts"

INK, MUTED, FAINT, BASELINE = "#222222", "#6a6a6a", "#d9d9d9", "#9e9e9e"

# Dataset colours (2026-11-15, user chose "slate"). First-person datasets saturated (BabyView blue, VEDB rust); curated datasets
# in three cool slate neutrals (THINGS darkest). Domains own the light, medium-chroma hues, so no dataset colour is close to a domain
# colour (OKLab ΔE ≥ 15 for every dataset-domain pair under normal vision; checked against the CDI and THINGS-27
# palettes). "Real objects only" versions keep their dataset's colour and are marked by hatching or open markers.
DS_COLOR = {
    "bv": "#1F5CA8", "bv_real": "#1F5CA8",
    "vedb": "#C2511F", "vedb_real": "#C2511F",
    "things": "#29343F", "imagenet": "#4E5966", "lvis": "#75828F",
}
DS_LABEL = {"bv": "BabyView", "bv_real": "BabyView (real)", "vedb": "VEDB", "vedb_real": "VEDB (real)",
            "things": "THINGS", "imagenet": "ImageNet", "lvis": "LVIS"}

DOMAIN_ORDER = ["animals", "body_parts", "clothing", "food_drink", "furniture_rooms", "household", "outside",
                "people", "toys", "vehicles", "other"]
DOMAIN_COLOR = {
    "animals": "#4DB8A8", "body_parts": "#E87A5F", "clothing": "#9B7EC8", "food_drink": "#E8A54C",
    "furniture_rooms": "#6BAB7A", "household": "#D97B9E", "outside": "#5B9BD5", "people": "#E8C44C",
    "toys": "#B07CC8", "vehicles": "#6BA3D5", "other": "#D8C7A0",
}
# Visual format of a box: real objects in neutral grey, depictions in hues not used for datasets.
FMT_COLOR = {"real": "#cfcfcf", "toy": "#7E57C2", "picture": "#26A69A", "drawing": "#26A69A", "photo": "#80CBC4",
             "screen": "#F9A825"}

DOMAIN_LABEL = {"animals": "animals", "body_parts": "body parts", "clothing": "clothing", "food_drink": "food & drink",
                "furniture_rooms": "furniture & rooms", "household": "household", "outside": "outside",
                "people": "people", "toys": "toys", "vehicles": "vehicles", "other": "other"}

class _Fallback(dict):
    """Domain lookups for unknown or singleton domains ("other:<category>") fall back to "other"."""
    def __missing__(self, k):
        return dict.__getitem__(self, "other")


import os as _os
if _os.environ.get("QOV_DOMAINS", "cdi") == "things27":   # THINGS high-level categories with >= 5 core categories
    DOMAIN_ORDER = ["animal", "food", "clothing", "furniture", "home decor", "container", "tool", "electronic device",
                    "office supply", "vehicle", "people", "other"]
    # "sports equipment" dropped 2026-11-14: 4 core categories after the 183 -> 182 core change (VEDB adults only)
    DOMAIN_COLOR = {"animal": "#4DB8A8", "food": "#E8A54C", "clothing": "#9B7EC8", "furniture": "#6BAB7A",
                    "home decor": "#D97B9E", "container": "#C9A227", "tool": "#A4C639", "electronic device": "#E87A5F",
                    "office supply": "#80A4C8", "sports equipment": "#B07CC8", "vehicle": "#6BA3D5",
                    "people": "#E8C44C", "other": "#D8C7A0"}
    DOMAIN_LABEL = {d: d for d in DOMAIN_ORDER}
DOMAIN_COLOR = _Fallback(DOMAIN_COLOR)
DOMAIN_LABEL = _Fallback(DOMAIN_LABEL)
DOMAIN_LABEL.update({"other": "other"})


_patched = False


def _bold_legend(orig):
    def legend(self, *args, **kw):
        prop = dict(kw.pop("prop", {}) or {})
        prop.setdefault("weight", "bold")
        prop.setdefault("size", kw.pop("fontsize", plt.rcParams["legend.fontsize"]))
        return orig(self, *args, prop=prop, **kw)
    return legend


def apply(base=10):
    """Lato, bold axis labels, tick labels and legend text, no grid lines, light spines; call before plotting."""
    global _patched
    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Lato", "font.size": base, "axes.titlesize": base + 1, "axes.titleweight": "bold",
        "axes.labelsize": base + 1, "axes.labelweight": "bold", "figure.titlesize": base + 2,
        "figure.titleweight": "bold", "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300, "figure.facecolor": "white",
    })
    if not _patched:
        matplotlib.axes.Axes.legend = _bold_legend(matplotlib.axes.Axes.legend)
        _tick_init = matplotlib.axis.Tick.__init__
        def tick_init(self, *a, **k):                    # bold tick labels (no rcParam for tick label weight)
            _tick_init(self, *a, **k); self.label1.set_fontweight("bold"); self.label2.set_fontweight("bold")
        matplotlib.axis.Tick.__init__ = tick_init
        _grid = matplotlib.axes.Axes.grid                # no grid lines in any figure, even where a script asks for them
        matplotlib.axes.Axes.grid = lambda self, visible=None, which="major", axis="both", **kw: _grid(self, False, which=which, axis=axis)
        matplotlib.figure.Figure.legend = _bold_legend(matplotlib.figure.Figure.legend)
        _patched = True


def label(ds):
    return DS_LABEL.get(ds, ds)


def bh(p):
    """Benjamini-Hochberg q-values (NaN-free input)."""
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def ptext(p, name="p"):
    """'p = .034' / 'p < .001' (two decimals above .1, three below)."""
    if p is None or p != p:
        return ""
    if p < .001:
        return f"{name} < .001"
    return f"{name} = " + (f"{p:.2f}" if p >= .1 else f"{p:.3f}").lstrip("0")


def stars(p, ns="n.s."):
    """Significance marker: *** p < .001, ** p < .01, * p < .05, else `ns` (pass ns="" to hide)."""
    if p is None or p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ns


def bracket(ax, x1, x2, y, text, h=None, fontsize=9):
    """Comparison bracket from x1 to x2 at height y (data units) with a star label on top."""
    if not text:
        return
    h = h if h is not None else .012 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=INK, lw=.9, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * .3, text, ha="center", va="bottom", fontsize=fontsize, color=INK,
            fontweight="normal" if text == "n.s." else "bold")


def color_domain_ticks(ticklabels, domains):
    """Colour tick labels by CDI domain (manuscript colours)."""
    for t, d in zip(ticklabels, domains):
        t.set_color(DOMAIN_COLOR.get(d, DOMAIN_COLOR["other"]))


def units(ax, x, v, color, width=0.28, s=10, alpha=None, seed=0, horizontal=False, zorder=1):
    """Every unit (child, session, category) as a jittered dot around position x. alpha defaults by count."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    j = x + np.random.default_rng(seed).uniform(-width / 2, width / 2, len(v))
    a = alpha if alpha is not None else (.55 if len(v) <= 60 else .25 if len(v) <= 300 else .12)
    xy = (v, j) if horizontal else (j, v)
    ax.scatter(*xy, s=s, color=color, alpha=a, lw=0, zorder=zorder)


def mean_se(ax, x, m, se, color=INK, horizontal=False, width=0.34, zorder=4):
    """Mean as a short thick line, ± 1 SE as a whisker."""
    if horizontal:
        ax.plot([m, m], [x - width / 2, x + width / 2], color=color, lw=2.6, zorder=zorder, solid_capstyle="butt")
        ax.errorbar(m, x, xerr=se, fmt="none", ecolor=color, elinewidth=1.6, capsize=3, zorder=zorder)
    else:
        ax.plot([x - width / 2, x + width / 2], [m, m], color=color, lw=2.6, zorder=zorder, solid_capstyle="butt")
        ax.errorbar(x, m, yerr=se, fmt="none", ecolor=color, elinewidth=1.6, capsize=3, zorder=zorder)


import matplotlib.text as _mtext


def _text_rect(t, r):
    """Oriented rectangle (4 corners, window px) of a text label: unrotated size, rotated about its centre."""
    bb = _mtext.Text.get_window_extent(t, r); rot = t.get_rotation() % 180   # text only (annotations: no arrow)
    if rot in (0, 90):
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    t.set_rotation(0); b0 = _mtext.Text.get_window_extent(t, r); t.set_rotation(rot)
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]); w, h = b0.width / 2, b0.height / 2
    a = np.radians(rot); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return c + np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ R.T


def _rects_overlap(A, B, pad):
    """Separating-axis test for two convex quadrilaterals, shrunk by `pad` px."""
    for P in (A, B):
        for k in range(4):
            e = P[(k + 1) % 4] - P[k]; n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n) or 1
            pa, pb = A @ n, B @ n
            if pa.max() - pad <= pb.min() or pb.max() - pad <= pa.min():
                return False
    return True


def text_overlaps(fig, name="", pad=1.0, log=True):
    """Pairs of visible text labels (titles, axis and tick labels, annotations, legend text) whose drawn outlines overlap
    by more than `pad` pixels (rotated labels tested on their rotated outline). Printed as 'OVERLAP <figure>: "a" x "b"'."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()                                    # tick labels outside the view limits are not drawn
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 * abs(hi - lo) <= tk.get_loc() <= hi + 1e-9 * abs(hi - lo)):
                    hidden.update({id(tk.label1), id(tk.label2)})
    items = []
    for t in fig.findobj(mtext.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or t.get_alpha() == 0 or id(t) in hidden:
            continue
        try:
            bb = mtext.Text.get_window_extent(t, r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((s.replace("\n", " ")[:40], bb, _text_rect(t, r)))
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) <= pad or min(a.y1, b.y1) - max(a.y0, b.y0) <= pad:
                continue
            if _rects_overlap(items[i][2], items[j][2], pad):
                out.append((items[i][0], items[j][0]))
    if log:
        for a, b in out:
            print(f'OVERLAP {name}: "{a}" x "{b}"')
    return out


def wrap_label(s, width=9):
    """Two-line tick label for long domain names: 'food & drink' -> 'food\n& drink', 'electronic device' -> 'electronic\ndevice'."""
    s = DOMAIN_LABEL.get(s, s) if isinstance(DOMAIN_LABEL, dict) else s
    if " & " in s:
        return s.replace(" & ", "\n& ")
    if len(s) > width and " " in s:
        return s.replace(" ", "\n", 1)
    return s
