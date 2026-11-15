"""Manuscript figure style for the revision figures: Lato, bold axis labels and legends, one colour per dataset and
the manuscript's CDI domain colours (shared with analysis/talk-figures/unr-2026/talk_style.py).

  import fig_style as FS; FS.apply()          # before plotting
  FS.DS_COLOR["bv"], FS.DOMAIN_COLOR["animals"], FS.label("vedb")
"""
from pathlib import Path

import matplotlib
import matplotlib.axis
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONTS = Path(__file__).resolve().parents[1] / "talk-figures" / "unr-2026" / "fonts"

INK, MUTED, FAINT, BASELINE = "#222222", "#6a6a6a", "#d9d9d9", "#9e9e9e"

# Egocentric datasets saturated (BabyView navy, VEDB orange), curated datasets in lighter tints; real-object versions
# of BabyView and VEDB are lighter tints of their dataset's colour.
DS_COLOR = {
    "bv": "#0D47A1", "bv_real": "#6F9BD8",
    "vedb": "#E07B39", "vedb_real": "#F2B48A",
    "things": "#90CAF9", "imagenet": "#B9A6E0", "lvis": "#4DB38A",
}
DS_LABEL = {"bv": "BabyView", "bv_real": "BabyView (real)", "vedb": "VEDB", "vedb_real": "VEDB (real)",
            "things": "THINGS", "imagenet": "ImageNet", "lvis": "LVIS"}

DOMAIN_ORDER = ["animals", "body_parts", "clothing", "food_drink", "furniture_rooms", "household", "outside",
                "people", "toys", "vehicles", "other"]
DOMAIN_COLOR = {
    "animals": "#4DB8A8", "body_parts": "#E87A5F", "clothing": "#9B7EC8", "food_drink": "#E8A54C",
    "furniture_rooms": "#6BAB7A", "household": "#D97B9E", "outside": "#5B9BD5", "people": "#E8C44C",
    "toys": "#B07CC8", "vehicles": "#6BA3D5", "other": "#8B9A9E",
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
                    "home decor": "#D97B9E", "container": "#C9A227", "tool": "#8D6E63", "electronic device": "#E87A5F",
                    "office supply": "#80A4C8", "sports equipment": "#B07CC8", "vehicle": "#3F7FC0",
                    "people": "#E8C44C", "other": "#8B9A9E"}
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
