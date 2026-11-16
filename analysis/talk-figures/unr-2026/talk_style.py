"""Shared look for the UNR talk figures: Lato (as in the deck), the manuscript's CDI domain
colours, and one colour per dataset used on every slide."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1]
OUT = HERE / "figures"            # data figures (safe to commit)
PRIVATE = HERE / "figures" / "private"  # anything holding BabyView frames (git-ignored)

for f in (HERE / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

INK, MUTED, FAINT = "#222222", "#6a6a6a", "#d9d9d9"

# Same dataset colours as qwen-open-vocab/fig_style.py (2026-11-15): first-person datasets saturated and dark, curated
# datasets in slate neutrals, so no dataset colour is close to a domain colour.
DS_COLOR = {
    "things": "#29343F",
    "imagenet": "#4E5966",
    "lvis": "#75828F",
    "vedb": "#C2511F",
    "bv": "#1F5CA8",
}
DS_LABEL = {"things": "THINGS", "imagenet": "ImageNet-1k", "lvis": "LVIS",
            "vedb": "VEDB (adult)", "bv": "BabyView (child)"}

DOMAIN_ORDER = ["animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
                "household", "outside", "toys", "vehicles"]
DOMAIN_COLOR = {
    "animals": "#4DB8A8", "body_parts": "#E87A5F", "clothing": "#9B7EC8",
    "food_drink": "#E8A54C", "furniture_rooms": "#6BAB7A", "household": "#D97B9E",
    "outside": "#5B9BD5", "toys": "#B07CC8", "vehicles": "#6BA3D5", "other": "#D8C7A0",
}
DOMAIN_LABEL = {"animals": "animals", "body_parts": "body parts", "clothing": "clothing",
                "food_drink": "food & drink", "furniture_rooms": "furniture",
                "household": "household", "outside": "outside", "toys": "toys",
                "vehicles": "vehicles"}

# Content area of a 4:3 slide below the title bar.
SLIDE_W, SLIDE_H = 9.4, 5.6


def setup():
    plt.rcParams.update({
        "font.family": "Lato", "font.size": 14, "axes.titlesize": 15,
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "figure.facecolor": "white",
    })


TEXT_SCALE = 1.3          # every text element is enlarged by this factor at save time


def bolden(fig, scale=TEXT_SCALE):
    """Talk legibility: enlarge all text, bold tick labels, axis labels, titles and legends."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_fontsize(t.get_fontsize() * scale)
    for ax in fig.axes:
        for t in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title,
                                                                 ax._left_title, ax._right_title]:
            t.set_fontweight("bold")
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_fontweight("bold")
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_fontweight("bold")
    if not fig.get_constrained_layout():
        fig.tight_layout()


def save(fig, name, private=False, scale=TEXT_SCALE):
    bolden(fig, scale)
    try:                                                  # text-overlap check (qwen-open-vocab/fig_style.py)
        import fig_style
        fig_style.text_overlaps(fig, name)
    except ImportError:
        pass
    d = PRIVATE if private else OUT
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", d / f"{name}.png")
