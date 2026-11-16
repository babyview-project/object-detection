"""Per child / per session format and truncation shares, for figures that show every unit with a mean ± SE.

Input: out/private/bv_vedb_renamed/unit_category_format.csv.gz (build_unit_format_table.py; Tier 1 categories).
A unit enters a cell (domain, or all categories) when it has >= MIN_BOXES boxes there; the figure value is the unweighted
mean over those units, ± 1 SE (SD / sqrt(n)). Pooled shares (every box counted once) stay in the result tables.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_BOXES = 20
# categories that are depictions or toys by definition: left out of the by-domain format shares (as crossdataset_analysis)
INTRINSIC = {"picture", "photo", "drawing", "painting", "poster", "sticker", "screen", "television", "toy", "block", "puzzle",
             "puzzle piece", "doll", "teddy bear", "lego", "lego block", "stuffed animal", "figurine", "cartoon", "map", "calendar",
             "book", "picture book", "card", "tablet", "phone", "laptop", "computer monitor", "monitor"}
_CACHE = {}


def table():
    if "t" not in _CACHE:
        _CACHE["t"] = pd.read_csv(HERE / "out" / "private" / "bv_vedb_renamed" / "unit_category_format.csv.gz", dtype={"unit": str})
    return _CACHE["t"].copy()


def per_unit(domain_of=None, categories=None, exclude_intrinsic=False, min_boxes=MIN_BOXES):
    """One row per (dataset, unit[, domain]): boxes and shares depicted / toy / picture / screen / edge (truncated)."""
    t = table()
    if categories is not None:
        t = t[t.category.isin(set(categories))]
    if exclude_intrinsic:
        t = t[~t.category.isin(INTRINSIC)]
    keys = ["dataset", "unit"]
    if domain_of is not None:
        t["domain"] = t.category.map(lambda c: domain_of.get(c, "other")); keys.append("domain")
    g = t.groupby(keys)[["boxes", "toy", "picture", "screen", "edge"]].sum().reset_index()
    g = g[g.boxes >= min_boxes]
    for c in ("toy", "picture", "screen", "edge"):
        g[c] = g[c] / g.boxes
    g["depicted"] = g.toy + g.picture + g.screen
    return g


def summary(u, by, col):
    """Mean ± SE over units of `col`, per `by` group."""
    s = u.groupby(by)[col].agg(["mean", "std", "count"]).rename(columns={"count": "n"})
    s["se"] = s["std"] / np.sqrt(s.n)
    return s
