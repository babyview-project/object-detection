"""Label variant for the open-category analyses: QOV_LABELS=original (default) or renamed.

renamed: boxes take `label_renamed` from <store>/boxes_index_renamed.csv.gz (annotation/qwen_pilot/make_renamed_index.py:
generic labels replaced by the second-pass names, computer mice split from mice). Outputs go to parallel folders with a
"_renamed" suffix, so the two versions can be compared; the Tier 1 category list is the original one in both.
"""
import os
from pathlib import Path

import pandas as pd

LABELS = os.environ.get("QOV_LABELS", "original")
assert LABELS in ("original", "renamed"), LABELS
SUFFIX = "" if LABELS == "original" else "_renamed"
# Domain scheme: cdi (default; categories.csv `domain`) or things27 (THINGS high-level categories, Hebart et al., 2019:
# data/category_domains_things27.csv `domain_analysis`, built by build_things27_domains.py). things27 outputs go to
# parallel folders with a further "_things27" suffix; their inputs are copies made by make_things27_inputs.py.
DOMAINS = os.environ.get("QOV_DOMAINS", "cdi")
assert DOMAINS in ("cdi", "things27"), DOMAINS
if DOMAINS == "things27":
    SUFFIX += "_things27"


def index_path(store: Path) -> Path:
    return Path(store) / ("boxes_index.csv.gz" if LABELS == "original" else "boxes_index_renamed.csv.gz")


def read_index(store: Path, usecols=None, **kw) -> pd.DataFrame:
    """The store's box index with `label` = the variant's label (row order = shard order)."""
    cols = None if usecols is None else [c for c in usecols if c != "label"] + (["label"] if LABELS == "original" else ["label_renamed"])
    ix = pd.read_csv(index_path(store), usecols=cols, **kw)
    if LABELS == "renamed":
        ix = ix.drop(columns=[c for c in ("label", "renamed_from") if c in ix.columns]).rename(columns={"label_renamed": "label"})
    return ix


def out_dir(path: Path) -> Path:
    """Parallel output folder for the variant: out/private/open -> out/private/open_renamed."""
    path = Path(path)
    return path.with_name(path.name + SUFFIX)


def manuscript_figures_dir(default: Path) -> Path:
    """Where manuscript figures are copied: the draft's figures/ (cdi) or figures_things27/ next to it (things27)."""
    default = Path(default)
    return default if DOMAINS == "cdi" else default.with_name(default.name + "_things27")
