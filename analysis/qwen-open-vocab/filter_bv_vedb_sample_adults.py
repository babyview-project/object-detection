#!/usr/bin/env python3
"""Drop VEDB minors from the BabyView / VEDB embedding sample without redrawing it.

The sample (out/private/bv_vedb_renamed/sample_{clip,dinov3}.npz, build_bv_vedb_data.py) is referenced by row position
from three places: the blur sharpness scores (sharpness.csv.gz, one row per sample row) and SAM 3's chunks and masked
embeddings (annotation/qwen_pilot/sam3_crop_sample.py, embed_masked_sample.py: `idx` = sample row). Redrawing the
sample would orphan the SAM 3 outputs (GPU), so instead the original sample loses the rows of VEDB sessions recorded
by minors (vedb-1612/vedb_adult_sessions.txt), and every positional reference is remapped to the new row numbers:

  sample_{clip,dinov3}.npz, sharpness.csv.gz   rows kept in order          -> --out (default out/private/bv_vedb_renamed)
  SAM 3 w*_*.npz, embeddings_{clip,dinov3}.npz  rows of dropped boxes removed, idx remapped -> SAM_ADULTS

Run after build_bv_vedb_data.py (adult store), which rewrites frames / presence / geometry and draws a new sample;
this script then replaces that sample with the filtered original (--src, the pre-exclusion copy).

  python filter_bv_vedb_sample_adults.py --src /data2/j7yang/object-detection-offload/backups/pre_vedb_adults_20261115/qwen-open-vocab/out/private/bv_vedb_renamed
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ADULTS = HERE.parents[0] / "vedb-1612" / "vedb_adult_sessions.txt"
SAM = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb")
SAM_ADULTS = SAM.with_name(SAM.name + "_adults")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="pre-exclusion bv_vedb_renamed folder (sample + sharpness)")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "private" / "bv_vedb_renamed")
    a = ap.parse_args()
    assert a.src.resolve() != a.out.resolve(), "--src must be the untouched pre-exclusion copy"
    adult_units = {l.split("_")[1] for l in ADULTS.read_text().split()}

    ref = None
    for m in ("clip", "dinov3"):
        z = dict(np.load(a.src / f"sample_{m}.npz", allow_pickle=True))
        keep = (z["dataset"] != "vedb") | np.isin(z["unit"], list(adult_units))
        if ref is None:
            ref = keep
            n_v = (z["dataset"] == "vedb").sum()
            print(f"sample: {len(keep):,} rows; VEDB {n_v:,} -> {(keep & (z['dataset'] == 'vedb')).sum():,}; "
                  f"dropped {(~keep).sum():,} rows from {len(set(z['unit'][~keep]))} minor sessions")
        assert (keep == ref).all(), "clip and dinov3 samples differ in row order"
        np.savez(a.out / f"sample_{m}.npz", **{k: v[keep] for k, v in z.items()})
    sh = pd.read_csv(a.src / "sharpness.csv.gz")
    assert len(sh) == len(ref)
    sh[ref].to_csv(a.out / "sharpness.csv.gz", index=False)

    new_idx = np.full(len(ref), -1, dtype=np.int64)
    new_idx[ref] = np.arange(ref.sum())
    SAM_ADULTS.mkdir(exist_ok=True)
    for f in sorted(glob.glob(str(SAM / "*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        k = new_idx[z["idx"]] >= 0
        z = {c: v[k] for c, v in z.items()}
        z["idx"] = new_idx[z["idx"]]
        np.savez(SAM_ADULTS / Path(f).name, **z)
    for f in SAM.iterdir():
        if f.suffix != ".npz" and f.is_file():
            shutil.copy2(f, SAM_ADULTS / f.name)
    (SAM_ADULTS / "ADULTS_ONLY.txt").write_text(
        f"{SAM.name} without VEDB minors' boxes; idx remapped to the filtered sample "
        "(analysis/qwen-open-vocab/filter_bv_vedb_sample_adults.py).\n")
    e = np.load(SAM_ADULTS / "embeddings_dinov3.npz")
    print(f"SAM 3 embeddings: {len(e['idx']):,} rows, idx max {e['idx'].max():,} < {ref.sum():,} -> {SAM_ADULTS}")


if __name__ == "__main__":
    main()
