#!/usr/bin/env python3
"""Box and frame tables for every child whose frames are all annotated (full 2025.2 run + subject runs).

The full run (run_fullrun.sh) works through the manifest in unit-id order, so children finish one after
another while it runs. A child counts as complete when every manifest frame of it has a parsed record
(missing <= --tolerance of its frames, for the odd parse failure). Reads the shard files as they are, so it is
safe while the run writes (a half-written last line is skipped).

Output (private: unit ids name videos), out/private/:
  children_objects.pkl   one row per box (near-duplicates flagged in `dup`): unit_id, subject, label, cat, basic,
                         domain, domain_source, generic, format, box, area, age_months
  children_frames.pkl    one row per frame: unit_id, subject, video_id, date, age_months, n_objects, at_cap, marching
  children_status.csv    per child: manifest frames, annotated frames, complete

  python build_children_cache.py          # analysis env (vislearnlabpy); ~10-20 min single-threaded
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
HERE = Path(__file__).resolve().parent
QP = HERE.parents[1] / "annotation" / "qwen_pilot"        # pipeline: loaders, embeddings, manifest
sys.path.insert(0, str(QP))
import qwen_data as qd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
TAGS = ["fullrun2025.2_t1_pp0"]          # plus every subj_*_t1_pp0 run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=0.001)
    args = ap.parse_args()
    m = pd.read_csv(QP.parent / "gemini_full_run" / "out" / "private" / "manifest.csv", usecols=["unit_id", "video_id"])
    m["subject"] = m.video_id.str.split("_").str[0]

    tags = TAGS + sorted(p.name.split("__p1_")[1].removesuffix(".jsonl")
                         for p in qd.PRIV.glob(f"{qd.MODEL_TAG}__detect_open_format__p1_subj_*_t1_pp0.jsonl"))
    recs: dict[str, dict] = {}
    for t in tags:
        recs |= qd.load_records("detect_open_format", t)
    ok = {u for u, r in recs.items() if r.get("error") is None}
    st = m.assign(done=m.unit_id.isin(ok)).groupby("subject").done.agg(frames="size", annotated="sum")
    st["complete"] = st.annotated >= st.frames * (1 - args.tolerance)
    st.to_csv(OUT / "children_status.csv")
    keep = set(st.index[st.complete])
    print(f"{len(keep)} complete children: {sorted(keep)}", flush=True)

    units = set(m.unit_id[m.subject.isin(keep)])
    recs = {u: r for u, r in recs.items() if u in units}
    fr = qd.frames_table(recs)
    obj = qd.objects_table(recs).merge(fr[["unit_id", "subject", "age_months"]], on="unit_id")
    fr.to_pickle(OUT / "children_frames.pkl")
    obj.to_pickle(OUT / "children_objects.pkl")
    print(f"{len(fr):,} frames, {len(obj):,} boxes ({int(obj.dup.sum()):,} near-duplicates), "
          f"{obj.cat.nunique():,} categories; domains from the hierarchy table for "
          f"{(obj.domain_source == 'hierarchy').mean():.1%} of boxes")


if __name__ == "__main__":
    main()
