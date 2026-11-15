#!/usr/bin/env python3
"""Score the Qwen (temperature 1.0, full run) pre-fill against the RAs' CVAT corrections (gold set v2, cvat_qwen_t1).

RAs got every Qwen box on each frame pre-filled (name + format) and corrected the frame: delete wrong boxes, fix
box / name / format, draw what Qwen missed. A pre-filled box the RA left alone counts as agreement.

Matching, per frame: an RA box with source="file" and the pre-fill's exact coordinates is an untouched pre-fill box.
The rest (pre-fill boxes the RA edited or deleted, RA boxes edited or drawn) are matched one-to-one by IoU (Hungarian):
IoU >= 0.5, or >= 0.3 when the name is unchanged (a box the RA resized a lot). Unmatched pre-fill = deleted (false
positive); unmatched RA box = added (miss).

Reviewed frames: CVAT does not record which frames an RA has looked at, and untouched frames count as agreement, so
only frames up to the last frame the RA changed count as reviewed (frames after it may not have been opened yet);
override with --reviewed Mira_batch1=139. The latest export per RA and batch is used (date in the file name).

Measures (pooled over boxes and frame-weighted with the sample's inclusion weights, frames_qwen_t1.csv):
  precision      pre-fill boxes kept (matched) / pre-fill boxes
  recall         matched / (matched + added)
  name kept      matched boxes whose name the RA left unchanged (raw, and after qwen_data.norm_label)
  format kept    matched boxes whose format the RA left unchanged; confusion matrix
  box IoU        pre-fill vs final box for matched boxes
  strict         matched with name and format unchanged and IoU >= 0.5 (precision and recall)
Each measure also without person, without body parts (BODY_PARTS) and for body parts only.
Inter-rater: on the shared frames both RAs reviewed, their final boxes matched the same way.

Outputs (out/private/ra_scores/: frame ids): boxes.csv (every pre-fill and RA box with its outcome), summary.csv,
by_rater_batch.csv, by_rater_batch_no_body_parts.csv, format_confusion.csv, by_name.csv, name_changes.csv, interrater.csv,
interrater_pairs.csv (Emma / Mira / Qwen pre-fill, pairwise on the shared frames, with and without body parts)
  python score_ra_qwen_t1.py [--reviewed Mira_batch1=139]          # analysis env (vislearnlabpy)
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
RA_DIR = PRIV / "cvat_qwen_t1_ra_annotations"
PREFILL = PRIV / "cvat_qwen_t1"
OUT = PRIV / "ra_scores"
sys.path.insert(0, str(HERE.parent / "qwen_pilot"))
IOU_MATCH, IOU_SAME_NAME, IOU_STRICT = 0.5, 0.3, 0.5
# body parts: the body_parts domain of qwen_hierarchy.py (minus furniture leg, face id, superhero face, skull, tattoo,
# sweat) plus RA spellings it does not list; a box counts as a body part if its Qwen or RA name is one
BODY_PARTS = set("""adult hand|adult s leg|ankle|arm|arms|baby feet|baby hand|back|bear paw|beard|belly button|body parts|
cheek|child hand|child s foot|child s hand|child s leg|chin|claw|ear|elbow|eye|eyebrow|eyelash|eyelid|eyes|face|feet|finger|
fingernail|fingers|fingertip|fist|foot|footn|forehead|hair|hair bun|hand|hands|head|human foot|human hand|human leg|knee|
kneecap|kneww|leg|legs|lip|lips|mouth|mustache|neck|nipple|nose|palm|paw|person s arm|person s finger|person s hand|ponytail|
shin|shoulder|skin|smiling face|stomach|teeth|thigh|thumb|thumbnail|toe|toes|tongue|tooth|torso""".replace("\n", "").split("|"))


def read_xml(xml: bytes) -> pd.DataFrame:
    rows = []
    for im in ET.fromstring(xml).findall("image"):
        for k, b in enumerate(im.findall("box")):
            a = {x.get("name"): (x.text or "").strip() for x in b.findall("attribute")}
            rows.append(dict(frame_id=int(im.get("id")), frame_uid=im.get("name").rsplit(".", 1)[0], k=k,
                             source=b.get("source"), name=a.get("name", "").lower(), format=a.get("format", "unset"),
                             x0=float(b.get("xtl")), y0=float(b.get("ytl")), x1=float(b.get("xbr")), y1=float(b.get("ybr"))))
    frames = [(int(im.get("id")), im.get("name").rsplit(".", 1)[0]) for im in ET.fromstring(xml).findall("image")]
    return pd.DataFrame(rows), pd.DataFrame(frames, columns=["frame_id", "frame_uid"])


def read_zip(p: Path):
    with zipfile.ZipFile(p) as z:
        return read_xml(z.read("annotations.xml"))


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    i = ix * iy
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return i / np.maximum(area(A)[:, None] + area(B)[None, :] - i, 1e-9)


def match(P: pd.DataFrame, R: pd.DataFrame, exact: bool = True) -> list[tuple]:
    """(p index or None, r index or None, iou) for one frame."""
    out, used_p, used_r = [], set(), set()
    if exact:                      # untouched pre-fill boxes: source file, same coordinates
        for ri, r in R[R.source == "file"].iterrows():
            hit = P[(~P.index.isin(used_p)) & np.isclose(P.x0, r.x0, atol=.02) & np.isclose(P.y0, r.y0, atol=.02)
                    & np.isclose(P.x1, r.x1, atol=.02) & np.isclose(P.y1, r.y1, atol=.02)]
            if len(hit):
                pi = hit.index[0]; used_p.add(pi); used_r.add(ri); out.append((pi, ri, 1.0))
    p_left, r_left = [i for i in P.index if i not in used_p], [i for i in R.index if i not in used_r]
    if p_left and r_left:
        cols = ["x0", "y0", "x1", "y1"]
        M = iou_matrix(P.loc[p_left, cols].values, R.loc[r_left, cols].values)
        same = P.loc[p_left, "name"].values[:, None] == R.loc[r_left, "name"].values[None, :]
        ok = (M >= IOU_MATCH) | (same & (M >= IOU_SAME_NAME))
        a, b = linear_sum_assignment(-(np.where(ok, M, 0) + same * 1e-3))
        for i, j in zip(a, b):
            if ok[i, j]:
                used_p.add(p_left[i]); used_r.add(r_left[j]); out.append((p_left[i], r_left[j], float(M[i, j])))
    out += [(pi, None, np.nan) for pi in P.index if pi not in used_p]
    out += [(None, ri, np.nan) for ri in R.index if ri not in used_r]
    return out


def latest_exports() -> dict[str, Path]:
    best = {}
    for p in sorted(RA_DIR.glob("*.zip")):
        m = re.match(r"(\w+_batch\d+)_(\d{4}-\d{1,2}-\d{1,2}|final)", p.stem)
        if m:
            key, tag = m.groups()
            if tag != "final":                       # 2026-11-4 -> 2026-11-04, so dates compare as strings
                y, mo, d = tag.split("-"); tag = f"{y}-{int(mo):02d}-{int(d):02d}"
            if key not in best or tag == "final" or (best[key][0] != "final" and tag > best[key][0]):
                best[key] = (tag, p)
    return {k: v[1] for k, v in best.items()}


def norm(s: str) -> str:
    try:
        from qwen_data import norm_label
        return norm_label(s)
    except Exception:
        return s.strip().lower()


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", nargs="*", default=[], help="BATCH=LAST_FRAME_ID overrides, e.g. Mira_batch1=139")
    args = ap.parse_args()
    over = dict(s.split("=") for s in args.reviewed)
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(PRIV / "frames_qwen_t1.csv").set_index("frame_uid")
    rows, reviewed, in_batch = [], {}, {}
    for key, p in sorted(latest_exports().items()):
        R, Rf = read_zip(p)
        P, Pf = read_zip(PREFILL / key / "annotations.zip")
        # CVAT's XML export drops frames left with no boxes (RA deleted every Qwen box): take the frame list from the pre-fill
        Rf = pd.concat([Rf, Pf]).drop_duplicates("frame_id").sort_values("frame_id").reset_index(drop=True)
        in_batch.setdefault(key.split("_")[0], set()).update(Rf.frame_uid)
        changed = set(R.loc[R.source != "file", "frame_id"])
        deleted = set(P.frame_id) - set(R.frame_id) | {f for f, g in P.groupby("frame_id")
                                                       if len(g) != (R.frame_id == f).sum()}
        last = int(over.get(key, max(changed | deleted) if (changed | deleted) else -1))
        frames = Rf[Rf.frame_id <= last]
        reviewed[key] = dict(export=p.name, last_frame_id=last, frames_reviewed=len(frames), frames_in_batch=len(Rf),
                             frames_changed=len({f for f in changed | deleted if f <= last}))
        print(f"{key}: {p.name}, reviewed frames 0-{last} ({len(frames)} of {len(Rf)})", flush=True)
        for fid, uid in frames.itertuples(index=False):
            Pi, Ri = P[P.frame_id == fid], R[R.frame_id == fid]
            for pi, ri, iou in match(Pi, Ri):
                pr, rr = (P.loc[pi] if pi is not None else None), (R.loc[ri] if ri is not None else None)
                rows.append(dict(rater=key.split("_")[0], batch=key, frame_uid=uid, frame_id=fid,
                                 outcome="matched" if pr is not None and rr is not None else ("deleted" if rr is None else "added"),
                                 untouched=bool(rr is not None and rr["source"] == "file" and iou == 1.0),
                                 q_name=pr["name"] if pr is not None else "", q_format=pr["format"] if pr is not None else "",
                                 ra_name=rr["name"] if rr is not None else "", ra_format=rr["format"] if rr is not None else "",
                                 iou=iou, **{f"q_{c}": (pr[c] if pr is not None else np.nan) for c in ("x0", "y0", "x1", "y1")},
                                 **{f"ra_{c}": (rr[c] if rr is not None else np.nan) for c in ("x0", "y0", "x1", "y1")}))
    B = pd.DataFrame(rows)
    nm = {s: norm(s) for s in set(B.q_name) | set(B.ra_name)}
    m = B.outcome == "matched"
    B["name_kept"] = np.where(m, B.q_name == B.ra_name, np.nan)
    B["name_kept_norm"] = np.where(m, B.q_name.map(nm) == B.ra_name.map(nm), np.nan)
    B["format_kept"] = np.where(m, B.q_format == B.ra_format, np.nan)
    B["strict"] = np.where(m, (B.name_kept_norm == 1) & (B.format_kept == 1) & (B.iou >= IOU_STRICT), np.nan)
    B["subject_id"] = B.frame_uid.map(fr.subject_id); B["age_months"] = B.frame_uid.map(fr.age_months)
    B["weight"] = B.frame_uid.map(fr.weight)
    shared = set.intersection(*in_batch.values()) if len(in_batch) == 2 else set()      # frames given to both RAs
    B["shared"] = B.frame_uid.isin(shared)
    B.to_csv(OUT / "boxes.csv", index=False)
    pd.DataFrame(reviewed).T.to_csv(OUT / "reviewed_frames.csv")

    def measures(g: pd.DataFrame) -> dict:
        q = g[g.outcome != "added"]; ra = g[g.outcome != "deleted"]; mm = g[g.outcome == "matched"]
        n_m, n_q, n_a = len(mm), len(q), (g.outcome == "added").sum()
        w = lambda s: wmean(s.astype(float), g.loc[s.index, "weight"])
        return {"frames": g.frame_uid.nunique(), "qwen_boxes": n_q, "ra_boxes": len(ra), "matched": n_m,
                "deleted": n_q - n_m, "added": n_a,
                "frames_untouched": int(sum((x.outcome == "matched").all() and x.untouched.all() for _, x in g.groupby("frame_uid"))),
                "precision": n_m / n_q if n_q else np.nan, "recall": n_m / (n_m + n_a) if n_m + n_a else np.nan,
                "precision_w": w(q.outcome == "matched"), "recall_w": w(ra.outcome == "matched"),
                "box_untouched": mm.untouched.mean(), "name_kept": mm.name_kept.mean(), "name_kept_norm": mm.name_kept_norm.mean(),
                "format_kept": mm.format_kept.mean(), "iou_median": mm.iou.median(), "iou_ge_0.5": (mm.iou >= .5).mean(),
                "strict_precision": mm.strict.sum() / n_q if n_q else np.nan,
                "strict_recall": mm.strict.sum() / (n_m + n_a) if n_m + n_a else np.nan}
    no_person = B[(B.q_name != "person") & (B.ra_name != "person")]
    bp = B.q_name.isin(BODY_PARTS) | B.ra_name.isin(BODY_PARTS)
    B_nobp = B[~bp]
    S = pd.DataFrame([{"set": "all", **measures(B)}, {"set": "without person", **measures(no_person)},
                      {"set": "without body parts", **measures(B_nobp)},
                      {"set": "without body parts or person", **measures(B_nobp[(B_nobp.q_name != "person") & (B_nobp.ra_name != "person")])},
                      {"set": "body parts only", **measures(B[bp])}])
    S["f1"] = 2 * S.precision * S.recall / (S.precision + S.recall)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame([{"batch": k, **measures(g)} for k, g in B.groupby("batch")]).to_csv(OUT / "by_rater_batch.csv", index=False)
    pd.DataFrame([{"batch": k, **measures(g)} for k, g in B_nobp.groupby("batch")]).to_csv(OUT / "by_rater_batch_no_body_parts.csv", index=False)
    mm = B[B.outcome == "matched"]
    pd.crosstab(mm.q_format, mm.ra_format, margins=True).to_csv(OUT / "format_confusion.csv")
    q = B[B.outcome != "added"].assign(kept=lambda x: x.outcome == "matched")
    by = q.groupby(q.q_name.map(nm)).agg(qwen_boxes=("kept", "size"), precision=("kept", "mean"))
    added = B[B.outcome == "added"].groupby(B.ra_name.map(nm)).size().rename("added")
    by = by.join(added, how="outer").fillna({"qwen_boxes": 0, "added": 0})
    by["recall"] = (by.precision * by.qwen_boxes) / (by.precision.fillna(0) * by.qwen_boxes + by.added)
    by.sort_values(["qwen_boxes", "added"], ascending=False).to_csv(OUT / "by_name.csv")
    ch = mm[mm.name_kept_norm == 0].groupby(["q_name", "ra_name"]).size().rename("n").sort_values(ascending=False)
    ch.to_csv(OUT / "name_changes.csv")

    # inter-rater: shared frames both RAs reviewed; final boxes of one vs the other
    ir = []
    sh = B[B.shared == True]
    both = set.intersection(*[set(g.frame_uid) for _, g in sh.groupby("rater")]) if sh.rater.nunique() == 2 else set()
    final = sh[sh.outcome != "deleted"]
    cols = ["ra_x0", "ra_y0", "ra_x1", "ra_y1"]
    for uid in sorted(both):
        a = final[(final.frame_uid == uid) & (final.rater == "Emma")].rename(columns=dict(zip(cols, ["x0", "y0", "x1", "y1"]))) \
            .assign(name=lambda x: x.ra_name, source="manual").reset_index(drop=True)
        b = final[(final.frame_uid == uid) & (final.rater == "Mira")].rename(columns=dict(zip(cols, ["x0", "y0", "x1", "y1"]))) \
            .assign(name=lambda x: x.ra_name, source="manual").reset_index(drop=True)
        for i, j, iou in match(a, b, exact=False):
            ir.append(dict(frame_uid=uid, both=i is not None and j is not None, emma_only=j is None, mira_only=i is None, iou=iou,
                           name_same=(nm[a.loc[i, "name"]] == nm[b.loc[j, "name"]]) if i is not None and j is not None else np.nan,
                           format_same=(a.loc[i, "ra_format"] == b.loc[j, "ra_format"]) if i is not None and j is not None else np.nan))
    I = pd.DataFrame(ir)
    if len(I):
        nb, ne, nmi = I.both.sum(), I.emma_only.sum(), I.mira_only.sum()
        pd.DataFrame([{"shared_frames_both_reviewed": len(both), "boxes_both": nb, "emma_only": ne, "mira_only": nmi,
                       "box_agreement_f1": 2 * nb / (2 * nb + ne + nmi), "name_same": I.name_same.mean(),
                       "format_same": I.format_same.mean(), "iou_median": I.iou.median()}]).to_csv(OUT / "interrater.csv", index=False)

    # the same agreement for every pair on the shared frames (Emma, Mira, Qwen's pre-fill), with and without body parts
    def boxes_of(who, nobp):
        if who == "Qwen":
            g = sh[(sh.outcome != "added") & sh.frame_uid.isin(both)].drop_duplicates(["frame_uid", "q_x0", "q_y0", "q_x1", "q_y1", "q_name"])
            g = g.assign(x0=g.q_x0, y0=g.q_y0, x1=g.q_x1, y1=g.q_y1, name=g.q_name, fmt=g.q_format)
            bp = g.q_name.map(nm).isin(BODY_PARTS)
        else:
            g = final[(final.rater == who) & final.frame_uid.isin(both)]
            g = g.assign(x0=g.ra_x0, y0=g.ra_y0, x1=g.ra_x1, y1=g.ra_y1, name=g.ra_name, fmt=g.ra_format)
            bp = g.ra_name.map(nm).isin(BODY_PARTS) | g.q_name.map(nm).isin(BODY_PARTS)
        return (g[~bp] if nobp else g).assign(source="manual")
    pairs = []
    for nobp in (False, True):
        for x, y in (("Emma", "Mira"), ("Qwen", "Emma"), ("Qwen", "Mira")):
            X, Y = boxes_of(x, nobp), boxes_of(y, nobp)
            n = {"both": 0, "a_only": 0, "b_only": 0}; nm_same, fm_same, ious = [], [], []
            for uid in sorted(both):
                a = X[X.frame_uid == uid].reset_index(drop=True); b = Y[Y.frame_uid == uid].reset_index(drop=True)
                for i, j, iou in match(a, b, exact=False):
                    if i is not None and j is not None:
                        n["both"] += 1; ious.append(iou)
                        nm_same.append(nm[a.loc[i, "name"]] == nm[b.loc[j, "name"]]); fm_same.append(a.loc[i, "fmt"] == b.loc[j, "fmt"])
                    else:
                        n["a_only" if j is None else "b_only"] += 1
            pairs.append({"pair": f"{x} vs {y}", "body_parts": "excluded" if nobp else "included", "frames": len(both),
                          "boxes_both": n["both"], "a_only": n["a_only"], "b_only": n["b_only"],
                          "box_f1": 2 * n["both"] / (2 * n["both"] + n["a_only"] + n["b_only"]),
                          "name_same": np.mean(nm_same), "format_same": np.mean(fm_same), "iou_median": np.median(ious)})
    pd.DataFrame(pairs).to_csv(OUT / "interrater_pairs.csv", index=False)
    print(S.round(3).T.to_string())
    print(pd.read_csv(OUT / "reviewed_frames.csv", index_col=0).to_string())


if __name__ == "__main__":
    main()
