#!/usr/bin/env python3
"""Zero-shot visual-format classifier (real / toy / picture / screen) on existing CLIP image embeddings.

For datasets without format labels (THINGS, LVIS, ImageNet). Each item is compared with format prompts built from its own
label (prompt ensembles: "a photo of a {x}" / "a photo of a toy {x}" / "a drawing of a {x}" / "a {x} on a TV screen", ...),
giving four cosine similarities; a 4-feature multinomial logistic regression turns them into a format. It is fitted and
checked on BabyView boxes against Qwen's own format labels (photo + drawing = picture): half the sampled videos to fit,
half to test. Labels are normalized first (qwen_data.norm_label: "toy car" -> car, "picture of a dog" -> dog) so the
label does not give the format away; generic labels (toy, animal, ...) are skipped.

  python format_classifier.py validate [--videos 800]      # BabyView: fit + held-out check -> results/format_classifier_*.csv
The classifier used downstream is the supervised probe (L2-normalized CLIP + DINOv3 image embeddings + the four prompt
similarities, logistic regression, no class balancing): 88.5% held-out agreement with Qwen; held-out format shares within
~1 point of Qwen's (real 58.7 vs 56.8, toy 26.0 vs 27.1, picture 13.2 vs 13.5, screen 2.1 vs 2.7%).
  python format_classifier.py apply --dataset things        # THINGS images -> results/format_things.csv
Analysis env (vislearnlabpy; CLIP text encoder on CPU).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QP = HERE.parents[1] / "annotation" / "qwen_pilot"
sys.path.insert(0, str(QP))
STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
THINGS = Path("/data2/dataset/babyview/related_data/things")
RES = HERE / "results"
FORMATS = ["real", "toy", "picture", "screen"]
PROMPTS = {
    "real": ["a photo of a {x}", "a photo of a real {x}", "a close-up photo of a {x}"],
    "toy": ["a photo of a toy {x}", "a photo of a plastic toy {x}", "a photo of a stuffed toy {x}"],
    "picture": ["a drawing of a {x}", "an illustration of a {x} in a picture book", "a printed photo of a {x}"],
    "screen": ["a {x} on a TV screen", "a photo of a screen showing a {x}", "a {x} on a phone screen"],
}
GENERIC = {"toy", "animal", "object", "item", "thing", "character", "food", "vehicle", "figure", "picture", "drawing",
           "photo", "screen", "cartoon", "painting", "illustration", "poster", "sticker"}


def text_bank(labels: list[str]) -> dict[str, np.ndarray]:
    """label -> (4, 512) L2-normalized mean prompt embedding per format (CLIP ViT-B/32 text tower, CPU)."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    torch.set_num_threads(1)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    texts = [(l, f, p.format(x=l)) for l in labels for f in FORMATS for p in PROMPTS[f]]
    embs = []
    with torch.inference_mode():
        for i in range(0, len(texts), 512):
            t = tok([x[2] for x in texts[i:i + 512]], padding=True, truncation=True, return_tensors="pt")
            e = m.get_text_features(**t)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).numpy())
    E = np.concatenate(embs)
    out, k = {}, 0
    for l in labels:
        per = []
        for f in FORMATS:
            n = len(PROMPTS[f]); v = E[k:k + n].mean(0); k += n
            per.append(v / np.linalg.norm(v))
        out[l] = np.stack(per)
    return out


def features(X: np.ndarray, labels: list[str], bank: dict) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    T = np.stack([bank[l] for l in labels])                      # n x 4 x 512
    return np.einsum("nd,nfd->nf", Xn, T) * 100.0                 # CLIP logit scale


def bv_sample(n_videos: int, seed: int = 0) -> pd.DataFrame:
    import qwen_data as qd
    stems = sorted(p.stem for p in (STORE / "shards").glob("*.npz") if not p.name.endswith(".tmp.npz"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(stems, n_videos, replace=False)
    rows, X, XD = [], [], []
    for s in pick:
        with np.load(STORE / "shards" / f"{s}.npz", allow_pickle=True) as z:
            if len(z["image_id"]) == 0:
                continue
            labs = [qd.norm_label(str(l)) for l in z["label"]]
            fmt = pd.Series(z["format"].astype(str)).replace({"photo": "picture", "drawing": "picture"}).values
            keep = np.array([l not in GENERIC and l != "person" and f in FORMATS for l, f in zip(labs, fmt)])
            idx = np.flatnonzero(keep)
            if len(idx) > 300:                                    # cap per video, keep all depictions
                dep = idx[fmt[idx] != "real"]; real = idx[fmt[idx] == "real"]
                idx = np.concatenate([dep, rng.choice(real, max(0, 300 - len(dep)), replace=False)]) if len(real) else dep
            for i in idx:
                rows.append({"video": s, "label": labs[i], "format": fmt[i]})
            X.append(z["clip_emb"][idx].astype(np.float32)); XD.append(z["dino_emb"][idx].astype(np.float32))
    return pd.DataFrame(rows), np.concatenate(X), np.concatenate(XD)


def validate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    df, X, XD = bv_sample(args.videos)
    print(f"BabyView sample: {len(df):,} boxes from {df.video.nunique()} videos; formats {df.format.value_counts().to_dict()}", flush=True)
    bank = text_bank(sorted(df.label.unique()))
    F = features(X, list(df.label), bank)
    vids = df.video.unique(); rng = np.random.default_rng(1); fit_v = set(rng.choice(vids, len(vids) // 2, replace=False))
    tr = df.video.isin(fit_v).values
    zs = np.array(FORMATS)[F.argmax(1)]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(F[tr], df.format[tr])
    pr = clf.predict(F[~tr])
    y = df.format[~tr].values
    rep = []
    for name, p in (("zero-shot argmax", zs[~tr]), ("calibrated (logistic on 4 similarities)", pr)):
        r = classification_report(y, p, labels=FORMATS, output_dict=True, zero_division=0)
        for f in FORMATS:
            rep.append({"method": name, "format": f, "precision": r[f]["precision"], "recall": r[f]["recall"],
                        "f1": r[f]["f1-score"], "support": r[f]["support"]})
        rep.append({"method": name, "format": "accuracy", "precision": r["accuracy"]})
        print(f"\n{name}: held-out accuracy {r['accuracy']:.3f}\n" +
              pd.DataFrame(confusion_matrix(y, p, labels=FORMATS), index=[f"Qwen {f}" for f in FORMATS],
                           columns=[f"pred {f}" for f in FORMATS]).to_string())
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rep).to_csv(RES / "format_classifier_validation_bv.csv", index=False)
    # share of each format per dataset is what we compare: estimated vs Qwen share on held-out BV
    est = pd.Series(pr).value_counts(normalize=True).reindex(FORMATS).fillna(0)
    true = pd.Series(y).value_counts(normalize=True).reindex(FORMATS).fillna(0)
    print("\nheld-out BV format shares, Qwen vs classifier:\n", pd.DataFrame({"qwen": true, "classifier": est}).round(3).to_string())
    # supervised probe: L2-normalized CLIP + DINOv3 image embeddings + the 4 prompt similarities (no class balancing,
    # so predicted shares are comparable to Qwen's)
    def l2(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Z = np.hstack([l2(X), l2(XD), F / 100.0])
    probe = LogisticRegression(max_iter=3000, C=1.0).fit(Z[tr], df.format[tr])
    pp = probe.predict(Z[~tr])
    r = classification_report(y, pp, labels=FORMATS, output_dict=True, zero_division=0)
    print(f"\nprobe (CLIP + DINOv3 + prompts): held-out accuracy {r['accuracy']:.3f}\n" +
          pd.DataFrame(confusion_matrix(y, pp, labels=FORMATS), index=[f"Qwen {f}" for f in FORMATS],
                       columns=[f"pred {f}" for f in FORMATS]).to_string())
    for f in FORMATS:
        rep.append({"method": "probe (CLIP + DINOv3 + prompts)", "format": f, "precision": r[f]["precision"],
                    "recall": r[f]["recall"], "f1": r[f]["f1-score"], "support": r[f]["support"]})
    rep.append({"method": "probe (CLIP + DINOv3 + prompts)", "format": "accuracy", "precision": r["accuracy"]})
    pd.DataFrame(rep).to_csv(RES / "format_classifier_validation_bv.csv", index=False)
    est2 = pd.Series(pp).value_counts(normalize=True).reindex(FORMATS).fillna(0)
    print("\nheld-out BV format shares: Qwen / probe\n", pd.DataFrame({"qwen": true, "probe": est2}).round(3).to_string())
    probe_all = LogisticRegression(max_iter=3000, C=1.0).fit(Z, df.format)
    pickle.dump({"clf": probe_all, "prompts": PROMPTS, "formats": FORMATS, "features": "clip+dinov3+prompts"},
                open(RES / "format_classifier.pkl", "wb"))


def apply(args):
    model = pickle.load(open(RES / "format_classifier.pkl", "rb"))
    if args.dataset == "things":
        z = np.load(THINGS / "embeddings" / "things_images_clip.npz", allow_pickle=True)
        t = pd.read_csv(THINGS / "concepts-metadata_things.tsv", sep="\t", usecols=["Word", "uniqueID"]).set_index("uniqueID").Word
        labels = [str(t.get(c, c)).lower() for c in z["concept"]]
        X = z["emb"].astype(np.float32)
        zd = np.load(THINGS / "embeddings" / "things_images_dinov3.npz", allow_pickle=True)
        assert (zd["path"] == z["path"]).all()
        XD = zd["emb"].astype(np.float32)
        ids = z["path"]
        items = z["concept"]
    else:
        raise SystemExit("apply: only --dataset things for now (LVIS / ImageNet after the VEDB run)")
    bank = text_bank(sorted(set(labels)))
    F = features(X, labels, bank)
    def l2(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    p = model["clf"].predict_proba(np.hstack([l2(X), l2(XD), F / 100.0]))
    out = pd.DataFrame(p, columns=[f"p_{c}" for c in model["clf"].classes_])
    out.insert(0, "item", items); out.insert(1, "label", labels); out.insert(2, "id", ids)
    out["format"] = model["clf"].classes_[p.argmax(1)]
    out.to_csv(RES / f"format_{args.dataset}.csv", index=False)
    print(f"{args.dataset}: {len(out):,} images; predicted formats {out.format.value_counts(normalize=True).round(3).to_dict()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["validate", "apply"])
    ap.add_argument("--videos", type=int, default=800)
    ap.add_argument("--dataset", default="things")
    args = ap.parse_args()
    validate(args) if args.step == "validate" else apply(args)


if __name__ == "__main__":
    main()
