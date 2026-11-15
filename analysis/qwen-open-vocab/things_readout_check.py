#!/usr/bin/env python3
"""Why do CLIP and DINOv3 disagree on THINGS? Re-embed the THINGS images mapped to the 183 core categories in four ways
and recompute the THINGS RDM and its domain clustering.

Variants (same models as embed_things.py; L2-normalized image vectors, averaged per concept, pooled to categories by
mapping/map_things.csv, z-scored across the core categories, cosine RDMs):
  full / center     whole image, or the central 60% of each side (the object region in THINGS' centred photos)
  dinov3 cls / patch   DINOv3 global token (pooler_output, as in the main analyses) or the mean of its patch tokens

Reported per variant: delta / SD and AUC of domain clustering, CLIP-DINOv3 RDM agreement, Spearman with BabyView's RDM
(same model, main readout), and with a text reference RDM (CLIP text embeddings of "a photo of a {category}.").

Outputs: out/private/things_readout/things_variants_{clip,dinov3}.npz, results/crossdataset_renamed_core/things_readout_check.csv
  OMP_NUM_THREADS=2 nice -n 19 python things_readout_check.py [--device cpu] [--analyze-only]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HOME", "/ccn2/u/khaiaw/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
THINGS_ROOT = Path("/data2/dataset/babyview/related_data/things")
import variant as V
OUT = HERE / "out" / "private" / "things_readout"
if V.DOMAINS != "cdi":
    OUT = V.out_dir(OUT)
RES = V.out_dir(HERE / "results" / "crossdataset"); RES = RES.with_name(RES.name + "_core")
CORE = [l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_core.txt") if l.strip()]
CENTER = 0.6


def embed(device, batch):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    m = pd.read_csv(HERE / "mapping" / "map_things.csv")
    concepts = set(m[m.category.isin(CORE)].id)
    paths = sorted(p for p in (THINGS_ROOT / "object_images").glob("*/*.jpg") if p.parent.name in concepts)
    print(f"{len(paths):,} images of {len(concepts)} concepts", flush=True)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    dino = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(device).eval()
    nreg = getattr(dino.config, "num_register_tokens", 0)

    def center(im):
        w, h = im.size
        return im.crop((round(w * (1 - CENTER) / 2), round(h * (1 - CENTER) / 2), round(w * (1 + CENTER) / 2), round(h * (1 + CENTER) / 2)))

    out = {k: [] for k in ("clip_full", "clip_center", "dinov3_cls_full", "dinov3_patch_full", "dinov3_cls_center", "dinov3_patch_center")}
    for s in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        for tag, I in (("full", ims), ("center", [center(i) for i in ims])):
            with torch.inference_mode():
                c = clip.get_image_features(**{k: v.to(device) for k, v in clip_proc(images=I, return_tensors="pt").items()})
                c = c.pooler_output if hasattr(c, "pooler_output") else c
                o = dino(**{k: v.to(device) for k, v in dproc(images=I, return_tensors="pt").items()})
            out[f"clip_{tag}"].append(c.float().cpu().numpy())
            out[f"dinov3_cls_{tag}"].append(o.pooler_output.float().cpu().numpy())
            out[f"dinov3_patch_{tag}"].append(o.last_hidden_state[:, 1 + nreg:].mean(1).float().cpu().numpy())
        if (s // batch) % 10 == 0:
            print(f"  {s + len(ims):,}/{len(paths):,}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "things_variants.npz", concept=np.array([p.parent.name for p in paths]),
             **{k: np.concatenate(v).astype(np.float32) for k, v in out.items()})


def analyze():
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import mannwhitneyu, spearmanr
    z = np.load(OUT / "things_variants.npz", allow_pickle=True)
    conc = z["concept"]
    m = pd.read_csv(HERE / "mapping" / "map_things.csv")
    m = m[m.category.isin(CORE)]
    dom = pd.read_csv(V.out_dir(HERE / "out" / "private" / "open") / "categories.csv", index_col=0).domain.to_dict()
    zr = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-10)
    iu = np.triu_indices(len(CORE), 1)
    sem = np.array([dom[c] for c in CORE]); same = sem[iu[0]] == sem[iu[1]]

    def cat_rdm(X):
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        S = pd.DataFrame(X).groupby(conc).sum(); N = pd.Series(conc).value_counts()
        cs = {}
        for r in m.itertuples():                                   # same pooling as build_crossdataset_tables.pool
            if r.id in S.index:
                a, n = cs.get(r.category, (0, 0)); cs[r.category] = (a + S.loc[r.id].values, n + N[r.id])
        M = np.stack([cs[c][0] / cs[c][1] for c in CORE])
        return squareform(pdist(zr(M), "cosine"))[iu]

    def stats(x):
        return ((x[~same].mean() - x[same].mean()) / x.std(), mannwhitneyu(x[~same], x[same]).statistic / (same.sum() * (~same).sum()), x.std())

    tabs = {mm: np.load(HERE / "out" / "crossdataset_renamed" / f"tables_{mm}.npz", allow_pickle=True) for mm in ("clip", "dinov3")}
    def ref(mm, d):
        t = tabs[mm]; T = pd.DataFrame(t[f"{d}__means"], index=list(t[f"{d}__categories"]))
        return squareform(pdist(zr(T.loc[CORE].values.astype(float)), "cosine"))[iu]
    bv = {mm: ref(mm, "bv") for mm in tabs}; th0 = {mm: ref(mm, "things") for mm in tabs}
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"); cm_ = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    with torch.inference_mode():
        t = cm_.get_text_features(**tok([f"a photo of a {c}." for c in CORE], padding=True, return_tensors="pt"))
        t = (t.pooler_output if hasattr(t, "pooler_output") else t).numpy()
    tx = squareform(pdist(zr(t / np.linalg.norm(t, axis=1, keepdims=True)), "cosine"))[iu]
    R = {k: cat_rdm(z[k]) for k in z.files if k != "concept"}
    rows = []
    for k, x in R.items():
        mm = "clip" if k.startswith("clip") else "dinov3"
        dsd, auc, sd = stats(x)
        other = R["dinov3_cls_" + k.split("_")[-1]] if mm == "clip" else R["clip_" + k.split("_")[-1]]
        rows.append(dict(variant=k, model=mm, delta_sd=dsd, auc=auc, rdm_sd=sd, rho_other_model=spearmanr(x, other).statistic,
                         rho_bv=spearmanr(x, bv[mm]).statistic, rho_text=spearmanr(x, tx).statistic,
                         rho_main_things=spearmanr(x, th0[mm]).statistic))
    for mm in ("clip", "dinov3"):
        dsd, auc, sd = stats(bv[mm])
        rows.append(dict(variant=f"BabyView (main, {mm})", model=mm, delta_sd=dsd, auc=auc, rdm_sd=sd, rho_bv=1.0,
                         rho_text=spearmanr(bv[mm], tx).statistic))
    T = pd.DataFrame(rows); RES.mkdir(parents=True, exist_ok=True); T.to_csv(RES / "things_readout_check.csv", index=False)
    pd.set_option("display.width", 220); print(T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        embed(a.device, a.batch)
    analyze()
