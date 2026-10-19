#!/usr/bin/env python3
"""Pool 868-hour and 2025.2 BabyView category embeddings (CLIP, DINOv3).

Pooled-crop mean == crop-count-weighted mean of the two runs' raw category means
(each is a mean of L2-normalized crops). Then z-score across categories per set.
valid85 = pooled valid129 means restricted to included_categories_valid85.txt.
Run from analysis/manuscript-2026/. Writes exemplar_set_embeddings/combined/{valid129,valid85}/.
"""
import json, shutil, subprocess, numpy as np, pandas as pd
from pathlib import Path
E = Path('exemplar_set_embeddings'); OUT = E / 'combined'
p = [l for l in subprocess.check_output(['git','show','--name-only','--format=','0c7edcb'],text=True).split() if l.endswith('valid129/exemplar_embedding_run.json')][0]
j868 = json.loads(subprocess.check_output(['git','show',f'0c7edcb:{p}'],text=True, cwd='../..'))
j25 = json.load(open(E/'2025.2/valid129/exemplar_embedding_run.json'))
def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-10)
def load(f):
    df = pd.read_csv(f); df['category'] = df.category.str.strip().str.lower(); return df.set_index('category')
cats85 = [l.strip().lower() for l in open('../../data/included_categories_valid85.txt') if l.strip()]
for m, diag in [('clip','diagnostics_clip'),('dinov3','diagnostics_dinov3')]:
    a = load(E/f'valid129/bv_{m}_exemplar_avg_raw_within_valid129.csv')
    b = load(E/f'2025.2/valid129/bv_{m}_exemplar_avg_raw_within_valid129.csv').loc[a.index]
    n1 = pd.Series({k: v['found'] for k, v in j868[diag]['n_paths'].items()}).loc[a.index].astype(float)
    n2 = pd.Series(j25['models'][m]['n_crops']).loc[a.index].astype(float)
    A, B = a.to_numpy(), b.to_numpy()
    cos = (A*B).sum(1) / np.linalg.norm(A,axis=1) / np.linalg.norm(B,axis=1)
    print(f'[{m}] 868h crops={int(n1.sum()):,} 2025.2 crops={int(n2.sum()):,} | per-category cos(868h mean, 2025.2 mean): mean={cos.mean():.3f} min={cos.min():.3f} ({a.index[cos.argmin()]})')
    w = (n1 / (n1 + n2)).to_numpy()[:, None]
    C = w * A + (1 - w) * B
    print(f'   868h share of crops per category: median={np.median(w):.2f} range=[{w.min():.2f},{w.max():.2f}]')
    for cs, idx in [('valid129', list(a.index)), ('valid85', [c for c in a.index if c in cats85])]:
        d = OUT / cs; d.mkdir(parents=True, exist_ok=True)
        Ci = C[[list(a.index).index(c) for c in idx]]
        df = pd.DataFrame(zs(Ci), columns=a.columns); df.insert(0, 'category', idx)
        df.to_csv(d/f'bv_{m}_exemplar_avg_zscore_within_{cs}.csv', index=False)
        t = d/f'things_{m}_exemplar_avg_zscore_within_{cs}.csv'
        shutil.copyfile(E/f'2025.2/{cs}/things_{m}_exemplar_avg_zscore_within_{cs}.csv', t)
        print(f'   wrote {cs}: {len(idx)} cats')
