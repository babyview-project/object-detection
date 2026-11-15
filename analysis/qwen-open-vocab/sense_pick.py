#!/usr/bin/env python3
"""Qwen picks the WordNet sense of each Tier 1 category (for build_crossdataset_mapping.py).

  python sense_pick.py prepare     # analysis env (nltk): mapping/sense_candidates.json
  CUDA_VISIBLE_DEVICES=7 python sense_pick.py run      # qwen env (vLLM, text only): mapping/sense_choices.csv

prepare: for each category, every WordNet noun sense whose lemma is the category (spelling variants: high chair ->
highchair), of any lexical file, with its definition; if the word is not in WordNet, the senses of its head noun,
marked approximate (light switch -> switch). Context: the category's domain and its most frequent Qwen labels.
run: Qwen3.8-27B (temperature 0, thinking off, JSON with one integer) chooses the option that is the everyday object
in a young child's home videos, or 0 if none fits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "mapping"
STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
PROMPT = """A category of objects seen in young children's head-camera videos at home: "{cat}" (domain: {domain}; \
labels it groups include: {examples}).
Which WordNet sense below is this everyday object? Choose the option number, or 0 if none of them is this object.
{options}
Return JSON {{"choice": <number>}}."""


def prepare():
    from nltk.corpus import wordnet as wn
    sys.path.insert(0, str(HERE))
    from build_open_means import basic_of
    cats = pd.read_csv(HERE / "out" / "private" / "open" / "categories.csv", index_col=0)
    ix = pd.read_csv(STORE / "boxes_index.csv.gz", usecols=["label"])
    vc = ix.label.astype(str).value_counts()
    b = basic_of(vc.index)
    ex = pd.DataFrame({"label": vc.index, "n": vc.values, "cat": [b[l] for l in vc.index]})
    ex = ex[ex.cat.isin(cats.index)].sort_values("n", ascending=False).groupby("cat").label.apply(lambda s: list(s.head(5)))
    out = {}
    for c, r in cats.iterrows():
        senses, approx = [], False
        for lemma in dict.fromkeys([c.replace(" ", "_"), c.replace(" ", ""), c.replace(" ", "-")]):
            senses += [s for s in wn.synsets(lemma, "n") if lemma.lower() in [l.lower() for l in s.lemma_names()] and s not in senses]
        if not senses and " " in c:
            head = c.split()[-1]
            senses, approx = [s for s in wn.synsets(head, "n") if head in s.lemma_names()], True
        out[c] = {"domain": r.domain, "examples": ex.get(c, [c]), "approximate": approx,
                  "options": [{"synset": s.name(), "lex": s.lexname(), "definition": s.definition(),
                               "lemmas": s.lemma_names()[:4]} for s in senses[:12]]}
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / "sense_candidates.json", "w"), indent=1)
    print(f"{len(out)} categories; {sum(1 for v in out.values() if not v['options'])} with no candidate sense; "
          f"{sum(v['approximate'] for v in out.values())} via the head noun")


def run():
    import os
    from types import SimpleNamespace
    sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from run_qwen_pilot import make_llm
    cand = json.load(open(OUT / "sense_candidates.json"))
    todo = {c: v for c, v in cand.items() if v["options"]}
    msgs, keys = [], []
    for c, v in todo.items():
        opts = "\n".join(f"{i + 1}) {o['synset']} [{', '.join(o['lemmas'])}]: {o['definition']}" for i, o in enumerate(v["options"]))
        msgs.append([{"role": "user", "content": PROMPT.format(cat=c, domain=v["domain"], examples=", ".join(v["examples"]),
                                                                options=opts)}])
        keys.append(c)
    llm = make_llm(SimpleNamespace(mtp=0, tp=1, max_model_len=4096, gpu_mem=0.9, max_num_seqs=128, enforce_eager=False))
    sps = [SamplingParams(temperature=0.0, max_tokens=16, seed=0, structured_outputs=StructuredOutputsParams(json={
        "type": "object", "properties": {"choice": {"type": "integer", "enum": list(range(len(todo[c]["options"]) + 1))}},
        "required": ["choice"], "additionalProperties": False})) for c in keys]
    res = llm.chat(msgs, sps, chat_template_kwargs={"enable_thinking": False})
    rows = []
    for c, r in zip(keys, res):
        try:
            k = int(json.loads(r.outputs[0].text)["choice"])
        except Exception:
            k = -1
        o = todo[c]["options"][k - 1] if k > 0 else None
        rows.append({"category": c, "choice": k, "synset": o["synset"] if o else "", "definition": o["definition"] if o else "",
                     "approximate": todo[c]["approximate"], "n_options": len(todo[c]["options"])})
    for c, v in cand.items():
        if not v["options"]:
            rows.append({"category": c, "choice": 0, "synset": "", "definition": "", "approximate": False, "n_options": 0})
    pd.DataFrame(rows).to_csv(OUT / "sense_choices.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"{len(d)} categories; sense chosen for {(d.choice > 0).sum()}, none fits {(d.choice == 0).sum()}, failed {(d.choice < 0).sum()}")


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
