#!/usr/bin/env python3
"""Basic level, semantic domain and generic flag for Qwen's open categories (text only, Qwen3.8-27B locally).

For every category of a run (--min-frames 1 = all), Qwen reads the WHOLE name and returns the everyday
basic-level name ("christmas tree" -> tree, "red cup" -> cup, "computer mouse" -> computer mouse, "hamster
cage" -> cage), the semantic domain of what the thing is (CDI domains + other), and whether the label is
generic ("toy", "animal", "object"). Final domain: the CDI domain when the basic-level name is a CDI word
listed under one domain (CDI's convention, e.g. teddy bear -> animals), else Qwen's. Also kept for review:
the compound-head rule (longest proper word suffix that is itself a category) and the earlier automatic
domain (qwen_data, head-noun / CLIP based); `flag` marks rows where they disagree with the new answer.

Output (labels only, no frames): results/category_hierarchy_<tag>.csv
  cat, frames, boxes, old_domain, rule_parent, qwen_basic, qwen_domain, qwen_generic, cdi_domain, domain, flag

  python qwen_hierarchy.py --prepare                      # analysis env
  CUDA_VISIBLE_DEVICES=2 python qwen_hierarchy.py         # qwen env
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"

DOMAINS = ["animals", "body_parts", "clothing", "food_drink", "furniture_rooms", "household", "outside", "people",
           "toys", "vehicles", "other"]
PROMPT = """An object label from annotations of a young child's home videos: "{cat}".
Judge the WHOLE name, not just its last word: a "computer mouse" is a device, not an animal; a "hamster cage" is a cage; a "hot dog" is food; "cat litter" is a household thing; a "fish tank" is a container.
Return JSON:
- "basic_level": the everyday basic-level name for this kind of thing, the word a parent would use with a toddler, lowercase. Drop colours, materials and sizes ("red cup" -> "cup") and subtypes ("christmas tree" -> "tree", "sippy cup" -> "cup", "high chair" -> "chair", "water bottle" -> "bottle", "polar bear" -> "bear"). Keep names that are their own kind of thing ("computer mouse" -> "computer mouse", "hamster cage" -> "cage", "bean bag" -> "bean bag", "teddy bear" -> "teddy bear", "pant leg" -> "pants", "hot dog" -> "hot dog"). For a sticker, figurine, logo or picture of something, name what it shows ("cat sticker" -> "cat"). If the label is already basic level, repeat it.
- "domain": what the thing IS, one of: animals (animals, including toy or pictured animals and teddy bears), body_parts, clothing (clothes, shoes, accessories worn), food_drink, furniture_rooms (furniture, rooms and room parts such as wall, floor, door, window, and large appliances), household (household objects: containers, tools, utensils, devices, decorations, pictures, linens, things for pets), outside (plants, nature, sky, outdoor places and things), people (people and human or human-like characters), toys (toys, games, books, play materials), vehicles, other (letters, numbers, shapes, text, logos, icons, anything else).
- "generic": true if the label is a general word that does not say what the thing is ("toy", "animal", "object", "item", "thing", "character", "food", "vehicle"), else false."""
SCHEMA = {"type": "object", "properties": {"basic_level": {"type": "string"},
                                           "domain": {"type": "string", "enum": DOMAINS},
                                           "generic": {"type": "boolean"}},
          "required": ["basic_level", "domain", "generic"], "additionalProperties": False}


def cdi_domain(word: str) -> str | None:
    """CDI domain of a word (spaces removed, as CDI joins compounds), None if not a CDI word or listed under two domains."""
    import qwen_data as qd
    d = pd.read_csv(qd.OD_ROOT / "data" / "cdi_words.csv", usecols=["uni_lemma", "category"])
    d["u"] = d.uni_lemma.astype(str).str.strip().str.lower()
    doms = d.groupby("u").category.unique()
    w = word.replace(" ", "")
    return doms[w][0] if w in doms.index and len(doms[w]) == 1 else None


def categories(tag: str, min_frames: int) -> pd.DataFrame:
    p = PRIV / f"qwen_all_cats_{tag}.csv.gz"          # written by --prepare (analysis env: needs WordNet)
    b = pd.read_csv(p)
    c = b.groupby("cat").agg(frames=("unit_id", "nunique"), boxes=("unit_id", "size"), old_domain=("domain", "first"))
    return c[c.frames >= min_frames].sort_values("frames", ascending=False)


def rule_parent(cat: str, cats: set[str]) -> str:
    w = cat.split()
    for k in range(1, len(w)):
        h = " ".join(w[k:])
        if h in cats:
            return h
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--prepare", action="store_true", help="write the category table (analysis env, needs WordNet)")
    args = ap.parse_args()
    if args.prepare:
        import qwen_data as qd
        o = qd.dedup(qd.objects_table(qd.load_records("detect_open_format", args.tag)))
        o[["unit_id", "cat", "domain"]].to_csv(PRIV / f"qwen_all_cats_{args.tag}.csv.gz", index=False)
        print(f"{o.cat.nunique():,} categories"); return
    c = categories(args.tag, args.min_frames)
    cats = set(c.index)
    c["rule_parent"] = [rule_parent(x, cats) for x in c.index]
    print(f"{len(c)} categories in >= {args.min_frames} frames", flush=True)

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from run_qwen_pilot import make_llm
    llm = make_llm(SimpleNamespace(mtp=0, tp=1, max_model_len=4096, gpu_mem=0.9, max_num_seqs=128, enforce_eager=False))
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    msgs = [[{"role": "user", "content": PROMPT.format(cat=x)}] for x in c.index]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    basic, dom, generic = [], [], []
    for x, o in zip(c.index, outs):
        try:
            d = json.loads(o.outputs[0].text)
            basic.append(str(d["basic_level"]).strip().lower() or x); dom.append(d["domain"]); generic.append(bool(d["generic"]))
        except Exception:
            basic.append(x); dom.append(None); generic.append(None)
    c["qwen_basic"], c["qwen_domain"], c["qwen_generic"] = basic, dom, generic
    c["cdi_domain"] = [cdi_domain(x) for x in c.qwen_basic]
    c["domain"] = c.cdi_domain.fillna(c.qwen_domain)
    c["flag"] = (c.rule_parent != c.qwen_basic) | (c.old_domain != c.domain)
    c.index.name = "cat"
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"category_hierarchy_{args.tag}.csv"
    c.to_csv(out)
    print(f"wrote {out}: {len(c)} categories; basic level differs from the name for {(c.qwen_basic != c.index).sum()}; "
          f"domain changed for {(c.old_domain != c.domain).sum()}; generic {int(c.qwen_generic.fillna(False).sum())}; "
          f"flagged {int(c.flag.sum())} ({int(c[c.frames >= 30].flag.sum())} with >= 30 frames)")


if __name__ == "__main__":
    main()
