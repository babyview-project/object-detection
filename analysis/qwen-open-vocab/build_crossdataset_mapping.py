#!/usr/bin/env python3
"""Map the Tier 1 open categories onto THINGS, LVIS and ImageNet (and name the YOLOE / VEDB correspondence).

Same method as analysis/comparison-datasets (notebook 01): each category gets WordNet anchor sense(s); a dataset item
(THINGS concept, LVIS category, ImageNet class) maps to a category when an anchor is the item's synset or one of its
hypernyms; each item counts only for its closest category (fewest hypernym steps; tie -> the category with more BV
boxes), so no exemplar is shared between categories.

Anchors:
  - CDI words: the hand-checked senses, added anchors and exclusions of comparison-datasets (anchors_lvis.json,
    which layers LVIS changes on the ImageNet anchors).
  - every other category: the sense Qwen chose (sense_pick.py: Qwen3.8-27B reads the category, its domain and example
    labels, and picks among all the word's WordNet senses or "none fits"); a head-noun sense for a compound WordNet lacks
    is not used for pooling (name matches only). The rule-based pick below is kept as rule_anchor and disagreements are
    flagged for review. Rule (also the fallback when sense_choices.csv is absent): among the word's concrete noun senses (artifact, animal, food, plant, body, object, person,
    substance; no named individuals; no person senses unless the domain is people), the first whose lexical file fits
    the category's domain (e.g. household -> artifact), else the first concrete one; senses whose main name is the word come
    before senses that only list it as a synonym (poster is not a post horse). Flagged for review when the chosen
    sense is not the word's first sense, when no sense fits the domain, or when no concrete sense exists.
VEDB needs no mapping (its Qwen labels go through the same normalization: build_open_means.basic_of). YOLOE classes
correspond by name (valid129 class == Tier 1 category after the same synonyms).

Outputs (mapping/):
  tier1_anchors.csv        category, domain, anchors, origin, definition, review flags
  map_things.csv, map_lvis.csv, map_imagenet.csv     item -> category, anchor, hypernym distance, exemplar count
  coverage.csv             per category: items and exemplars per dataset, YOLOE class, review flags
  review.csv               categories to check (flags) with their anchors and largest mapped items

  python build_crossdataset_mapping.py        # analysis env (nltk WordNet)
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

HERE = Path(__file__).resolve().parent
OD = HERE.parents[1]
CD = OD / "analysis" / "comparison-datasets" / "outputs" / "mapping"
THINGS = Path("/data2/dataset/babyview/related_data/things")
LVIS_JSON = Path("/data2/j7yang/lvis/annotations/lvis_v1_val.json")
OUT = HERE / "mapping"
CONCRETE = {"noun.artifact", "noun.animal", "noun.food", "noun.plant", "noun.body", "noun.object", "noun.person",
            "noun.substance"}
DOMAIN_LEX = {"animals": {"noun.animal"}, "body_parts": {"noun.body"}, "food_drink": {"noun.food", "noun.plant"},
              "people": {"noun.person"}, "clothing": {"noun.artifact"}, "furniture_rooms": {"noun.artifact"},
              "household": {"noun.artifact", "noun.object", "noun.substance"}, "toys": {"noun.artifact"},
              "vehicles": {"noun.artifact"}, "outside": {"noun.plant", "noun.object", "noun.artifact", "noun.substance"}}


# Fixes after the 2026-11-14 audit of exact-name THINGS concepts (core categories whose own THINGS concept was unmapped):
# the toy senses of ball and balloon (THINGS "ball" = ball.n.06, a plaything; "balloon" = balloon.n.02, the rubber
# balloon; the hand-checked balloon anchor was the hot-air balloon), and THINGS "plant" (plant.n.02, a potted plant).
EXTRA_ANCHORS = {"ball": ["ball.n.06"], "balloon": ["balloon.n.02"]}
EXTRA_EXCLUDE = {"balloon": ["hot-air_balloon.n.01"], "ball": ["snowball.n.04"]}
EXTRA_ITEMS = {"things": {"plant": "plant"}}        # item id -> category, if the item is otherwise unmapped


def anchors_for(cats: pd.DataFrame):
    cd = json.load(open(CD / "anchors_lvis.json"))
    cd_anchor, cd_excl = cd["anchors"], cd.get("exclude", {})
    qp = OUT / "sense_choices.csv"
    qsense = pd.read_csv(qp).fillna("").set_index("category") if qp.exists() else pd.DataFrame()
    rows = []
    for c, r in cats.iterrows():
        senses = []
        for lemma in dict.fromkeys([c.replace(" ", "_"), c.replace(" ", ""), c.replace(" ", "-")]):   # high chair -> highchair
            senses += [s for s in wn.synsets(lemma, "n") if lemma.lower() in [l.lower() for l in s.lemma_names()] and s not in senses]
        flags = []
        if c in cd_anchor and cd_anchor[c]:
            anc, origin = cd_anchor[c], "comparison-datasets (hand-checked)"
        else:
            conc = [s for s in senses if s.lexname() in CONCRETE and not s.instance_hypernyms()]   # no named individuals
            primary = {c.replace(" ", "_").lower(), c.replace(" ", "").lower(), c.replace(" ", "-").lower()}
            conc = sorted(conc, key=lambda s: s.lemma_names()[0].lower() not in primary)   # word as the sense's main name first
            pref = DOMAIN_LEX.get(r.domain, {"noun.artifact", "noun.object"})
            if r.domain != "people":
                conc_np = [s for s in conc if s.lexname() != "noun.person"] or conc
            else:
                conc_np = conc
            fit = [s for s in conc_np if s.lexname() in pref]
            if conc_np:
                pick = (fit or conc_np)[0]
                anc, origin = [pick.name()], "domain-fitting concrete sense" if fit else "first concrete sense"
                if pick != senses[0]:
                    flags.append("not first sense")
                if len({s.lexname() for s in conc_np}) > 1 and not fit:
                    flags.append("concrete senses in several lexical files")
                if not fit and r.domain in DOMAIN_LEX:
                    flags.append(f"lexical file {pick.lexname()} vs domain {r.domain}")
            elif senses:
                anc, origin = [senses[0].name()], "first sense (not concrete)"
                flags.append("no concrete sense")
            else:
                anc, origin = [], "none"
                flags.append("not in WordNet")
        rule = ";".join(anc)
        if origin != "comparison-datasets (hand-checked)" and c in qsense.index:   # Qwen's sense choice overrides the rule
            q = qsense.loc[c]
            flags = [f for f in flags if f == "not in WordNet"]
            if q.approximate and q.synset:
                anc, origin = [], "none (Qwen: head-noun sense only)"
                flags.append(f"approximate: {q.synset} (head noun; name matches only)")
            elif q.synset:
                anc, origin = [q.synset], "Qwen-chosen sense"
                if q.synset not in rule.split(";"):
                    flags.append(f"Qwen and rule disagree (rule: {rule or 'none'})")
            else:
                anc, origin = [], "none (Qwen: no sense fits)"
                if rule:
                    flags.append(f"Qwen: no sense fits (rule: {rule})")
        if c in EXTRA_ANCHORS:
            anc = list(anc) + [x for x in EXTRA_ANCHORS[c] if x not in anc]
            origin += " + audit fix"
        rows.append({"category": c, "domain": r.domain, "boxes": r.boxes, "anchors": ";".join(anc), "origin": origin, "rule_anchor": rule,
                     "definition": wn.synset(anc[0]).definition() if anc else "",
                     "exclude": ";".join(list(cd_excl.get(c, [])) + EXTRA_EXCLUDE.get(c, [])), "review_flags": "; ".join(flags)})
    return pd.DataFrame(rows)


def hypernym_distances(syn) -> dict[str, int]:
    """Every hypernym (incl. instance hypernyms) of a synset with its fewest-steps distance (0 = itself)."""
    d, q = {syn.name(): 0}, deque([syn])
    while q:
        s = q.popleft()
        for h in s.hypernyms() + s.instance_hypernyms():
            if h.name() not in d:
                d[h.name()] = d[s.name()] + 1
                q.append(h)
    return d


def map_items(items: pd.DataFrame, anc: pd.DataFrame) -> pd.DataFrame:
    """items: id, name, synset, n -> closest category (exclusive)."""
    a2c, excl = {}, {}
    for r in anc.itertuples():
        for a in filter(None, r.anchors.split(";")):
            a2c.setdefault(a, []).append(r.category)
        for e in filter(None, r.exclude.split(";")):
            excl.setdefault(r.category, set()).add(e)
    boxes = anc.set_index("category").boxes
    rows = []
    for it in items.itertuples():
        try:
            syn = wn.synset(it.synset)
        except Exception:
            continue
        best = None
        for h, dist in hypernym_distances(syn).items():
            for c in a2c.get(h, []):
                if it.synset in excl.get(c, set()) or it.name in excl.get(c, set()):
                    continue
                key = (dist, -boxes.get(c, 0))
                if best is None or key < best[0]:
                    best = (key, c, h, dist)
        if best:
            rows.append({"id": it.id, "name": it.name, "synset": it.synset, "n_exemplars": it.n, "category": best[1],
                         "anchor": best[2], "hypernym_distance": best[3]})
    return pd.DataFrame(rows)


def name_key(x: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"_\([^)]*\)", "", str(x)).replace("_", " ").replace("-", " ")).strip().lower()


def name_fallback(items: pd.DataFrame, m: pd.DataFrame, anc: pd.DataFrame, synonyms: dict | None = None) -> pd.DataFrame:
    """Categories with no WordNet anchor: an item whose name (or listed synonym) equals the category, if still unmapped."""
    unanch = set(anc.loc[anc.anchors.fillna("") == "", "category"])
    done = set(m.id) if len(m) else set()
    rows = []
    for it in items.itertuples():
        if it.id in done:
            continue
        keys = {name_key(it.name)} | {name_key(v) for v in (synonyms or {}).get(it.id, [])}
        hit = sorted(keys & unanch)
        if hit:
            rows.append({"id": it.id, "name": it.name, "synset": it.synset, "n_exemplars": it.n, "category": hit[0],
                         "anchor": "name match", "hypernym_distance": 0})
    return pd.concat([m, pd.DataFrame(rows)], ignore_index=True) if rows else m


def items_things():
    t = pd.read_csv(THINGS / "concepts-metadata_things.tsv", sep="\t", usecols=["Word", "uniqueID", "Wordnet ID4"])
    n = pd.read_csv(THINGS / "embeddings" / "things_clip_concept_mean_raw.csv", usecols=["concept", "n_images"]).set_index("concept").n_images
    return pd.DataFrame({"id": t.uniqueID, "name": t.Word.str.lower(), "synset": t["Wordnet ID4"].astype(str).str.strip(),
                         "n": t.uniqueID.map(n).fillna(0).astype(int)})


def items_lvis():
    cats = json.load(open(LVIS_JSON))["categories"]
    df = pd.DataFrame({"id": [c["id"] for c in cats], "name": [c["name"] for c in cats], "synset": [c["synset"] for c in cats],
                       "n": [c.get("instance_count", 0) for c in cats]})   # val counts; train counts in comparison-datasets
    df.attrs["synonyms"] = {c["id"]: c.get("synonyms", []) for c in cats}
    return df


def items_imagenet():
    import numpy as np
    z = np.load(CD / "imagenet_class_embeddings_clip.npz", allow_pickle=True)
    rows = []
    for w, n in zip(z["wnids"], z["counts"]):
        s = wn.synset_from_pos_and_offset("n", int(w[1:]))
        rows.append({"id": w, "name": s.lemma_names()[0].replace("_", " "), "synset": s.name(), "n": int(n)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    cats = pd.read_csv(HERE / "out" / "private" / "open" / "categories.csv", index_col=0)
    anc = anchors_for(cats)
    anc.to_csv(OUT / "tier1_anchors.csv", index=False)
    cov = anc[["category", "domain", "boxes", "origin", "review_flags"]].set_index("category")
    maps = {}
    for name, fn in (("things", items_things), ("lvis", items_lvis), ("imagenet", items_imagenet)):
        items = fn()
        m = map_items(items, anc)
        m = name_fallback(items, m, anc, items.attrs.get("synonyms"))
        add = [i for i in EXTRA_ITEMS.get(name, {}) if i not in set(m.id)]
        if add:
            it = items.set_index("id").loc[add]
            m = pd.concat([m, pd.DataFrame({"id": add, "name": it.name.values, "synset": it.synset.values,
                                            "n_exemplars": it.n.values, "category": [EXTRA_ITEMS[name][i] for i in add],
                                            "anchor": "audit fix", "hypernym_distance": 0})], ignore_index=True)
        m.to_csv(OUT / f"map_{name}.csv", index=False)
        maps[name] = m
        g = m.groupby("category").agg(items=("id", "size"), exemplars=("n_exemplars", "sum"),
                                      exact=("hypernym_distance", lambda x: (x == 0).sum()))
        cov[f"{name}_items"] = g["items"].reindex(cov.index).fillna(0).astype(int)
        cov[f"{name}_exact"] = g["exact"].reindex(cov.index).fillna(0).astype(int)
        cov[f"{name}_exemplars"] = g["exemplars"].reindex(cov.index).fillna(0).astype(int)
        print(f"{name}: {len(m)} items mapped to {m.category.nunique()} categories "
              f"({(m.hypernym_distance == 0).sum()} exact-sense items)")
    valid129 = {l.strip() for l in open(OD / "data" / "included_categories_valid129.txt") if l.strip()}
    cov["yoloe_valid129"] = cov.index.isin(valid129)
    cov.to_csv(OUT / "coverage.csv")
    rev = anc[anc["review_flags"] != ""].copy()
    def gloss(x):
        try:
            return "; ".join(f"{a}: {wn.synset(a).definition()}" for a in x.split(";") if a)
        except Exception:
            return ""
    rev["chosen"] = [gloss(x) for x in rev.anchors]
    rev["rule"] = [gloss(x) for x in rev.rule_anchor]
    for name, m in maps.items():
        top = m.sort_values("n_exemplars", ascending=False).groupby("category").name.apply(lambda s: ", ".join(s.head(4)))
        rev[f"{name}_items"] = rev.category.map(top).fillna("")
    rev["decision"] = ""          # for the reviewer: ok / use rule / other synset name / no match
    cols = ["category", "domain", "boxes", "origin", "chosen", "rule", "review_flags", "things_items", "lvis_items",
            "imagenet_items", "decision"]
    rev.sort_values("boxes", ascending=False)[cols].to_csv(OUT / "review.csv", index=False)
    both = cov[(cov.things_items > 0) & (cov.lvis_items > 0)]
    print(f"{len(cov)} Tier 1 categories; in THINGS {(cov.things_items > 0).sum()}, LVIS {(cov.lvis_items > 0).sum()}, "
          f"ImageNet {(cov.imagenet_items > 0).sum()}, THINGS and LVIS {len(both)}, all three "
          f"{((cov.things_items > 0) & (cov.lvis_items > 0) & (cov.imagenet_items > 0)).sum()}; YOLOE valid129 {cov.yoloe_valid129.sum()}")
    print(f"review sheet: {len(rev)} flagged categories -> {OUT / 'review.csv'}")


if __name__ == "__main__":
    main()
