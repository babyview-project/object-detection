#!/usr/bin/env python3
"""Superordinate domains for the open categories from the THINGS high-level categories (Hebart et al., 2019; the 27
manually curated categories, category27_manual.tsv), filled in from WordNet where THINGS gives no label.

Per category (Tier 1 list; the core list is a subset), in order:
  1 exact   the THINGS concept with the category's own name (map_things.csv, hypernym_distance 0) has a label
  2 mapped  majority label over the THINGS concepts mapped to the category (map_things.csv)
  3 wordnet walk up the category's WordNet hypernym path (sense_choices.csv; else the exact THINGS concept's synset);
            at the first ancestor with >= MIN_DESC labeled THINGS concepts below it, take their majority label if it
            holds >= MIN_SHARE of them
THINGS labels overlap; nested labels fold into their parent (bird -> animal; fruit, vegetable, dessert -> food;
clothing accessory -> clothing; kitchen appliance -> electronic device; kitchen tool -> tool), and remaining ties
break by PRECEDENCE. Rule 3 is validated by leave-one-out on the labeled THINGS concepts.

Output: data/category_domains_things27.csv (category, domain_things27, source, detail, cdi_domain)
  python build_things27_domains.py
"""
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
T = Path("/data2/dataset/babyview/related_data/things")
FOLD = {"bird": "animal", "fruit": "food", "vegetable": "food", "dessert": "food", "clothing accessory": "clothing",
        "kitchen appliance": "electronic device", "kitchen tool": "tool"}
PRECEDENCE = ["animal", "plant", "body part", "food", "drink", "vehicle", "part of car", "clothing", "furniture",
              "musical instrument", "toy", "sports equipment", "weapon", "medical equipment", "electronic device",
              "office supply", "home decor", "tool", "container", "insect"]
PRECEDENCE = ["insect"] + [p for p in PRECEDENCE if p != "insect"]
MIN_DESC, MIN_SHARE = 3, 0.5
MIN_CORE = 5


def fold(labels):
    return {FOLD.get(l, l) for l in labels}


def pick(counter):
    """Most frequent folded label; ties by precedence."""
    if not counter:
        return None
    top = max(counter.values())
    return min((l for l, n in counter.items() if n == top), key=lambda l: PRECEDENCE.index(l) if l in PRECEDENCE else 99)


def main():
    meta = pd.read_csv(T / "concepts-metadata_things.tsv", sep="\t")
    c27 = pd.read_csv(T / "category27_manual.tsv", sep="\t"); c27.index = meta.uniqueID
    lab = {u: fold([c for c in c27.columns if c27.loc[u, c] == 1]) for u in c27.index}
    syn = {}
    for u, k, w in zip(meta.uniqueID, meta["Wordnet ID2"], meta["WordNet ID"]):
        try:                                    # sense key first (robust to WordNet version), offset as fallback
            syn[u] = wn.lemma_from_key(str(k)).synset()
        except Exception:
            try:
                syn[u] = wn.synset_from_pos_and_offset("n", int(str(w).lstrip("n")))
            except Exception:
                pass
    syn = {u: s for u, s in syn.items() if s is not None}
    anc = {u: {a for p in s.hypernym_paths() for a in p} for u, s in syn.items()}
    labeled = [u for u in syn if lab.get(u)]

    def by_wordnet(s, exclude=None):
        # the synset itself (labeled concepts below it), then its ancestors upward along the longest path
        path = max(s.hypernym_paths(), key=len)[::-1]
        for a in path:
            desc = [u for u in labeled if u != exclude and a in anc[u]]
            if len(desc) >= MIN_DESC:
                cnt = Counter(l for u in desc for l in lab[u])
                l = pick(cnt)
                if cnt[l] / len(desc) >= MIN_SHARE:
                    return l, f"{a.name()} ({cnt[l]}/{len(desc)} labeled THINGS concepts)"
                return None, f"{a.name()}: no majority ({cnt.most_common(2)})"
        return None, "no ancestor"

    # leave-one-out validation of rule 3 on labeled THINGS concepts with a single folded label
    ok = n = 0
    for u in labeled:
        if len(lab[u]) != 1:
            continue
        l, _ = by_wordnet(syn[u], exclude=u)
        n += 1; ok += l in lab[u]
    print(f"WordNet rule, leave-one-out on {n} single-label THINGS concepts: {ok / n:.1%} correct")

    tier1 = [l.strip() for l in open(DATA / "included_categories_qwen_open_tier1.txt") if l.strip()]
    core = {l.strip() for l in open(DATA / "included_categories_qwen_core.txt") if l.strip()}
    mt = pd.read_csv(HERE / "mapping" / "map_things.csv")
    sc = pd.read_csv(HERE / "mapping" / "sense_choices.csv").set_index("category").synset
    cdi = pd.read_csv(HERE / "out" / "private" / "open_renamed" / "categories.csv", index_col=0).domain
    by_name = {}
    for u, w in zip(meta.uniqueID, meta.Word):
        by_name.setdefault(str(w).lower(), []).append(u)
    anc_tab = pd.read_csv(HERE / "mapping" / "tier1_anchors.csv").set_index("category").anchors.fillna("")
    anchor = {}
    for c, a in anc_tab.items():
        try:
            anchor[c] = wn.synset(a.split(";")[0]) if a else None
        except Exception:
            anchor[c] = None
    rows = []
    for c in tier1:
        g = mt[mt.category == c]
        ex = g[g.hypernym_distance == 0].id.tolist()
        r = dict(category=c, core=c in core, cdi_domain=cdi.get(c, ""))
        exl = Counter(l for u in ex for l in lab.get(u, ()))
        mp = Counter(l for u in g.id for l in lab.get(u, ()))
        # THINGS concepts with the category's own name that the exclusive mapping sent elsewhere (bowl, lamp, ...):
        # accepted when the concept's sense is the category's anchor or in the same WordNet lexical file
        nm = [u for u in by_name.get(c, []) if lab.get(u)]
        a0 = anchor.get(c)
        nm = [u for u in nm if u in syn and (a0 is None and len(nm) == 1 or a0 is not None and
                                              (syn[u] == a0 or syn[u].lexname() == a0.lexname()))]
        nml = Counter(l for u in nm for l in lab[u])
        if cdi.get(c, "") == "people":
            r.update(domain_things27=None, source="person", detail="people are not a domain")
        elif exl:
            r.update(domain_things27=pick(exl), source="exact", detail=",".join(ex))
        elif nml:
            r.update(domain_things27=pick(nml), source="name", detail=",".join(nm))
        elif mp:
            r.update(domain_things27=pick(mp), source="mapped", detail=f"{dict(mp)}")
        else:
            s = sc.get(c)
            s = wn.synset(s) if isinstance(s, str) else (syn.get(ex[0]) if ex and ex[0] in syn else None)
            if s is None:
                r.update(domain_things27=None, source="none", detail="no WordNet sense")
            else:
                l, d = by_wordnet(s)
                r.update(domain_things27=l, source="wordnet" if l else "none", detail=f"{s.name()} -> {d}")
        rows.append(r)
    out = pd.DataFrame(rows)
    # hand labels override the rules: the raters' agreed label (column "final" in mapping/domain_handmap_things27.csv)
    # when filled, else the provisional labels (mapping/domain_handmap_things27_provisional.csv); "none" = no group
    hm = pd.read_csv(HERE / "mapping" / "domain_handmap_things27.csv")
    final = hm.set_index("category")["final"].dropna() if "final" in hm else pd.Series(dtype=str)
    prov = pd.read_csv(HERE / "mapping" / "domain_handmap_things27_provisional.csv").set_index("category").label
    hand, how = (final, "hand (raters)") if len(final) else (prov, "hand (provisional)")
    for c, l in hand.items():
        i = out.index[out.category == c]
        out.loc[i, ["domain_things27", "source"]] = [None if l == "none" else l, how]
    # display domains: THINGS groups with >= MIN_CORE core categories; the rest "other" (never a domain in clustering)
    size = out[out.core].domain_things27.value_counts()
    keep = set(size[size >= MIN_CORE].index)
    out["domain_display"] = [d if d in keep else "other" for d in out.domain_things27]
    out["domain_analysis"] = [d if d in keep else f"other:{c}" for d, c in zip(out.domain_things27, out.category)]
    out.loc[out.source == "person", ["domain_display", "domain_analysis"]] = ["people", "people"]
    print(f"display domains (>= {MIN_CORE} core categories):", size[size >= MIN_CORE].to_dict())
    out.to_csv(DATA / "category_domains_things27.csv", index=False)
    k = out[out.core]
    print(k.source.value_counts().to_dict())
    print(k.domain_things27.value_counts(dropna=False).to_dict())
    print(k[k.source.isin(["wordnet", "none"])][["category", "domain_things27", "detail"]].to_string(index=False))
    print(k[k.source == "mapped"][["category", "domain_things27", "detail"]].to_string(index=False))
    print("tier1:", out.source.value_counts().to_dict())


if __name__ == "__main__":
    main()
