# CDI vocabulary lists used in Yang et al.

## Plain lists (one word per line)
- `cdi_concrete_nouns_295.txt` — initial detection vocabulary (N = 295 unique uni_lemmas)
- `included_categories_valid129.txt` — primary analysis set (N = 129)
- `included_categories_valid85.txt` — high-precision robustness subset (N = 85; precision > .60)

## Same lists with CDI semantic domains
- `cdi_concrete_nouns_295_with_semantic.csv`
- `included_categories_valid129_with_semantic.csv`
- `included_categories_valid85_with_semantic.csv`

CSV columns: `category`, `cdi_semantic`

## Construction notes
### 295
Source: American English CDI Words & Sentences (WS) / Wordbank (Marchman et al., 2023).
1. Keep `lexical_category == "nouns"`
2. Exclude multi-word `uni_lemma`s (containing a space)
3. Add `person` (CDI lexical_category "other")
=> 295 unique words

### 129
From the 295 detection vocabulary, after YOLOE + CLIP filtering (similarity > .27):
exclude `person` and `picture`, and categories with fewer than 100 exemplars.

### 85
Subset of the 129 categories with crowdsourced precision > .60 at both
category and exemplar levels (see paper SI).
