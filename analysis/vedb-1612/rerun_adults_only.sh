#!/usr/bin/env bash
# Rerun every VEDB-dependent analysis with VEDB restricted to adult observers (>= 18 y; 709 of 717 sessions,
# 50 of 56 participants; vedb_adult_sessions.txt, build_adult_stores.py). November 2026.
#
# Every VEDB loader reads an "_adults" view of its store (build_adult_stores.py) or, for the BV/VEDB embedding sample
# and its SAM 3 outputs, the filtered original (qwen-open-vocab/filter_bv_vedb_sample_adults.py). Nothing upstream of
# the stores (frames, YOLOE, CLIP/DINOv3, Qwen) is recomputed.
#
# Tracks (A and B in parallel, C after both):
#   A  YOLOE + CLIP pipeline (SI): vedb-1612 Stage 0 and BV-VEDB scripts, comparison-datasets notebooks 05, 07-12, 14-16
#   B  Qwen open vocabulary (main text; renamed labels): VEDB builds, then the CDI and THINGS-27 domain chains
#   C  figures: talk (UNR) figures and decks, manuscript figures (CDI and THINGS-27)
#
# Core set: excluding minors leaves 'baseball' with 28 adult VEDB boxes (< 30), so the cross-dataset core drops from 183 to
# 182 categories (user decision 2026-11-14: apply the rule). data/included_categories_qwen_core.txt was edited to 182 and
# everything derived from it is rebuilt here (THINGS-27 domains: 'sports equipment' falls to 4 core categories < 5 and
# becomes singletons; BabyView child-month means). The real-objects core (164) is unchanged.
#
#   nohup bash rerun_adults_only.sh [all|resume] > /dev/null 2>&1 &   # log: qwen-open-vocab/out/private/vedb_adults/chain.log
#   resume: skip track A and the VEDB builds (first run, 2026-11-14), wait for the sample filter, continue from the core check
set -u
R=/home/j7yang/babyview-projects/vss2026/object-detection/analysis
QOV=$R/qwen-open-vocab; CMP=$R/comparison-datasets; VD=$R/vedb-1612; TALK=$R/talk-figures/unr-2026
E=/data/j7yang/anaconda3/envs/vislearnlabpy/bin
BASEPY=/data/j7yang/anaconda3/bin/python                      # has pyarrow (development_within_child.py)
PPTX=${PPTX:-/tmp/claude-1240/-home-j7yang-babyview-projects-vss2026-object-detection/97646d8b-d29f-407e-a226-0ccda6bde0dd/scratchpad/pptxlib}
BK=/data2/j7yang/object-detection-offload/backups/pre_vedb_adults_20261115   # pre-exclusion snapshot
CACHE_BK=$BK/comparison-datasets-vedb-caches
LOGD=$QOV/out/private/vedb_adults; mkdir -p $LOGD; L=$LOGD/chain.log
export PATH=$E:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export QOV_LABELS=renamed

step() { echo "$(date -Is) $*" >> $L; }
run() {   # run <log name> <command...>
  local log=$1; shift; step "start [$log] $*"
  if nice -n 19 "$@" >> $LOGD/$log.log 2>&1; then step "ok    [$log] $*"; else step "FAILED [$log] $*"; return 1; fi
}
nbx() {   # Qwen notebook: nbx <notebook> <executed name> <log>
  run $3 jupyter nbconvert --to notebook --execute $1.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \
      --ExecutePreprocessor.timeout=28800 --output-dir out/private --output $2
}
cmpnb() { # comparison-datasets notebook with a parameter line replaced: cmpnb <nn> <regex of line> <new line> [inplace]
  local src; src=$(ls $CMP/notebooks/$1_*.ipynb)
  local dst=$src; [ "${4:-}" = inplace ] || dst=$CMP/notebooks/_tmp_$1_$RANDOM.ipynb
  python - "$src" "$dst" "$2" "$3" <<'EOF'
import json, re, sys
src, dst, pat, new = sys.argv[1:]
nb = json.load(open(src)); n = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        lines = "".join(c["source"]).split("\n")
        for i, l in enumerate(lines):
            if re.match(pat, l) and n == 0:
                lines[i] = new; n += 1
        c["source"] = "\n".join(lines)
assert n == 1, f"parameter line {pat!r} not found in {src}"
json.dump(nb, open(dst, "w"), indent=1)
EOF
  run cmp_$1 jupyter nbconvert --to notebook --execute --inplace "$dst" --ExecutePreprocessor.kernel_name=vislearnlabpy \
      --ExecutePreprocessor.timeout=28800
  local rc=$?; [ "$dst" = "$src" ] || rm -f "$dst"; return $rc
}

track_a() {
  step "== A: YOLOE pipeline (SI)"
  cd $VD
  run yoloe_vedb python -u run_analyses.py || { step "A stopped: Stage 0 failed"; return; }
  for s in bv_vedb_comparison.py build_bv_vedb_cosine_ranking.py plot_bv_vedb_rdms.py plot_bv_vedb_category_cosine.py \
           plot_bv_vedb_category_montage.py plot_bv_vedb_cluster_delta_bars.py plot_rdms.py; do
    run yoloe_vedb python -u $s
  done
  # VEDB caches of notebooks 11-16 are rebuilt (BV / LVIS / THINGS caches reused); old ones kept in the snapshot
  mkdir -p $CACHE_BK
  for f in $CMP/outputs/*/cache/*vedb* $CMP/outputs/context_split/cache/cross_half_rdms.npz; do
    [ -e "$f" ] || continue
    d=$CACHE_BK/$(basename $(dirname $(dirname $f))); mkdir -p $d; mv "$f" $d/
  done
  step "moved VEDB caches -> $CACHE_BK"
  cmpnb 05 '^DATASET = ' 'DATASET = "imagenet"'
  cmpnb 05 '^DATASET = ' 'DATASET = "lvis"' inplace
  cmpnb 07 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "imagenet"]'
  cmpnb 07 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "lvis"]' inplace
  cmpnb 08 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "things", "imagenet"]'
  cmpnb 08 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "things", "lvis"]'
  cmpnb 08 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "things", "imagenet", "lvis"]' inplace
  for s in '"bv", "vedb", "imagenet"' '"bv", "vedb", "lvis"' '"bv", "vedb", "things", "imagenet"' \
           '"bv", "vedb", "things", "imagenet", "lvis"'; do
    cmpnb 09 '^DATASETS = \[' "DATASETS = [$s]"
  done
  cmpnb 09 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "things", "lvis"]' inplace
  cmpnb 10 '^DATASETS = \[' 'DATASETS = ["bv", "vedb", "imagenet", "lvis"]' inplace
  for n in 11 12 14 15 16; do cmpnb $n '^RECOMPUTE_CACHE = ' 'RECOMPUTE_CACHE = False' inplace; done
  step "== A done"
}

domain_chain() {   # CDI or THINGS-27 analyses after the builds; $1 = cdi | things27
  local D=$1 T=""; [ $D = things27 ] && T=_things27
  export QOV_DOMAINS=$D
  cd $QOV
  ( export QOV_CATS=core
    nbx crossdataset_analysis crossdataset_analysis_executed_renamed${T}_core.ipynb ${D}_cross
    run ${D}_cross python -u ego_vs_curated.py
    run ${D}_cross python -u build_blur_control.py analyze
    run ${D}_cross python -u analyze_background_control.py
    run ${D}_cross python -u things_readout_check.py --analyze-only  # CDI: things_readout/things_variants.npz was
                                         # replaced by the 7,384-image re-embedding in things_readout_renamed_things27
    nbx bv_vedb_analysis bv_vedb_analysis_executed_renamed${T}_core.ipynb ${D}_cross ) &
  ( export QOV_CATS=core
    run ${D}_controls python -u controls_analysis.py --sections cooc geom size count perm prev things ) &
  ( nbx crossdataset_analysis crossdataset_analysis_executed_renamed${T}.ipynb ${D}_tier1
    nbx bv_vedb_analysis bv_vedb_analysis_executed_renamed${T}.ipynb ${D}_tier1
    if [ $D = cdi ]; then
      run ${D}_tier1 python -u controls_analysis.py --sections prev          # non-core prevalence (fig_qwen_talk)
      python build_cooccurrence_nb.py >> $LOGD/${D}_tier1.log 2>&1
      nbx cooccurrence_analysis cooccurrence_analysis_executed_renamed.ipynb ${D}_tier1
    fi ) &
  ( run ${D}_dev python -u scene_within_child_density.py
    run ${D}_dev $BASEPY -u development_within_child.py
    [ $D = cdi ] && run ${D}_dev $BASEPY -u development_size_by_domain.py ) &   # CDI-only analysis; reads parquet
  wait
  export QOV_CATS=core
  run ${D}_controls python -u controls_figures.py
  unset QOV_CATS
  step "== B $D chain done"
}

track_b() {
  step "== B: Qwen builds (VEDB adults)"
  cd $QOV; unset QOV_DOMAINS
  run b_open python -u build_open_means.py --dataset vedb --workers 8 || { step "B stopped: open means failed"; return; }
  ( run b_bvvedb python -u build_bv_vedb_data.py --workers 8 && \
    run b_bvvedb python -u filter_bv_vedb_sample_adults.py --src $BK/qwen-open-vocab/out/private/bv_vedb_renamed ) &
  run b_controls python -u build_controls_data.py --workers 16 &
  run b_cooc python -u build_cooccurrence_data.py &
  run b_format python -u build_format_tables.py vedb &          # steps are named; no argument = no-op
  run b_cross python -u build_crossdataset_tables.py &
  wait
  grep -q "FAILED \[b_" $L && { step "B stopped: a build failed"; return; }
  track_b_tail
}

track_b_tail() {
  cd $QOV; unset QOV_DOMAINS
  # the core set (data/included_categories_qwen_core.txt) = categories with exemplars in all five datasets;
  # stop if the adult VEDB tables change it, since the frozen list would then disagree with the live one
  python - >> $LOGD/b_cross.log 2>&1 <<'EOF' || { step "B stopped: core set changed (see b_cross.log)"; return; }
import numpy as np
from pathlib import Path
z = np.load("out/crossdataset_renamed/tables_clip.npz", allow_pickle=True)
sets = [set(z[f"{d}__categories"]) for d in ("bv", "vedb", "things", "lvis", "imagenet")]
live = set.intersection(*sets)
frozen = {l.strip() for l in open(Path.cwd().parents[1] / "data" / "included_categories_qwen_core.txt") if l.strip()}
print("core check: live", len(live), "frozen", len(frozen), "lost", sorted(frozen - live), "new", sorted(live - frozen))
raise SystemExit(0 if live == frozen else 1)
EOF
  step "core set matches data/included_categories_qwen_core.txt"
  run b_core python -u build_things27_domains.py                     # THINGS-27 domains of the core categories
  run b_core $BASEPY -u build_child_month_means.py --workers 8      # BabyView month means on the core (writes parquet)
  # THINGS-27 inputs: the copies of VEDB-dependent folders are stale; make_things27_inputs.py only copies missing ones
  rm -rf out/private/open_vedb_renamed_things27 out/private/bv_vedb_renamed_things27 out/private/controls_renamed_things27 \
         out/private/dev_within_child_renamed_things27
  # fig_style.py: THINGS-27 DOMAIN_ORDER no longer lists "sports equipment" (4 core categories < 5)
  run b_t27 python -u make_things27_inputs.py
  QOV_DOMAINS=things27 run b_t27 python -u build_crossdataset_tables.py
  cp out/crossdataset_renamed/format_by_category.csv out/crossdataset_renamed_things27/   # per category, domain-free
  domain_chain cdi &
  domain_chain things27 &
  wait
  cp results/controls_renamed/prevalence_spearman.csv results/controls_renamed_things27/   # domain-free; as before
  step "== B done"
}

track_c() {
  step "== C: figures"
  cd $TALK; unset QOV_DOMAINS QOV_CATS
  run c_talk python -u make_talk_montages.py
  run c_talk python -u fig_qwen_talk.py
  cd $QOV
  run c_fig python -u make_background_examples.py
  run c_fig python -u make_overview_examples_figure.py
  for D in cdi things27; do
    export QOV_DOMAINS=$D; T=""; [ $D = things27 ] && T=_things27
    M=""; [ $D = cdi ] && M=montages
    run c_fig_$D python -u make_manuscript_figures.py long_tail formats similarity rdms clustering children age resp $M
    run c_fig_$D python -u make_dinov3_manuscript_figures.py
    run c_fig_$D python -u make_manuscript_figures.py copy
    for x in png pdf; do                                   # within-child figure is not in copy_in()
      s=figures/bv_vedb_renamed$T/dev_within_child.$x; [ -e $s ] || continue
      mkdir -p figures/manuscript$T; cp $s figures/manuscript$T/figS_within_child_development.$x
      cp $s $R/manuscript-2026/revision_private/manuscript_revision/figures$T/figS_within_child_development.$x
    done
  done
  for D in cdi things27; do                                  # letter figures R1d / R1j (overwrite the older versions above)
    QOV_DOMAINS=$D run c_fig_$D $BASEPY -u background_response_figures.py
  done
  unset QOV_DOMAINS
  step "== C done"
}

MODE=${1:-all}
if [ $MODE = resume ]; then
  step "resume: waiting for the BV/VEDB sample filter"
  until grep -qE "(ok|FAILED) +\[b_bvvedb\] python -u filter_bv_vedb" $L || grep -q "FAILED \[b_bvvedb\] python -u build_bv" $L; do sleep 30; done
  grep -q "FAILED \[b_bvvedb\]" $L && { step "resume stopped: sample build failed"; exit 1; }
  ( track_b_tail )
else
  : > $L
  step "start: VEDB adults-only rerun ($(wc -l < $VD/vedb_adult_sessions.txt) sessions)"
  track_a &
  track_b &
  wait
fi
grep -qE "B stopped|resume stopped" $L && { step "figures skipped: track B stopped"; exit 1; }
track_c
step "chain complete; failures: $(grep -c FAILED $L)"
