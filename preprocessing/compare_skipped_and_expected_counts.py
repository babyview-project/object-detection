"""
Compare CLIP vs DINOv3 skipped-embeddings CSVs and check non-skipped count vs reference list.

Usage (run where /data2 is mounted, e.g. cluster):
  python preprocessing/compare_skipped_and_expected_counts.py

Paths are hardcoded below; edit BASE if needed.
"""
import pandas as pd
from pathlib import Path
from typing import Optional

BASE = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
CLIP_SKIPPED = BASE / "skipped_embeddings_filtered-0.27_clip.csv"
DINOV3_SKIPPED = BASE / "skipped_embeddings_filtered-0.27_dinov3.csv"
REFERENCE_LIST = BASE / "clip_image_embeddings_filtered-by-clip-0.27_exclude-people_exclude-subject-00270001.txt"
CLIP_EMBEDDINGS_DIR = BASE / "clip_embeddings_new"
DINOV3_EMBEDDINGS_DIR = BASE / "facebook_dinov3-vitb16-pretrain-lvd1689m"
CLIP_OUTPUT_DIR = BASE / "clip_embeddings_grouped_by_age-mo_filtered-0.27"
DINOV3_OUTPUT_DIR = BASE / "dinov3_embeddings_grouped_by_age-mo_filtered-0.27"


def main():
    print("=" * 60)
    print("1. Comparing skipped-embeddings CSVs (CLIP vs DINOv3)")
    print("=" * 60)

    if not CLIP_SKIPPED.exists():
        print(f"  Missing: {CLIP_SKIPPED}")
        clip_df = None
    else:
        clip_df = pd.read_csv(CLIP_SKIPPED)
        print(f"  CLIP skipped:   {len(clip_df):,} rows")

    if not DINOV3_SKIPPED.exists():
        print(f"  Missing: {DINOV3_SKIPPED}")
        dinov3_df = None
    else:
        dinov3_df = pd.read_csv(DINOV3_SKIPPED)
        print(f"  DINOv3 skipped: {len(dinov3_df):,} rows")

    if clip_df is not None and dinov3_df is not None:
        clip_names = set(clip_df["embedding_name"])
        dinov3_names = set(dinov3_df["embedding_name"])
        common_skipped = clip_names & dinov3_names
        only_clip = clip_names - dinov3_names
        only_dinov3 = dinov3_names - clip_names

        print(f"\n  Embedding names skipped in both:     {len(common_skipped):,}")
        print(f"  Skipped in CLIP only:                {len(only_clip):,}")
        print(f"  Skipped in DINOv3 only:              {len(only_dinov3):,}")

        if clip_names == dinov3_names:
            print("\n  Content match: same set of embedding_name in both CSVs.")
        else:
            print("\n  Content mismatch: different sets of embedding_name.")
            if only_clip:
                print(f"    Sample CLIP-only: {list(only_clip)[:3]}")
            if only_dinov3:
                print(f"    Sample DINOv3-only: {list(only_dinov3)[:3]}")

        # Same (embedding_name, category, skip_reason)?
        clip_tuples = set(
            zip(clip_df["embedding_name"], clip_df["category"], clip_df["skip_reason"])
        )
        dinov3_tuples = set(
            zip(dinov3_df["embedding_name"], dinov3_df["category"], dinov3_df["skip_reason"])
        )
        if clip_tuples == dinov3_tuples:
            print("  Row match: same (embedding_name, category, skip_reason) in both.")
        else:
            in_both = clip_tuples & dinov3_tuples
            print(f"  Rows identical in both: {len(in_both):,} / {len(clip_tuples):,}")

    print("\n" + "=" * 60)
    print("2. Non-skipped count vs reference list")
    print("=" * 60)

    if not REFERENCE_LIST.exists():
        print(f"  Missing: {REFERENCE_LIST}")
        ref_count = None
    else:
        ref_lines = REFERENCE_LIST.read_text(encoding="utf-8").strip().splitlines()
        ref_lines = [ln.strip() for ln in ref_lines if ln.strip()]
        ref_count = len(ref_lines)
        print(f"  Reference list: {REFERENCE_LIST.name}")
        print(f"  Line count (expected non-skipped for CLIP): {ref_count:,}")

    # Non-skipped from reproducibility manifest (processed - skipped from run)
    def non_skipped_from_manifest(output_dir: Path) -> Optional[int]:
        manifest = output_dir / "reproducibility_manifest.txt"
        if not manifest.exists():
            return None
        text = manifest.read_text(encoding="utf-8")
        processed = skipped = None
        for line in text.splitlines():
            if line.startswith("embeddings_processed="):
                processed = int(line.split("=", 1)[1])
            elif line.startswith("embeddings_skipped="):
                skipped = int(line.split("=", 1)[1])
        if processed is not None and skipped is not None:
            return processed - skipped
        return None

    clip_non_skipped = non_skipped_from_manifest(CLIP_OUTPUT_DIR)
    dinov3_non_skipped = non_skipped_from_manifest(DINOV3_OUTPUT_DIR)
    if clip_non_skipped is not None:
        print(f"  CLIP non-skipped (from manifest):   {clip_non_skipped:,}")
    if dinov3_non_skipped is not None:
        print(f"  DINOv3 non-skipped (from manifest): {dinov3_non_skipped:,}")

    if ref_count is not None and clip_non_skipped is not None:
        diff = clip_non_skipped - ref_count
        if diff == 0:
            print(f"\n  Match: CLIP non-skipped count equals reference list count ({ref_count:,}).")
        else:
            print(f"\n  Difference (CLIP non-skipped - reference): {diff:+,}")

    # 3. Where do the reference names go? (diagnostic for ref_count != non_skipped)
    if REFERENCE_LIST.exists() and clip_df is not None:
        print("\n" + "=" * 60)
        print("3. Reference list vs CLIP skipped (where do ref names go?)")
        print("=" * 60)
        ref_names = set(REFERENCE_LIST.read_text(encoding="utf-8").strip().splitlines())
        ref_names = {n.strip() for n in ref_names if n.strip()}
        clip_skipped_names = set(clip_df["embedding_name"])
        ref_that_we_skipped = ref_names & clip_skipped_names
        ref_that_we_did_not_skip = ref_names - clip_skipped_names
        print(f"  Reference list size:                    {len(ref_names):,}")
        print(f"  Reference names that we SKIPPED:        {len(ref_that_we_skipped):,}")
        print(f"  Reference names we did NOT skip:       {len(ref_that_we_did_not_skip):,}")
        if clip_non_skipped is not None:
            print(f"  CLIP non-skipped (from manifest):       {clip_non_skipped:,}")
            print(f"  => Ref-not-skipped should equal non-skipped if ref = 'embeddings we use'.")
        if ref_that_we_skipped:
            reasons = clip_df[clip_df["embedding_name"].isin(ref_that_we_skipped)]["skip_reason"].value_counts()
            print(f"  Skip reasons for those ref names we skipped:")
            for reason, count in reasons.items():
                print(f"    - {reason}: {count:,}")

    print("=" * 60)


if __name__ == "__main__":
    main()
