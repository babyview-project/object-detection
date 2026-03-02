"""
Group embeddings by category, subject, and age_mo; compute per-group averages and save.

Run (defaults use project paths):
  # All categories from allowlist file (default), person always excluded
  python preprocessing/group_embeddings_by_agemo.py

  # All categories (no allowlist), 8 workers for speed
  python preprocessing/group_embeddings_by_agemo.py --no-categories-file --num-workers 8

  # Single category test
  python preprocessing/group_embeddings_by_agemo.py --test-category zipper

  # Custom paths
  python preprocessing/group_embeddings_by_agemo.py --embeddings-dir /path/to/embeddings --metadata-csv /path/to/metadata.csv --output-dir /path/to/out

  # DINOv3 embeddings (uses dinov3 defaults; same pipeline)
  python preprocessing/group_embeddings_by_agemo.py --embedding-type dinov3 --no-categories-file --num-workers 8

  # Different confidence threshold (0.26 or 0.28); used with --embedding-type for metadata/output paths
  python preprocessing/group_embeddings_by_agemo.py --embedding-type clip --threshold 0.28 --no-categories-file
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Group embeddings by category, subject, and age_mo')
parser.add_argument('--test-category', type=str, default=None,
                    help='Test on a single category (e.g., "zipper"). If not specified, processes all categories.')
parser.add_argument('--embeddings-dir', type=str, 
                    default="/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_new",
                    help='Directory containing category subfolders with embedding files')
parser.add_argument('--metadata-csv', type=str,
                    default="/home/j7yang/babyview-projects/vss2026/object-detection/frame_data/merged_frame_detections_with_metadata_filtered-0.27.csv",
                    help='Path to the metadata CSV file')
parser.add_argument('--output-dir', type=str,
                    default="/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_grouped_by_age-mo_filtered-0.27",
                    help='Output directory for grouped embeddings')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of parallel workers to process categories (default: 1). Use e.g. 8 to speed up on multi-core machines.')
parser.add_argument('--categories-file', type=str, 
                    default="/home/j7yang/babyview-projects/vss2026/object-detection/data/things_bv_overlap_categories_exclude_zero_precisions.txt",
                    help='Path to text file with category names (one per line). If specified, only these categories will be processed.')
parser.add_argument('--no-categories-file', action='store_true',
                    help='Process all categories; ignore --categories-file and do not load any category exclusion/allowlist.')
parser.add_argument('--skipped-csv', type=str, default=None,
                    help='Path to save skipped embeddings (default: <base-embeddings>/skipped_embeddings_filtered-<threshold>_<embedding-type>.csv)')
parser.add_argument('--tracked-csv', type=str, default=None,
                    help='Path to save embedding-to-group mapping (default: <output-dir>/embedding_to_grouped_mapping.csv)')
parser.add_argument('--embedding-type', type=str, choices=['clip', 'dinov3'], default=None,
                    help='Preset paths for clip or dinov3 embeddings (overrides --embeddings-dir and --output-dir defaults)')
parser.add_argument('--threshold', type=float, default=0.27,
                    help='Confidence threshold for metadata/output paths when using --embedding-type (e.g. 0.26, 0.27, 0.28). Default: 0.27')
args = parser.parse_args()

# Apply embedding-type presets (same filtering: metadata filtered-{threshold} + categories file for both)
_BASE_EMBEDDINGS = "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings"
_FRAME_DATA_DIR = "/home/j7yang/babyview-projects/vss2026/object-detection/frame_data"
_CATEGORIES_FILE = "/home/j7yang/babyview-projects/vss2026/object-detection/data/things_bv_overlap_categories_exclude_zero_precisions.txt"
_threshold = args.threshold
if args.embedding_type == 'dinov3':
    args.embeddings_dir = f"{_BASE_EMBEDDINGS}/facebook_dinov3-vitb16-pretrain-lvd1689m"
    args.output_dir = f"{_BASE_EMBEDDINGS}/dinov3_embeddings_grouped_by_age-mo_filtered-{_threshold}"
    args.metadata_csv = f"{_FRAME_DATA_DIR}/merged_frame_detections_with_metadata_filtered-{_threshold}.csv"
    args.categories_file = _CATEGORIES_FILE
elif args.embedding_type == 'clip':
    args.embeddings_dir = f"{_BASE_EMBEDDINGS}/clip_embeddings_new"
    args.output_dir = f"{_BASE_EMBEDDINGS}/clip_embeddings_grouped_by_age-mo_filtered-{_threshold}"
    args.metadata_csv = f"{_FRAME_DATA_DIR}/merged_frame_detections_with_metadata_filtered-{_threshold}.csv"
    args.categories_file = _CATEGORIES_FILE
_embedding_type_label = args.embedding_type if args.embedding_type else 'clip'

# Worker for one category (must be top-level for multiprocessing pickling)
def _process_one_category(pack):
    """Process a single category; return (group_sums_counts, processed, skipped, skip_reasons, skipped_list, tracked_list)."""
    category_folder, file_paths, metadata_lookup, pattern_str, excluded_subject_ids = pack
    category = category_folder.name
    pattern = re.compile(pattern_str) if isinstance(pattern_str, str) else pattern_str
    group_sums_counts = {}  # key (category, subject_id, age_mo) -> (sum_array, count)
    processed = 0
    skipped = 0
    skip_reasons = {'pattern_mismatch': 0, 'category_mismatch': 0, 'missing_metadata': 0, 'missing_age_mo': 0, 'load_error': 0, 'excluded_subject': 0}
    skipped_list = []  # (embedding_name, category, skip_reason)
    tracked_list = []  # (embedding_name, category, subject_id, age_mo) for CSV mapping
    for embedding_file in file_paths:
        processed += 1
        embedding_name = embedding_file.name
        if excluded_subject_ids and any(f"_{sid}_" in embedding_name or f"_{sid}." in embedding_name for sid in excluded_subject_ids):
            skipped += 1
            skip_reasons['excluded_subject'] += 1
            skipped_list.append((embedding_name, category, 'excluded_subject'))
            continue
        match = pattern.match(embedding_name)
        if not match:
            skipped += 1
            skip_reasons['pattern_mismatch'] += 1
            skipped_list.append((embedding_name, category, 'pattern_mismatch'))
            continue
        parsed_category, confidence, subject_id, gcp_name, frame_id = match.groups()
        if parsed_category != category:
            skipped += 1
            skip_reasons['category_mismatch'] += 1
            skipped_list.append((embedding_name, category, 'category_mismatch'))
            continue
        if embedding_name not in metadata_lookup:
            skipped += 1
            skip_reasons['missing_metadata'] += 1
            skipped_list.append((embedding_name, category, 'missing_metadata'))
            continue
        metadata = metadata_lookup[embedding_name]
        age_mo = metadata['age_mo']
        if age_mo is None:
            skipped += 1
            skip_reasons['missing_age_mo'] += 1
            skipped_list.append((embedding_name, category, 'missing_age_mo'))
            continue
        try:
            embedding = np.load(embedding_file, mmap_mode='r')
            embedding_array = np.asarray(embedding, dtype=np.float64)
            key = (category, subject_id, age_mo)
            if key not in group_sums_counts:
                group_sums_counts[key] = [embedding_array.copy(), 1]
            else:
                group_sums_counts[key][0] += embedding_array
                group_sums_counts[key][1] += 1
            tracked_list.append((embedding_name, category, subject_id, age_mo))
            del embedding, embedding_array
        except Exception:
            skipped += 1
            skip_reasons['load_error'] += 1
            skipped_list.append((embedding_name, category, 'load_error'))
            continue
    return group_sums_counts, processed, skipped, skip_reasons, skipped_list, tracked_list


# Define paths
embeddings_dir = Path(args.embeddings_dir)
metadata_csv = Path(args.metadata_csv)
output_dir = Path(args.output_dir)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading metadata...")
# Load metadata - we'll need all columns to save back with the new column
with tqdm(total=1, desc="Loading metadata CSV", unit="file") as pbar:
    metadata_df = pd.read_csv(
        metadata_csv,
        dtype={'age_mo': 'float64', 'subject_id': 'str', 'class_name': 'str', 'original_embedding_name': 'str'}
    )
    pbar.update(1)
    pbar.set_postfix({"rows": len(metadata_df)})

# Create lookup dictionary: embedding_name -> (age_mo, subject_id, class_name)
# Use vectorized operations for much faster processing
print("Building metadata lookup...")
# Drop duplicates on original_embedding_name to handle potential duplicates
metadata_unique = metadata_df.dropna(subset=['original_embedding_name']).drop_duplicates(
    subset=['original_embedding_name'], keep='first'
)

# Use vectorized operations instead of iterrows (much faster!)
with tqdm(total=1, desc="Building lookup", unit="step") as pbar:
    # Convert age_mo to int, handling NaN
    metadata_unique = metadata_unique.copy()
    metadata_unique['age_mo_int'] = metadata_unique['age_mo'].fillna(-1).astype(int)
    metadata_unique.loc[metadata_unique['age_mo_int'] == -1, 'age_mo_int'] = None
    
    # Create lookup dict using zip (much faster than iterrows)
    metadata_lookup = {
        name: {
            'age_mo': int(age) if pd.notna(age) and age != -1 else None,
            'subject_id': str(subj),
            'class_name': str(cls)
        }
        for name, age, subj, cls in zip(
            metadata_unique['original_embedding_name'],
            metadata_unique['age_mo_int'],
            metadata_unique['subject_id'],
            metadata_unique['class_name']
        )
    }
    pbar.update(1)

print(f"Loaded {len(metadata_lookup)} unique metadata entries")

# Add new column to metadata_df with the grouped embedding name (vectorized)
# Format: {category}/{subject_id}_{age_mo}_month_level_avg.npy
print("\nAdding new_embedding_name_grouped_by_age_mo column...")
# Use vectorized operations instead of apply()
with tqdm(total=1, desc="Creating grouped names", unit="step") as pbar:
    mask_valid = (
        metadata_df['age_mo'].notna() & 
        metadata_df['class_name'].notna() & 
        metadata_df['subject_id'].notna()
    )
    metadata_df['new_embedding_name_grouped_by_age_mo'] = None
    metadata_df.loc[mask_valid, 'new_embedding_name_grouped_by_age_mo'] = (
        metadata_df.loc[mask_valid, 'class_name'].astype(str) + '/' +
        metadata_df.loc[mask_valid, 'subject_id'].astype(str) + '_' +
        metadata_df.loc[mask_valid, 'age_mo'].astype(int).astype(str) + 
        '_month_level_avg.npy'
    )
    pbar.update(1)
    pbar.set_postfix({"valid_rows": mask_valid.sum()})

# Dictionary to store running averages and counts for incremental averaging
# Structure: (category, subject_id, age_mo) -> {'sum': array, 'count': int}
# This avoids storing all embeddings in memory
embeddings_by_group = defaultdict(lambda: {'sum': None, 'count': 0})

# Pattern to parse filenames: {category}_{confidence}_{subject_id}_{gcp_name}_processed_{frame_id}.npy
_filename_pattern_str = r'^(.+?)_([\d.]+)_(\d+)_(.+?)_processed_(\d+)\.npy$'
filename_pattern = re.compile(_filename_pattern_str)

# Iterate through all category folders
all_category_folders = [f for f in embeddings_dir.iterdir() if f.is_dir()]

# Load allowed categories from file unless --no-categories-file is set
allowed_categories = None
if not args.no_categories_file and args.categories_file:
    categories_file = Path(args.categories_file)
    if categories_file.exists():
        print(f"Loading allowed categories from: {categories_file}")
        with open(categories_file, 'r') as f:
            allowed_categories = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(allowed_categories)} categories from file")
    else:
        print(f"Warning: Categories file not found: {categories_file}")
        print("  Processing all categories instead.")
elif args.no_categories_file:
    print("Using --no-categories-file: processing all categories (no category allowlist).")

# Categories to always exclude (e.g. person), regardless of --categories-file or --no-categories-file
ALWAYS_EXCLUDED_CATEGORIES = {'person'}
# Subjects to always exclude (e.g. different filename format / special data)
ALWAYS_EXCLUDED_SUBJECT_IDS = {'00270001'}

# Filter categories based on test category, categories file, or use all
if args.test_category:
    # Test mode: only process the specified category
    category_folders = [f for f in all_category_folders if f.name == args.test_category]
    if not category_folders:
        print(f"Error: Category '{args.test_category}' not found in embeddings directory.")
        print(f"Available categories: {sorted([f.name for f in all_category_folders])[:10]}...")
        exit(1)
    print(f"\nTEST MODE: Processing only category '{args.test_category}'...")
elif allowed_categories is not None:
    # Filter to only categories in the allowed list
    category_folders = [f for f in all_category_folders if f.name in allowed_categories]
    print(f"\nFILTERED MODE: Processing {len(category_folders)} categories from allowed list...")
    # Check if any categories in the file don't exist
    missing_categories = allowed_categories - set(f.name for f in all_category_folders)
    if missing_categories:
        print(f"Warning: {len(missing_categories)} categories from file not found in embeddings directory:")
        print(f"  {sorted(list(missing_categories))[:10]}{'...' if len(missing_categories) > 10 else ''}")
else:
    # Process all categories
    category_folders = all_category_folders
    print(f"\nProcessing all {len(category_folders)} categories...")

# Always exclude certain categories (e.g. person)
excluded_present = [f for f in category_folders if f.name in ALWAYS_EXCLUDED_CATEGORIES]
if excluded_present:
    category_folders = [f for f in category_folders if f.name not in ALWAYS_EXCLUDED_CATEGORIES]
    print(f"Excluding always-ignored categories: {sorted(ALWAYS_EXCLUDED_CATEGORIES)} ({len(excluded_present)} folder(s) skipped).")
    print(f"Categories to process: {len(category_folders)}")
if ALWAYS_EXCLUDED_SUBJECT_IDS:
    print(f"Excluding always-ignored subject(s): {sorted(ALWAYS_EXCLUDED_SUBJECT_IDS)} (files with this subject ID in filename will be skipped).")

# Count total embeddings for overall progress (optimized: count while getting file lists)
print("Counting embeddings...")
category_file_lists = {}
total_embeddings = 0
for cat_folder in tqdm(category_folders, desc="Counting files", leave=False):
    files = list(cat_folder.glob("*.npy"))
    category_file_lists[cat_folder] = files
    total_embeddings += len(files)

print(f"Total embeddings to process: {total_embeddings:,}")
print()

# Track overall progress and skip reasons
embeddings_processed = 0
embeddings_skipped = 0
skip_reasons = {
    'pattern_mismatch': 0,
    'category_mismatch': 0,
    'missing_metadata': 0,
    'missing_age_mo': 0,
    'load_error': 0,
    'excluded_subject': 0
}
skipped_records = []  # list of (embedding_name, category, skip_reason)
tracked_records = []  # list of (embedding_name, category, subject_id, age_mo) for mapping CSV

num_workers = max(1, args.num_workers)
pbar_files = tqdm(total=total_embeddings, desc="Embeddings", unit="file", smoothing=0.01)

if num_workers > 1:
    # Parallel: process categories in worker processes, then merge
    print(f"Processing {len(category_folders)} categories with {num_workers} workers...")
    task_packs = [
        (cat_folder, category_file_lists[cat_folder], metadata_lookup, _filename_pattern_str, ALWAYS_EXCLUDED_SUBJECT_IDS)
        for cat_folder in category_folders
    ]
    with Pool(num_workers) as pool:
        for group_sums_counts, proc, skp, reasons, skipped_list, tracked_list in tqdm(
            pool.imap(_process_one_category, task_packs, chunksize=1),
            total=len(task_packs), desc="Categories", unit="cat", leave=False
        ):
            pbar_files.update(proc + skp)
            embeddings_processed += proc
            embeddings_skipped += skp
            skipped_records.extend(skipped_list)
            tracked_records.extend(tracked_list)
            for k, v in reasons.items():
                skip_reasons[k] += v
            for key, (sum_arr, count) in group_sums_counts.items():
                existing = embeddings_by_group[key]
                if existing['sum'] is None:
                    existing['sum'] = sum_arr
                    existing['count'] = count
                else:
                    existing['sum'] = existing['sum'] + sum_arr
                    existing['count'] += count
    del task_packs
    gc.collect()
else:
    # Sequential
    for category_folder in tqdm(category_folders, desc="Categories", unit="cat", leave=False):
        category = category_folder.name
        embedding_files = category_file_lists[category_folder]
        for embedding_file in embedding_files:
            pbar_files.update(1)
            pbar_files.set_postfix(cat=category)
            embeddings_processed += 1
            embedding_name = embedding_file.name
            if ALWAYS_EXCLUDED_SUBJECT_IDS and any(f"_{sid}_" in embedding_name or f"_{sid}." in embedding_name for sid in ALWAYS_EXCLUDED_SUBJECT_IDS):
                embeddings_skipped += 1
                skip_reasons['excluded_subject'] += 1
                skipped_records.append((embedding_name, category, 'excluded_subject'))
                continue
            match = filename_pattern.match(embedding_name)
            if not match:
                embeddings_skipped += 1
                skip_reasons['pattern_mismatch'] += 1
                skipped_records.append((embedding_name, category, 'pattern_mismatch'))
                continue
            parsed_category, confidence, subject_id, gcp_name, frame_id = match.groups()
            if parsed_category != category:
                embeddings_skipped += 1
                skip_reasons['category_mismatch'] += 1
                skipped_records.append((embedding_name, category, 'category_mismatch'))
                continue
            if embedding_name not in metadata_lookup:
                embeddings_skipped += 1
                skip_reasons['missing_metadata'] += 1
                skipped_records.append((embedding_name, category, 'missing_metadata'))
                continue
            metadata = metadata_lookup[embedding_name]
            age_mo = metadata['age_mo']
            if age_mo is None:
                embeddings_skipped += 1
                skip_reasons['missing_age_mo'] += 1
                skipped_records.append((embedding_name, category, 'missing_age_mo'))
                continue
            try:
                embedding = np.load(embedding_file, mmap_mode='r')
                embedding_array = np.asarray(embedding, dtype=np.float64)
                key = (category, subject_id, age_mo)
                group_data = embeddings_by_group[key]
                if group_data['sum'] is None:
                    group_data['sum'] = embedding_array.copy()
                    group_data['count'] = 1
                else:
                    group_data['sum'] += embedding_array
                    group_data['count'] += 1
                tracked_records.append((embedding_name, category, subject_id, age_mo))
                del embedding, embedding_array
            except Exception as e:
                print(f"Error loading {embedding_file}: {e}")
                embeddings_skipped += 1
                skip_reasons['load_error'] += 1
                skipped_records.append((embedding_name, category, 'load_error'))
                continue
        gc.collect()

pbar_files.close()

# Save skipped embeddings with reasons (default: embeddings root)
if skipped_records:
    skipped_csv = Path(args.skipped_csv) if args.skipped_csv else Path(f"{_BASE_EMBEDDINGS}/skipped_embeddings_filtered-{args.threshold}_{_embedding_type_label}.csv")
    skipped_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped_df = pd.DataFrame(skipped_records, columns=['embedding_name', 'category', 'skip_reason'])
    skipped_df.to_csv(skipped_csv, index=False)
    print(f"\nSaved {len(skipped_records):,} skipped embeddings to: {skipped_csv}")

# Save embedding-to-group mapping CSV (which tracked embeddings belong to which grouped embedding)
if tracked_records:
    tracked_csv = Path(args.tracked_csv) if args.tracked_csv else (output_dir / "embedding_to_grouped_mapping.csv")
    tracked_csv.parent.mkdir(parents=True, exist_ok=True)
    tracked_df = pd.DataFrame(
        tracked_records,
        columns=['original_embedding_name', 'category', 'subject_id', 'age_mo']
    )
    tracked_df['grouped_embedding_name'] = (
        tracked_df['category'].astype(str) + '/' +
        tracked_df['subject_id'].astype(str) + '_' +
        tracked_df['age_mo'].astype(int).astype(str) + '_month_level_avg.npy'
    )
    tracked_df.to_csv(tracked_csv, index=False)
    print(f"Saved {len(tracked_df):,} tracked embeddings → grouped mapping to: {tracked_csv}")

# Calculate and save averaged embeddings
# Since we used incremental averaging, we just need to divide sum by count
print(f"\nCalculating final averages and saving...")
print(f"Total groups to process: {len(embeddings_by_group):,}")
print(f"Embeddings processed: {embeddings_processed:,} (skipped: {embeddings_skipped:,})")
print()
for (category, subject_id, age_mo), group_data in tqdm(embeddings_by_group.items(), desc="Averaging and saving", unit="group"):
    if group_data['count'] == 0 or group_data['sum'] is None:
        continue
    
    # Calculate mean: sum / count (use in-place division for memory efficiency)
    avg_embedding = group_data['sum'] / group_data['count']
    
    # Convert to float32 to save space (reduces file size by ~50%)
    # Use astype with copy=False when possible, but we need copy here since we're deleting sum
    avg_embedding = avg_embedding.astype(np.float32, copy=True)
    
    # Create category subfolder in output directory
    category_output_dir = output_dir / category
    category_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename: {subject_id}_{age_mo}_month_level_avg.npy
    output_filename = f"{subject_id}_{age_mo}_month_level_avg.npy"
    output_path = category_output_dir / output_filename
    
    # Save averaged embedding
    np.save(output_path, avg_embedding)
    
    # Free memory
    del avg_embedding, group_data['sum']

# Save the updated metadata CSV with the new column
# Use chunked writing for large files to avoid memory issues
print("\nSaving updated metadata CSV...")
# Use chunked writing for very large files to avoid memory issues
chunk_size = 500000  # Process 500k rows at a time for large files

if len(metadata_df) > chunk_size:
    print(f"Large file detected ({len(metadata_df):,} rows), using chunked writing...")
    for i in tqdm(range(0, len(metadata_df), chunk_size), desc="Writing CSV chunks", unit="chunk"):
        chunk = metadata_df.iloc[i:i+chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(metadata_csv, mode=mode, header=header, index=False)
else:
    with tqdm(total=1, desc="Writing CSV", unit="file") as pbar:
        metadata_df.to_csv(metadata_csv, index=False)
        pbar.update(1)
        pbar.set_postfix({"rows": len(metadata_df)})
print(f"Updated metadata saved to: {metadata_csv}")

print(f"\n{'='*60}")
print(f"Done! Summary:")
print(f"  - Averaged embeddings saved to: {output_dir}")
if tracked_records:
    tracked_csv = Path(args.tracked_csv) if args.tracked_csv else (output_dir / "embedding_to_grouped_mapping.csv")
    print(f"  - Embedding→group mapping CSV: {tracked_csv}")
print(f"  - Total groups processed: {len(embeddings_by_group):,}")
print(f"  - Embeddings processed: {embeddings_processed:,}")
print(f"  - Embeddings skipped: {embeddings_skipped:,}")
if embeddings_processed > 0:
    print(f"  - Success rate: {(embeddings_processed - embeddings_skipped) / embeddings_processed * 100:.1f}%")
if embeddings_skipped > 0:
    print(f"\n  Skip reasons breakdown:")
    print(f"    - Filename pattern mismatch: {skip_reasons['pattern_mismatch']:,}")
    print(f"    - Category mismatch: {skip_reasons['category_mismatch']:,}")
    print(f"    - Missing in metadata: {skip_reasons['missing_metadata']:,}")
    print(f"    - Missing age_mo: {skip_reasons['missing_age_mo']:,}")
    print(f"    - Load error: {skip_reasons['load_error']:,}")
    print(f"    - Excluded subject: {skip_reasons['excluded_subject']:,}")
    if skipped_records:
        skipped_csv = Path(args.skipped_csv) if args.skipped_csv else Path(f"{_BASE_EMBEDDINGS}/skipped_embeddings_filtered-{args.threshold}_{_embedding_type_label}.csv")
        print(f"  - Skipped list saved to: {skipped_csv}")
print(f"{'='*60}")