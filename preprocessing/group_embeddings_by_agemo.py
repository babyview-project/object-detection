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
                    default="/home/j7yang/babyview-projects/vss2026/object-detection/frame_data/merged_frame_detections_with_metadata.csv",
                    help='Path to the metadata CSV file')
parser.add_argument('--output-dir', type=str,
                    default="/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_grouped_by_age-mo",
                    help='Output directory for grouped embeddings')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of parallel workers for processing categories (default: 1, sequential)')
parser.add_argument('--categories-file', type=str, 
                    default="/home/j7yang/babyview-projects/vss2026/object-detection/data/things_bv_overlap_categories_exclude_zero_precisions.txt",
                    help='Path to text file with category names (one per line). If specified, only these categories will be processed. Default: things_bv_overlap_categories_exclude_zero_precisions.txt')
args = parser.parse_args()

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
filename_pattern = re.compile(r'^(.+?)_([\d.]+)_(\d+)_(.+?)_processed_(\d+)\.npy$')

# Iterate through all category folders
all_category_folders = [f for f in embeddings_dir.iterdir() if f.is_dir()]

# Load allowed categories from file if specified
allowed_categories = None
if args.categories_file:
    categories_file = Path(args.categories_file)
    if categories_file.exists():
        print(f"Loading allowed categories from: {categories_file}")
        with open(categories_file, 'r') as f:
            allowed_categories = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(allowed_categories)} categories from file")
    else:
        print(f"Warning: Categories file not found: {categories_file}")
        print("  Processing all categories instead.")

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
    'load_error': 0
}

for category_folder in tqdm(category_folders, desc="Processing categories", unit="category"):
    category = category_folder.name
    
    # Use pre-computed file list (avoid double glob)
    embedding_files = category_file_lists[category_folder]
    
    for embedding_file in tqdm(embedding_files, desc=f"  {category}", leave=False, unit="file"):
        embeddings_processed += 1
        
        # Parse filename
        match = filename_pattern.match(embedding_file.name)
        if not match:
            embeddings_skipped += 1
            skip_reasons['pattern_mismatch'] += 1
            continue
        
        parsed_category, confidence, subject_id, gcp_name, frame_id = match.groups()
        
        # Verify category matches folder name
        if parsed_category != category:
            embeddings_skipped += 1
            skip_reasons['category_mismatch'] += 1
            continue
        
        # Look up metadata using the embedding filename
        embedding_name = embedding_file.name
        if embedding_name not in metadata_lookup:
            embeddings_skipped += 1
            skip_reasons['missing_metadata'] += 1
            continue
        
        metadata = metadata_lookup[embedding_name]
        age_mo = metadata['age_mo']
        
        # Skip if age_mo is missing
        if age_mo is None:
            embeddings_skipped += 1
            skip_reasons['missing_age_mo'] += 1
            continue
        
        # Load embedding and update running average incrementally
        # This avoids storing all embeddings in memory
        try:
            # Use mmap_mode='r' for large files to avoid loading full array into memory
            # This is especially useful for very large embedding files
            embedding = np.load(embedding_file, mmap_mode='r')
            # Convert to float64 in-place if needed, but keep original dtype if possible
            embedding_array = np.asarray(embedding, dtype=np.float64)
            
            key = (category, subject_id, age_mo)
            group_data = embeddings_by_group[key]
            
            if group_data['sum'] is None:
                # First embedding for this group - copy to avoid mmap issues
                group_data['sum'] = embedding_array.copy()
                group_data['count'] = 1
            else:
                # Incremental update: new_sum = old_sum + new_value
                group_data['sum'] += embedding_array
                group_data['count'] += 1
            
            # Free memory immediately
            del embedding, embedding_array
        except Exception as e:
            print(f"Error loading {embedding_file}: {e}")
            embeddings_skipped += 1
            skip_reasons['load_error'] += 1
            continue
    
    # Periodic garbage collection after each category to free memory
    gc.collect()

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
print(f"{'='*60}")