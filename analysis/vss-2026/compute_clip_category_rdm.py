#!/usr/bin/env python3
"""
Compute pairwise RDM (Representational Dissimilarity Matrix) for CLIP or DINOv3 category embeddings.
Reads embeddings from .npy files listed in a text file, calculates average category embeddings,
and generates RDMs organized by animate, small, bodyparts, and big objects.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def preflight_checks():
    """
    Run pre-flight checks to ensure all dependencies work correctly.
    Raises an exception if any critical check fails.
    """
    print("Running pre-flight checks...")
    
    # Check pandas CSV functionality
    try:
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        test_path = Path('/tmp/pandas_test.csv')
        test_df.to_csv(test_path)
        if test_path.exists():
            test_path.unlink()  # Clean up
        print("  ✓ Pandas CSV export test passed")
    except Exception as e:
        raise RuntimeError(
            f"Pandas CSV export test failed: {e}\n"
            f"This indicates a corrupted pandas installation. "
            f"Please reinstall pandas: conda install pandas --force-reinstall"
        ) from e
    
    # Check numpy
    try:
        test_arr = np.array([1, 2, 3])
        test_path = Path('/tmp/numpy_test.npy')
        np.save(test_path, test_arr)
        if test_path.exists():
            test_path.unlink()  # Clean up
        print("  ✓ NumPy file I/O test passed")
    except Exception as e:
        raise RuntimeError(f"NumPy file I/O test failed: {e}") from e
    
    # Check matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        plt.close(fig)
        print("  ✓ Matplotlib test passed")
    except Exception as e:
        raise RuntimeError(f"Matplotlib test failed: {e}") from e
    
    print("All pre-flight checks passed!\n")
    return True

def safe_to_csv(df, path, **kwargs):
    """
    Safely save DataFrame to CSV with error handling and fallback.
    Args:
        df: DataFrame to save
        path: Path to save CSV file
        **kwargs: Additional arguments to pass to to_csv
    Returns:
        True if successful, False if failed (with warning printed)
    """
    try:
        df.to_csv(path, **kwargs)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to save CSV to {path}: {e}")
        print(f"  This is non-critical - NPZ files contain all the data.")
        return False

def load_embedding_paths(txt_path):
    """
    Load embedding file paths from text file
    Args:
        txt_path: Path to text file containing embedding paths (one per line)
    Returns:
        List of embedding file paths
    """
    print(f"Loading embedding paths from {txt_path}...")
    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    print(f"Found {len(paths)} embedding paths")
    return paths

def scan_embedding_directory(embeddings_dir):
    """
    Scan directory for all .npy embedding files
    Args:
        embeddings_dir: Base directory containing category subdirectories with .npy files
    Returns:
        List of embedding file paths (relative to embeddings_dir)
    """
    embeddings_dir = Path(embeddings_dir)
    print(f"Scanning {embeddings_dir} for .npy files...")
    
    # Find all .npy files recursively
    npy_files = list(embeddings_dir.rglob("*.npy"))
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {embeddings_dir}")
    
    # Convert to relative paths
    paths = [str(f.relative_to(embeddings_dir)) for f in npy_files]
    paths.sort()  # Sort for consistency
    
    print(f"Found {len(paths)} embedding files in {len(set(f.parent for f in npy_files))} category directories")
    return paths

def match_embedding_paths_from_list(reference_list_path, target_embeddings_dir):
    """
    Match embedding paths from a reference list (e.g., CLIP list) to target directory (e.g., DINOv3).
    Extracts filenames from reference list and looks for matching files in target directory.
    Args:
        reference_list_path: Path to text file with reference embedding paths (e.g., CLIP list)
        target_embeddings_dir: Target directory to search for matching files (e.g., DINOv3 directory)
    Returns:
        List of matched embedding file paths (relative to target_embeddings_dir)
    """
    target_embeddings_dir = Path(target_embeddings_dir)
    
    print(f"Loading reference embedding list from {reference_list_path}...")
    # Load reference paths
    with open(reference_list_path, 'r') as f:
        reference_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(reference_paths)} reference paths")
    print(f"Matching filenames in {target_embeddings_dir}...")
    
    # Extract (category, filename) pairs from reference paths
    reference_mapping = {}
    for ref_path in reference_paths:
        ref_path_obj = Path(ref_path)
        if len(ref_path_obj.parts) >= 2:
            category = ref_path_obj.parts[-2]  # Parent directory = category
            filename = ref_path_obj.name  # Filename
            reference_mapping[(category, filename)] = ref_path
    
    print(f"Extracted {len(reference_mapping)} unique (category, filename) pairs")
    
    # Build a lookup of available files in target directory
    # Structure: {category: {filename: relative_path}}
    target_files = {}
    if target_embeddings_dir.exists():
        for npy_file in target_embeddings_dir.rglob("*.npy"):
            rel_path = npy_file.relative_to(target_embeddings_dir)
            if len(rel_path.parts) >= 2:
                category = rel_path.parts[0]
                filename = rel_path.name
                if category not in target_files:
                    target_files[category] = {}
                target_files[category][filename] = str(rel_path)
    else:
        raise ValueError(f"Target embeddings directory does not exist: {target_embeddings_dir}")
    
    print(f"Found files in {len(target_files)} categories in target directory")
    
    # Match reference paths to target paths
    matched_paths = []
    matched_count = 0
    missing_count = 0
    
    for (category, filename), ref_path in reference_mapping.items():
        if category in target_files and filename in target_files[category]:
            matched_paths.append(target_files[category][filename])
            matched_count += 1
        else:
            missing_count += 1
    
    print(f"Matched {matched_count} files ({matched_count/len(reference_mapping)*100:.1f}%)")
    if missing_count > 0:
        print(f"Warning: {missing_count} files from reference list not found in target directory")
    
    matched_paths.sort()  # Sort for consistency
    
    return matched_paths

def extract_category_from_path(path):
    """
    Extract category name from embedding file path
    Path format: .../category/category_*.npy
    Args:
        path: Full path to embedding file
    Returns:
        Category name
    """
    # Extract category from path (second to last component before filename)
    path_parts = Path(path).parts
    # Find the category directory (should be the parent of the .npy file)
    if len(path_parts) >= 2:
        return path_parts[-2]  # Category is the parent directory name
    return None

def load_single_embedding(args):
    """
    Load a single embedding file (worker function for parallel processing)
    Args:
        args: Tuple of (path, embeddings_dir, is_absolute)
    Returns:
        Tuple of (category, embedding) or (None, None) if failed
    """
    path, embeddings_dir, is_absolute = args
    
    try:
        # Handle both absolute and relative paths
        if is_absolute:
            full_path = Path(path)
        else:
            full_path = Path(embeddings_dir) / path
        
        # Check existence before processing (faster than loading then failing)
        if not full_path.exists():
            return None, None
        
        # Extract category from path (cache path parts to avoid repeated Path operations)
        path_parts = full_path.parts
        if len(path_parts) < 2:
            return None, None
        category = path_parts[-2]  # Category is the parent directory name
        
        # Load embedding (embeddings are typically small, so no need for memory mapping)
        embedding = np.load(full_path)
        # Ensure it's 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        return category, embedding
    except Exception:
        return None, None

def load_embeddings_by_category(embedding_paths, embeddings_dir, num_workers=None, use_parallel=True):
    """
    Load embeddings grouped by category
    Args:
        embedding_paths: List of embedding file paths (relative or absolute)
        embeddings_dir: Base directory for embeddings
        num_workers: Number of parallel workers (None = auto-detect)
        use_parallel: Whether to use parallel loading (default: True)
    Returns:
        Dictionary mapping category names to lists of embeddings
    """
    print("Loading embeddings by category...")
    embeddings_by_category = defaultdict(list)
    
    embeddings_dir = Path(embeddings_dir)
    
    # Determine number of workers
    # For I/O-bound operations (file reading), use more workers for faster storage
    # Modern SSDs can handle 16-32 concurrent reads efficiently
    if num_workers is None:
        num_workers = min(16, mp.cpu_count())  # Increased from 8 to 16 for better throughput
    
    # Pre-process paths to determine if they're absolute
    path_args = []
    for path in embedding_paths:
        is_absolute = Path(path).is_absolute()
        path_args.append((path, str(embeddings_dir), is_absolute))
    
    if use_parallel and len(embedding_paths) > 100:
        # Use parallel processing for large datasets
        print(f"Using {num_workers} parallel workers for I/O-bound operations...")
        print(f"Processing {len(path_args):,} embedding files...")
        
        # For very large datasets, use larger chunks to reduce overhead
        # Larger chunks = fewer futures = less overhead
        # Optimal chunk size: balance between memory and overhead
        chunk_size = max(5000, num_workers * 200)  # Increased from 1000 to 5000 for better throughput
        total_chunks = (len(path_args) + chunk_size - 1) // chunk_size
        
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for I/O-bound operations (file reading)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process in chunks to avoid creating millions of futures
            for chunk_idx in range(0, len(path_args), chunk_size):
                chunk = path_args[chunk_idx:chunk_idx + chunk_size]
                chunk_num = (chunk_idx // chunk_size) + 1
                
                # Submit chunk and collect results more efficiently
                # Use map for better performance than submit+as_completed
                results = list(executor.map(load_single_embedding, chunk))
                
                # Process results
                for category, embedding in results:
                    if category is not None and embedding is not None:
                        embeddings_by_category[category].append(embedding)
                        successful += 1
                    else:
                        failed += 1
                
                # Show progress less frequently to reduce I/O overhead
                if chunk_num % 50 == 0 or chunk_num == total_chunks:
                    progress = (chunk_num / total_chunks) * 100
                    loaded_count = successful
                    print(f"Progress: {progress:.1f}% ({chunk_num}/{total_chunks} chunks, {loaded_count:,} loaded)")
        
        if failed > 0:
            print(f"\nWarning: Failed to load {failed:,} embeddings out of {len(path_args):,}")
        print(f"Successfully loaded {successful:,} embeddings")
    else:
        # Sequential processing for small datasets
        print("Using sequential loading...")
        for args in tqdm(path_args, desc="Loading embeddings"):
            category, embedding = load_single_embedding(args)
            if category is not None and embedding is not None:
                embeddings_by_category[category].append(embedding)
    
    # Print statistics
    print(f"\nLoaded embeddings for {len(embeddings_by_category)} categories:")
    for category, emb_list in sorted(embeddings_by_category.items()):
        print(f"  {category}: {len(emb_list)} embeddings")
    
    return embeddings_by_category

def compute_category_averages(embeddings_by_category):
    """
    Compute average embedding for each category
    Args:
        embeddings_by_category: Dictionary mapping category names to lists of embeddings
    Returns:
        Dictionary mapping category names to average embeddings (numpy arrays)
        List of category names in order
    """
    print("\nComputing category average embeddings...")
    category_averages = {}
    categories = []
    
    for category, emb_list in sorted(embeddings_by_category.items()):
        if len(emb_list) == 0:
            continue
        
        # Stack embeddings and compute mean
        emb_array = np.array(emb_list)
        avg_embedding = np.mean(emb_array, axis=0)
        category_averages[category] = avg_embedding
        categories.append(category)
        
        print(f"  {category}: {len(emb_list)} embeddings -> average embedding shape {avg_embedding.shape}")
    
    return category_averages, categories

def save_category_averages(category_averages, categories, output_dir):
    """
    Save category average embeddings to file
    Args:
        category_averages: Dictionary mapping category names to average embeddings
        categories: List of category names in order
        output_dir: Output directory for saving files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nSaving category average embeddings...")
    
    # Stack embeddings in order
    embeddings = np.array([category_averages[cat] for cat in categories])
    
    # Save as NPZ file (easy to load later)
    npz_path = output_dir / 'category_average_embeddings.npz'
    np.savez(npz_path, 
             embeddings=embeddings, 
             categories=np.array(categories))
    print(f"  Saved to {npz_path}")
    
    # Save as CSV file (for easy inspection) - non-critical, so use safe wrapper
    csv_path = output_dir / 'category_average_embeddings.csv'
    # Create DataFrame with categories as index and embedding dimensions as columns
    emb_df = pd.DataFrame(embeddings, index=categories)
    if safe_to_csv(emb_df, csv_path):
        print(f"  Saved to {csv_path}")
    
    # Save category names with embedding info
    info_path = output_dir / 'category_average_info.txt'
    with open(info_path, 'w') as f:
        f.write("Category Average Embeddings Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total categories: {len(categories)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n\n")
        f.write("Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"  {i}: {cat}\n")
    print(f"  Saved info to {info_path}")

def compute_similarity_matrix(category_averages, categories):
    """
    Compute pairwise cosine similarity matrix
    Args:
        category_averages: Dictionary mapping category names to average embeddings
        categories: List of category names in order
    Returns:
        Similarity matrix (numpy array)
    """
    print("\nComputing cosine similarity matrix...")
    
    # Stack embeddings in order
    embeddings = np.array([category_averages[cat] for cat in categories])
    
    # Normalize each embedding vector to unit length (axis=1: across columns/features for each row/category)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Compute cosine similarity = dot product of normalized embeddings
    # This gives true cosine similarity (bounded to [-1, 1])
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    return similarity_matrix

def compute_distance_matrix(similarity_matrix):
    """
    Compute cosine distance matrix from similarity matrix
    Distance = 1 - Cosine Similarity
    Args:
        similarity_matrix: numpy array of shape (n_categories, n_categories)
    Returns:
        distance_matrix: numpy array of shape (n_categories, n_categories)
    """
    print("Computing cosine distance matrix from similarity...")
    
    # Distance = 1 - Similarity
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is exactly 0
    np.fill_diagonal(distance_matrix, 0)
    
    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    return distance_matrix

def load_category_types(cdi_path):
    """
    Load category type information from CDI words CSV
    Args:
        cdi_path: Path to cdi_words.csv file
    Returns:
        Dictionary mapping category names to type information
    """
    print(f"\nLoading category types from {cdi_path}...")
    cdi_df = pd.read_csv(cdi_path)
    
    category_types = {}
    for _, row in cdi_df.iterrows():
        category_types[row['uni_lemma']] = {
            'is_animate': bool(row.get('is_animate', 0)),
            'is_bodypart': bool(row.get('is_bodypart', 0)),
            'is_small': bool(row.get('is_small', 0)),
            'is_big': bool(row.get('is_big', 0))
        }
    
    print(f"Loaded type information for {len(category_types)} categories")
    return category_types

def organize_categories_by_type(categories, category_types):
    """
    Organize categories into groups: animate, small, bodyparts, big
    Args:
        categories: List of category names
        category_types: Dictionary mapping category names to type information
    Returns:
        Dictionary with keys 'animate', 'small', 'bodyparts', 'big', 'others'
        Each value is a sorted list of category names
    """
    print("\nOrganizing categories by type...")
    
    organized = {
        'animate': [],
        'small': [],
        'bodyparts': [],
        'big': [],
        'others': []
    }
    
    for cat in categories:
        if cat not in category_types:
            organized['others'].append(cat)
            continue
        
        types = category_types[cat]
        if types['is_animate']:
            organized['animate'].append(cat)
        elif types['is_bodypart']:
            organized['bodyparts'].append(cat)
        elif types['is_small']:
            organized['small'].append(cat)
        elif types['is_big']:
            organized['big'].append(cat)
        else:
            organized['others'].append(cat)
    
    # Sort within each group
    for key in organized:
        organized[key] = sorted(organized[key])
        print(f"  {key}: {len(organized[key])} categories")
    
    return organized

def create_organized_rdm(distance_matrix, categories, organized_categories, output_dir, embedding_type='clip'):
    """
    Create RDM heatmap organized by category type groups
    Args:
        distance_matrix: Distance matrix
        categories: List of category names in original order
        organized_categories: Dictionary of organized categories
        output_dir: Output directory for saving plots
        embedding_type: Type of embeddings ('clip' or 'dinov3')
    """
    output_dir = Path(output_dir)
    
    print("\nCreating organized RDM heatmap...")
    
    # Create ordered list: animate -> bodyparts -> small -> big -> others
    ordered_categories = (
        organized_categories['animate'] +
        organized_categories['bodyparts'] +
        organized_categories['small'] +
        organized_categories['big'] +
        organized_categories['others']
    )
    
    # Create mapping from category to index
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Get ordered indices
    ordered_indices = [cat_to_idx[cat] for cat in ordered_categories if cat in cat_to_idx]
    
    # Reorder distance matrix
    reordered_matrix = distance_matrix[np.ix_(ordered_indices, ordered_indices)]
    reordered_categories = [cat for cat in ordered_categories if cat in cat_to_idx]
    
    # Calculate boundaries between groups
    boundaries = []
    current_pos = 0
    for group_name in ['animate', 'bodyparts', 'small', 'big', 'others']:
        group_cats = [cat for cat in organized_categories[group_name] if cat in cat_to_idx]
        if len(group_cats) > 0:
            if current_pos > 0:
                boundaries.append(current_pos)
            current_pos += len(group_cats)
    
    # Calculate figure size
    n_categories = len(reordered_categories)
    fig_size = max(20, n_categories * 0.5)
    
    plt.figure(figsize=(fig_size, fig_size))
    
    ax = sns.heatmap(reordered_matrix, 
                xticklabels=reordered_categories,
                yticklabels=reordered_categories,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=2,
                square=True,
                cbar_kws={'label': 'Cosine Distance', 'shrink': 0.8})
    
    # Add lines to separate category groups
    for boundary in boundaries:
        ax.axhline(y=boundary, color='black', linewidth=2)
        ax.axvline(x=boundary, color='black', linewidth=2)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Cosine Distance', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.title(f'{embedding_type.upper()} Category RDM (Organized by Type)\nAnimate → Bodyparts → Small → Big → Others', 
              fontsize=24, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    plt.savefig(output_dir / 'rdm_organized_by_type.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    
    print(f"Saved organized RDM to {output_dir / 'rdm_organized_by_type.png'}")
    
    # Save category organization info
    with open(output_dir / 'category_organization.txt', 'w') as f:
        f.write("Category Organization by Type:\n\n")
        for group_name in ['animate', 'bodyparts', 'small', 'big', 'others']:
            group_cats = [cat for cat in organized_categories[group_name] if cat in cat_to_idx]
            if len(group_cats) > 0:
                f.write(f"\n{group_name.upper()} ({len(group_cats)} categories):\n")
                for cat in group_cats:
                    f.write(f"  - {cat}\n")

def create_full_rdm(distance_matrix, categories, output_dir, embedding_type='clip'):
    """
    Create full RDM heatmap with all categories
    Args:
        distance_matrix: Distance matrix
        categories: List of category names
        output_dir: Output directory for saving plots
        embedding_type: Type of embeddings ('clip' or 'dinov3')
    """
    output_dir = Path(output_dir)
    
    print("\nCreating full RDM heatmap...")
    
    n_categories = len(categories)
    fig_size = max(20, n_categories * 0.5)
    
    plt.figure(figsize=(fig_size, fig_size))
    
    ax = sns.heatmap(distance_matrix, 
                xticklabels=categories,
                yticklabels=categories,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=2,
                square=True,
                cbar_kws={'label': 'Cosine Distance', 'shrink': 0.8})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Cosine Distance', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.title(f'{embedding_type.upper()} Category Representational Distance Matrix (RDM)\n(Cosine Distance: 1 - Cosine Similarity)', 
              fontsize=24, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    plt.savefig(output_dir / 'rdm_full.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    
    print(f"Saved full RDM to {output_dir / 'rdm_full.png'}")

def save_data(similarity_matrix, distance_matrix, categories, output_dir):
    """
    Save similarity matrix, distance matrix, and category names
    Args:
        similarity_matrix: Similarity matrix
        distance_matrix: Distance matrix
        categories: List of category names
        output_dir: Output directory for saving files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nSaving data files...")
    
    # Save as numpy arrays
    np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)
    np.save(output_dir / 'distance_matrix.npy', distance_matrix)
    
    # Save as CSV with category names - use safe wrapper
    sim_df = pd.DataFrame(similarity_matrix, index=categories, columns=categories)
    if safe_to_csv(sim_df, output_dir / 'similarity_matrix.csv'):
        print(f"  Saved similarity matrix CSV")
    
    dist_df = pd.DataFrame(distance_matrix, index=categories, columns=categories)
    if safe_to_csv(dist_df, output_dir / 'distance_matrix.csv'):
        print(f"  Saved distance matrix CSV")
    
    # Save category names with indices
    cat_df = pd.DataFrame({'index': range(len(categories)), 'category': categories})
    if safe_to_csv(cat_df, output_dir / 'category_names.txt', index=False, sep='\t'):
        print(f"  Saved category names")
    
    print(f"Saved data files to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Compute pairwise RDM for CLIP or DINOv3 category embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # For CLIP embeddings (with embedding list file):
  python compute_clip_category_rdm.py \\
    --embedding_list /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_new/clip_image_embeddings_doc_normalized_filtered-by-clip-0.26.txt \\
    --embeddings_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_new \\
    --output_dir ./clip_rdm_results \\
    --cdi_path ./data/cdi_words.csv \\
    --embedding_type clip
  
  # For DINOv3 embeddings (auto-scan directory - no list file needed):
  python compute_clip_category_rdm.py \\
    --embeddings_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m \\
    --output_dir ./dinov3_rdm_results \\
    --cdi_path ./data/cdi_words.csv \\
    --embedding_type dinov3
  
  # For DINOv3 embeddings (match from CLIP list - ensures same images):
  python compute_clip_category_rdm.py \\
    --embedding_list /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_new/clip_image_embeddings_doc_normalized_filtered-by-clip-0.24.txt \\
    --embeddings_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m \\
    --output_dir ./dinov3_rdm_results_24 \\
    --cdi_path ./data/cdi_words.csv \\
    --embedding_type dinov3 \\
    --match_from_list
  
  # For DINOv3 embeddings (with embedding list file, if you want to filter):
  python compute_clip_category_rdm.py \\
    --embedding_list /path/to/dinov3_embeddings_list.txt \\
    --embeddings_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m \\
    --output_dir ./dinov3_rdm_results \\
    --cdi_path ./data/cdi_words.csv \\
    --embedding_type dinov3
        """
    )
    
    parser.add_argument(
        '--embedding_list',
        type=str,
        default=None,
        help='Path to text file containing embedding paths (one per line). If not provided, will scan embeddings_dir for all .npy files. If provided and --match_from_list is used, will match filenames from this list to embeddings_dir.'
    )
    parser.add_argument(
        '--match_from_list',
        action='store_true',
        help='If set, match filenames from embedding_list to embeddings_dir (useful for using CLIP list with DINOv3 embeddings)'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        required=True,
        help='Base directory for embeddings (used if paths in list are relative)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./clip_rdm_results',
        help='Output directory for RDM results and visualizations (default: ./clip_rdm_results)'
    )
    parser.add_argument(
        '--cdi_path',
        type=str,
        default='./data/cdi_words.csv',
        help='Path to CDI words CSV file (default: ./data/cdi_words.csv)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers for loading embeddings (default: auto-detect, max 16)'
    )
    parser.add_argument(
        '--no_parallel',
        action='store_true',
        help='Disable parallel loading (use sequential loading instead)'
    )
    parser.add_argument(
        '--embedding_type',
        type=str,
        choices=['clip', 'dinov3'],
        default='clip',
        help='Type of embeddings: "clip" or "dinov3" (default: clip)'
    )
    
    args = parser.parse_args()
    
    # Run pre-flight checks before starting long computation
    try:
        preflight_checks()
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print("PRE-FLIGHT CHECK FAILED")
        print(f"{'='*60}")
        print(str(e))
        print(f"\nPlease fix the issue before running the script.")
        sys.exit(1)
    
    # Load embedding paths
    if args.embedding_list:
        if args.match_from_list:
            # Match filenames from reference list to target directory
            embedding_paths = match_embedding_paths_from_list(args.embedding_list, args.embeddings_dir)
        else:
            # Use paths directly from list
            embedding_paths = load_embedding_paths(args.embedding_list)
    else:
        print("No embedding list provided, scanning directory for all .npy files...")
        embedding_paths = scan_embedding_directory(args.embeddings_dir)
    
    # Load embeddings by category
    embeddings_by_category = load_embeddings_by_category(
        embedding_paths, 
        args.embeddings_dir,
        num_workers=args.num_workers,
        use_parallel=not args.no_parallel
    )
    
    # Compute category averages
    category_averages, categories = compute_category_averages(embeddings_by_category)
    
    # Save category averages
    output_dir = Path(args.output_dir)
    save_category_averages(category_averages, categories, output_dir)
    
    # Compute similarity and distance matrices
    similarity_matrix = compute_similarity_matrix(category_averages, categories)
    distance_matrix = compute_distance_matrix(similarity_matrix)
    
    # Save data (RDM matrices)
    save_data(similarity_matrix, distance_matrix, categories, output_dir)
    
    # Load category types and organize
    cdi_path = Path(args.cdi_path)
    if cdi_path.exists():
        category_types = load_category_types(cdi_path)
        organized_categories = organize_categories_by_type(categories, category_types)
        
        # Create organized RDM
        create_organized_rdm(distance_matrix, categories, organized_categories, output_dir, args.embedding_type)
    else:
        print(f"\nWarning: CDI path {cdi_path} does not exist. Skipping organized RDM.")
        print(f"  Expected location: {cdi_path.absolute()}")
    
    # Create full RDM
    create_full_rdm(distance_matrix, categories, output_dir, args.embedding_type)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Total categories: {len(categories)}")
    print(f"Embedding dimension: {list(category_averages.values())[0].shape[0]}")
    print(f"Mean similarity: {similarity_matrix.mean():.4f}")
    print(f"Mean distance: {distance_matrix.mean():.4f}")
    print(f"Min distance: {distance_matrix[distance_matrix > 0].min():.4f}")
    print(f"Max distance: {distance_matrix.max():.4f}")
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main()

