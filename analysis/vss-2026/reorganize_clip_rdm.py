#!/usr/bin/env python3
"""
Reorganize CLIP category RDM by loading saved average embeddings,
filtering categories (by exclusion or inclusion), and reorganizing by type groups.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import squareform

def load_excluded_categories(exclusion_file):
    """Load list of categories to exclude"""
    print(f"Loading excluded categories from {exclusion_file}...")
    with open(exclusion_file, 'r') as f:
        excluded = set(line.strip() for line in f if line.strip())
    print(f"Found {len(excluded)} categories to exclude")
    return excluded

def load_included_categories(inclusion_file):
    """Load list of categories to include"""
    print(f"Loading included categories from {inclusion_file}...")
    with open(inclusion_file, 'r') as f:
        included = set(line.strip() for line in f if line.strip())
    print(f"Found {len(included)} categories to include")
    return included

def load_category_averages(npz_path):
    """Load saved category average embeddings"""
    print(f"Loading category averages from {npz_path}...")
    data = np.load(npz_path)
    
    # Check available keys
    available_keys = list(data.keys())
    print(f"Available keys in NPZ file: {available_keys}")
    
    # Get embeddings
    if 'embeddings' in available_keys:
        embeddings = data['embeddings']
    else:
        raise KeyError(f"Expected 'embeddings' key in NPZ file. Available keys: {available_keys}")
    
    # Get categories (try multiple possible key names)
    categories = None
    for key_name in ['categories', 'category', 'labels', 'label']:
        if key_name in available_keys:
            categories = data[key_name]
            # Convert numpy array of strings to list
            if categories.dtype == 'object':
                categories = [str(cat) for cat in categories]
            else:
                categories = categories.tolist()
            print(f"Found categories under key '{key_name}'")
            break
    
    if categories is None:
        # If no categories, create generic names based on indices
        print(f"Warning: No 'categories', 'category', 'labels', or 'label' key found. Creating generic category names.")
        n_categories = embeddings.shape[0]
        categories = [f"category_{i}" for i in range(n_categories)]
    
    print(f"Loaded {len(categories)} categories with embeddings of shape {embeddings.shape}")
    return embeddings, categories

def load_category_organization_from_file(org_file_path, categories):
    """
    Load category organization from a text file (matching code snippet approach)
    File format: lines starting with "  - " followed by category name
    Args:
        org_file_path: Path to organization text file
        categories: List of available categories (set for fast lookup)
    Returns:
        List of category names in the order specified in the file (only those that exist)
        Array of indices (for reference, but may not be used)
    """
    print(f"\nLoading category organization from {org_file_path}...")
    org_df = pd.read_csv(org_file_path, header=None)
    new_order = org_df.values.flatten()
    
    # Extract category names from lines starting with "  - "
    new_order = [x[4:99] for x in new_order if x[0:4] == "  - "]
    
    # Convert categories to set for fast lookup
    categories_set = set(categories)
    
    # Filter to only include categories that exist
    ordered_categories = [x for x in new_order if x in categories_set]
    
    # Count how many were skipped
    skipped = len(new_order) - len(ordered_categories)
    if skipped > 0:
        print(f"Warning: {skipped} categories from organization file not found in available categories")
    
    print(f"Loaded {len(ordered_categories)} categories from organization file")
    return ordered_categories, None  # Return None for indices as they're not needed

def load_category_types(cdi_path):
    """Load category type information from CDI words CSV"""
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

def cluster_categories_within_group(group_categories, cat_to_embedding, method='ward', metric='euclidean'):
    """
    Perform hierarchical clustering within a group of categories.
    
    Args:
        group_categories: List of category names in the group
        cat_to_embedding: Dictionary mapping category names to embeddings
        method: Linkage method for clustering (default: 'ward')
        metric: Distance metric (default: 'euclidean', ignored if method='ward')
    
    Returns:
        List of category names reordered according to clustering dendrogram
    """
    if len(group_categories) <= 1:
        return group_categories
    
    # Get embeddings for this group
    group_embeddings = np.array([cat_to_embedding[cat] for cat in group_categories])
    
    # Compute distance matrix (1 - cosine similarity)
    # First normalize embeddings
    normalized_embeddings = (group_embeddings - group_embeddings.mean(axis=0)) / (group_embeddings.std(axis=0) + 1e-10)
    similarity_matrix = cosine_similarity(normalized_embeddings)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form for linkage
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    if method == 'ward':
        # Ward method requires Euclidean distance, so we'll use the distance matrix directly
        # Convert similarity to distance and ensure it's Euclidean-like
        linkage_matrix = linkage(condensed_distances, method=method)
    else:
        linkage_matrix = linkage(condensed_distances, method=method, metric=metric)
    
    # Get optimal leaf ordering for better visualization
    try:
        linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_distances)
    except:
        # If optimal leaf ordering fails, use original linkage
        pass
    
    # Extract the order from the dendrogram
    # The dendrogram function returns a dictionary with 'leaves' key
    dendro_dict = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro_dict['leaves']
    
    # Reorder categories according to clustering
    clustered_categories = [group_categories[i] for i in leaf_order]
    
    return clustered_categories

def filter_and_organize_categories(categories, embeddings, excluded_categories, category_types, included_categories=None, use_clustering=True):
    """Filter categories (by exclusion or inclusion) and organize by type, with optional hierarchical clustering within groups"""
    print("\nFiltering and organizing categories...")
    
    # Filter categories based on inclusion or exclusion
    filtered_indices = []
    filtered_categories = []
    filtered_embeddings = []
    
    if included_categories is not None:
        # Use inclusion list: only include categories in the list
        categories_set = set(categories)
        matched_categories = included_categories & categories_set
        
        if len(matched_categories) == 0:
            print(f"\nERROR: No categories from inclusion file matched categories in NPZ file!")
            print(f"  Categories in NPZ file: {len(categories)} total")
            print(f"  Categories in inclusion file: {len(included_categories)} total")
            print(f"  Matched: {len(matched_categories)}")
            print(f"\n  First 10 categories in NPZ file:")
            for cat in list(categories)[:10]:
                print(f"    - '{cat}'")
            print(f"\n  First 10 categories in inclusion file:")
            for cat in list(included_categories)[:10]:
                print(f"    - '{cat}'")
            raise ValueError("No matching categories found between inclusion file and NPZ file. Check for name mismatches, whitespace, or case sensitivity.")
        
        for idx, cat in enumerate(categories):
            if cat in included_categories:
                filtered_indices.append(idx)
                filtered_categories.append(cat)
                filtered_embeddings.append(embeddings[idx])
        print(f"After filtering: {len(filtered_categories)} categories (included from {len(included_categories)} specified categories, {len(matched_categories)} matched)")
    else:
        # Use exclusion list: exclude specified categories
        for idx, cat in enumerate(categories):
            if cat not in excluded_categories:
                filtered_indices.append(idx)
                filtered_categories.append(cat)
                filtered_embeddings.append(embeddings[idx])
        print(f"After filtering: {len(filtered_categories)} categories (excluded {len(excluded_categories)} categories)")
    
    # Organize by type: animals, bodyparts, big objects, small objects
    organized = {
        'animals': [],
        'bodyparts': [],
        'big_objects': [],
        'small_objects': [],
        'others': []
    }
    
    # Create mapping from category to embedding
    cat_to_embedding = {cat: emb for cat, emb in zip(filtered_categories, filtered_embeddings)}
    
    for cat in filtered_categories:
        if cat not in category_types:
            organized['others'].append(cat)
            continue
        
        types = category_types[cat]
        if types['is_animate']:
            organized['animals'].append(cat)
        elif types['is_bodypart']:
            organized['bodyparts'].append(cat)
        elif types['is_big']:
            organized['big_objects'].append(cat)
        elif types['is_small']:
            organized['small_objects'].append(cat)
        else:
            organized['others'].append(cat)
    
    # Cluster or sort within each group
    for key in organized:
        if use_clustering and len(organized[key]) > 1:
            print(f"  Clustering {key} ({len(organized[key])} categories)...")
            organized[key] = cluster_categories_within_group(organized[key], cat_to_embedding)
        else:
            organized[key] = sorted(organized[key])
        print(f"  {key}: {len(organized[key])} categories")
    
    # Create ordered list
    ordered_categories = (
        organized['animals'] +
        organized['bodyparts'] +
        organized['big_objects'] +
        organized['small_objects'] +
        organized['others']
    )
    
    # Create ordered embeddings array
    ordered_embeddings = np.array([cat_to_embedding[cat] for cat in ordered_categories])
    
    return ordered_categories, ordered_embeddings, organized, cat_to_embedding

def save_dendrograms(organized_categories, cat_to_embedding, output_dir):
    """Save dendrogram plots for each category group"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nSaving dendrograms for each category group...")
    
    for group_name in ['animals', 'bodyparts', 'big_objects', 'small_objects']:
        group_categories = organized_categories[group_name]
        if len(group_categories) <= 1:
            print(f"  Skipping {group_name}: only {len(group_categories)} category")
            continue
        
        # Get embeddings for this group
        group_embeddings = np.array([cat_to_embedding[cat] for cat in group_categories])
        
        # Compute distance matrix
        normalized_embeddings = (group_embeddings - group_embeddings.mean(axis=0)) / (group_embeddings.std(axis=0) + 1e-10)
        similarity_matrix = cosine_similarity(normalized_embeddings)
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed form
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='ward')
        try:
            linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_distances)
        except:
            pass
        
        # Create dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, 
                  labels=group_categories,
                  leaf_rotation=90,
                  leaf_font_size=10)
        plt.title(f'Hierarchical Clustering Dendrogram: {group_name.upper()}\n({len(group_categories)} categories)',
                 fontsize=16, pad=20)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        # Save as PNG
        output_path_png = output_dir / f'dendrogram_{group_name}.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"  Saved dendrogram for {group_name} to {output_path_png}")
        
        # Save as PDF
        output_path_pdf = output_dir / f'dendrogram_{group_name}.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight', pad_inches=0.2)
        print(f"  Saved dendrogram for {group_name} to {output_path_pdf}")
        
        plt.close()

def compute_similarity_matrix(embeddings):
    """Compute pairwise similarity matrix from z-score normalized embeddings"""
    print("\nComputing similarity matrix...")
    
    # Z-score normalization: subtract mean and divide by std (per feature dimension)
    # This matches the code snippet approach
    # Add small epsilon to avoid division by zero
    std = embeddings.std(axis=0)
    normalized_embeddings = (embeddings - embeddings.mean(axis=0)) / (std + 1e-10)
    
    # Compute cosine similarity using sklearn
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    return similarity_matrix

def compute_distance_matrix(similarity_matrix):
    """Compute distance matrix from similarity matrix"""
    print("Computing distance matrix from similarity...")
    
    # Distance = 1 - Similarity
    distance_matrix = 1 - similarity_matrix

    np.fill_diagonal(distance_matrix, 0)
    
    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    return distance_matrix

def create_organized_rdm(distance_matrix, categories, organized_categories, output_dir):
    """Create RDM heatmap organized by category type groups"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating organized RDM heatmap...")
    
    # Calculate boundaries between groups
    boundaries = []
    current_pos = 0
    for group_name in ['animals', 'bodyparts', 'big_objects', 'small_objects', 'others']:
        group_cats = organized_categories[group_name]
        if len(group_cats) > 0:
            if current_pos > 0:
                boundaries.append(current_pos)
            current_pos += len(group_cats)
    
    # Calculate figure size
    n_categories = len(categories)
    fig_size = max(20, n_categories * 0.5)
    
    # Use fixed colormap range to match example RDM
    vmin = 0
    vmax = 2
    
    # Calculate stats for reference
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    data_min = np.min(distance_matrix[mask])
    data_max = np.max(distance_matrix[mask])
    data_mean = np.mean(distance_matrix[mask])
    
    print(f"Distance matrix stats: min={data_min:.4f}, max={data_max:.4f}, mean={data_mean:.4f}")
    print(f"Using colormap range: vmin={vmin:.4f}, vmax={vmax:.4f}")
    
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use a better colormap that shows clusters more clearly
    # 'viridis' or 'plasma' work well, or 'coolwarm' for symmetric data
    ax = sns.heatmap(distance_matrix, 
                xticklabels=categories,
                yticklabels=categories,
                cmap='viridis',  # Changed from 'RdYlBu_r' to 'viridis' for better visibility
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Distance (1 - Cosine Similarity)', 'shrink': 0.8})
    
    # Add lines to separate category groups
    for boundary in boundaries:
        ax.axhline(y=boundary, color='white', linewidth=2, linestyle='--')
        ax.axvline(x=boundary, color='white', linewidth=2, linestyle='--')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Distance (1 - Cosine Similarity)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.title('CLIP Category RDM (Organized by Type)\nAnimals → Bodyparts → Big Objects → Small Objects → Others', 
              fontsize=24, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    output_path = output_dir / 'rdm_organized_filtered.png'
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    
    print(f"Saved organized RDM to {output_path}")
    
    # Also create a version with 'coolwarm' colormap for comparison
    plt.figure(figsize=(fig_size, fig_size))
    # Calculate mean for centering coolwarm colormap
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    data_mean = np.mean(distance_matrix[mask])
    ax = sns.heatmap(distance_matrix, 
                xticklabels=categories,
                yticklabels=categories,
                cmap='coolwarm',
                center=data_mean,  # Center colormap at mean
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Distance (1 - Cosine Similarity)', 'shrink': 0.8})
    
    for boundary in boundaries:
        ax.axhline(y=boundary, color='black', linewidth=2, linestyle='--')
        ax.axvline(x=boundary, color='black', linewidth=2, linestyle='--')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Distance (1 - Cosine Similarity)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.title('CLIP Category RDM (Organized by Type)\nAnimals → Bodyparts → Big Objects → Small Objects → Others', 
              fontsize=24, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    output_path_coolwarm = output_dir / 'rdm_organized_filtered_coolwarm.png'
    plt.savefig(output_path_coolwarm, 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    
    print(f"Saved organized RDM (coolwarm) to {output_path_coolwarm}")
    
    # Save category organization info
    with open(output_dir / 'category_organization_filtered.txt', 'w') as f:
        f.write("Category Organization by Type (After Filtering):\n\n")
        for group_name in ['animals', 'bodyparts', 'big_objects', 'small_objects', 'others']:
            group_cats = organized_categories[group_name]
            if len(group_cats) > 0:
                f.write(f"\n{group_name.upper()} ({len(group_cats)} categories):\n")
                for cat in group_cats:
                    f.write(f"  - {cat}\n")

def extract_top_pairs(distance_matrix, categories, n_pairs, most_similar=True):
    """
    Extract top N most similar or most dissimilar pairs
    Args:
        distance_matrix: Distance matrix
        categories: List of category names
        n_pairs: Number of pairs to extract
        most_similar: If True, extract most similar (lowest distance), else most dissimilar (highest distance)
    Returns:
        List of tuples: (category1, category2, distance)
    """
    n = len(categories)
    
    # Get upper triangle (excluding diagonal) to avoid duplicates
    upper_triangle = np.triu(np.ones((n, n)), k=1).astype(bool)
    distances_flat = distance_matrix[upper_triangle]
    indices_flat = np.where(upper_triangle)
    
    # Get indices sorted by distance
    if most_similar:
        sorted_indices = np.argsort(distances_flat)[:n_pairs]
    else:
        sorted_indices = np.argsort(distances_flat)[-n_pairs:][::-1]  # Reverse to get highest first
    
    pairs = []
    for idx in sorted_indices:
        i, j = indices_flat[0][idx], indices_flat[1][idx]
        pairs.append((categories[i], categories[j], distances_flat[idx]))
    
    return pairs

def plot_top_pairs(distance_matrix, categories, n_pairs, output_dir, most_similar=True):
    """
    Plot top N pairs as a focused heatmap
    Args:
        distance_matrix: Full distance matrix
        categories: List of category names
        n_pairs: Number of pairs to plot
        output_dir: Output directory
        most_similar: If True, plot most similar pairs, else most dissimilar
    """
    output_dir = Path(output_dir)
    
    # Extract top pairs
    pairs = extract_top_pairs(distance_matrix, categories, n_pairs, most_similar=most_similar)
    
    # Get unique categories from pairs
    unique_cats = sorted(set([cat for pair in pairs for cat in pair[:2]]))
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    
    # Create submatrix for these categories
    n_unique = len(unique_cats)
    submatrix = np.zeros((n_unique, n_unique))
    
    for i, cat1 in enumerate(unique_cats):
        for j, cat2 in enumerate(unique_cats):
            # Find original indices
            orig_i = categories.index(cat1)
            orig_j = categories.index(cat2)
            submatrix[i, j] = distance_matrix[orig_i, orig_j]
    
    # Create figure
    fig_size = max(12, n_unique * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use fixed colormap range to match example RDM
    vmin = 0
    vmax = 2
    
    ax = sns.heatmap(submatrix,
                    xticklabels=unique_cats,
                    yticklabels=unique_cats,
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    square=True,
                    cbar_kws={'label': 'Distance (1 - Cosine Similarity)', 'shrink': 0.8},
                    annot=False)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Distance (1 - Cosine Similarity)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    pair_type = "Most Similar" if most_similar else "Most Dissimilar"
    plt.title(f'Top {n_pairs} {pair_type} Category Pairs\n({len(unique_cats)} unique categories)', 
              fontsize=18, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    pair_suffix = "similar" if most_similar else "dissimilar"
    output_path = output_dir / f'top_{n_pairs}_{pair_suffix}_pairs.png'
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()
    
    print(f"Saved top {n_pairs} {pair_suffix} pairs plot to {output_path}")
    
    # Save pairs list to text file
    pairs_file = output_dir / f'top_{n_pairs}_{pair_suffix}_pairs.txt'
    with open(pairs_file, 'w') as f:
        f.write(f"Top {n_pairs} {pair_type} Category Pairs\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Category 1':<30} {'Category 2':<30} {'Distance':<10}\n")
        f.write("-" * 70 + "\n")
        for cat1, cat2, dist in pairs:
            f.write(f"{cat1:<30} {cat2:<30} {dist:.6f}\n")
    
    print(f"Saved pairs list to {pairs_file}")

def save_data(similarity_matrix, distance_matrix, categories, output_dir):
    """Save similarity matrix, distance matrix, and category names"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nSaving data files...")
    
    # Save as numpy arrays
    np.save(output_dir / 'similarity_matrix_filtered.npy', similarity_matrix)
    np.save(output_dir / 'distance_matrix_filtered.npy', distance_matrix)
    
    # Save as CSV with category names
    sim_df = pd.DataFrame(similarity_matrix, index=categories, columns=categories)
    sim_df.to_csv(output_dir / 'similarity_matrix_filtered.csv')
    print(f"  Saved similarity matrix CSV")
    
    dist_df = pd.DataFrame(distance_matrix, index=categories, columns=categories)
    dist_df.to_csv(output_dir / 'distance_matrix_filtered.csv')
    print(f"  Saved distance matrix CSV")
    
    # Save category names with indices
    cat_df = pd.DataFrame({'index': range(len(categories)), 'category': categories})
    cat_df.to_csv(output_dir / 'category_names_filtered.txt', index=False, sep='\t')
    print(f"  Saved category names")
    
    print(f"Saved data files to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Reorganize CLIP category RDM by loading saved embeddings and excluding categories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using type-based organization with exclusion (default):
  python reorganize_clip_rdm.py \\
    --npz_path ./clip_rdm_results_24/category_average_embeddings.npz \\
    --exclusion_file ../data/categories_with_zero_precision_simple.txt \\
    --cdi_path ../data/cdi_words.csv \\
    --output_dir ./clip_rdm_results_24_filtered
  
  # Using inclusion file to specify which categories to include:
  python reorganize_clip_rdm.py \\
    --npz_path ./clip_rdm_results_24/category_average_embeddings.npz \\
    --inclusion_file ../data/categories_to_include.txt \\
    --cdi_path ../data/cdi_words.csv \\
    --output_dir ./clip_rdm_results_24_filtered
  
  # Using organization file (matching code snippet approach):
  python reorganize_clip_rdm.py \\
    --npz_path ./clip_rdm_results_24/category_average_embeddings.npz \\
    --exclusion_file ../data/categories_with_zero_precision_simple.txt \\
    --organization_file ./clip_rdm_results/category_organization.txt \\
    --output_dir ./clip_rdm_results_24_filtered
        """
    )
    
    parser.add_argument(
        '--npz_path',
        type=str,
        required=True,
        help='Path to NPZ file containing saved category average embeddings'
    )
    parser.add_argument(
        '--exclusion_file',
        type=str,
        default=None,
        help='Path to text file containing categories to exclude (one per line). Ignored if --inclusion_file is provided.'
    )
    parser.add_argument(
        '--inclusion_file',
        type=str,
        default=None,
        help='Path to text file containing categories to include (one per line). If provided, only these categories will be included in the RDM. Takes precedence over --exclusion_file.'
    )
    parser.add_argument(
        '--cdi_path',
        type=str,
        default='./data/cdi_words.csv',
        help='Path to CDI words CSV file (default: ./data/cdi_words.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./clip_rdm_results_filtered',
        help='Output directory for reorganized RDM results (default: ./clip_rdm_results_filtered)'
    )
    parser.add_argument(
        '--organization_file',
        type=str,
        default=None,
        help='Optional path to text file containing category organization (lines starting with "  - "). If provided, will use this order instead of organizing by type.'
    )
    parser.add_argument(
        '--no_clustering',
        action='store_true',
        help='Disable hierarchical clustering within each category group (default: clustering enabled)'
    )
    parser.add_argument(
        '--save_dendrograms',
        action='store_true',
        help='Save dendrogram plots for each category group'
    )
    
    args = parser.parse_args()
    
    # Validate that either exclusion_file or inclusion_file is provided
    if not args.exclusion_file and not args.inclusion_file:
        parser.error("Either --exclusion_file or --inclusion_file must be provided")
    
    # Load included or excluded categories
    included_categories = None
    excluded_categories = set()
    
    if args.inclusion_file:
        # Use inclusion file (takes precedence)
        included_categories = load_included_categories(args.inclusion_file)
    elif args.exclusion_file:
        # Use exclusion file
        excluded_categories = load_excluded_categories(args.exclusion_file)
    
    # Load category averages
    embeddings, categories = load_category_averages(args.npz_path)
    
    # Organize categories - either from file or by type
    if args.organization_file:
        # Use organization from file (matching code snippet approach)
        org_path = Path(args.organization_file)
        if not org_path.exists():
            print(f"Warning: Organization file {org_path} does not exist. Falling back to type-based organization.")
            args.organization_file = None
        
    if args.organization_file:
        # Filter categories based on inclusion or exclusion
        if included_categories is not None:
            categories_set = set(categories)
            matched_categories = included_categories & categories_set
            
            if len(matched_categories) == 0:
                print(f"\nERROR: No categories from inclusion file matched categories in NPZ file!")
                print(f"  Categories in NPZ file: {len(categories)} total")
                print(f"  Categories in inclusion file: {len(included_categories)} total")
                print(f"  Matched: {len(matched_categories)}")
                print(f"\n  First 10 categories in NPZ file:")
                for cat in list(categories)[:10]:
                    print(f"    - '{cat}'")
                print(f"\n  First 10 categories in inclusion file:")
                for cat in list(included_categories)[:10]:
                    print(f"    - '{cat}'")
                raise ValueError("No matching categories found between inclusion file and NPZ file. Check for name mismatches, whitespace, or case sensitivity.")
            
            filtered_categories = [cat for cat in categories if cat in included_categories]
            print(f"After filtering: {len(filtered_categories)} categories (included from {len(included_categories)} specified categories, {len(matched_categories)} matched)")
        else:
            filtered_categories = [cat for cat in categories if cat not in excluded_categories]
            print(f"After filtering: {len(filtered_categories)} categories (excluded {len(excluded_categories)} categories)")
        
        if len(filtered_categories) == 0:
            raise ValueError("No categories remaining after filtering. Cannot proceed with empty category list.")
        
        filtered_embeddings = np.array([embeddings[categories.index(cat)] for cat in filtered_categories])
        
        # Load organization from file
        ordered_categories, ordered_indices = load_category_organization_from_file(org_path, filtered_categories)
        
        # Create ordered embeddings array (only include categories that exist after filtering)
        cat_to_embedding = {cat: emb for cat, emb in zip(filtered_categories, filtered_embeddings)}
        ordered_embeddings = np.array([cat_to_embedding[cat] for cat in ordered_categories if cat in cat_to_embedding])
        # Filter ordered_categories to only include those that exist
        ordered_categories = [cat for cat in ordered_categories if cat in cat_to_embedding]
        
        # Create a dummy organized_categories structure for compatibility
        organized_categories = {'animals': [], 'bodyparts': [], 'big_objects': [], 'small_objects': [], 'others': ordered_categories}
    else:
        # Use type-based organization (original approach)
        cdi_path = Path(args.cdi_path)
        if not cdi_path.exists():
            print(f"Warning: CDI path {cdi_path} does not exist. Cannot organize by type.")
            print(f"  Expected location: {cdi_path.absolute()}")
            return
        
        category_types = load_category_types(cdi_path)
        
        # Filter and organize categories
        use_clustering = not args.no_clustering
        ordered_categories, ordered_embeddings, organized_categories, cat_to_embedding = filter_and_organize_categories(
            categories, embeddings, excluded_categories, category_types, included_categories=included_categories, use_clustering=use_clustering
        )
        
        # Save dendrograms if requested
        if args.save_dendrograms and use_clustering:
            save_dendrograms(organized_categories, cat_to_embedding, args.output_dir)
    
    # Validate that we have categories before proceeding
    if len(ordered_categories) == 0:
        raise ValueError("No categories remaining after filtering. Cannot compute RDM with empty category list.")
    
    if ordered_embeddings.shape[0] == 0:
        raise ValueError("No embeddings remaining after filtering. Cannot compute RDM with empty embeddings.")
    
    # Compute similarity and distance matrices
    similarity_matrix = compute_similarity_matrix(ordered_embeddings)
    distance_matrix = compute_distance_matrix(similarity_matrix)
    
    # Save data
    save_data(similarity_matrix, distance_matrix, ordered_categories, args.output_dir)
    
    # Create organized RDM
    create_organized_rdm(distance_matrix, ordered_categories, organized_categories, args.output_dir)
    
    # Plot top pairs
    print("\n" + "="*60)
    print("PLOTTING TOP PAIRS")
    print("="*60)
    for n in [20, 30, 50]:
        print(f"\nPlotting top {n} most similar pairs...")
        plot_top_pairs(distance_matrix, ordered_categories, n, args.output_dir, most_similar=True)
        print(f"Plotting top {n} most dissimilar pairs...")
        plot_top_pairs(distance_matrix, ordered_categories, n, args.output_dir, most_similar=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Original categories: {len(categories)}")
    if included_categories is not None:
        print(f"Included categories: {len(included_categories)}")
    else:
        print(f"Excluded categories: {len(excluded_categories)}")
    print(f"Remaining categories: {len(ordered_categories)}")
    print(f"Embedding dimension: {ordered_embeddings.shape[1]}")
    print(f"Mean similarity: {similarity_matrix.mean():.4f}")
    print(f"Mean distance: {distance_matrix.mean():.4f}")
    print(f"Min distance: {distance_matrix[distance_matrix > 0].min():.4f}")
    print(f"Max distance: {distance_matrix.max():.4f}")
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
