#!/usr/bin/env python3
"""
Correlate within-category average embeddings between two embedding files.
For each category that exists in both files, computes correlation between
the average embedding vectors.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(file_path):
    """
    Load embeddings and categories from NPZ file.
    
    Args:
        file_path: Path to NPZ file
        
    Returns:
        Tuple of (embeddings array, categories list)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix != '.npz':
        raise ValueError(f"Expected .npz file, got: {file_path.suffix}")
    
    print(f"Loading embeddings from: {file_path}")
    data = np.load(file_path)
    
    available_keys = list(data.keys())
    print(f"  Available keys: {available_keys}")
    
    # Get embeddings
    if 'embeddings' not in available_keys:
        raise KeyError(f"Expected 'embeddings' key in NPZ file. Available keys: {available_keys}")
    
    embeddings = data['embeddings']
    print(f"  Embeddings shape: {embeddings.shape}")
    
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
            print(f"  Found categories under key '{key_name}': {len(categories)} categories")
            break
    
    if categories is None:
        raise KeyError(f"No category/label key found. Available keys: {available_keys}")
    
    if len(categories) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between number of categories ({len(categories)}) "
            f"and embeddings ({embeddings.shape[0]})"
        )
    
    return embeddings, categories


def compute_vector_correlations(vec1, vec2):
    """
    Compute multiple correlation/similarity metrics between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Dictionary with correlation statistics
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} vs {len(vec2)}")
    
    # Remove any NaN or Inf values
    mask = np.isfinite(vec1) & np.isfinite(vec2)
    vec1_clean = vec1[mask]
    vec2_clean = vec2[mask]
    
    if len(vec1_clean) == 0:
        return {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'cosine_similarity': np.nan,
            'euclidean_distance': np.nan,
            'n_valid': 0,
            'n_total': len(vec1)
        }
    
    if len(vec1_clean) < 3:
        # Not enough for correlation, but can compute cosine similarity
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
    else:
        # Compute Pearson correlation
        pearson_r, pearson_p = pearsonr(vec1_clean, vec2_clean)
        
        # Compute Spearman correlation
        spearman_r, spearman_p = spearmanr(vec1_clean, vec2_clean)
    
    # Compute cosine similarity (using all values, handling NaNs)
    # Reshape for sklearn cosine_similarity
    vec1_2d = vec1_clean.reshape(1, -1) if len(vec1_clean) > 0 else vec1.reshape(1, -1)
    vec2_2d = vec2_clean.reshape(1, -1) if len(vec2_clean) > 0 else vec2.reshape(1, -1)
    
    if vec1_2d.shape[1] == vec2_2d.shape[1] and vec1_2d.shape[1] > 0:
        cosine_sim = cosine_similarity(vec1_2d, vec2_2d)[0, 0]
    else:
        cosine_sim = np.nan
    
    # Compute Euclidean distance
    if len(vec1_clean) > 0:
        euclidean_dist = np.linalg.norm(vec1_clean - vec2_clean)
    else:
        euclidean_dist = np.nan
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'n_valid': len(vec1_clean),
        'n_total': len(vec1)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Correlate within-category average embeddings between two embedding files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Compare category average embeddings:
  python correlate_category_embeddings.py \\
    --embeddings1 ./clip_rdm_results_26/category_average_embeddings.npz \\
    --embeddings2 ../data/things_clip_embeddings.npz \\
    --output ./category_embedding_correlations.txt
  
  # Save detailed per-category results:
  python correlate_category_embeddings.py \\
    --embeddings1 ./clip_rdm_results_26/category_average_embeddings.npz \\
    --embeddings2 ../data/things_clip_embeddings.npz \\
    --output ./category_embedding_correlations.txt \\
    --save_per_category
        """
    )
    
    parser.add_argument(
        '--embeddings1',
        type=str,
        required=True,
        help='Path to first embeddings file (.npz)'
    )
    parser.add_argument(
        '--embeddings2',
        type=str,
        required=True,
        help='Path to second embeddings file (.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for correlation results (default: print to stdout)'
    )
    parser.add_argument(
        '--save_per_category',
        action='store_true',
        help='Save detailed per-category correlation results to CSV file'
    )
    parser.add_argument(
        '--min_correlation',
        type=float,
        default=None,
        help='Only report categories with correlation above this threshold'
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    print("="*60)
    print("LOADING EMBEDDINGS")
    print("="*60)
    embeddings1, categories1 = load_embeddings(args.embeddings1)
    print()
    embeddings2, categories2 = load_embeddings(args.embeddings2)
    
    # Check embedding dimensions
    if embeddings1.shape[1] != embeddings2.shape[1]:
        print(f"\nWarning: Embedding dimensions differ: {embeddings1.shape[1]} vs {embeddings2.shape[1]}")
        print("  Correlations will be computed on the minimum dimension.")
        min_dim = min(embeddings1.shape[1], embeddings2.shape[1])
        embeddings1 = embeddings1[:, :min_dim]
        embeddings2 = embeddings2[:, :min_dim]
    
    # Find matching categories
    print("\n" + "="*60)
    print("FINDING MATCHING CATEGORIES")
    print("="*60)
    categories1_set = set(categories1)
    categories2_set = set(categories2)
    matching_categories = sorted(categories1_set & categories2_set)
    
    print(f"Categories in file 1: {len(categories1)}")
    print(f"Categories in file 2: {len(categories2)}")
    print(f"Matching categories: {len(matching_categories)}")
    
    if len(matching_categories) == 0:
        print("\nERROR: No matching categories found between the two files!")
        print("\nFirst 10 categories in file 1:")
        for cat in list(categories1)[:10]:
            print(f"  - '{cat}'")
        print("\nFirst 10 categories in file 2:")
        for cat in list(categories2)[:10]:
            print(f"  - '{cat}'")
        raise ValueError("No matching categories found between the two files")
    
    # Create mapping from category to index
    cat_to_idx1 = {cat: idx for idx, cat in enumerate(categories1)}
    cat_to_idx2 = {cat: idx for idx, cat in enumerate(categories2)}
    
    # Compute correlations for each matching category
    print("\n" + "="*60)
    print("COMPUTING CORRELATIONS")
    print("="*60)
    
    per_category_results = []
    all_pearson_rs = []
    all_spearman_rs = []
    all_cosine_sims = []
    
    for cat in matching_categories:
        idx1 = cat_to_idx1[cat]
        idx2 = cat_to_idx2[cat]
        
        vec1 = embeddings1[idx1]
        vec2 = embeddings2[idx2]
        
        results = compute_vector_correlations(vec1, vec2)
        
        per_category_results.append({
            'category': cat,
            'pearson_r': results['pearson_r'],
            'pearson_p': results['pearson_p'],
            'spearman_r': results['spearman_r'],
            'spearman_p': results['spearman_p'],
            'cosine_similarity': results['cosine_similarity'],
            'euclidean_distance': results['euclidean_distance']
        })
        
        if not np.isnan(results['pearson_r']):
            all_pearson_rs.append(results['pearson_r'])
        if not np.isnan(results['spearman_r']):
            all_spearman_rs.append(results['spearman_r'])
        if not np.isnan(results['cosine_similarity']):
            all_cosine_sims.append(results['cosine_similarity'])
    
    # Compute summary statistics
    summary_stats = {
        'n_categories': len(matching_categories),
        'mean_pearson_r': np.nanmean(all_pearson_rs) if all_pearson_rs else np.nan,
        'std_pearson_r': np.nanstd(all_pearson_rs) if all_pearson_rs else np.nan,
        'median_pearson_r': np.nanmedian(all_pearson_rs) if all_pearson_rs else np.nan,
        'min_pearson_r': np.nanmin(all_pearson_rs) if all_pearson_rs else np.nan,
        'max_pearson_r': np.nanmax(all_pearson_rs) if all_pearson_rs else np.nan,
        'mean_spearman_r': np.nanmean(all_spearman_rs) if all_spearman_rs else np.nan,
        'std_spearman_r': np.nanstd(all_spearman_rs) if all_spearman_rs else np.nan,
        'median_spearman_r': np.nanmedian(all_spearman_rs) if all_spearman_rs else np.nan,
        'mean_cosine_sim': np.nanmean(all_cosine_sims) if all_cosine_sims else np.nan,
        'std_cosine_sim': np.nanstd(all_cosine_sims) if all_cosine_sims else np.nan,
        'median_cosine_sim': np.nanmedian(all_cosine_sims) if all_cosine_sims else np.nan,
    }
    
    # Generate output text
    output_text = f"""
CATEGORY EMBEDDING CORRELATION RESULTS
{'='*60}

Input Files:
  Embeddings 1: {args.embeddings1}
    Categories: {len(categories1)}, Embedding dim: {embeddings1.shape[1]}
  Embeddings 2: {args.embeddings2}
    Categories: {len(categories2)}, Embedding dim: {embeddings2.shape[1]}

Matching Categories: {summary_stats['n_categories']}

SUMMARY STATISTICS
{'='*60}

Pearson Correlation:
  Mean:   {summary_stats['mean_pearson_r']:.6f}
  Std:    {summary_stats['std_pearson_r']:.6f}
  Median: {summary_stats['median_pearson_r']:.6f}
  Min:    {summary_stats['min_pearson_r']:.6f}
  Max:    {summary_stats['max_pearson_r']:.6f}

Spearman Correlation:
  Mean:   {summary_stats['mean_spearman_r']:.6f}
  Std:    {summary_stats['std_spearman_r']:.6f}
  Median: {summary_stats['median_spearman_r']:.6f}

Cosine Similarity:
  Mean:   {summary_stats['mean_cosine_sim']:.6f}
  Std:    {summary_stats['std_cosine_sim']:.6f}
  Median: {summary_stats['median_cosine_sim']:.6f}
"""
    
    # Add top/bottom categories if requested
    if args.min_correlation is None:
        # Show top and bottom categories by Pearson correlation
        sorted_results = sorted(per_category_results, key=lambda x: x['pearson_r'] if not np.isnan(x['pearson_r']) else -np.inf, reverse=True)
        
        output_text += f"\nTOP 10 CATEGORIES BY PEARSON CORRELATION\n{'='*60}\n"
        for i, result in enumerate(sorted_results[:10], 1):
            output_text += f"{i:2d}. {result['category']:<30} r={result['pearson_r']:.6f}, cos={result['cosine_similarity']:.6f}\n"
        
        output_text += f"\nBOTTOM 10 CATEGORIES BY PEARSON CORRELATION\n{'='*60}\n"
        for i, result in enumerate(sorted_results[-10:], len(sorted_results)-9):
            output_text += f"{i:2d}. {result['category']:<30} r={result['pearson_r']:.6f}, cos={result['cosine_similarity']:.6f}\n"
    else:
        # Filter by minimum correlation
        filtered_results = [r for r in per_category_results 
                           if not np.isnan(r['pearson_r']) and r['pearson_r'] >= args.min_correlation]
        output_text += f"\nCATEGORIES WITH CORRELATION >= {args.min_correlation}\n{'='*60}\n"
        output_text += f"Found {len(filtered_results)} categories\n"
        sorted_filtered = sorted(filtered_results, key=lambda x: x['pearson_r'], reverse=True)
        for i, result in enumerate(sorted_filtered[:20], 1):
            output_text += f"{i:2d}. {result['category']:<30} r={result['pearson_r']:.6f}, cos={result['cosine_similarity']:.6f}\n"
    
    print(output_text)
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"\nResults saved to: {output_path}")
    
    # Save per-category results to CSV if requested
    if args.save_per_category:
        df = pd.DataFrame(per_category_results)
        csv_path = Path(args.output).with_suffix('.csv') if args.output else Path('category_embedding_correlations.csv')
        df.to_csv(csv_path, index=False)
        print(f"Per-category results saved to: {csv_path}")
    
    return summary_stats, per_category_results


if __name__ == "__main__":
    main()
