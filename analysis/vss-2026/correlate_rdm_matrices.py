#!/usr/bin/env python3
"""
Correlate the lower triangle of two RDM (Representational Dissimilarity Matrix) matrices.
Supports both CSV and NPY file formats as input.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform


def load_rdm_matrix(file_path):
    """
    Load RDM matrix from either CSV or NPY file.
    
    Args:
        file_path: Path to CSV or NPY file
        
    Returns:
        numpy array of the RDM matrix
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.npy':
        # Load NPY file
        matrix = np.load(file_path)
        print(f"Loaded NPY file: {file_path}")
        print(f"  Shape: {matrix.shape}")
    elif file_path.suffix == '.csv':
        # Load CSV file - extract numeric matrix (skip index/column names if present)
        df = pd.read_csv(file_path, index_col=0)
        matrix = df.values
        print(f"Loaded CSV file: {file_path}")
        print(f"  Shape: {matrix.shape}")
        print(f"  Categories: {len(df.index)}")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Expected .npy or .csv")
    
    # Validate that it's a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: shape {matrix.shape}")
    
    # Check if matrix is symmetric (within tolerance)
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        print(f"Warning: Matrix may not be symmetric. Max difference: {np.abs(matrix - matrix.T).max():.6f}")
    
    return matrix


def extract_lower_triangle(matrix, exclude_diagonal=True):
    """
    Extract lower triangle of a symmetric matrix.
    
    Args:
        matrix: Square numpy array
        exclude_diagonal: If True, exclude diagonal elements (default: True)
        
    Returns:
        1D array of lower triangle values
    """
    if exclude_diagonal:
        # Extract lower triangle excluding diagonal (k=1 means start below diagonal)
        lower_tri = matrix[np.tril_indices_from(matrix, k=-1)]
    else:
        # Extract lower triangle including diagonal
        lower_tri = matrix[np.tril_indices_from(matrix, k=0)]
    
    return lower_tri


def compute_correlations(vec1, vec2):
    """
    Compute Pearson and Spearman correlations between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
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
        raise ValueError("No valid (finite) values found after removing NaNs/Infs")
    
    if len(vec1_clean) < 3:
        raise ValueError(f"Not enough valid values for correlation: {len(vec1_clean)}")
    
    # Compute Pearson correlation
    pearson_r, pearson_p = pearsonr(vec1_clean, vec2_clean)
    
    # Compute Spearman correlation
    spearman_r, spearman_p = spearmanr(vec1_clean, vec2_clean)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_valid': len(vec1_clean),
        'n_total': len(vec1)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Correlate the lower triangle of two RDM matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using NPY files:
  python correlate_rdm_matrices.py \\
    --rdm1 ./clip_rdm_results_24/distance_matrix_filtered.npy \\
    --rdm2 ./dinov3_rdm_results_24/distance_matrix_filtered.npy \\
    --output ./rdm_correlation_results.txt
  
  # Using CSV files:
  python correlate_rdm_matrices.py \\
    --rdm1 ./clip_rdm_results_24/distance_matrix_filtered.csv \\
    --rdm2 ./dinov3_rdm_results_24/distance_matrix_filtered.csv \\
    --output ./rdm_correlation_results.txt
  
  # Include diagonal in correlation:
  python correlate_rdm_matrices.py \\
    --rdm1 ./clip_rdm_results_24/distance_matrix_filtered.npy \\
    --rdm2 ./dinov3_rdm_results_24/distance_matrix_filtered.npy \\
    --include_diagonal \\
    --output ./rdm_correlation_results.txt
        """
    )
    
    parser.add_argument(
        '--rdm1',
        type=str,
        required=True,
        help='Path to first RDM matrix file (.npy or .csv)'
    )
    parser.add_argument(
        '--rdm2',
        type=str,
        required=True,
        help='Path to second RDM matrix file (.npy or .csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for correlation results (default: print to stdout)'
    )
    parser.add_argument(
        '--include_diagonal',
        action='store_true',
        help='Include diagonal elements in correlation (default: exclude diagonal)'
    )
    parser.add_argument(
        '--use_upper_triangle',
        action='store_true',
        help='Use upper triangle instead of lower triangle (default: lower triangle)'
    )
    
    args = parser.parse_args()
    
    # Load matrices
    print("="*60)
    print("LOADING RDM MATRICES")
    print("="*60)
    matrix1 = load_rdm_matrix(args.rdm1)
    matrix2 = load_rdm_matrix(args.rdm2)
    
    # Check if matrices have same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError(
            f"Matrices must have the same shape: "
            f"RDM1 {matrix1.shape} vs RDM2 {matrix2.shape}"
        )
    
    print(f"\nBoth matrices have shape: {matrix1.shape}")
    print(f"Total elements: {matrix1.size}")
    
    # Extract triangle
    print("\n" + "="*60)
    print("EXTRACTING TRIANGLE")
    print("="*60)
    triangle_type = "upper" if args.use_upper_triangle else "lower"
    exclude_diag = not args.include_diagonal
    
    if args.use_upper_triangle:
        if exclude_diag:
            vec1 = matrix1[np.triu_indices_from(matrix1, k=1)]
            vec2 = matrix2[np.triu_indices_from(matrix2, k=1)]
        else:
            vec1 = matrix1[np.triu_indices_from(matrix1, k=0)]
            vec2 = matrix2[np.triu_indices_from(matrix2, k=0)]
    else:
        if exclude_diag:
            vec1 = extract_lower_triangle(matrix1, exclude_diagonal=True)
            vec2 = extract_lower_triangle(matrix2, exclude_diagonal=True)
        else:
            vec1 = extract_lower_triangle(matrix1, exclude_diagonal=False)
            vec2 = extract_lower_triangle(matrix2, exclude_diagonal=False)
    
    print(f"Extracted {triangle_type} triangle (diagonal {'included' if not exclude_diag else 'excluded'})")
    print(f"  Vector length: {len(vec1)}")
    print(f"  Expected length: {matrix1.shape[0] * (matrix1.shape[0] - 1) // 2 if exclude_diag else matrix1.shape[0] * (matrix1.shape[0] + 1) // 2}")
    
    # Compute correlations
    print("\n" + "="*60)
    print("COMPUTING CORRELATIONS")
    print("="*60)
    results = compute_correlations(vec1, vec2)
    
    # Print results
    output_text = f"""
RDM MATRIX CORRELATION RESULTS
{'='*60}

Input Files:
  RDM 1: {args.rdm1}
  RDM 2: {args.rdm2}

Matrix Information:
  Shape: {matrix1.shape}
  Triangle: {triangle_type} (diagonal {'included' if not exclude_diag else 'excluded'})
  Elements in correlation: {results['n_valid']} / {results['n_total']}

Correlation Results:
  Pearson r:  {results['pearson_r']:.6f}
  Pearson p:  {results['pearson_p']:.2e}
  Spearman r: {results['spearman_r']:.6f}
  Spearman p: {results['spearman_p']:.2e}

Statistics:
  RDM 1 - Mean: {vec1.mean():.6f}, Std: {vec1.std():.6f}, Min: {vec1.min():.6f}, Max: {vec1.max():.6f}
  RDM 2 - Mean: {vec2.mean():.6f}, Std: {vec2.std():.6f}, Min: {vec2.min():.6f}, Max: {vec2.max():.6f}
"""
    
    print(output_text)
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
