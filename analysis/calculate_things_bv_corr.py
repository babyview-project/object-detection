import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import pandas as pd

def load_matrix(file_path):
    """Load a .npy file and return the matrix"""
    try:
        matrix = np.load(file_path)
        print(f"Loaded matrix from {file_path}")
        print(f"Matrix shape: {matrix.shape}")
        return matrix
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def calculate_correlations(matrix1, matrix2, method='pearson'):
    """
    Calculate correlations between two matrices
    Args:
        matrix1: First matrix
        matrix2: Second matrix
        method: Correlation method ('pearson' or 'spearman')
    Returns:
        correlation: Correlation coefficient
        p_value: P-value
    """
    # Flatten matrices
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(flat1) | np.isnan(flat2))
    flat1 = flat1[mask]
    flat2 = flat2[mask]
    
    if method == 'pearson':
        correlation, p_value = pearsonr(flat1, flat2)
    elif method == 'spearman':
        correlation, p_value = spearmanr(flat1, flat2)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return correlation, p_value

def plot_correlation_scatter(matrix1, matrix2, output_path, method='pearson'):
    """
    Create a scatter plot of the correlation between two matrices
    Args:
        matrix1: First matrix
        matrix2: Second matrix
        output_path: Path to save the plot
        method: Correlation method used
    """
    # Flatten matrices
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(flat1) | np.isnan(flat2))
    flat1 = flat1[mask]
    flat2 = flat2[mask]
    
    # Calculate correlation
    correlation, p_value = calculate_correlations(matrix1, matrix2, method)
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(flat1, flat2, alpha=0.5, s=1)
    
    # Add correlation line
    z = np.polyfit(flat1, flat2, 1)
    p = np.poly1d(z)
    plt.plot(flat1, p(flat1), "r--", alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Matrix 1 Values')
    plt.ylabel('Matrix 2 Values')
    plt.title(f'{method.capitalize()} Correlation: {correlation:.3f} (p={p_value:.2e})')
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate correlation between two .npy files')
    parser.add_argument('--matrix1', required=True, help='Path to first .npy file')
    parser.add_argument('--matrix2', required=True, help='Path to second .npy file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--method', choices=['pearson', 'spearman'], default='pearson',
                      help='Correlation method (default: pearson)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load matrices
    matrix1 = load_matrix(args.matrix1)
    matrix2 = load_matrix(args.matrix2)
    
    # Check matrix shapes
    if matrix1.shape != matrix2.shape:
        print(f"Warning: Matrix shapes don't match: {matrix1.shape} vs {matrix2.shape}")
    
    # Calculate correlation
    correlation, p_value = calculate_correlations(matrix1, matrix2, args.method)
    
    # Print results
    print(f"\nResults:")
    print(f"{args.method.capitalize()} correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.2e}")
    
    # Save results to text file
    results_file = output_dir / 'correlation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Matrix 1: {args.matrix1}\n")
        f.write(f"Matrix 2: {args.matrix2}\n")
        f.write(f"Matrix 1 shape: {matrix1.shape}\n")
        f.write(f"Matrix 2 shape: {matrix2.shape}\n")
        f.write(f"\n{args.method.capitalize()} correlation: {correlation:.4f}\n")
        f.write(f"P-value: {p_value:.2e}\n")
    
    # Create scatter plot
    plot_path = output_dir / f'correlation_scatter_{args.method}.png'
    plot_correlation_scatter(matrix1, matrix2, plot_path, args.method)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
