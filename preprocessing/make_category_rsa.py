import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt

def load_embeddings(embedding_path: str) -> np.ndarray:
    """Load embeddings from .npy file"""
    embedding = np.load(embedding_path)
    # Print shape for debugging
    print(f"Loading {embedding_path}, shape: {embedding.shape}")
    
    # Ensure embedding is 2D with shape (1, feature_dim)
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 512)  # Assuming CLIP's 512 dimension
    return embedding

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute correlation matrix between embeddings"""
    print(f"Computing similarity matrix for embeddings shape: {embeddings.shape}")
    
    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    
    # Normalize embeddings
    normalized = embeddings / norms
    
    # Compute correlation matrix
    similarity_matrix = normalized @ normalized.T
    return similarity_matrix

def compute_category_rsa(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute RSA within each category"""
    categories = df['text'].unique()
    category_rsa = {}
    
    for category in categories:
        # Get embeddings for this category
        category_df = df[df['text'] == category]
        category_embeddings = []
        valid_paths = []
        
        for path in category_df['embedding_path']:
            try:
                embedding = load_embeddings(path)
                # Skip embeddings with zero norm
                if np.linalg.norm(embedding) < 1e-10:
                    continue
                category_embeddings.append(embedding)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading embedding from {path}: {e}")
                continue
            
        if not category_embeddings:
            print(f"Warning: No valid embeddings found for category {category}")
            continue
            
        print(f"Category {category}: {len(category_embeddings)} valid embeddings out of {len(category_df)}")
        
        category_embeddings = np.vstack(category_embeddings)
        similarity_matrix = compute_similarity_matrix(category_embeddings)
        category_rsa[category] = similarity_matrix
        
    return category_rsa

def compute_cross_category_rsa(df: pd.DataFrame) -> np.ndarray:
    """Compute RSA between category averages"""
    categories = df['text'].unique()
    category_means = []
    
    for category in categories:
        # Get mean embedding for this category
        category_df = df[df['text'] == category]
        category_embeddings = []
        
        for path in category_df['embedding_path']:
            try:
                embedding = load_embeddings(path)
                if np.linalg.norm(embedding) < 1e-10:
                    print(f"Skipping zero-norm embedding: {path}")
                    continue
                category_embeddings.append(embedding)
            except Exception as e:
                print(f"Error loading embedding from {path}: {e}")
                continue
        
        if not category_embeddings:
            print(f"No valid embeddings for category: {category}")
            continue
            
        # Stack embeddings and compute mean
        category_embeddings = np.vstack(category_embeddings)
        mean_embedding = np.mean(category_embeddings, axis=0, keepdims=True)
        category_means.append(mean_embedding)
    
    # Stack all category means
    category_means = np.vstack(category_means)
    print(f"Final category means shape: {category_means.shape}")
    
    cross_category_similarity = compute_similarity_matrix(category_means)
    return cross_category_similarity

def visualize_rsa_matrix(similarity_matrix: np.ndarray, 
                        labels: List[str], 
                        title: str,
                        output_path: str):
    """
    Create and save a heatmap visualization of the RSA matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='viridis',
                vmin=-1,
                vmax=1,
                center=0,
                square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Read input CSV
    df = pd.read_csv('/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_filtered-by-size-0.05/embeddings/image_embeddings/clip_image_embeddings_npy.csv')  # Update path as needed
    
    # Create output directories
    output_dir = Path('rsa_results')
    plot_dir = output_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    # Compute and visualize within-category RSA
    category_rsa = compute_category_rsa(df)
    
    for category, rsa_matrix in category_rsa.items():
        # Save matrix
        np.save(output_dir / f'{category}_rsa.npy', rsa_matrix)
        
        # Create labels (indices for within-category)
        n_samples = rsa_matrix.shape[0]
        labels = [f'{i+1}' for i in range(n_samples)]
        
        # Visualize
        visualize_rsa_matrix(
            rsa_matrix,
            labels,
            f'Within-category RSA: {category}',
            plot_dir / f'{category}_rsa.png'
        )
    
    # Compute and visualize cross-category RSA
    cross_category_rsa = compute_cross_category_rsa(df)
    
    # Save matrix
    np.save(output_dir / 'cross_category_rsa.npy', cross_category_rsa)
    
    # Visualize cross-category RSA
    categories = df['text'].unique()
    visualize_rsa_matrix(
        cross_category_rsa,
        categories,
        'Cross-category RSA',
        plot_dir / 'cross_category_rsa.png'
    )

if __name__ == '__main__':
    main()
