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

def load_category_types(cdi_path: str = '../data/cdi_words.csv') -> Dict[str, Dict[str, bool]]:
    """Load category types from CDI words file"""
    cdi_df = pd.read_csv(cdi_path)
    category_types = {}
    
    for _, row in cdi_df.iterrows():
        category_types[row['uni_lemma']] = {
            'is_animate': row['is_animate'],
            'is_small': row['is_small'],
            'is_big': row['is_big']
        }
    return category_types

def sort_categories(categories: List[str], category_types: Dict[str, Dict[str, bool]]) -> List[str]:
    """Sort categories by type (animate -> small -> big -> others)"""
    # Create category groups
    animate = []
    small = []
    big = []
    others = []
    
    for cat in categories:
        if cat not in category_types:
            others.append(cat)
            continue
            
        types = category_types[cat]
        if types['is_animate']:
            animate.append(cat)
        elif types['is_small']:
            small.append(cat)
        elif types['is_big']:
            big.append(cat)
        else:
            others.append(cat)
    
    # Sort within each group alphabetically
    return sorted(animate) + sorted(small) + sorted(big) + sorted(others)

def visualize_rsa_matrix(similarity_matrix: np.ndarray, 
                        labels: List[str], 
                        title: str,
                        output_path: str,
                        is_cross_category: bool = False):
    """
    Create and save a heatmap visualization of the RSA matrix
    """
    # Adjust figure size based on number of labels
    if is_cross_category:
        plt.figure(figsize=(15, 12))  # Larger figure for cross-category
    else:
        plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='viridis',
                vmin=0,  # Changed from -1 to 0
                vmax=1,
                square=True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust font sizes
    plt.title(title, fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Read input CSV and CDI words
    df = pd.read_csv('/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_mask/embeddings/image_embeddings/clip_image_embeddings_npy.csv')
    category_types = load_category_types()
    
    # Create output directories
    output_dir = Path('../analysis/rsa_results')
    plot_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Compute cross-category RSA
    cross_category_rsa = compute_cross_category_rsa(df)
    
    # Save matrix
    np.save(output_dir / 'cross_category_rsa.npy', cross_category_rsa)
    
    # Get categories and sort them by type
    categories = df['text'].unique()
    sorted_categories = sort_categories(categories, category_types)
    
    # Reorder the cross-category RSA matrix according to sorted categories
    category_to_idx = {cat: i for i, cat in enumerate(categories)}
    sorted_indices = [category_to_idx[cat] for cat in sorted_categories]
    sorted_rsa = cross_category_rsa[sorted_indices][:, sorted_indices]
    
    # Visualize sorted cross-category RSA
    visualize_rsa_matrix(
        sorted_rsa,
        sorted_categories,
        'Cross-category RSA\n(Grouped by: Animate → Small → Big → Others)',
        plot_dir / 'cross_category_rsa.png',
        is_cross_category=True
    )

if __name__ == '__main__':
    main()
