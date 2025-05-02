import argparse
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import cv2
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def detect_blur(image_path, threshold=100):
    """
    Detect if an image is blurry using Laplacian variance
    Args:
        image_path: Path to the image
        threshold: Blur threshold (lower means more strict)
    Returns:
        bool: True if image is not blurry, False if blurry
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Return True if image is not blurry (variance above threshold)
        return laplacian_var > threshold
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def process_image(args):
    """Helper function for parallel processing"""
    img_path, threshold = args
    return detect_blur(img_path, threshold)

def filter_blurry_images(embedding_store, threshold=100):
    """
    Filter out blurry images from the embedding store using parallel processing
    Args:
        embedding_store: EmbeddingStore object
        threshold: Blur detection threshold
    Returns:
        Filtered EmbeddingStore object
    """
    print("Checking images for blur...")
    start_time = time.time()
    
    # Get image paths from embedding store
    image_paths = embedding_store.EmbeddingList.url
    total_images = len(image_paths)
    print(f"Total images to process: {total_images}")
    
    # Prepare arguments for parallel processing
    args = [(path, threshold) for path in image_paths]
    
    # Use parallel processing to check images
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_processes} processes for parallel processing")
    
    # Process images with progress bar
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_image, args),
            total=total_images,
            desc="Filtering blurry images",
            unit="images"
        ))
    
    # Get valid indices
    valid_indices = [i for i, is_valid in enumerate(results) if is_valid]
    num_valid = len(valid_indices)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nBlur filtering completed in {elapsed_time:.2f} seconds")
    print(f"Found {num_valid} non-blurry images out of {total_images} total images")
    print(f"Filtered out {total_images - num_valid} blurry images")
    print(f"Processing speed: {total_images/elapsed_time:.2f} images/second")
    
    # Create filtered embedding store
    filtered_store = EmbeddingStore()
    filtered_store.EmbeddingList = embedding_store.EmbeddingList[valid_indices].copy()
    
    return filtered_store

def compute_rsa_matrix(embeddings):
    """
    Compute RSA matrix for a set of embeddings
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
    Returns:
        RSA matrix of shape (n_samples, n_samples)
    """
    # Compute correlation matrix
    n_samples = embeddings.shape[0]
    rsa_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            r, _ = pearsonr(embeddings[i], embeddings[j])
            rsa_matrix[i, j] = r
            
    return rsa_matrix

def compute_category_rsa(embedding_store, min_images=0):
    """
    Compute between category RSA using mean category embeddings
    Args:
        embedding_store: EmbeddingStore object containing embeddings and categories
        min_images: Minimum number of images required per category (default: 4)
    Returns:
        between_rsa: Dictionary mapping category pairs to mean between-category RSA
        corr_matrix: Full correlation matrix
        unique_valid_categories: List of valid category names
        category_means: Mean embeddings for each category
    """
    start_time = time.time()
    
    # Get normalized embeddings and categories
    print("Loading embeddings and categories...")
    embeddings = np.array(embedding_store.EmbeddingList.normed_embedding)  # Convert to numpy array
    categories = np.array(embedding_store.EmbeddingList.text)  # Convert to numpy array
    
    # Debug: Print first few embeddings and their categories
    print("\nFirst few embeddings and their categories:")
    for i in range(min(5, len(embeddings))):
        print(f"Embedding {i}:")
        print(f"  Category: {categories[i]}")
        print(f"  Embedding shape: {embeddings[i].shape}")
        print(f"  First few values: {embeddings[i][:5]}")
    
    print(f"Time to load: {time.time() - start_time:.2f} seconds")
    
    # Get unique categories and count their occurrences
    unique_categories, category_counts = np.unique(categories, return_counts=True)
    
    print("\nCategory Information:")
    print(f"Total number of categories: {len(unique_categories)}")
    print(f"Categories with counts:")
    for cat, count in zip(unique_categories, category_counts):
        print(f"  {cat}: {count} images")
    
    # Filter categories to only include those with at least min_images
    valid_categories = unique_categories[category_counts >= min_images]
    print(f"\nFound {len(valid_categories)} categories with at least {min_images} images")
    
    if len(valid_categories) == 0:
        print("\nERROR: No valid categories found! This might be due to:")
        print("1. No categories matching the minimum image threshold")
        print("2. Categories in embeddings don't match exactly with CSV categories")
        print("3. Case sensitivity issues in category names")
        print("\nPlease check the category names in both the embeddings and CSV file.")
        raise ValueError("No valid categories found")
    
    # Calculate expected number of pairs
    n_categories = len(valid_categories)
    n_pairs = (n_categories * (n_categories - 1)) // 2
    print(f"Will compute {n_pairs} category pairs")
    
    # Get indices for valid categories
    valid_indices = np.where(np.isin(categories, valid_categories))[0]
    print("\nDebug information for valid_indices:")
    print("Shape of valid_indices:", valid_indices.shape)
    print("First few valid_indices:", valid_indices[:5] if len(valid_indices) > 0 else "No valid indices")
    print("Number of valid indices:", len(valid_indices))
    
    # Filter embeddings and categories to only include valid ones
    valid_embeddings = embeddings[valid_indices]
    valid_categories = categories[valid_indices]
    
    # Get unique valid categories
    unique_valid_categories = np.unique(valid_categories)
    print(f"\nNumber of unique categories after filtering: {len(unique_valid_categories)}")
    
    # Debug: Verify the mapping between embeddings and categories
    print("\nVerifying embedding-category mapping:")
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        print(f"Index {idx}:")
        print(f"  Original category: {categories[idx]}")
        print(f"  Valid category: {valid_categories[i]}")
        print(f"  Embedding shape: {valid_embeddings[i].shape}")
    
    # Compute mean embedding for each category using truly vectorized operations
    print("\nComputing mean embeddings...")
    mean_start = time.time()
    
    # Create a mapping from category to index
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_valid_categories)}
    
    # Create an array of category indices
    cat_indices = np.array([cat_to_idx[cat] for cat in valid_categories])
    
    # Compute means using numpy's bincount
    n_categories = len(unique_valid_categories)
    n_features = valid_embeddings.shape[1]
    
    # Initialize array for means
    category_means = np.zeros((n_categories, n_features))
    
    # For each feature, compute means using bincount
    for i in range(n_features):
        category_means[:, i] = np.bincount(cat_indices, weights=valid_embeddings[:, i], minlength=n_categories) / np.bincount(cat_indices, minlength=n_categories)
    
    # Debug: Print mean embeddings for first few categories
    print("\nMean embeddings for first few categories:")
    for i, cat in enumerate(unique_valid_categories[:5]):
        print(f"Category {cat}:")
        print(f"  Mean embedding shape: {category_means[i].shape}")
        print(f"  First few values: {category_means[i][:5]}")
    
    print(f"Time to compute means: {time.time() - mean_start:.2f} seconds")
    
    # Compute between-category RSA using mean embeddings
    print("\nComputing between-category RSA...")
    rsa_start = time.time()
    
    # Compute correlation matrix using np.corrcoef
    corr_matrix = np.corrcoef(category_means)
    
    # Convert to dictionary format
    between_rsa = {}
    for i, cat1 in enumerate(unique_valid_categories):
        for j, cat2 in enumerate(unique_valid_categories):
            if i < j:  # Only store upper triangle to avoid duplicates
                pair = f"{cat1}-{cat2}"
                between_rsa[pair] = corr_matrix[i, j]
    
    print(f"Time to compute RSA: {time.time() - rsa_start:.2f} seconds")
    print(f"Total pairs processed: {len(between_rsa)}")
    print(f"Expected number of pairs: {(len(unique_valid_categories) * (len(unique_valid_categories) - 1)) // 2}")
    
    print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")
    return between_rsa, corr_matrix, unique_valid_categories, category_means

def sort_categories(categories, category_types):
    """Sort categories by type (animate -> bodypart -> place -> big -> small -> others)"""
    # Create category groups
    animate = []
    bodypart = []
    place = []
    big = []
    small = []
    others = []
    
    for cat in categories:
        if cat not in category_types:
            others.append(cat)
            continue
            
        types = category_types[cat]
        if types['is_animate']:
            animate.append(cat)
        elif types['is_bodypart']:
            bodypart.append(cat)
        elif types['is_place']:
            place.append(cat)
        elif types['is_big']:
            big.append(cat)
        elif types['is_small']:
            small.append(cat)
        else:
            others.append(cat)
    
    # Sort within each group alphabetically
    return sorted(animate) + sorted(bodypart) + sorted(place) + sorted(big) + sorted(small) + sorted(others)

def plot_top_pairs_heatmap(distance_matrix, sorted_categories, between_rsa, output_dir, n_pairs, category_types):
    """
    Create a heatmap showing the top N most similar category pairs
    Args:
        distance_matrix: Full distance matrix
        sorted_categories: List of sorted category names
        between_rsa: Dictionary of between-category RSA values
        output_dir: Output directory for saving the plot
        n_pairs: Number of top pairs to show
        category_types: Dictionary containing category type information
    """
    # Convert between_rsa to list of tuples (pair, distance) and sort by distance
    pairs_distances = [(pair, 1 - rsa) for pair, rsa in between_rsa.items()]
    pairs_distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)
    
    # Get top N pairs
    top_pairs = pairs_distances[:n_pairs]
    
    # Get unique categories from top pairs
    unique_cats = set()
    for pair, _ in top_pairs:
        cat1, cat2 = pair.split('-')
        unique_cats.add(cat1)
        unique_cats.add(cat2)
    
    # Sort the unique categories using the same sorting rules
    sorted_unique_cats = sort_categories(list(unique_cats), category_types)
    
    # Get indices of these categories in the sorted list
    cat_indices = [sorted_categories.index(cat) for cat in sorted_unique_cats]
    
    # Create submatrix for these categories
    submatrix = distance_matrix[np.ix_(cat_indices, cat_indices)]
    
    # Create heatmap
    plt.figure(figsize=(15, 15))
    
    # Create the heatmap
    ax = sns.heatmap(submatrix, 
                xticklabels=sorted_unique_cats,
                yticklabels=sorted_unique_cats,
                cmap='viridis',
                vmin=0,
                vmax=2,
                square=True,
                cbar_kws={'label': 'Distance (1 - RSA)'})
    
    # Adjust the layout to prevent label cutoff
    plt.title(f'Top {n_pairs} Most Similar Category Pairs\n(Animate → Bodypart → Place → Big → Small → Others)', pad=20, fontsize=16)
    
    # Rotate x-axis labels and adjust their alignment
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Add more space for labels
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    
    # Save with high DPI and tight bounding box
    plt.savefig(output_dir / f'top_{n_pairs}_pairs_heatmap.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    
    # Save the top pairs to a text file
    with open(output_dir / f'top_{n_pairs}_pairs.txt', 'w') as f:
        f.write(f"Top {n_pairs} Most Similar Category Pairs:\n")
        for pair, distance in top_pairs:
            rsa = 1 - distance
            f.write(f"{pair}: RSA={rsa:.4f}, Distance={distance:.4f}\n")

def visualize_rsa_results(between_rsa, corr_matrix, categories, output_dir):
    """
    Create visualizations for RSA results
    Args:
        between_rsa: Dictionary of between-category RSA values
        corr_matrix: Full correlation matrix
        categories: List of category names in the same order as corr_matrix
        output_dir: Path to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load category types
    cdi_path = '/home/j7yang/babyview-projects/object-detection/data/cdi_words.csv'
    cdi_df = pd.read_csv(cdi_path)
    category_types = {}
    for _, row in cdi_df.iterrows():
        category_types[row['uni_lemma']] = {
            'is_animate': row['is_animate'],
            'is_bodypart': row['is_bodypart'],
            'is_place': row['is_place'],
            'is_small': row['is_small'],
            'is_big': row['is_big']
        }
    
    # Sort categories
    sorted_categories = sort_categories(categories, category_types)
    
    # Get sorting indices
    sort_idx = np.array([list(categories).index(cat) for cat in sorted_categories])
    
    # Sort correlation matrix
    sorted_corr = corr_matrix[sort_idx][:, sort_idx]
    
    # Convert to distance matrix
    distance_matrix = 1 - sorted_corr
    
    # Create full heatmap
    plt.figure(figsize=(30, 30))
    
    # Create the heatmap
    ax = sns.heatmap(distance_matrix, 
                xticklabels=sorted_categories,
                yticklabels=sorted_categories,
                cmap='viridis',
                vmin=0,
                vmax=2,
                square=True,
                cbar_kws={'label': 'Distance (1 - RSA)'})
    
    # Adjust the layout to prevent label cutoff
    plt.title('Between-Category Distance (1 - RSA)', pad=20, fontsize=16)
    
    # Rotate x-axis labels and adjust their alignment
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Add more space for labels
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    
    # Save with high DPI and tight bounding box
    plt.savefig(output_dir / 'between_category_distance_heatmap.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    
    # Create heatmaps for top pairs
    for n_pairs in [20, 50, 100]:
        # Get top N pairs
        pairs_distances = [(pair, 1 - rsa) for pair, rsa in between_rsa.items()]
        pairs_distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)
        top_pairs = pairs_distances[:n_pairs]
        
        # Get unique categories from top pairs
        unique_cats = set()
        for pair, _ in top_pairs:
            cat1, cat2 = pair.split('-')
            unique_cats.add(cat1)
            unique_cats.add(cat2)
        
        # Sort these categories independently
        sorted_unique_cats = sort_categories(list(unique_cats), category_types)
        
        # Get indices of these categories in the full sorted list
        cat_indices = [sorted_categories.index(cat) for cat in sorted_unique_cats]
        
        # Create submatrix using the full distance matrix
        submatrix = distance_matrix[np.ix_(cat_indices, cat_indices)]
        
        # Create heatmap
        plt.figure(figsize=(15, 15))
        
        # Create the heatmap
        ax = sns.heatmap(submatrix, 
                    xticklabels=sorted_unique_cats,
                    yticklabels=sorted_unique_cats,
                    cmap='viridis',
                    vmin=0,
                    vmax=2,
                    square=True,
                    cbar_kws={'label': 'Distance (1 - RSA)'})
        
        # Adjust the layout to prevent label cutoff
        plt.title(f'Top {n_pairs} Most Similar Category Pairs\n(Animate → Bodypart → Place → Big → Small → Others)', pad=20, fontsize=16)
        
        # Rotate x-axis labels and adjust their alignment
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # Add more space for labels
        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
        
        # Save with high DPI and tight bounding box
        plt.savefig(output_dir / f'top_{n_pairs}_pairs_heatmap.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()
        
        # Save the top pairs to a text file
        with open(output_dir / f'top_{n_pairs}_pairs.txt', 'w') as f:
            f.write(f"Top {n_pairs} Most Similar Category Pairs:\n")
            for pair, distance in top_pairs:
                rsa = 1 - distance
                f.write(f"{pair}: RSA={rsa:.4f}, Distance={distance:.4f}\n")
    
    # Save numerical results
    with open(output_dir / 'rsa_results.txt', 'w') as f:
        f.write("Between-Category RSA (and Distance):\n")
        for pair in sorted(between_rsa.keys()):
            rsa = between_rsa[pair]
            distance = 1 - rsa
            f.write(f"{pair}: RSA={rsa:.4f}, Distance={distance:.4f}\n")

def filter_embeddings_by_categories(embedding_store, categories_csv_path):
    """
    Filter embeddings to only include those whose text matches categories in the CSV file
    Args:
        embedding_store: EmbeddingStore object
        categories_csv_path: Path to CSV file containing categories
    Returns:
        Filtered EmbeddingStore object
    """
    print(f"Loading categories from {categories_csv_path}...")
    categories_df = pd.read_csv(categories_csv_path)
    valid_categories = set(categories_df['category'].unique())
    
    print(f"Found {len(valid_categories)} unique categories in CSV")
    print("First few categories from CSV:", list(valid_categories)[:5])
    
    # Get indices of embeddings with valid categories
    valid_indices = []
    unmatched_categories = set()
    for i, text in enumerate(embedding_store.EmbeddingList.text):
        if text in valid_categories:
            valid_indices.append(i)
        else:
            unmatched_categories.add(text)
    
    print(f"Found {len(valid_indices)} embeddings matching categories from CSV")
    print(f"Found {len(unmatched_categories)} unique categories in embeddings that don't match CSV")
    print("First few unmatched categories:", list(unmatched_categories)[:5])
    
    # Create filtered embedding store
    filtered_store = EmbeddingStore()
    filtered_store.EmbeddingList = embedding_store.EmbeddingList.__class__()  # Create new instance of same class
    for idx in valid_indices:
        filtered_store.EmbeddingList.append(embedding_store.EmbeddingList[idx])
    
    # Debug: Check filtered store contents
    print("\nDebug: Filtered store contents:")
    print(f"Number of embeddings: {len(filtered_store.EmbeddingList)}")
    texts = [item.text for item in filtered_store.EmbeddingList]
    print(f"Number of texts: {len(texts)}")
    print("First few texts in filtered store:", texts[:5])
    print("Unique categories in filtered store:", len(set(texts)))
    print("First few unique categories:", list(set(texts))[:5])
    
    return filtered_store

def main():
    parser = argparse.ArgumentParser(description='Compute between category RSA')
    parser.add_argument("--doc_path", required=True, help="Path to the .doc file containing embeddings")
    parser.add_argument("--output_dir", required=True, help="Output directory for RSA results and visualizations")
    parser.add_argument("--min_images", type=int, default=0, help="Minimum number of images required per category (default: 0)")
    parser.add_argument("--categories_csv", required=True, help="Path to CSV file containing categories to include")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load embeddings from .doc file
    print(f"Loading embeddings from {args.doc_path}")
    embedding_store = EmbeddingStore.from_doc(args.doc_path)
    
    # Filter embeddings by categories from CSV
    print("Filtering embeddings by categories...")
    filtered_store = filter_embeddings_by_categories(embedding_store, args.categories_csv)
    
    # Save filtered embeddings to a new .doc file
    filtered_doc_path = output_dir / "filtered_embeddings.doc"
    print(f"Saving filtered embeddings to {filtered_doc_path}")
    filtered_store.to_doc(str(filtered_doc_path))
    
    # Load the filtered embeddings
    print("Loading filtered embeddings...")
    filtered_store = EmbeddingStore.from_doc(str(filtered_doc_path))
    
    # Compute RSA
    print("Computing RSA...")
    between_rsa, corr_matrix, categories, category_means = compute_category_rsa(filtered_store, args.min_images)
    
    # Save RDM matrix and category means
    print("Saving RDM matrix and category means...")
    np.save(output_dir / "rdm_matrix.npy", corr_matrix)
    np.save(output_dir / "rdm_matrix_lower_triangle.npy", np.tril(corr_matrix))
    np.save(output_dir / "category_means.npy", category_means)
    
    # Save category names
    with open(output_dir / "category_names.txt", "w") as f:
        for cat in categories:
            f.write(f"{cat}\n")
    
    # Visualize results
    print("Creating visualizations...")
    visualize_rsa_results(between_rsa, corr_matrix, categories, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 