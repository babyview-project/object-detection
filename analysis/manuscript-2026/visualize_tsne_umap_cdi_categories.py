import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from collections import defaultdict
import pandas as pd
import random
from matplotlib.colors import ListedColormap

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_cdi_words(cdi_path):
    """
    Load CDI words mapping from CSV file
    Args:
        cdi_path: Path to the cdi_words.csv file
    Returns:
        cdi_df: DataFrame with CDI words and categories
        word_to_category: dict mapping uni_lemma to category
    """
    print(f"Loading CDI words from {cdi_path}")
    cdi_df = pd.read_csv(cdi_path)
    
    # Create mapping from uni_lemma to category
    word_to_category = dict(zip(cdi_df['uni_lemma'], cdi_df['category']))
    
    print(f"Loaded {len(cdi_df)} CDI words")
    print(f"Unique CDI categories: {sorted(cdi_df['category'].unique())}")
    
    return cdi_df, word_to_category

def load_grouped_embeddings(grouped_embeddings_dir, categories_file=None, excluded_subject=None):
    """
    Load grouped age-month embeddings (same structure as vss-2026 and individual_analyses).
    Directory layout: {grouped_embeddings_dir}/{category}/{subject_id}_{age_mo}_month_level_avg.npy
    Computes one average embedding per category (mean across all subject/age_mo files).
    Args:
        grouped_embeddings_dir: Path to grouped embeddings root (e.g. clip_embeddings_grouped_by_age-mo_normalized)
        categories_file: Optional path to text file with category names to include (one per line)
        excluded_subject: Optional subject_id to exclude from averaging (e.g. "00270001")
    Returns:
        class_embeddings: numpy array of shape (n_categories, embedding_dim)
        class_names: list of category names
        class_counts: dict mapping category name to number of subject/age_mo files (exemplar count)
    """
    grouped_embeddings_dir = Path(grouped_embeddings_dir)
    if not grouped_embeddings_dir.exists():
        raise FileNotFoundError(f"Grouped embeddings directory not found: {grouped_embeddings_dir}")

    allowed_categories = None
    if categories_file:
        categories_path = Path(categories_file)
        if categories_path.exists():
            with open(categories_path, "r") as f:
                allowed_categories = set(line.strip() for line in f if line.strip())
            print(f"Using {len(allowed_categories)} categories from {categories_file}")

    category_folders = [f for f in grouped_embeddings_dir.iterdir() if f.is_dir()]
    if allowed_categories:
        category_folders = [f for f in category_folders if f.name in allowed_categories]

    print(f"Loading grouped embeddings from {grouped_embeddings_dir} ({len(category_folders)} categories)...")

    class_embeddings = []
    class_names = []
    class_counts = {}

    for category_folder in sorted(category_folders):
        category = category_folder.name
        embedding_files = list(category_folder.glob("*.npy"))
        embeddings_list = []

        for emb_file in embedding_files:
            stem = emb_file.stem  # e.g. 00320001_12_month_level_avg
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject_id = parts[0]
            try:
                age_mo = int(parts[1])
            except ValueError:
                continue
            if excluded_subject and subject_id == excluded_subject:
                continue
            try:
                emb = np.load(emb_file)
                if emb.ndim > 1:
                    emb = emb.flatten()
                embeddings_list.append(emb)
            except Exception as e:
                print(f"Warning: failed to load {emb_file}: {e}")
                continue

        if len(embeddings_list) == 0:
            continue

        emb_array = np.array(embeddings_list)
        avg_embedding = np.mean(emb_array, axis=0)
        class_embeddings.append(avg_embedding)
        class_names.append(category)
        class_counts[category] = len(embeddings_list)

    class_embeddings = np.array(class_embeddings)
    print(f"Loaded {len(class_names)} category-level embeddings (avg over subject/age_mo)")
    print(f"Embedding dimension: {class_embeddings.shape[1]}")
    print(f"Exemplar counts per category: min={min(class_counts.values())}, max={max(class_counts.values())}")

    return class_embeddings, class_names, class_counts

def match_classes_to_cdi(class_names, word_to_category):
    """
    Match class names (category names) to CDI categories.
    Args:
        class_names: array or list of class/category names from embeddings
        word_to_category: dict mapping uni_lemma to CDI category
    Returns:
        matched_indices: list of indices into class_names for matched classes
        matched_classes: list of matched class names
        matched_categories: list of corresponding CDI categories
    """
    matched_indices = []
    matched_classes = []
    matched_categories = []

    if hasattr(class_names, "__iter__") and not isinstance(class_names, np.ndarray):
        class_names = np.array(class_names)
    unique_classes = np.unique(class_names)

    for i, class_name in enumerate(unique_classes):
        # Try exact match first
        if class_name in word_to_category:
            matched_indices.append(i)
            matched_classes.append(class_name)
            matched_categories.append(word_to_category[class_name])
        # Try removing common suffixes/prefixes for partial matching
        else:
            # Remove common variations
            clean_name = class_name.lower().strip()
            variations = [
                clean_name,
                clean_name.rstrip('s'),  # remove plural s
                clean_name.rstrip('es'), # remove plural es
                clean_name.replace('_', ' '),
                clean_name.replace('-', ' '),
            ]
            
            found_match = False
            for variation in variations:
                if variation in word_to_category:
                    matched_indices.append(i)
                    matched_classes.append(class_name)
                    matched_categories.append(word_to_category[variation])
                    found_match = True
                    break
            
            if not found_match:
                print(f"No CDI match found for class: {class_name}")
    
    print(f"Matched {len(matched_classes)} classes to CDI categories")
    return matched_indices, matched_classes, matched_categories


def compute_embedding_2d(embeddings, method="tsne", random_state=42):
    """
    Compute 2D embedding from high-dimensional embeddings using t-SNE or UMAP.
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: "tsne" or "umap"
        random_state: random seed for reproducibility
    Returns:
        coords_2d: numpy array of shape (n_samples, 2)
    """
    n = len(embeddings)
    if method == "tsne":
        perplexity = min(30, max(1, n - 1))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        coords_2d = reducer.fit_transform(embeddings)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")
    return coords_2d


def create_embedding_visualization_cdi(coords_2d, method, class_names, cdi_categories, class_counts, output_dir):
    """
    Create 2D embedding visualization (t-SNE or UMAP) colored by CDI categories.
    Args:
        coords_2d: numpy array of shape (n_samples, 2) from compute_embedding_2d
        method: "tsne" or "umap" (used for labels and filenames)
        class_names: list of class names
        cdi_categories: list of CDI categories
        class_counts: dict with counts per class
        output_dir: output directory for saving plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    method_upper = method.upper() if method == "umap" else "t-SNE"

    # Get unique CDI categories and assign colors
    unique_categories = sorted(list(set(cdi_categories)))
    n_categories = len(unique_categories)
    
    # Use a colormap with enough distinct colors
    if n_categories <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
    elif n_categories <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_categories))
    
    category_to_color = dict(zip(unique_categories, colors))
    point_colors = [category_to_color[cat] for cat in cdi_categories]

    # Display names for CDI categories (for cluster labels)
    def _category_display_name(cat):
        names = {
            "body_parts": "Body parts",
            "food_drink": "Food/Drink",
            "furniture_rooms": "Furniture",
            "household": "Household items",
            "outside": "Outdoor",
            "people": "People",
            "toys": "Toys",
            "vehicles": "Vehicles",
            "animals": "Animals",
            "clothing": "Clothing",
        }
        return names.get(cat, cat.replace("_", " ").title())

    # Label positions at periphery of each cluster
    def _cluster_label_positions(xy, cdi_cats, unique_cats, padding_frac=0.02, max_offset_frac=0.15):
        global_center = np.mean(xy, axis=0)
        x_range = np.ptp(xy[:, 0])
        y_range = np.ptp(xy[:, 1])
        data_span = max(x_range, y_range)
        padding = padding_frac * data_span
        max_offset = max_offset_frac * data_span
        positions = {}
        for cat in unique_cats:
            idx = [i for i, c in enumerate(cdi_cats) if c == cat]
            if not idx:
                continue
            pts = xy[idx]
            centroid = np.array([np.median(pts[:, 0]), np.median(pts[:, 1])])
            vec = centroid - global_center
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                direction = np.array([1.0, 0.0])
            else:
                direction = vec / norm
            radius = np.max(np.linalg.norm(pts - centroid, axis=1))
            offset = min(radius + padding, max_offset)
            label_xy = centroid + offset * direction
            positions[cat] = (float(label_xy[0]), float(label_xy[1]))
        return positions
    
    # Calculate dot sizes based on class counts
    sizes = [class_counts[class_name] for class_name in class_names]
    sizes = np.array(sizes)
    if len(sizes) > 1:
        sizes_norm = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 450
    else:
        sizes_norm = np.array([100])
    
    # Full visualization with per-point labels
    plt.figure(figsize=(20, 16))
    for i, category in enumerate(unique_categories):
        cat_indices = [j for j, cat in enumerate(cdi_categories) if cat == category]
        if cat_indices:
            plt.scatter(coords_2d[cat_indices, 0], coords_2d[cat_indices, 1],
                       c=[point_colors[j] for j in cat_indices],
                       s=[sizes_norm[j] for j in cat_indices],
                       label=category, alpha=0.7, edgecolors='black', linewidth=0.5)
    for i, class_name in enumerate(class_names):
        plt.annotate(class_name, 
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    plt.title(f'{method_upper} Visualization of Object Classes by CDI Categories\n'
              '(Grouped embeddings: dot size = number of subject/age_mo exemplars per category)', 
              fontsize=16, fontweight='bold')
    plt.xlabel(f'{method_upper} Dimension 1', fontsize=12)
    plt.ylabel(f'{method_upper} Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f'{method}_cdi_categories.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{method}_cdi_categories.pdf', bbox_inches='tight')
    plt.close()
    
    # Clean version with color-coded cluster labels only
    fig, ax = plt.subplots(figsize=(16, 12))
    for i, category in enumerate(unique_categories):
        cat_indices = [j for j, cat in enumerate(cdi_categories) if cat == category]
        if cat_indices:
            ax.scatter(coords_2d[cat_indices, 0], coords_2d[cat_indices, 1],
                      c=[point_colors[j] for j in cat_indices],
                      s=[sizes_norm[j] for j in cat_indices],
                      label=category, alpha=0.7, edgecolors='black', linewidth=0.5)
    label_positions = _cluster_label_positions(
        coords_2d, cdi_categories, unique_categories,
        padding_frac=0.02, max_offset_frac=0.15
    )
    # Include both data and label positions in axis limits so no labels are clipped
    x_vals = list(coords_2d[:, 0]) + [p[0] for p in label_positions.values()]
    y_vals = list(coords_2d[:, 1]) + [p[1] for p in label_positions.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    x_margin = 0.06 * x_span
    y_margin = 0.06 * y_span
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    for cat in unique_categories:
        if cat not in label_positions:
            continue
        x, y = label_positions[cat]
        color = category_to_color[cat]
        if hasattr(color, '__len__') and len(color) >= 3:
            color = tuple(color[:3])
        ax.annotate(
            _category_display_name(cat),
            (x, y),
            fontsize=15,
            fontweight='bold',
            ha='center',
            va='center',
            color=color,
            clip_on=False,
        )
    ax.set_title(f'{method_upper} Visualization of Object Classes by CDI Categories\n'
                 '(Grouped embeddings: dot size = number of subject/age_mo exemplars)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method_upper} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method_upper} Dimension 2', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f'{method}_cdi_categories_clean.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{method}_cdi_categories_clean.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # Save coordinates CSV (column names reflect method)
    coord_df = pd.DataFrame({
        'class_name': class_names,
        'cdi_category': cdi_categories,
        f'{method}_x': coords_2d[:, 0],
        f'{method}_y': coords_2d[:, 1],
        'exemplar_count': [class_counts[name] for name in class_names]
    })
    coord_df.to_csv(output_dir / f'{method}_cdi_coordinates.csv', index=False)
    
    # Category statistics (same for any method)
    category_stats = pd.DataFrame({
        'cdi_category': unique_categories,
        'num_classes': [sum(1 for cat in cdi_categories if cat == category) 
                       for category in unique_categories],
        'total_exemplars': [sum(class_counts[class_names[i]] for i, cat in enumerate(cdi_categories) if cat == category)
                           for category in unique_categories]
    })
    category_stats.to_csv(output_dir / 'cdi_category_stats.csv', index=False)
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Files created:")
    print(f"- {method}_cdi_categories.png, .pdf: Full visualization with labels")
    print(f"- {method}_cdi_categories_clean.png, .pdf: Clean visualization with color-coded cluster labels")
    print(f"- {method}_cdi_coordinates.csv: 2D coordinates and metadata")
    print(f"- cdi_category_stats.csv: Statistics per CDI category")


# Default base path for grouped embeddings (same parent for CLIP and DinoV3)
GROUPED_EMBEDDINGS_BASE = "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings"
CLIP_GROUPED_DIR = f"{GROUPED_EMBEDDINGS_BASE}/clip_embeddings_grouped_by_age-mo_normalized"
DINOV3_GROUPED_DIR = f"{GROUPED_EMBEDDINGS_BASE}/facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo_normalized"


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # object-detection/

    parser = argparse.ArgumentParser(
        description="Create t-SNE or UMAP visualization of grouped embeddings colored by CDI categories"
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "umap", "both"],
        default="tsne",
        help="Dimensionality reduction method: 'tsne', 'umap', or 'both' (runs both and writes to separate dirs)",
    )
    parser.add_argument(
        "--embedding_type",
        choices=["clip", "dinov3"],
        default="clip",
        help="Embedding model: 'clip' or 'dinov3'. Sets default grouped_embeddings_dir and output_dir if not overridden.",
    )
    parser.add_argument(
        "--grouped_embeddings_dir",
        default=None,
        help="Path to grouped age-month embeddings; structure: {dir}/{category}/{subject_id}_{age_mo}_month_level_avg.npy (default: CLIP or DinoV3 path per --embedding_type)",
    )
    parser.add_argument(
        "--cdi_path",
        default=None,
        help="Path to the CDI words CSV file (default: project data/cdi_words.csv)",
    )
    parser.add_argument(
        "--categories_file",
        default=None,
        help="Optional path to text file with category names to include, one per line (default: project data/things_bv_overlap_categories_exclude_zero_precisions.txt)",
    )
    parser.add_argument(
        "--excluded_subject",
        default="00270001",
        help="Subject ID to exclude from averaging (default: 00270001, set to empty string to include all)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for visualizations (default: {method}_cdi_results_{embedding_type}; ignored when --method both)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for t-SNE/UMAP",
    )

    args = parser.parse_args()

    if args.method == "umap" and not HAS_UMAP:
        parser.error("UMAP requires umap-learn. Install with: pip install umap-learn")

    # Set embedding-type-specific defaults when not overridden
    if args.grouped_embeddings_dir is None:
        args.grouped_embeddings_dir = DINOV3_GROUPED_DIR if args.embedding_type == "dinov3" else CLIP_GROUPED_DIR

    # Default paths relative to project root
    if args.cdi_path is None:
        args.cdi_path = project_root / "data" / "cdi_words.csv"
    else:
        args.cdi_path = Path(args.cdi_path)
    if args.categories_file is None:
        default_categories = project_root / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"
        args.categories_file = default_categories if default_categories.exists() else None
    else:
        args.categories_file = Path(args.categories_file) if args.categories_file else None
    excluded_subject = args.excluded_subject if args.excluded_subject else None

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Load CDI words
    cdi_df, word_to_category = load_cdi_words(args.cdi_path)

    # Load grouped embeddings (one average per category)
    class_embeddings, class_names, class_counts = load_grouped_embeddings(
        args.grouped_embeddings_dir,
        categories_file=args.categories_file,
        excluded_subject=excluded_subject,
    )

    # Match category names to CDI categories
    _, matched_classes, matched_categories = match_classes_to_cdi(class_names, word_to_category)

    if len(matched_classes) == 0:
        print("No classes matched to CDI categories!")
        return

    # Build final arrays for matched classes only
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    final_embeddings = np.array([class_embeddings[name_to_idx[mc]] for mc in matched_classes])
    final_class_names = matched_classes
    final_cdi_categories = matched_categories
    final_class_counts = {name: class_counts[name] for name in matched_classes}

    methods_to_run = ["tsne", "umap"] if args.method == "both" else [args.method]

    for method in methods_to_run:
        if args.method == "both":
            output_dir = script_dir / f"{method}_cdi_results_{args.embedding_type}"
        else:
            output_dir = Path(args.output_dir) if args.output_dir is not None else script_dir / f"{method}_cdi_results_{args.embedding_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {method.upper()} ---")
        print(f"Output dir: {output_dir}")

        print(f"Computing {method.upper()} embedding...")
        coords_2d = compute_embedding_2d(final_embeddings, method=method, random_state=args.random_seed)

        create_embedding_visualization_cdi(
            coords_2d,
            method,
            final_class_names,
            final_cdi_categories,
            final_class_counts,
            output_dir,
        )

    print(f"\nMatched {len(final_class_names)} classes to CDI categories")
    print(f"CDI categories found: {sorted(set(final_cdi_categories))}")


if __name__ == "__main__":
    main()
