import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

def load_category_embeddings(csv_path):
    """Load category embeddings from CSV file"""
    df = pd.read_csv(csv_path)
    categories = df.iloc[:, 0]  # First column contains category names
    embeddings = df.iloc[:, 1:].values  # Rest of the columns are embeddings
    return categories, embeddings

def load_category_types(csv_path):
    """Load category types from CSV file"""
    df = pd.read_csv(csv_path)
    # Create a dictionary mapping uni_lemma to category type
    category_types = {}
    for _, row in df.iterrows():
        category_types[row['uni_lemma']] = {
            'is_animate': row['is_animate'],
            'is_bodypart': row['is_bodypart'],
            'is_place': row['is_place'],
            'is_big': row['is_big'],
            'is_small': row['is_small']
        }
    return category_types

def get_category_color(category, category_types):
    """Get color for a category based on its type"""
    if category not in category_types:
        return '#C0BDBD'  # Default color for unknown categories
    
    cat_type = category_types[category]
    if cat_type['is_animate']:
        return '#8250A0'  # purple
    elif cat_type['is_bodypart']:
        return '#CE1717'  # red
    elif cat_type['is_place']:
        return '#4E9A13'  # green
    elif cat_type['is_big']:
        return '#3A53A4'  # blue
    elif cat_type['is_small']:
        return '#FAA41A'  # orange
    else:
        return '#C0BDBD'  # gray

def calculate_within_category_similarity(categories1, embeddings1, categories2, embeddings2):
    """Calculate within-category similarity between two sets of embeddings"""
    # Find common categories
    common_categories = set(categories1) & set(categories2)
    print(f"Found {len(common_categories)} common categories")
    
    # Create mapping from category to index
    cat_to_idx1 = {cat: idx for idx, cat in enumerate(categories1)}
    cat_to_idx2 = {cat: idx for idx, cat in enumerate(categories2)}
    
    # Calculate correlations for each common category
    results = []
    for cat in common_categories:
        idx1 = cat_to_idx1[cat]
        idx2 = cat_to_idx2[cat]
        
        # Get embeddings for this category
        emb1 = embeddings1[idx1]
        emb2 = embeddings2[idx2]
        
        # Calculate correlation
        corr, pval = pearsonr(emb1, emb2)
        
        results.append({
            'category': cat,
            'correlation': corr,
            'p_value': pval
        })
    
    return pd.DataFrame(results)

def plot_category_similarities(df, output_dir, category_types):
    """Create visualizations of category-wise similarities"""
    # Sort by correlation
    df_sorted = df.sort_values('correlation', ascending=False)
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=10, label='Animate'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Body Part'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Place'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Big Object'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Small Object'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Other')
    ]
    
    # Set font sizes
    title_fontsize = 25
    label_fontsize = 25
    tick_fontsize = 25
    
    # Create bar plot for top 30 categories
    plt.figure(figsize=(15, 12))
    top_30 = df_sorted.head(30)
    colors = [get_category_color(cat, category_types) for cat in top_30['category']]
    # Reverse the order for plotting to show highest correlation at the top
    plt.barh(range(len(top_30)-1, -1, -1), top_30['correlation'], color=colors)
    plt.yticks(range(len(top_30)-1, -1, -1), top_30['category'], fontsize=tick_fontsize,fontweight='bold')
    plt.xticks(fontsize=tick_fontsize)
    plt.xlabel('Correlation', fontsize=label_fontsize)
    #plt.title('Top 30 Categories: Within-Category Similarity between BabyView and THINGS', fontsize=title_fontsize)
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_30_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plot for bottom 30 categories
    plt.figure(figsize=(15, 12))
    bottom_30 = df_sorted.tail(30)
    colors = [get_category_color(cat, category_types) for cat in bottom_30['category']]
    # Reverse the order for plotting to show highest correlation at the top
    plt.barh(range(len(bottom_30)-1, -1, -1), bottom_30['correlation'], color=colors)
    plt.yticks(range(len(bottom_30)-1, -1, -1), bottom_30['category'], fontsize=tick_fontsize,fontweight='bold')
    #plt.title('Bottom 30 Categories: Within-Category Similarity between BabyView and THINGS', fontsize=title_fontsize)
    plt.xlabel('Correlation', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / 'bottom_30_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='correlation', bins=30)
    plt.axvline(df['correlation'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["correlation"].mean():.3f}')
    plt.title('Distribution of Within-Category Similarities between BabyView and THINGS', fontsize=title_fontsize)
    plt.xlabel('Correlation', fontsize=label_fontsize)
    plt.ylabel('Count', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.savefig(output_dir / 'category_similarities_hist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of top and bottom categories with color coding
    n_categories = 10
    top_categories = df.nlargest(n_categories, 'correlation')
    bottom_categories = df.nsmallest(n_categories, 'correlation')
    
    plt.figure(figsize=(12, 6))
    
    # Plot top categories
    for i, (_, row) in enumerate(top_categories.iterrows()):
        color = get_category_color(row['category'], category_types)
        plt.scatter(row['correlation'], i, color=color, label=row['category'] if i == 0 else "")
        plt.text(row['correlation'], i, row['category'], 
                verticalalignment='center', horizontalalignment='left', fontsize=tick_fontsize)
    
    # Plot bottom categories
    for i, (_, row) in enumerate(bottom_categories.iterrows()):
        color = get_category_color(row['category'], category_types)
        plt.scatter(row['correlation'], i + n_categories, color=color, label=row['category'] if i == 0 else "")
        plt.text(row['correlation'], i + n_categories, row['category'], 
                verticalalignment='center', horizontalalignment='left', fontsize=tick_fontsize)
    
    plt.title('Top and Bottom Categories: Within-Category Similarity between BabyView and THINGS', fontsize=title_fontsize)
    plt.xlabel('Correlation', fontsize=label_fontsize)
    plt.yticks([])
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=label_fontsize)
    plt.tick_params(axis='x', which='major', labelsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_categories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a combined legend plot
    plt.figure(figsize=(6, 4))
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='center', ncol=2, fontsize=label_fontsize)
    plt.savefig(output_dir / 'category_type_legend.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate within-category similarity between two CSV files')
    parser.add_argument('--csv1', required=True, help='Path to first CSV file with category embeddings')
    parser.add_argument('--csv2', required=True, help='Path to second CSV file with category embeddings')
    parser.add_argument('--category_types', required=True, help='Path to CSV file with category types')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading first CSV file...")
    categories1, embeddings1 = load_category_embeddings(args.csv1)
    print(f"Loaded {len(categories1)} categories")
    
    print("\nLoading second CSV file...")
    categories2, embeddings2 = load_category_embeddings(args.csv2)
    print(f"Loaded {len(categories2)} categories")
    
    # Load category types
    print("Loading category types...")
    category_types = load_category_types(args.category_types)
    
    # Calculate within-category similarity
    print("\nCalculating within-category similarity...")
    results_df = calculate_within_category_similarity(
        categories1, embeddings1, categories2, embeddings2
    )
    
    # Sort results by correlation in descending order
    results_df = results_df.sort_values('correlation', ascending=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean correlation: {results_df['correlation'].mean():.3f}")
    print(f"Median correlation: {results_df['correlation'].median():.3f}")
    print(f"Min correlation: {results_df['correlation'].min():.3f}")
    print(f"Max correlation: {results_df['correlation'].max():.3f}")
    
    # Create and save summary statistics
    summary_stats = {
        'statistic': ['mean', 'median', 'min', 'max'],
        'correlation': [
            results_df['correlation'].mean(),
            results_df['correlation'].median(),
            results_df['correlation'].min(),
            results_df['correlation'].max()
        ],
        'p_value': [
            results_df['p_value'].mean(),
            results_df['p_value'].median(),
            results_df['p_value'].min(),
            results_df['p_value'].max()
        ]
    }
    pd.DataFrame(summary_stats).to_csv(output_dir / 'correlation_summary_stats.csv', index=False)
    
    # Save results
    print("\nSaving results...")
    results_df.to_csv(output_dir / 'within_category_correlations.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_category_similarities(results_df, output_dir, category_types)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()