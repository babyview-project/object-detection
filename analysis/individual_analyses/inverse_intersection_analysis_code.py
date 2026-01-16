# ====================================================================
# INVERSE ANALYSIS: For a given number of categories X, find maximum number of subjects N
# where all N subjects have all X categories
# ====================================================================
# Add this code to the end of the intersection analysis cell in notebook 06
# (right after the "Recommended thresholds" section)

print("\n" + "="*60)
print("INVERSE ANALYSIS: Maximum Subjects for Given Category Count")
print("="*60)
print("For each number of categories X, find the maximum number of subjects N")
print("where all N subjects have all X overlapping categories.\n")

# Get unique category counts from intersection_df
unique_category_counts = sorted(intersection_df['intersection_size'].unique(), reverse=True)

# Store results
inverse_analysis = []

for n_categories in unique_category_counts:
    # Find the threshold where intersection_size == n_categories (exact match)
    exact_match = intersection_df[intersection_df['intersection_size'] == n_categories]
    
    if len(exact_match) > 0:
        # Use the row with maximum n_subjects that gives exactly n_categories
        best_row = exact_match.loc[exact_match['n_subjects'].idxmax()]
        threshold = best_row['n_subjects']
        target_categories = set(best_row['categories'])
    else:
        # No exact match, find the smallest threshold that gives us at least n_categories
        candidate_rows = intersection_df[intersection_df['intersection_size'] >= n_categories]
        if len(candidate_rows) == 0:
            continue
        min_threshold_row = candidate_rows.loc[candidate_rows['n_subjects'].idxmin()]
        threshold = min_threshold_row['n_subjects']
        all_candidate_cats = min_threshold_row['categories']
        # Take first n_categories
        target_categories = set(all_candidate_cats[:n_categories])
    
    # Now find all subjects that have ALL target_categories
    subjects_with_all_cats = []
    for subject_id, subject_cats in subject_category_sets.items():
        if target_categories.issubset(subject_cats):
            subjects_with_all_cats.append(subject_id)
    
    n_subjects = len(subjects_with_all_cats)
    
    inverse_analysis.append({
        'n_categories': n_categories,
        'n_subjects': n_subjects,
        'threshold_used': threshold,
        'categories': sorted(list(target_categories)),
        'subject_ids': sorted(subjects_with_all_cats)
    })
    
    print(f"  X={n_categories:3d} categories → N={n_subjects:2d} subjects (threshold={threshold:2d})")

# Create DataFrame
inverse_df = pd.DataFrame(inverse_analysis)

# Save to CSV
inverse_df_expanded = inverse_df.copy()
# Convert lists to strings for CSV
inverse_df_expanded['categories'] = inverse_df_expanded['categories'].apply(lambda x: ', '.join(x))
inverse_df_expanded['subject_ids'] = inverse_df_expanded['subject_ids'].apply(lambda x: ', '.join(x))
inverse_df_expanded.to_csv(csv_dir / "inverse_intersection_analysis.csv", index=False)
print(f"\nSaved inverse analysis data to {csv_dir / 'inverse_intersection_analysis.csv'}")

# Plot: Number of categories vs maximum number of subjects
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(inverse_df['n_categories'], inverse_df['n_subjects'], 
        marker='o', linewidth=2, markersize=6, label='Max subjects with all X categories')
ax.set_xlabel('Number of Overlapping Categories (X)', fontsize=12)
ax.set_ylabel('Maximum Number of Subjects (N)', fontsize=12)
ax.set_title('Inverse Intersection Analysis\n(Maximum Subjects N for Given Category Count X)', fontsize=14)
ax.grid(True, alpha=0.3)

# Add annotations for key points
for idx, row in inverse_df.iterrows():
    if row['n_categories'] in [81, 99, 120, 126, 137, 140, 144, 147, 150, 153, 155, 158, 160, 163]:
        ax.annotate(f"N={int(row['n_subjects'])}", 
                   xy=(row['n_categories'], row['n_subjects']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, rotation=45, ha='left')

plt.tight_layout()
plt.savefig(output_dir / "inverse_intersection_analysis.png", dpi=150, bbox_inches='tight')
print(f"Saved inverse analysis plot to {output_dir / 'inverse_intersection_analysis.png'}")
plt.close()

# Create a summary table for easy reference
print("\n" + "="*60)
print("Summary: Maximum Subjects for Given Category Counts")
print("="*60)
print(f"{'Categories (X)':<15} {'Max Subjects (N)':<18} {'Threshold Used':<15}")
print("-"*60)
for _, row in inverse_df.iterrows():
    print(f"{row['n_categories']:<15} {int(row['n_subjects']):<18} {int(row['threshold_used']):<15}")

# Also create a reverse lookup: for common category counts, show max subjects
print("\n" + "="*60)
print("Quick Reference: Common Category Counts")
print("="*60)
common_counts = [148, 150, 140, 120, 100, 81]
for x in common_counts:
    matching = inverse_df[inverse_df['n_categories'] == x]
    if len(matching) > 0:
        row = matching.iloc[0]
        print(f"  X={x:3d} categories → N={int(row['n_subjects']):2d} subjects")
    else:
        # Find closest
        closest = inverse_df.iloc[(inverse_df['n_categories'] - x).abs().argsort()[:1]]
        if len(closest) > 0:
            row = closest.iloc[0]
            print(f"  X={x:3d} categories → N={int(row['n_subjects']):2d} subjects (closest: {int(row['n_categories'])} categories)")
