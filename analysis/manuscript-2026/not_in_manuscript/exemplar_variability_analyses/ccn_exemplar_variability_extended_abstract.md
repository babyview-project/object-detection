# Exemplar Variability in Infant-Centric Visual Experience: Convergent Evidence from CLIP and DINOv3 Embedding Geometry

**Authors**  
[Author 1], [Author 2], [Author 3]  
[Affiliation(s)]  
[Contact email]

## Introduction

Computational and developmental accounts of category learning often prioritize category frequency: how often an object type appears in visual input. However, frequency alone does not specify the representational challenge a learner faces. Two categories can occur equally often yet differ dramatically in within-category variability (e.g., pose, viewpoint, scale, occlusion, illumination, and scene context). These differences are likely consequential for category formation, boundary sharpness, and generalization.

Here, we characterize exemplar variability in infant-centric visual data using two distinct pretrained vision models (CLIP and DINOv3) and three complementary analyses: (1) montage-based qualitative inspection, (2) category-wise distance-to-centroid (global dispersion), and (3) k-nearest-neighbor (kNN) structure (local coherence). This combination lets us ask whether variability profiles are robust across embedding families and whether qualitative and quantitative indices converge.

Our central claim is that exemplar variability is a measurable and psychologically meaningful property of naturalistic category input, and that this property is visible in both global and local embedding geometry across models.

## Methods

**Data and category set.**  
We analyzed object exemplars from infant-perspective visual experience (BabyView), organized into [N categories] with [N total exemplars / category range]. We focused on categories meeting minimum sample criteria ([criterion]) to ensure stable per-category estimates.

**Embeddings.**  
Each exemplar crop was embedded using:
- **CLIP** ([model variant])
- **DINOv3** ([model variant])

Embeddings were [state preprocessing: e.g., L2-normalized], and all analyses were run separately for each model.

**Analysis 1: Montage visualization (qualitative).**  
For selected categories spanning the variability range, we generated exemplar montages to inspect visual heterogeneity (appearance, context, scale, and viewpoint). Montage panels served as an interpretable bridge between raw images and geometry-based metrics.

**Analysis 2: Distance-to-centroid (global variability).**  
For each category c, we computed a centroid mu_c as the mean embedding of exemplars in that category. We then quantified dispersion by the average (or median) distance from each exemplar to mu_c:

V_c = (1 / |c|) * sum_{x_i in c} d(x_i, mu_c)

where d is [cosine distance / Euclidean distance]. Higher V_c indicates broader within-category spread. We summarized category rankings and [optionally] semantic-group aggregates.

**Analysis 3: kNN neighborhood structure (local variability).**  
For each exemplar, we identified its k nearest neighbors in embedding space (k = [value]) and computed local-structure metrics, including [same-category neighbor proportion / within-category neighbor distance dispersion / both]. Category-level kNN scores were obtained by averaging exemplar-level values. This analysis captures local manifold continuity and category boundary sharpness, complementing centroid-based global spread.

**Cross-model comparison.**  
To evaluate robustness, we compared CLIP and DINOv3 category variability profiles via [rank correlation / overlap of top- and bottom-variability categories / both].

## Results

**1) Montage evidence reveals meaningful variability differences.**  
Montages showed clear qualitative contrasts across categories. Some categories exhibited tight visual regularity (consistent object shape, framing, and context), while others spanned broader appearance and contextual variation (multiple viewpoints, clutter levels, co-occurring objects, and scale regimes). Categories identified as visually diverse in montages were typically those ranked as high-variability by geometric metrics in both embedding models.

**2) Distance-to-centroid reveals stable global spread differences.**  
Distance-to-centroid analyses in CLIP and DINOv3 both produced wide category-level dispersion ranges, indicating substantial heterogeneity in within-category structure. Despite model-specific scaling differences, categories with larger centroid distances in one model tended to be large in the other, suggesting a shared signal rather than architecture-specific noise.
- **CLIP:** [insert mean/SD/range; top 5 and bottom 5 categories]
- **DINOv3:** [insert mean/SD/range; top 5 and bottom 5 categories]
- **Cross-model alignment:** [insert Spearman rho, CI/p-value if available]

**3) kNN analyses show complementary local-geometry effects.**  
kNN metrics converged with centroid findings while adding local detail. Categories with large global dispersion often showed lower neighborhood coherence (e.g., lower same-category neighbor proportion and/or higher local distance dispersion), consistent with fragmented or overlapping local manifolds. Categories with compact centroid structure generally exhibited stronger local cohesion.
- **CLIP kNN:** [insert central tendency + notable categories]
- **DINOv3 kNN:** [insert central tendency + notable categories]
- **Cross-model agreement:** [insert correspondence statistic]

**4) Convergence across qualitative and quantitative views.**  
Across both embedding models, montage-based impressions, centroid dispersion, and kNN coherence pointed to similar category-level variability profiles. This three-way convergence supports the interpretation that exemplar variability is an intrinsic property of the data distribution rather than an artifact of a single metric.

## Discussion

The present results extend category-statistics analyses beyond occurrence counts by quantifying representational structure within categories. Two implications follow.

First, **learning-relevant input statistics are multidimensional**: frequency and variability jointly characterize the learner’s visual environment. Categories with broad exemplar spread may impose different generalization demands than equally frequent but compact categories.

Second, **variability appears robust across embedding families**. CLIP and DINOv3 differ in training objectives and representational biases, yet both recover similar compact-vs-variable category distinctions. This suggests that observed variability profiles reflect stable image-distribution properties likely to matter for both models and biological learners.

Methodologically, combining montage inspection with centroid and kNN analyses provides a practical framework for interpretable representational profiling. Global and local geometry are not redundant: centroid distance indexes overall spread around a category center, while kNN captures neighborhood continuity and boundary sharpness. Their joint use offers a richer account of within-category structure than either alone.

## Limitations and Future Directions

Three limitations are important: (1) variability estimates depend on embedding choice and preprocessing decisions, (2) uneven category sample sizes can influence metric stability, and (3) crop quality and annotation noise may inflate apparent spread in some categories.

Future work should test (a) temporal and subject-specific variability trajectories, (b) additional embedding families and self-supervised objectives, and (c) direct links between variability metrics and downstream category-learning performance in both computational models and human developmental data.

## Conclusion

Exemplar variability in infant-centric visual data is systematic, measurable, and convergent across CLIP and DINOv3. Qualitative montages, global centroid dispersion, and local kNN structure jointly indicate that categories differ not only in how often they appear, but also in how representationally diverse they are. Modeling category learning in natural environments therefore benefits from treating variability as a core statistic of visual experience.

## Figure Plan

- **Figure 1:** Montage panels for selected high- vs low-variability categories.
- **Figure 2:** Category-wise distance-to-centroid (CLIP and DINOv3 side-by-side).
- **Figure 3:** Category-wise kNN coherence/diversity metrics (CLIP and DINOv3 side-by-side).

