import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_pca_analysis(features_df, output_folder):
    """Perform PCA analysis (binary 0/1 variables are not standardized; save subplots separately)."""
    
    # Select numeric features
    feature_columns = [
        'qa_turns', 'is_multi_turn', 'total_time_minutes', 'avg_qa_time_minutes',
        'total_question_chars', 'avg_question_length',
        'if_non_class', 'avg_hours_to_assignment', 'avg_hours_since_release',
        'course_progress_ratio', 'calendar_week_since_2025_0217',
        'hours_to_next_class', 'hours_from_last_class', 'has_copy_keywords', 'copy_keywords_count',
        'is_exam_week','day_period','is_weekend',
        'is_in_class_time','question_type_why_how'
    ]
    
    # Binary columns (not standardized)
    binary_cols = [
        'is_multi_turn', 'if_non_class', 'has_copy_keywords',
        'is_exam_week', 'is_weekend', 'is_in_class_time', 'question_type_why_how',
    ]
    binary_cols_present = [c for c in binary_cols if c in feature_columns and c in features_df.columns]
    continuous_cols = [c for c in feature_columns if c not in binary_cols_present]

    X = features_df[feature_columns].copy()

    # Handle missing values and infinities
    print("Processing missing and infinite values...")
    X = X.fillna(X.median(numeric_only=True))
    time_features = ['hours_to_next_class', 'hours_from_last_class']
    for feature in time_features:
        if feature in X.columns:
            inf_mask = np.isinf(X[feature])
            if inf_mask.any():
                finite_values = X[feature][np.isfinite(X[feature])]
                replacement_value = finite_values.max() * 1.5 if not finite_values.empty else 168.0
                X.loc[inf_mask, feature] = replacement_value

    for col in X.columns:
        inf_mask = np.isinf(X[col])
        if inf_mask.any():
            median_val = X[col][~inf_mask].median()
            X.loc[inf_mask, col] = median_val

    # Add small noise to constant continuous columns
    if continuous_cols:
        constant_cols = X[continuous_cols].columns[X[continuous_cols].var() == 0].tolist()
        if constant_cols:
            for col in constant_cols:
                X[col] += np.random.normal(0, 1e-8, len(X))

    # Standardize continuous columns
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    if continuous_cols:
        X_cont_scaled = scaler.fit_transform(X[continuous_cols])
        X_scaled_df = pd.DataFrame(X_cont_scaled, columns=continuous_cols, index=X.index)
    else:
        X_scaled_df = pd.DataFrame(index=X.index)

    # Add binary columns without scaling
    for col in binary_cols_present:
        X_scaled_df[col] = X[col].astype(float)

    # Keep original column order
    X_scaled_df = X_scaled_df[feature_columns]
    X_scaled = X_scaled_df.values

    # Replace NaN/infinite if any remain
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA
    print("Performing PCA...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Print PCA summary
    print("\n=== PCA Analysis Results ===")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:6]):
        print(f"PC{i+1}: {ratio:.4f}")
    print(f"Cumulative (first 3): {pca.explained_variance_ratio_[:3].sum():.4f}")

    components_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=feature_columns
    )
    print("\nFeature loadings for first 3 components:")
    print(components_df.round(3))

    # ====== SAVE INDIVIDUAL PLOTS ======

    # 1️⃣ Explained Variance Ratio
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, color='skyblue', edgecolor='black')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_explained_variance_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2️⃣ Cumulative Explained Variance Ratio
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_cumulative_explained_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3️⃣ Feature Loadings Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(components_df.T, annot=True, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'})
    plt.title('Feature Loadings in Principal Components')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_feature_loadings_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4️⃣ 2D PCA Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30, edgecolor='black')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA 2D Projection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_2d_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved individual PCA plots to: {output_folder}")
    print("  - pca_explained_variance_ratio.png")
    print("  - pca_cumulative_explained_variance.png")
    print("  - pca_feature_loadings_heatmap.png")
    print("  - pca_2d_projection.png")

    return X_scaled, X_pca, pca, scaler, feature_columns
