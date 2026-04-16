import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe by encoding categoricals and converting to numeric."""
    # Drop ID columns if they exist
    drop_cols = ['CustomerID', 'RegistrationDate', 'LastLoginIP']
    df_enc = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Encode remaining categorical columns
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object' or df_enc[col].dtype.name == 'category':
            df_enc[col] = pd.Categorical(df_enc[col]).codes.astype(float)
            df_enc[col] = df_enc[col].replace(-1, np.nan)
        df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce')
    
    return df_enc


def print_strong_correlations(corr: pd.DataFrame, threshold: float = 0.5):
    """
    Print all unique feature pairs whose |Pearson r| is between threshold and 1
    (excluding the diagonal, i.e. a feature with itself).
    """
    cols = corr.columns.tolist()
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if threshold <= abs(r) < 1.0:
                pairs.append((cols[i], cols[j], r))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  STRONG CORRELATIONS  |r| >= {threshold}  ({len(pairs)} pairs found)")
    print(sep)
    print(f"  {'Feature A':<30} {'Feature B':<30} {'r':>6}")
    print(f"  {'-'*30} {'-'*30} {'-'*6}")
    for feat_a, feat_b, r in pairs:
        print(f"  {feat_a:<30} {feat_b:<30} {r:+.4f}")
    print(f"{sep}\n")


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Compute and plot the Pearson correlation matrix for all features.
    Also prints to the terminal all pairs with |r| between 0.5 and 1.
    Saves to `save_path` if provided, otherwise displays interactively.
    """
    df_clean = prepare_dataframe(df)
    corr = df_clean.corr(method='pearson')

    # Print strong correlations in the terminal
    print_strong_correlations(corr, threshold=0.1)

    # Plot heatmap
    n = len(corr)
    fig_size = max(20, n * 0.45)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr,
        ax=ax,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        linewidths=0.4,
        linecolor='#e0e0e0',
        square=True,
        cbar_kws={'shrink': 0.6, 'label': 'Pearson r'}
    )

    ax.set_title('Pearson Correlation Matrix — All Features', fontsize=16, pad=16)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8, rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved -> {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return corr


# =============================================================================
# MULTICOLLINEARITY DETECTION FUNCTIONS
# =============================================================================

def get_high_correlation_features(corr: pd.DataFrame, threshold: float = 0.8) -> list:
    """
    Identify feature pairs with |correlation| > threshold.
    Returns list of tuples (feature_a, feature_b, correlation_value).
    """
    cols = corr.columns.tolist()
    high_corr_pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                high_corr_pairs.append((cols[i], cols[j], r))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return high_corr_pairs


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    VIF > 10 indicates severe multicollinearity.
    Requires: pip install statsmodels
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_clean = prepare_dataframe(df)
    df_numeric = df_clean.select_dtypes(include=[np.number]).dropna()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i)
                       for i in range(df_numeric.shape[1])]

    vif_data = vif_data.sort_values("VIF", ascending=False)
    return vif_data


def recommend_features_to_drop(df: pd.DataFrame, corr_threshold: float = 0.8,
                                  vif_threshold: float = 10.0) -> dict:
    """
    Recommend features to drop based on correlation threshold and VIF.

    Strategy:
    1. Identify highly correlated pairs (|r| > corr_threshold)
    2. For each pair, drop the one with higher average correlation
       (or lower business sense - manual review recommended)
    3. Also flag features with VIF > vif_threshold

    Returns dict with recommendations and statistics.
    """
    df_clean = prepare_dataframe(df)
    corr = df_clean.corr(method='pearson')

    high_corr_pairs = get_high_correlation_features(corr, threshold=corr_threshold)

    features_to_drop = set()
    correlation_conflicts = []

    for feat_a, feat_b, r in high_corr_pairs:
        if feat_a in features_to_drop or feat_b in features_to_drop:
            continue

        avg_corr_a = corr[feat_a].abs().mean()
        avg_corr_b = corr[feat_b].abs().mean()

        if avg_corr_a > avg_corr_b:
            drop_feature, keep_feature = feat_a, feat_b
        else:
            drop_feature, keep_feature = feat_b, feat_a

        features_to_drop.add(drop_feature)
        correlation_conflicts.append({
            'pair': (feat_a, feat_b),
            'correlation': r,
            'drop': drop_feature,
            'keep': keep_feature,
            'reason': f"High correlation (r={r:.3f}), {drop_feature} has higher avg |r|"
        })

    try:
        vif_data = calculate_vif(df)
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
    except Exception:
        high_vif_features = []
        vif_data = pd.DataFrame()

    return {
        'high_correlation_pairs': high_corr_pairs,
        'correlation_conflicts': correlation_conflicts,
        'recommended_drops_correlation': list(features_to_drop),
        'high_vif_features': high_vif_features,
        'vif_data': vif_data,
        'summary': {
            'total_features': len(df_clean.columns),
            'high_correlation_pairs_found': len(high_corr_pairs),
            'features_recommended_to_drop_corr': len(features_to_drop),
            'features_with_high_vif': len(high_vif_features)
        }
    }


def print_multicollinearity_report(df: pd.DataFrame, corr_threshold: float = 0.8,
                                    vif_threshold: float = 10.0):
    """
    Print a comprehensive multicollinearity report including:
    - High correlation pairs
    - VIF scores
    - Recommended features to drop
    """
    results = recommend_features_to_drop(df, corr_threshold, vif_threshold)

    sep = "=" * 70
    print(f"\n{sep}")
    print("         MULTICOLLINEARITY ANALYSIS REPORT")
    print(sep)

    summary = results['summary']
    print(f"\n  Total features analyzed: {summary['total_features']}")
    print(f"  Correlation threshold: |r| > {corr_threshold}")
    print(f"  VIF threshold: VIF > {vif_threshold}")

    print(f"\n  HIGH CORRELATION PAIRS (|r| > {corr_threshold}):")
    print(f"  {'-' * 50}")
    if results['high_correlation_pairs']:
        for feat_a, feat_b, r in results['high_correlation_pairs']:
            marker = " ***" if abs(r) > 0.9 else ""
            print(f"    {feat_a:<30} vs {feat_b:<30} r = {r:+.3f}{marker}")
        print(f"\n  *** = severe correlation (|r| > 0.9)")
    else:
        print("    No high correlation pairs found.")

    print(f"\n  RECOMMENDED DROPS (Correlation-based):")
    print(f"  {'-' * 50}")
    if results['recommended_drops_correlation']:
        for conflict in results['correlation_conflicts']:
            print(f"    Drop: {conflict['drop']:<30} Keep: {conflict['keep']}")
            print(f"      Reason: {conflict['reason']}")
    else:
        print("    No features recommended to drop based on correlation.")

    if not results['vif_data'].empty:
        print(f"\n  VIF SCORES (VIF > {vif_threshold} = severe multicollinearity):")
        print(f"  {'-' * 50}")
        for _, row in results['vif_data'].iterrows():
            marker = " ***" if row['VIF'] > vif_threshold else ""
            print(f"    {row['Feature']:<35} VIF = {row['VIF']:>8.2f}{marker}")
        if results['high_vif_features']:
            print(f"\n  *** = VIF > {vif_threshold} (severe)")

    print(f"\n{sep}\n")


def plot_correlation_heatmap_highlights(df: pd.DataFrame, threshold: float = 0.8,
                                         save_path: str = None):
    """
    Plot correlation heatmap highlighting values above threshold.
    """
    df_clean = prepare_dataframe(df)
    corr = df_clean.corr(method='pearson')

    n = len(corr)
    fig_size = max(16, n * 0.4)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    mask = mask & (corr.abs() < threshold)

    sns.heatmap(
        corr,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        linewidths=0.3,
        square=True,
        mask=mask,
        cbar_kws={'shrink': 0.6, 'label': f'|r| >= {threshold}'}
    )

    ax.set_title(f'High Correlations Highlighted (|r| >= {threshold})', fontsize=14, pad=12)
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=7, rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved -> {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return corr


if __name__ == "__main__":
    import os
    
    # Create outputs directory if needed
    os.makedirs('outputs', exist_ok=True)
    
    # Load preprocessed data (feature engineering already done in preprocessing.py)
    df = pd.read_csv("data/processed/retail_customers_processed.csv")
    print(f"Loaded {len(df.columns)} features from processed data")
    
    # Generate correlation analysis
    print("\n" + "="*70)
    print("GENERATING CORRELATION MATRIX HEATMAP")
    print("="*70)
    plot_correlation_matrix(df, save_path='outputs/correlation_matrix.png')

    print("\n" + "="*70)
    print("MULTICOLLINEARITY REPORT")
    print("="*70)
    print_multicollinearity_report(df, corr_threshold=0.8, vif_threshold=10.0)

    print("\n" + "="*70)
    print("GENERATING HIGH CORRELATION HIGHLIGHT HEATMAP")
    print("="*70)
    plot_correlation_heatmap_highlights(df, threshold=0.8,
                                         save_path='outputs/correlation_highlights.png')
