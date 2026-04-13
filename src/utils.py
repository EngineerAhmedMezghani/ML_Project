import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


NUMERIC_FEATURES = [
    'MonetaryAvg', 'MonetaryStd', 'MonetaryMin', 'MonetaryMax',
    'TotalQuantity', 'AvgQuantityPerTransaction', 'MinQuantity', 'MaxQuantity',
    'FirstPurchaseDaysAgo', 'PreferredDayOfWeek', 'PreferredHour',
    'PreferredMonth', 'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases',
    'UniqueProducts', 'UniqueDescriptions', 'AvgProductsPerTransaction',
    'UniqueCountries', 'NegativeQuantityCount', 'ZeroPriceCount',
    'CancelledTransactions', 'ReturnRatio', 'TotalTransactions',
    'UniqueInvoices', 'AvgLinesPerInvoice', 'Age', 'SupportTicketsCount',
    'SatisfactionScore', 'Churn', 'RegistrationDate_day',
    'RegistrationDate_month', 'RegistrationDate_year', 'MonetaryPerDay',
    'AvgBasketValue', 'TenureRatio'
]

CATEGORICAL_FEATURES = [
    'RFMSegment', 'AgeCategory', 'SpendingCategory', 'CustomerType',
    'FavoriteSeason', 'PreferredTimeOfDay', 'Region', 'LoyaltyLevel',
    'ChurnRiskCategory', 'WeekendPreference', 'BasketSizeCategory',
    'ProductDiversity', 'Gender', 'AccountStatus', 'Country'
]

DROP_FEATURES = ['CustomerID', 'RegistrationDate', 'LastLoginIP']


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.drop(columns=[c for c in DROP_FEATURES if c in df.columns])
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns:
            df_enc[col] = pd.Categorical(df_enc[col]).codes.astype(float)
            df_enc[col] = df_enc[col].replace(-1, np.nan)
    for col in df_enc.columns:
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

    # ── Print strong correlations in the terminal ──────────────────────
    print_strong_correlations(corr, threshold=0.1)

    # ── Plot heatmap ───────────────────────────────────────────────────
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

def feature_engineering(df: pd.DataFrame):
    """
    Perform feature engineering on the dataframe.
    """

    cols_to_drop = [
        'MonetaryMax',
        'MonetaryMin',
        'MonetaryStd',
        'MonetaryPerDay',
        'TotalTransactions',
        'MaxQuantity',
        'MinQuantity',
        'NegativeQuantityCount',
        'CancelledTransactions',
        'WeekendPreference',
        'WeekendPurchaseRatio',
        'RegistrationDate_year',
        'ProductDiversity'
    ]
    for i in cols_to_drop:
        print(i, df[i].dtype)

    df['MonetaryRange'] = df['MonetaryMax'] - df['MonetaryMin']
    df['MonetaryStability'] = df['MonetaryStd'] / (df['MonetaryPerDay'] + 1e-6)
    df['MonetaryAvg'] = (df['MonetaryMin'] + df['MonetaryMax']) / 2
    df['SpendingIntensity'] = df['MonetaryPerDay'] * df['TotalTransactions']
    df['QuantityRange'] = df['MaxQuantity'] - df['MinQuantity']   
    df['NegativeBehavior'] = df['NegativeQuantityCount'] + df['CancelledTransactions']
    df['RiskScore'] = df['ReturnRatio'] * df['TotalTransactions']
    #type is object here 
    #df['WeekendActivity'] = df['WeekendPreference'] * df['WeekendPurchaseRatio']
    df['CustomerExperience'] = df['RegistrationDate_year'] * df['TotalTransactions']
    #type is object here 
    #df['EngagementScore'] = df['TotalTransactions'] * df['MonetaryPerDay'] * df['ProductDiversity']

    df.drop(columns=cols_to_drop, inplace=True)
    return df


if __name__ == "__main__":
    # Replace with your actual data loading:
    df=pd.read_csv("data/processed/retail_customers_processed.csv")
    print(len(df.columns.unique()))
    # print(df['MonetaryTotal'].dtype)

    # Demo with synthetic data
    # np.random.seed(42)
    # n = 500
    # demo_data = {feat: np.random.randn(n) for feat in NUMERIC_FEATURES}
    # for feat in CATEGORICAL_FEATURES:
    #     demo_data[feat] = np.random.choice(['A', 'B', 'C'], size=n)
    # demo_data['CustomerID']       = np.arange(n)
    # demo_data['RegistrationDate'] = pd.date_range('2020-01-01', periods=n).astype(str)
    # demo_data['LastLoginIP']      = ['192.168.0.1'] * n
    # df = pd.DataFrame(demo_data)

    # plot_correlation_matrix(df, save_path='correlation_matrix.png')
    print(df.columns.unique())
    df_clean = feature_engineering(df)
    print(len(df_clean.columns.unique()))
    print(df_clean.columns.unique())