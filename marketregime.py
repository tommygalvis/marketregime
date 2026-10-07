"""
Market Regime Detection Using Singular Value Decomposition
A Quantitative Approach to Identifying Market States

This script implements a production-ready framework for detecting market
regimes using Singular Value Decomposition (SVD) on sector ETF returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats  # kept for potential extensions
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.precision', 4)

print("Environment configured successfully.")

# ---------------------------------------------------------------------
# 1. Data Acquisition
# ---------------------------------------------------------------------

# S&P 500 Sector ETFs
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Date range: Last 3 years of data
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=3 * 365)

print(f"Fetching data from {START_DATE.date()} to {END_DATE.date()}")
print(f"Sectors: {list(SECTOR_ETFS.values())}")


def fetch_sector_data(tickers: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch close prices for sector ETFs."""
    symbols = list(tickers.keys())

    # Download data
    data = yf.download(symbols, start=start, end=end, progress=False)

    # Handle different yfinance versions / structures
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data = data['Close']
        elif 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
    else:
        if 'Close' in data.columns:
            data = data['Close']
        elif 'Adj Close' in data.columns:
            data = data['Adj Close']

    # Handle single ticker case
    if isinstance(data, pd.Series):
        data = data.to_frame(symbols[0])

    # Rename columns to sector names
    data.columns = [tickers.get(col, col) for col in data.columns]

    return data.dropna()


prices = fetch_sector_data(SECTOR_ETFS, START_DATE, END_DATE)
print(f"\nFetched {len(prices)} trading days across {len(prices.columns)} sectors")
print(prices.tail())

# ---------------------------------------------------------------------
# 2. Returns Computation
# ---------------------------------------------------------------------


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Compute returns from price data.

    Args:
        prices: DataFrame of asset prices
        method: 'log' for log returns, 'simple' for arithmetic returns
    """
    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()


returns = compute_returns(prices, method='log')

print(f"\nReturns matrix shape: {returns.shape}")
print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"\nReturns statistics (annualized):")
print(f"Mean: {(returns.mean() * 252 * 100).round(2).to_dict()}")

# ---------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------

# Cumulative returns and rolling volatility
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Cumulative returns
cumulative = (1 + returns).cumprod()
cumulative.plot(ax=axes[0], alpha=0.8)
axes[0].set_title('Cumulative Returns by Sector', fontsize=14, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Cumulative Return')
axes[0].legend(loc='upper left', ncol=3, fontsize=9)
axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.3)

# Rolling volatility (21-day)
rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
rolling_vol.plot(ax=axes[1], alpha=0.8)
axes[1].set_title('21-Day Rolling Volatility (Annualized %)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Volatility (%)')
axes[1].legend(loc='upper right', ncol=3, fontsize=9)

plt.tight_layout()
plt.show()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation = returns.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax,
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Sector Return Correlations (Full Period)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

avg_pairwise_corr = correlation.values[np.triu_indices_from(correlation.values, 1)].mean()
print(f"Average pairwise correlation: {avg_pairwise_corr:.3f}")

# ---------------------------------------------------------------------
# 4. SVD Implementation
# ---------------------------------------------------------------------


class MarketRegimeDetector:
    """SVD-based market regime detection framework."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.mean = None
        self.explained_variance_ratio = None
        self.projections = None

    def fit(self, returns: np.ndarray) -> 'MarketRegimeDetector':
        """Perform SVD on the returns matrix."""
        # Center the data
        self.mean = returns.mean(axis=0)
        centered = returns - self.mean

        # Full SVD decomposition
        self.U, self.S, self.Vt = np.linalg.svd(centered, full_matrices=False)

        # Explained variance ratios
        variance = self.S ** 2
        self.explained_variance_ratio = variance / variance.sum()

        # Project data onto principal components
        self.projections = centered @ self.Vt[:self.n_components].T

        return self

    def get_loadings(self) -> np.ndarray:
        """Get asset loadings for each principal component."""
        return self.Vt[:self.n_components]

    def reconstruct(self, n_components: int = None) -> np.ndarray:
        """Reconstruct returns using specified number of components."""
        k = n_components or self.n_components
        return self.U[:, :k] @ np.diag(self.S[:k]) @ self.Vt[:k] + self.mean

    def reconstruction_error(self, n_components: int = None) -> float:
        """Compute reconstruction error (1 - R²)."""
        k = n_components or self.n_components
        return 1 - float(self.explained_variance_ratio[:k].sum())


R = returns.values
detector = MarketRegimeDetector(n_components=3)
detector.fit(R)

print("\nSVD Decomposition Complete")
print(f"Returns matrix shape: {R.shape}")
print(f"U shape: {detector.U.shape}")
print(f"S shape: {detector.S.shape}")
print(f"Vt shape: {detector.Vt.shape}")

# ---------------------------------------------------------------------
# 5. Variance Analysis & Loadings
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_components = min(10, len(detector.S))
variance_pct = detector.explained_variance_ratio[:n_components] * 100
cumulative_var = np.cumsum(variance_pct)

# Individual variance
colors = ['#2ecc71' if i < 3 else '#95a5a6' for i in range(n_components)]
axes[0].bar(range(1, n_components + 1), variance_pct, color=colors, edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('Variance Explained by Component', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(1, n_components + 1))

for i, v in enumerate(variance_pct):
    axes[0].text(i + 1, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

# Cumulative variance
axes[1].plot(range(1, n_components + 1), cumulative_var, 'o-', color='#3498db', linewidth=2, markersize=8)
axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
axes[1].fill_between(range(1, n_components + 1), cumulative_var, alpha=0.3)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, n_components + 1))
axes[1].legend()
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.show()

print("\nVariance Explained Summary:")
print("-" * 50)
for i in range(min(5, n_components)):
    print(f"PC{i+1}: {variance_pct[i]:6.2f}% (Cumulative: {cumulative_var[i]:6.2f}%)")

# Asset loadings
loadings = detector.get_loadings()
loadings_df = pd.DataFrame(
    loadings.T,
    index=returns.columns,
    columns=[f'PC{i+1}' for i in range(loadings.shape[0])]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    if i >= loadings_df.shape[1]:
        ax.axis('off')
        continue
    pc_name = f'PC{i+1}'
    colors_pc = ['#e74c3c' if v < 0 else '#2ecc71' for v in loadings_df[pc_name]]
    loadings_df[pc_name].plot(kind='barh', ax=ax, color=colors_pc, edgecolor='white')
    ax.set_title(f'{pc_name} Loadings ({variance_pct[i]:.1f}% var)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Loading')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 6. Regime Visualization via Projections
# ---------------------------------------------------------------------

projections_df = pd.DataFrame(
    detector.projections,
    index=returns.index,
    columns=[f'PC{i+1}' for i in range(detector.n_components)]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Time-colored scatter in PC1-PC2 space
time_colors = np.arange(len(projections_df))
scatter = axes[0].scatter(
    projections_df['PC1'],
    projections_df['PC2'],
    c=time_colors,
    cmap='viridis',
    alpha=0.6,
    s=20
)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
axes[0].set_xlabel('PC1 (Market Factor)')
axes[0].set_ylabel('PC2 (Rotation Factor)')
axes[0].set_title('Market Days in Principal Component Space', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Trading Day')

# Distance from origin as regime intensity
distances = np.sqrt(projections_df['PC1']**2 + projections_df['PC2']**2)
axes[1].plot(projections_df.index, distances, alpha=0.7, linewidth=0.8)
axes[1].fill_between(projections_df.index, distances, alpha=0.3)

threshold = distances.quantile(0.95)
high_stress = distances[distances > threshold]
axes[1].scatter(high_stress.index, high_stress.values, color='red', s=20,
                label='High stress (>95th pctl)', zorder=5)

axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Distance from Origin')
axes[1].set_title('Regime Intensity Over Time', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 7. Regime Classification
# ---------------------------------------------------------------------


def classify_regimes(projections: pd.DataFrame) -> pd.Series:
    """Classify market days into regimes based on PC1 and PC2 quadrants."""
    pc1_median = projections['PC1'].median()
    pc2_median = projections['PC2'].median()

    conditions = [
        (projections['PC1'] >= pc1_median) & (projections['PC2'] >= pc2_median),
        (projections['PC1'] >= pc1_median) & (projections['PC2'] < pc2_median),
        (projections['PC1'] < pc1_median) & (projections['PC2'] >= pc2_median),
        (projections['PC1'] < pc1_median) & (projections['PC2'] < pc2_median),
    ]

    regime_names = ['Risk-On Rally', 'Defensive Rally', 'Risk-Off Stress', 'Broad Decline']

    regimes_arr = np.select(conditions, regime_names, default='')
    return pd.Series(regimes_arr, index=projections.index, name='regime')


regimes = classify_regimes(projections_df)

print("Regime Distribution:")
print(regimes.value_counts())
print(f"\nTotal days: {len(regimes)}")

# Regime visualization
regime_colors = {
    'Risk-On Rally': '#2ecc71',
    'Defensive Rally': '#3498db',
    'Risk-Off Stress': '#e74c3c',
    'Broad Decline': '#9b59b6'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot colored by regime
for regime_name, color in regime_colors.items():
    mask = regimes == regime_name
    axes[0].scatter(
        projections_df.loc[mask, 'PC1'],
        projections_df.loc[mask, 'PC2'],
        c=color,
        label=regime_name,
        alpha=0.6,
        s=20
    )

axes[0].axhline(y=projections_df['PC2'].median(), color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=projections_df['PC1'].median(), color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('PC1 (Market Factor)')
axes[0].set_ylabel('PC2 (Rotation Factor)')
axes[0].set_title('Market Regimes in PC Space', fontsize=14, fontweight='bold')
axes[0].legend()

# Regime timeline
regime_numeric = regimes.map({name: i for i, name in enumerate(regime_colors.keys())})
for i, (regime_name, color) in enumerate(regime_colors.items()):
    mask = regimes == regime_name
    axes[1].scatter(
        regimes.index[mask],
        regime_numeric[mask],
        c=color,
        s=10,
        label=regime_name,
        alpha=0.7
    )

axes[1].set_xlabel('Date')
axes[1].set_ylabel('Regime')
axes[1].set_yticks(range(len(regime_colors)))
axes[1].set_yticklabels(list(regime_colors.keys()))
axes[1].set_title('Regime Timeline', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 8. Regime Characteristics Analysis
# ---------------------------------------------------------------------


def compute_regime_metrics(returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute comprehensive metrics for each regime."""
    metrics = []

    for regime in regimes.unique():
        mask = regimes == regime
        regime_returns = returns[mask]

        # Portfolio returns (equal-weighted)
        portfolio_returns = regime_returns.mean(axis=1)

        if portfolio_returns.std() > 0:
            sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        corr_mat = regime_returns.corr()
        avg_corr = corr_mat.values[np.triu_indices_from(corr_mat.values, 1)].mean()

        metrics.append({
            'Regime': regime,
            'Days': mask.sum(),
            'Frequency (%)': mask.sum() / len(regimes) * 100,
            'Avg Daily Return (bps)': portfolio_returns.mean() * 10000,
            'Volatility (ann. %)': portfolio_returns.std() * np.sqrt(252) * 100,
            'Sharpe Ratio': sharpe,
            'Avg Correlation': avg_corr,
            'Worst Day (%)': portfolio_returns.min() * 100,
            'Best Day (%)': portfolio_returns.max() * 100,
        })

    return pd.DataFrame(metrics).set_index('Regime')


regime_metrics = compute_regime_metrics(returns, regimes)
print("\nRegime Characteristics:")
print("=" * 80)
print(regime_metrics.round(2).to_string())

# Sector performance by regime
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, regime in enumerate(regime_colors.keys()):
    mask = regimes == regime
    regime_returns = returns[mask]

    ann_returns = regime_returns.mean() * 252 * 100
    colors_sect = ['#2ecc71' if r > 0 else '#e74c3c' for r in ann_returns]

    ann_returns.plot(kind='barh', ax=axes[i], color=colors_sect, edgecolor='white')
    axes[i].set_title(f'{regime}\n({mask.sum()} days)', fontsize=12, fontweight='bold')
    axes[i].axvline(x=0, color='black', linewidth=0.5)
    axes[i].set_xlabel('Annualized Return (%)')

plt.suptitle('Sector Performance by Regime', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 9. Regime Transitions & Durations
# ---------------------------------------------------------------------


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute regime transition probability matrix."""
    transitions = pd.crosstab(
        regimes.shift(1).dropna(),
        regimes.iloc[1:],
        normalize='index'
    ) * 100
    return transitions


transition_matrix = compute_transition_matrix(regimes)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    transition_matrix,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    ax=ax,
    cbar_kws={'label': 'Probability (%)'}
)
ax.set_title('Regime Transition Probabilities (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('To Regime')
ax.set_ylabel('From Regime')
plt.tight_layout()
plt.show()

print("\nKey Transition Insights:")
print("- Diagonal values show regime persistence (tendency to stay in same regime)")
print("- Off-diagonal values show transition probabilities")


def compute_regime_durations(regimes: pd.Series) -> dict:
    """Compute duration statistics for each regime."""
    durations = {regime: [] for regime in regimes.unique()}

    current_regime = regimes.iloc[0]
    current_duration = 1

    for regime in regimes.iloc[1:]:
        if regime == current_regime:
            current_duration += 1
        else:
            durations[current_regime].append(current_duration)
            current_regime = regime
            current_duration = 1

    durations[current_regime].append(current_duration)
    return durations


durations = compute_regime_durations(regimes)

duration_stats = []
for regime, durs in durations.items():
    duration_stats.append({
        'Regime': regime,
        'Occurrences': len(durs),
        'Avg Duration (days)': np.mean(durs),
        'Max Duration (days)': np.max(durs),
        'Min Duration (days)': np.min(durs),
    })

duration_df = pd.DataFrame(duration_stats).set_index('Regime')
print("\nRegime Duration Statistics:")
print("=" * 60)
print(duration_df.round(1).to_string())

# ---------------------------------------------------------------------
# 10. Rolling SVD Analysis
# ---------------------------------------------------------------------


def rolling_svd_analysis(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Compute rolling SVD statistics."""
    results = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i].values
        centered = window_returns - window_returns.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        variance_ratios = (S ** 2) / (S ** 2).sum()

        results.append({
            'date': returns.index[i],
            'pc1_var': variance_ratios[0],
            'pc2_var': variance_ratios[1],
            'pc3_var': variance_ratios[2],
            'top3_var': variance_ratios[:3].sum(),
            'effective_dim': 1 / (variance_ratios ** 2).sum(),  # Participation ratio
        })

    return pd.DataFrame(results).set_index('date')


print("\nComputing rolling SVD analysis (63-day window)...")
rolling_stats = rolling_svd_analysis(returns, window=63)
print("Done!")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# PC1 variance
axes[0].plot(rolling_stats.index, rolling_stats['pc1_var'] * 100, color='#3498db', linewidth=1)
axes[0].fill_between(rolling_stats.index, rolling_stats['pc1_var'] * 100, alpha=0.3)
axes[0].axhline(y=rolling_stats['pc1_var'].mean() * 100, color='red', linestyle='--',
                alpha=0.5, label='Mean')
axes[0].set_ylabel('PC1 Variance (%)')
axes[0].set_title('Rolling Market Factor Dominance (63-day window)', fontsize=12, fontweight='bold')
axes[0].legend()

# Top 3 variance
axes[1].plot(rolling_stats.index, rolling_stats['top3_var'] * 100, color='#2ecc71', linewidth=1)
axes[1].fill_between(rolling_stats.index, rolling_stats['top3_var'] * 100, alpha=0.3)
axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
axes[1].set_ylabel('Top 3 PC Variance (%)')
axes[1].set_title('Rolling Variance Captured by Top 3 Components', fontsize=12, fontweight='bold')
axes[1].legend()

# Effective dimension
axes[2].plot(rolling_stats.index, rolling_stats['effective_dim'], color='#9b59b6', linewidth=1)
axes[2].fill_between(rolling_stats.index, rolling_stats['effective_dim'], alpha=0.3)
axes[2].set_ylabel('Effective Dimension')
axes[2].set_xlabel('Date')
axes[2].set_title('Rolling Effective Dimensionality (Participation Ratio)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nRolling SVD Interpretation:")
print("- High PC1 variance → Strong market factor, sectors moving together")
print("- Low effective dimension → Market dominated by few factors (stress)")
print("- High effective dimension → Diversified returns, sector-specific moves")

# ---------------------------------------------------------------------
# 11. Summary & Conclusions
# ---------------------------------------------------------------------

print("=" * 80)
print("MARKET REGIME ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nData Period: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"Total Trading Days: {len(returns)}")
print(f"Assets Analyzed: {len(returns.columns)} sector ETFs")

print(f"\nPrincipal Component Analysis:")
print(f"  - PC1 explains {detector.explained_variance_ratio[0]*100:.1f}% of variance (market factor)")
print(f"  - Top 3 PCs explain {detector.explained_variance_ratio[:3].sum()*100:.1f}% of variance")

print("\nRegime Statistics:")
for regime_name, row in regime_metrics.sort_index().iterrows():
    print(f"  - {regime_name}: {row['Frequency (%)']:.1f}% of days, "
          f"Sharpe: {row['Sharpe Ratio']:.2f}")

avg_duration = duration_df['Avg Duration (days)'].mean()
max_persistence = transition_matrix.values.diagonal().max()

print("\nKey Insights:")
print(f"  - Average regime duration: {avg_duration:.1f} days")
print(f"  - Strongest regime persistence (max diagonal of transition matrix): "
      f"{max_persistence:.1f}%")
print("=" * 80)

