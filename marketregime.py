"""
Market Regime Detection Using Singular Value Decomposition
A Quantitative Approach to Identifying Market States

This script implements a production-ready framework for detecting market
regimes using Singular Value Decomposition (SVD) on sector ETF returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

# Date range: Last 6 years of data
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=6 * 365)

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
cumulative = np.exp(returns.cumsum())
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

    def fit(self, returns: np.ndarray, growth_idx: list = None) -> 'MarketRegimeDetector':
        """Perform SVD on the returns matrix.

        Args:
            returns: (days x assets) matrix of returns
            growth_idx: column indices of growth sectors, used to orient PC2
        """
        # Center the data
        self.mean = returns.mean(axis=0)
        centered = returns - self.mean

        # Full SVD decomposition
        self.U, self.S, self.Vt = np.linalg.svd(centered, full_matrices=False)

        # SVD signs are arbitrary and can flip between runs, so fix them:
        # PC1 points in the "market up" direction, and PC2 is positive
        # when growth sectors outperform
        signs = np.sign(self.Vt.sum(axis=1))
        if growth_idx is not None:
            signs[1] = np.sign(self.Vt[1, growth_idx].sum())
        signs[signs == 0] = 1
        self.Vt *= signs[:, None]
        self.U *= signs

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


# Growth sectors define the positive direction of PC2 (growth vs value)
GROWTH_SECTORS = ['Technology', 'Consumer Discretionary', 'Communication Services']

R = returns.values
growth_idx = [returns.columns.get_loc(s) for s in GROWTH_SECTORS]
detector = MarketRegimeDetector(n_components=3)
detector.fit(R, growth_idx=growth_idx)

print("\nSVD Decomposition Complete")
print(f"Returns matrix shape: {R.shape}")
print(f"U shape: {detector.U.shape}")
print(f"S shape: {detector.S.shape}")
print(f"Vt shape: {detector.Vt.shape}")

# PC2's meaning depends on the data window, so show what it is
pc2 = pd.Series(detector.Vt[1], index=returns.columns).sort_values()
print(f"\nPC2 positive side: {', '.join(pc2.index[-3:][::-1])}")
print(f"PC2 negative side: {', '.join(pc2.index[:3])}")

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
axes[0].set_ylabel('PC2 (Growth vs Value)')
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


# Regimes are classified from the trailing month of factor returns rather
# than a single day's return, so they describe periods, not up/down days
REGIME_WINDOW = 21     # trading days (~1 month)
MIN_TRAIN_DAYS = 252   # first year is only used to fit the first SVD


def compute_regime_features(returns: pd.DataFrame, detector: MarketRegimeDetector,
                            window: int = REGIME_WINDOW) -> pd.DataFrame:
    """In-sample regime features, using loadings from the full-sample SVD.

    Trend    = PC1 return over the window (positive = market rising)
    Rotation = PC2 return over the window (positive = growth beating value)

    The loadings are fit on all dates, so early labels use future data
    (look-ahead bias). Kept only for comparison with the walk-forward version.
    """
    factor_returns = returns.values @ detector.Vt[:2].T
    factors = pd.DataFrame(factor_returns, index=returns.index, columns=['Trend', 'Rotation'])
    return factors.rolling(window).sum().dropna()


def compute_walk_forward_features(returns: pd.DataFrame, growth_idx: list,
                                  window: int = REGIME_WINDOW,
                                  min_train: int = MIN_TRAIN_DAYS) -> pd.DataFrame:
    """Point-in-time regime features with no look-ahead bias.

    On each day, refit the SVD on returns up to and including that day
    (expanding window), then project the trailing window's returns onto
    those loadings. Every label could have been computed at that day's close.

    Uses raw (uncentered) returns so the sign reflects the actual direction.
    """
    R = returns.values
    trailing = returns.rolling(window).sum().values
    start = max(min_train, window) - 1

    rows = []
    for t in range(start, len(R)):
        model = MarketRegimeDetector(n_components=2).fit(R[:t + 1], growth_idx=growth_idx)
        rows.append(trailing[t] @ model.Vt[:2].T)

    return pd.DataFrame(rows, index=returns.index[start:], columns=['Trend', 'Rotation'])


def classify_regimes(features: pd.DataFrame) -> pd.Series:
    """Classify each day by the signs of its trailing trend and rotation."""
    up = features['Trend'] >= 0
    growth_leads = features['Rotation'] >= 0

    conditions = [
        up & growth_leads,
        up & ~growth_leads,
        ~up & growth_leads,
        ~up & ~growth_leads,
    ]

    # A selloff is named after the side falling hardest
    regime_names = ['Growth Rally', 'Value Rally', 'Value Selloff', 'Growth Selloff']

    regimes_arr = np.select(conditions, regime_names, default='')
    return pd.Series(regimes_arr, index=features.index, name='regime')


print(f"\nFitting walk-forward SVD ({len(returns) - MIN_TRAIN_DAYS + 1} daily refits)...")
regime_features = compute_walk_forward_features(returns, growth_idx)
regimes = classify_regimes(regime_features)
regime_returns = returns.loc[regimes.index]

# In-sample labels on the same dates, to measure the look-ahead effect
in_sample_regimes = classify_regimes(compute_regime_features(returns, detector)).loc[regimes.index]

print(f"\nRegimes from trailing {REGIME_WINDOW}-day trend (PC1) and rotation (PC2), "
      f"walk-forward from {regimes.index[0].date()}")
print(f"Agreement with full-sample (look-ahead) labels: {(regimes == in_sample_regimes).mean():.1%}")
print("Regime Distribution:")
print(regimes.value_counts())
print(f"\nTotal days: {len(regimes)}")

# Regime visualization
regime_colors = {
    'Growth Rally': '#2ecc71',
    'Value Rally': '#3498db',
    'Value Selloff': '#9b59b6',
    'Growth Selloff': '#e74c3c',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot colored by regime
for regime_name, color in regime_colors.items():
    mask = regimes == regime_name
    axes[0].scatter(
        regime_features.loc[mask, 'Trend'],
        regime_features.loc[mask, 'Rotation'],
        c=color,
        label=regime_name,
        alpha=0.6,
        s=12
    )

axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel(f'Trend: PC1 return, trailing {REGIME_WINDOW} days')
axes[0].set_ylabel(f'Rotation: PC2 return, trailing {REGIME_WINDOW} days')
axes[0].set_title('Market Regimes in Factor Space', fontsize=14, fontweight='bold')
axes[0].legend()

# Regime timeline: equal-weight sector portfolio with regimes shaded behind it
portfolio_curve = np.exp(regime_returns.mean(axis=1).cumsum())
axes[1].plot(portfolio_curve.index, portfolio_curve.values, color='black', linewidth=1)

starts = regimes.index[regimes != regimes.shift()]
ends = list(starts[1:]) + [regimes.index[-1]]
for start, end in zip(starts, ends):
    axes[1].axvspan(start, end, color=regime_colors[regimes[start]], alpha=0.3, linewidth=0)

axes[1].legend(handles=[Patch(color=c, alpha=0.5, label=n) for n, c in regime_colors.items()],
               loc='upper left', fontsize=9)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Equal-Weight Sector Portfolio')
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


regime_metrics = compute_regime_metrics(regime_returns, regimes)
print("\nRegime Characteristics (same days; descriptive, not predictive):")
print("=" * 80)
print(regime_metrics.round(2).to_string())

# Sector performance by regime
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, regime in enumerate(regime_colors.keys()):
    mask = regimes == regime

    ann_returns = regime_returns[mask].mean() * 252 * 100
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
# 10. Forward-Looking Test: Does the Regime Predict Anything?
# ---------------------------------------------------------------------

FORWARD_HORIZON = 21  # trading days


def evaluate_forward_returns(returns: pd.DataFrame, regimes: pd.Series,
                             horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    """Measure what happened AFTER each regime day, using the equal-weight
    sector portfolio. This is the real test of a regime signal.

    Next-day returns don't overlap, so their t-stat (regime vs all days) is a
    fair significance check. Forward windows overlap, so treat the
    multi-day columns as descriptive.
    """
    portfolio = returns.mean(axis=1)
    data = pd.DataFrame({
        'regime': regimes,
        'next_day': portfolio.shift(-1),
        'fwd_ret': portfolio.rolling(horizon).sum().shift(-horizon),
        'fwd_vol': portfolio.rolling(horizon).std().shift(-horizon) * np.sqrt(252),
    }).dropna()

    rows = []
    for name, group in [*data.groupby('regime'), ('All days', data)]:
        next_day = group['next_day']
        t_stat = ((next_day.mean() - data['next_day'].mean())
                  / (next_day.std() / np.sqrt(len(next_day))))
        rows.append({
            'Regime': name,
            'Days': len(group),
            'Next-Day Return (bps)': next_day.mean() * 10000,
            't-stat vs All': t_stat if name != 'All days' else np.nan,
            f'Next {horizon}d Return (%)': group['fwd_ret'].mean() * 100,
            f'Next {horizon}d Up (%)': (group['fwd_ret'] > 0).mean() * 100,
            f'Next {horizon}d Vol (ann. %)': group['fwd_vol'].mean() * 100,
        })

    order = [r for r in regime_colors if r in data['regime'].values] + ['All days']
    return pd.DataFrame(rows).set_index('Regime').loc[order]


forward_stats = evaluate_forward_returns(returns, regimes)
print("\nForward Returns by Regime, walk-forward (what happened next):")
print("=" * 80)
print(forward_stats.round(2).to_string())

max_t = forward_stats['t-stat vs All'].abs().max()
print(f"\nLargest |t-stat| on next-day returns: {max_t:.2f} "
      f"({'notable' if max_t > 2 else 'not significant'}; |t| > 2 is a rough bar for a real effect)")

# Same test with the full-sample (look-ahead) labels on the same dates
in_sample_stats = evaluate_forward_returns(returns, in_sample_regimes)
ret_col = f'Next {FORWARD_HORIZON}d Return (%)'
vol_col = f'Next {FORWARD_HORIZON}d Vol (ann. %)'
comparison = pd.DataFrame({
    'Days (WF)': forward_stats['Days'],
    'Days (IS)': in_sample_stats['Days'],
    f'Next {FORWARD_HORIZON}d % (WF)': forward_stats[ret_col],
    f'Next {FORWARD_HORIZON}d % (IS)': in_sample_stats[ret_col],
    't-stat (WF)': forward_stats['t-stat vs All'],
    't-stat (IS)': in_sample_stats['t-stat vs All'],
})
print("\nLook-ahead check: walk-forward (WF) vs full-sample in-sample (IS) labels, same dates")
print("=" * 80)
print(comparison.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
regime_names = list(forward_stats.drop('All days').index)
x = np.arange(len(regime_names))
width = 0.38
bar_colors = [regime_colors[r] for r in regime_names]

for ax, col, title in [
    (axes[0], ret_col, f'Average Return Over the Next {FORWARD_HORIZON} Days'),
    (axes[1], vol_col, f'Volatility Over the Next {FORWARD_HORIZON} Days'),
]:
    ax.bar(x - width / 2, forward_stats.loc[regime_names, col], width,
           color=bar_colors, edgecolor='white', label='Walk-forward')
    ax.bar(x + width / 2, in_sample_stats.reindex(regime_names)[col], width,
           color=bar_colors, alpha=0.35, edgecolor='white', hatch='//', label='In-sample (look-ahead)')
    ax.axhline(forward_stats.loc['All days', col], color='black', linestyle='--',
               linewidth=1, label='All days')
    ax.set_xticks(x, regime_names, fontsize=9)
    ax.set_ylabel(col)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 11. Rolling SVD Analysis
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
# 12. Summary & Conclusions
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

print(f"\nRegime Statistics (next {FORWARD_HORIZON} days after each regime day):")
for regime_name, row in forward_stats.drop('All days').iterrows():
    print(f"  - {regime_name}: {regime_metrics.loc[regime_name, 'Frequency (%)']:.1f}% of days, "
          f"next {FORWARD_HORIZON}d return {row[f'Next {FORWARD_HORIZON}d Return (%)']:+.2f}%, "
          f"next-day t-stat {row['t-stat vs All']:+.2f}")
all_days = forward_stats.loc['All days']
print(f"  - All days: next {FORWARD_HORIZON}d return {all_days[f'Next {FORWARD_HORIZON}d Return (%)']:+.2f}%")

avg_duration = duration_df['Avg Duration (days)'].mean()
max_persistence = transition_matrix.values.diagonal().max()

print("\nKey Insights:")
print(f"  - Average regime duration: {avg_duration:.1f} days")
print(f"  - Strongest regime persistence (max diagonal of transition matrix): "
      f"{max_persistence:.1f}%")
print("=" * 80)

