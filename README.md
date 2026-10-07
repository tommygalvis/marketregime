# Market Regime Detection with SVD

Detects market regimes by running Singular Value Decomposition (SVD) on daily log returns of the 11 S&P 500 sector SPDR ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC) over the last 3 years.

## What it does

1. Downloads sector ETF prices from Yahoo Finance (`yfinance`)
2. Computes log returns, cumulative returns, rolling volatility, and a correlation heatmap
3. Runs SVD (`MarketRegimeDetector`) and reports variance explained and sector loadings per component
4. Projects each trading day onto PC1 (market factor) and PC2 (rotation factor)
5. Classifies days into four regimes by PC1/PC2 quadrant:
   - Risk-On Rally
   - Defensive Rally
   - Risk-Off Stress
   - Broad Decline
6. Reports per-regime metrics (return, volatility, Sharpe, correlation), sector performance, transition probabilities, and durations
7. Runs a rolling 63-day SVD to track market-factor dominance and effective dimensionality

## Usage

```bash
pip install -r requirements.txt
python marketregime.py
```

Charts open in matplotlib windows and the summary statistics print to the console.
