# Market Regime Detection with SVD

Detects market regimes by running Singular Value Decomposition (SVD) on daily log returns of the 11 S&P 500 sector SPDR ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC) over the last 6 years.

## What it does

1. Downloads sector ETF prices from Yahoo Finance (`yfinance`)
2. Computes log returns, cumulative returns, rolling volatility, and a correlation heatmap
3. Runs SVD (`MarketRegimeDetector`) and reports variance explained and sector loadings per component. SVD signs are arbitrary, so they're fixed after fitting:
   - **PC1 (market factor):** positive when the market rises
   - **PC2 (growth vs value):** positive when growth sectors (Technology, Consumer Discretionary, Communication Services) beat value sectors
4. Classifies each day into a regime using the trailing 21-day returns of PC1 (trend) and PC2 (rotation):

   | Regime | Trend | Rotation |
   |---|---|---|
   | Growth Rally | Market rising | Growth leading |
   | Value Rally | Market rising | Value leading |
   | Value Selloff | Market falling | Value falling hardest |
   | Growth Selloff | Market falling | Growth falling hardest |

5. Reports per-regime characteristics, sector performance, transition probabilities, and durations
6. **Forward-looking test:** measures returns and volatility over the next day and next 21 days after each regime day, with a t-stat for whether a regime's next-day return differs from all days
7. Runs a rolling 63-day SVD to track market-factor dominance and effective dimensionality

## Usage

```bash
pip install -r requirements.txt
python marketregime.py
```

Charts open in matplotlib windows and the summary statistics print to the console.

## Limitations

The SVD is fit on the full sample, so regime labels for early dates use information from later dates (look-ahead bias). Refit on an expanding or rolling window before using the regimes in a backtest.
