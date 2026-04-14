"""
Market data loader using yfinance.
Downloads real price history and estimates historical volatility.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def _synthetic_spy_prices(start: str, end: str, seed: int = 0) -> pd.Series:
    """
    Generate a realistic SPY-like price series (GBM + vol clustering)
    for sandbox environments where yfinance is not available.
    """
    import pandas as pd
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    np.random.seed(seed)
    mu_daily = 0.10 / 252        # ~10% annual drift
    sigma_daily = 0.18 / np.sqrt(252)  # ~18% annual vol
    # Add mild vol clustering via GARCH-like variance
    vols = np.ones(n) * sigma_daily
    for i in range(1, n):
        shock = abs(np.random.normal()) * sigma_daily
        vols[i] = 0.94 * vols[i - 1] + 0.06 * shock
    rets = np.random.normal(mu_daily, vols)
    prices = 320.0 * np.exp(np.cumsum(rets))
    series = pd.Series(prices, index=dates, name="SPY_synthetic")
    return series


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted closing prices for a ticker."""
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError("empty")
        prices = data["Close"].squeeze()
        prices.name = ticker
        return prices
    except Exception:
        print(f"      [!] yfinance unavailable — using synthetic SPY-like data.")
        return _synthetic_spy_prices(start, end)


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def realized_vol(prices: pd.Series, window: int = 252) -> float:
    """
    Annualised historical (realised) volatility from the full series.
    Uses log returns, multiplied by sqrt(252) to annualise.
    """
    rets = log_returns(prices)
    return float(rets.std() * np.sqrt(252))


def rolling_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility (default 21-day lookback)."""
    rets = log_returns(prices)
    return rets.rolling(window).std() * np.sqrt(252)


def get_risk_free_rate() -> float:
    """
    Download the 13-week T-bill yield as a proxy for the risk-free rate.
    Falls back to 5 % if unavailable.
    """
    try:
        tbill = yf.download("^IRX", period="5d", auto_adjust=True, progress=False)
        if tbill.empty:
            return 0.05
        rate = float(tbill["Close"].iloc[-1]) / 100.0
        return rate
    except Exception:
        return 0.05


def slice_window(prices: pd.Series, start_idx: int, n_days: int) -> pd.Series:
    """Return a slice of prices of length n_days starting at start_idx."""
    return prices.iloc[start_idx : start_idx + n_days]
