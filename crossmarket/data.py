"""Loading market data and computing returns / realized volatility."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MARKETS: dict[str, str] = {
    "^GSPC": "S&P 500 (USA)",
    "^GDAXI": "DAX (Germany)",
    "^FCHI": "CAC 40 (France)",
    "^FTSE": "FTSE 100 (UK)",
    "^NSEI": "Nifty 50 (India)",
    "^N225": "Nikkei 225 (Japan)",
    "^KS11": "KOSPI (South Korea)",
    "^HSI": "Hang Seng (Hong Kong)",
}
TICKERS = list(MARKETS)
TRADING_DAYS = 252


def load_ohlcv(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read a yfinance CSV (two header rows: Price / Ticker)."""
    df = pd.read_csv(data_dir / f"{ticker}_stock_data.csv", header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.astype(float).sort_index()


def trading_mask(df: pd.DataFrame) -> pd.Series:
    """True on real trading days.

    The stored CSVs were forward-filled onto a business-day calendar, so
    holidays appear as exact copies of the previous row. Treating those as
    trading days injects zero returns and biases volatility downward.
    """
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    repeated = (df[cols] == df[cols].shift(1)).all(axis=1)
    return ~repeated


def load_closes(tickers: list[str] | None = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Close prices on a common calendar; non-trading days are NaN."""
    closes = {}
    for t in tickers or TICKERS:
        df = load_ohlcv(t, data_dir)
        closes[t] = df["Close"].where(trading_mask(df))
    return pd.DataFrame(closes)


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; returns spanning a holiday are attributed to the next trading day."""
    return np.log(closes.ffill()).diff().where(closes.notna()).iloc[1:]


def realized_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized rolling realized volatility, sqrt(252 * mean r^2) over trading days."""
    min_periods = max(1, int(window * 0.7))
    rv = (returns**2).rolling(window, min_periods=min_periods).mean()
    return np.sqrt(TRADING_DAYS * rv).dropna(how="any")


def download(tickers: list[str] | None = None, start: str = "2007-11-06",
             end: str | None = None, data_dir: Path = DATA_DIR) -> None:
    """Refresh the CSVs from Yahoo Finance (requires `yfinance` and network)."""
    import yfinance as yf

    for t in tickers or TICKERS:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {t}")
        df.to_csv(data_dir / f"{t}_stock_data.csv")
