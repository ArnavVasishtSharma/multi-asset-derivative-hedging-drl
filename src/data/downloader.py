"""
data/downloader.py
------------------
Downloads market data for the Multi-Asset DRL Hedging System:
  - Equity:  SPX / SPY daily OHLCV + options chain (yfinance)
  - FX:      EURUSD, USDJPY daily (FRED / Dukascopy via yfinance)
  - Rates:   US Treasury yields 1Y, 5Y, 10Y (FRED)
  - DeFi:    Uniswap v3 pool hourly data (placeholder: The Graph API)
  - IV:      VIX term structure proxy from option chain ATM vols
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
EQUITY_TICKERS  = ["^GSPC", "SPY"]          # SPX index, SPY ETF
FX_TICKERS      = ["EURUSD=X", "USDJPY=X"]  # yfinance FX symbols
FRED_SERIES     = {
    "rate_1y": "DGS1",
    "rate_5y": "DGS5",
    "rate_10y": "DGS10",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


# ── helpers ──────────────────────────────────────────────────────────────────
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(_ensure_dir(DATA_DIR), f"{name}.parquet")
    df.to_parquet(path)
    log.info(f"  Saved {name}: {df.shape}  →  {path}")
    return path


# ── downloaders ──────────────────────────────────────────────────────────────
def download_equities(start: str, end: str) -> pd.DataFrame:
    """Download SPX & SPY OHLCV from yfinance."""
    log.info("Downloading equity data (SPX, SPY) …")
    raw = yf.download(EQUITY_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].rename(columns={"^GSPC": "spx", "SPY": "spy"})
    volume = raw["Volume"].rename(columns={"^GSPC": "spx_vol", "SPY": "spy_vol"})
    df = pd.concat([close, volume], axis=1).dropna()
    _save(df, "equities")
    return df


def download_fx(start: str, end: str) -> pd.DataFrame:
    """Download FX rates from yfinance."""
    log.info("Downloading FX data (EURUSD, USDJPY) …")
    raw = yf.download(FX_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    df = raw["Close"].rename(columns={"EURUSD=X": "eurusd", "USDJPY=X": "usdjpy"}).dropna()
    _save(df, "fx")
    return df


def download_rates(start: str, end: str) -> pd.DataFrame:
    """Download US Treasury yields from FRED."""
    log.info("Downloading US Treasury yields (1Y, 5Y, 10Y) from FRED …")
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", start, end).squeeze()
            frames[name] = s
        except Exception as e:
            log.warning(f"  FRED series {series_id} failed: {e}. Using zero fill.")
            frames[name] = pd.Series(dtype=float, name=name)
    df = pd.DataFrame(frames).ffill().dropna()
    df = df / 100.0   # convert % → decimal
    _save(df, "rates")
    return df


def download_vix_term_structure(start: str, end: str) -> pd.DataFrame:
    """
    Proxy IV surface from VIX and VIXM (9-day, 30-day, 90-day).
    Full CBOE surface requires paid API – this gives a 3-point term structure
    that is extended to a 5×3 grid via interpolation in preprocessor.py.
    """
    log.info("Downloading VIX term structure proxy …")
    vix_tickers = ["^VIX", "^VIX3M", "^VVIX"]
    raw = yf.download(vix_tickers, start=start, end=end, auto_adjust=True, progress=False)
    df = raw["Close"].rename(columns={
        "^VIX": "vix_30d", "^VIX3M": "vix_90d", "^VVIX": "vvix"
    }).dropna()
    df["vix_30d"] = df["vix_30d"] / 100.0
    df["vix_90d"] = df["vix_90d"] / 100.0
    df["vvix"]    = df["vvix"]    / 100.0
    _save(df, "vix_term")
    return df


def download_spy_options_atm(date: Optional[str] = None) -> pd.DataFrame:
    """
    Download SPY option chain for ATM implied vols.
    For historical IV surface, iterate over multiple dates.
    """
    log.info("Downloading SPY option chain (current) …")
    spy = yf.Ticker("SPY")
    expirations = spy.options
    records = []
    for exp in expirations[:6]:   # nearest 6 expirations
        try:
            chain = spy.option_chain(exp)
            calls = chain.calls[["strike", "impliedVolatility", "lastTradeDate"]].copy()
            calls["expiry"] = exp
            calls["type"] = "call"
            puts  = chain.puts[["strike", "impliedVolatility", "lastTradeDate"]].copy()
            puts["expiry"] = exp
            puts["type"] = "put"
            records.append(pd.concat([calls, puts]))
        except Exception as e:
            log.warning(f"  Option chain for {exp} failed: {e}")
    if records:
        df = pd.concat(records, ignore_index=True)
        _save(df, "spy_options_chain")
        return df
    return pd.DataFrame()


def download_defi_uniswap_placeholder() -> pd.DataFrame:
    """
    Placeholder for Uniswap v3 pool data.
    In production: query The Graph API (https://thegraph.com/explorer/subgraphs/uniswap)
    Returns simulated ETH/USDC pool data for development.
    """
    log.info("Generating Uniswap v3 placeholder data (ETH/USDC pool) …")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    pool_price  = 1800 * np.exp(np.cumsum(np.random.normal(0, 0.03, n)))
    funding_rate = np.random.normal(0.0001, 0.0005, n)
    basis_spread = np.random.normal(0, 0.002, n)
    liquidity    = np.abs(np.random.normal(500e6, 50e6, n))
    df = pd.DataFrame({
        "pool_price":    pool_price,
        "funding_rate":  funding_rate,
        "basis_spread":  basis_spread,
        "liquidity":     liquidity,
    }, index=dates)
    _save(df, "defi_uniswap")
    return df


# ── main ─────────────────────────────────────────────────────────────────────
def download_all(start: str, end: str) -> dict:
    """
    Download all required datasets.

    Returns
    -------
    dict with keys: equities, fx, rates, vix_term, defi
    """
    log.info(f"=== Starting data download: {start} → {end} ===")
    data = {
        "equities": download_equities(start, end),
        "fx":       download_fx(start, end),
        "rates":    download_rates(start, end),
        "vix_term": download_vix_term_structure(start, end),
        "defi":     download_defi_uniswap_placeholder(),
    }
    log.info("=== Download complete ===")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download market data for DRL hedging")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    args = parser.parse_args()
    download_all(args.start, args.end)
