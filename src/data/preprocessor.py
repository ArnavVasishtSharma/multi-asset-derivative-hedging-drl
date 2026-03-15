"""
data/preprocessor.py
---------------------
Preprocesses raw market data into model-ready tensors:
  - Aligns all data to a common trading calendar
  - Constructs the rolling 3×3 correlation matrix (equity, FX, rates)
  - Builds a 5×5 IV surface grid via interpolation from the VIX term structure
  - Computes option Greeks (Δ, Γ, Θ, Vega) under Black-Scholes
  - Normalises all features to zero-mean, unit-variance
"""

import os
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

log = logging.getLogger(__name__)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROC_DIR   = os.path.join(DATA_DIR, "processed")


# ── Black-Scholes Greeks ─────────────────────────────────────────────────────
def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1, d2 from Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> Dict[str, float]:
    """
    Black-Scholes option Greeks.

    Parameters
    ----------
    S, K : spot and strike price
    T    : time to expiry (years)
    r    : risk-free rate (decimal)
    sigma: implied volatility (decimal)
    option_type: 'call' or 'put'

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, price
    """
    d1, d2 = _d1d2(S, K, T, r, sigma)
    phi  = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T + 1e-9))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T + 1e-9))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = phi / (S * sigma * np.sqrt(T + 1e-9))
    vega  = S * phi * np.sqrt(T) / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "price": price}


# ── IV Surface Construction ──────────────────────────────────────────────────
class IVSurfaceBuilder:
    """
    Builds a 5×5 implied-volatility surface grid
    [5 moneyness levels × 5 maturity buckets] from VIX term-structure data.

    Moneyness grid: 0.85, 0.925, 1.00, 1.075, 1.15   (K/S)
    Maturity grid:  30d, 60d, 90d, 120d, 180d          (days)
    """
    MONEYNESS   = np.array([0.85, 0.925, 1.00, 1.075, 1.15])
    MATURITIES  = np.array([30,   60,    90,   120,   180])

    def __init__(self):
        pass

    def build_surface(self, vix_30d: float, vix_90d: float, vvix: float) -> np.ndarray:
        """
        Interpolate a 5×5 IV surface from 2 VIX data points.
        Uses smile shape from VVIX (vol-of-vol proxy).

        Parameters
        ----------
        vix_30d : 30-day ATM implied vol (decimal)
        vix_90d : 90-day ATM implied vol (decimal)
        vvix    : vol-of-vol index (decimal)

        Returns
        -------
        iv_surface : np.ndarray of shape (5, 5)
        """
        # Term structure: interpolate / extrapolate across maturities
        known_mats = np.array([30, 90])
        known_atm  = np.array([vix_30d, vix_90d])
        # Linear interpolation + flat extrapolation in log space
        log_atm = np.interp(self.MATURITIES, known_mats, np.log(known_atm + 1e-6))
        atm_term = np.exp(log_atm)

        # Smile: quadratic skew parameterised by VVIX
        skew_coef = vvix * 0.3   # empirical scaling
        surface = np.zeros((len(self.MATURITIES), len(self.MONEYNESS)))
        for i, (mat, atm) in enumerate(zip(self.MATURITIES, atm_term)):
            t = mat / 365.0
            smile = atm + skew_coef * (self.MONEYNESS - 1.0)**2  # parabolic smile
            # OTM puts are more expensive (skew / smirk)
            smile += skew_coef * 0.5 * (1.0 - self.MONEYNESS)
            surface[i, :] = np.clip(smile, 0.01, 3.0)
        return surface   # shape (5, 5): maturities × moneyness


# ── Correlation Matrix ───────────────────────────────────────────────────────
def compute_rolling_correlation(
    equities: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Compute rolling 3×3 correlation matrix across:
      cols: [spx_ret, eurusd_ret, rate_10y_chg]
    Returns DataFrame with 9 correlation columns: corr_00 … corr_22.
    """
    # Compute log returns / changes
    spx_ret     = np.log(equities["spx"]).diff()
    eurusd_ret  = np.log(fx["eurusd"]).diff()
    rate10_chg  = rates["rate_10y"].diff()

    combo = pd.DataFrame({
        "spx_ret":    spx_ret,
        "eurusd_ret": eurusd_ret,
        "rate10_chg": rate10_chg,
    }).dropna()

    # Rolling correlation in flattened form
    out_records = []
    for i in range(window, len(combo)):
        window_data = combo.iloc[i-window:i]
        corr = window_data.corr().values.flatten()   # 9 values
        out_records.append({"date": combo.index[i], **{f"corr_{j}": v for j,v in enumerate(corr)}})
    out = pd.DataFrame(out_records).set_index("date")
    return out


# ── Main preprocess pipeline ─────────────────────────────────────────────────
def preprocess_all(start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Loads raw parquet files, aligns to common calendar,
    computes correlation matrices, IV surfaces, and Greeks.

    Returns
    -------
    master_df: DataFrame with all features aligned to trading days
    """
    raw_dir = os.path.join(DATA_DIR, "raw")

    equities = pd.read_parquet(os.path.join(raw_dir, "equities.parquet"))
    fx       = pd.read_parquet(os.path.join(raw_dir, "fx.parquet"))
    rates    = pd.read_parquet(os.path.join(raw_dir, "rates.parquet"))
    vix      = pd.read_parquet(os.path.join(raw_dir, "vix_term.parquet"))
    defi     = pd.read_parquet(os.path.join(raw_dir, "defi_uniswap.parquet"))

    # Align to common index (inner join)
    aligned = equities.join(fx, how="inner").join(rates, how="inner").join(vix, how="inner")
    aligned = aligned.loc[start:end]

    # Rolling correlation
    log.info("Computing rolling 60-day correlation matrix …")
    corr_df = compute_rolling_correlation(
        equities=aligned[["spx", "spy"]],
        fx=aligned[["eurusd"]].rename(columns={"eurusd": "eurusd"}),
        rates=aligned[["rate_1y", "rate_5y", "rate_10y"]],
        window=60,
    )

    # IV surface per row
    log.info("Building IV surfaces …")
    iv_builder = IVSurfaceBuilder()
    iv_grids = []
    for _, row in aligned.iterrows():
        surf = iv_builder.build_surface(
            vix_30d=row.get("vix_30d", 0.2),
            vix_90d=row.get("vix_90d", 0.22),
            vvix=row.get("vvix", 0.7),
        )
        iv_grids.append(surf.flatten())   # 25-dim vector

    iv_df = pd.DataFrame(iv_grids,
                          index=aligned.index,
                          columns=[f"iv_{i}" for i in range(25)])

    # Compute BS Greeks for ATM 30-day SPY put (core hedge instrument)
    log.info("Computing BS Greeks for ATM SPY put …")
    greeks_records = []
    for i, (date, row) in enumerate(aligned.iterrows()):
        S = row["spy"]
        K = S   # ATM
        T = 30 / 365.0
        r = row.get("rate_1y", 0.05)
        sigma = row.get("vix_30d", 0.2)
        g = bs_greeks(S, K, T, r, sigma, "put")
        greeks_records.append({"date": date, **g})
    greeks_df = pd.DataFrame(greeks_records).set_index("date")

    # Merge everything
    master = aligned.join(corr_df, how="left").join(iv_df, how="left").join(greeks_df, how="left")

    # DeFi – resample to business days and join
    defi_resampled = defi.resample("B").last().ffill()
    master = master.join(defi_resampled, how="left").ffill()

    # Normalise numeric columns (z-score, fit on training portion)
    feature_cols = [c for c in master.columns if master[c].dtype != object]
    train_end = int(len(master) * 0.8)
    means = master[feature_cols].iloc[:train_end].mean()
    stds  = master[feature_cols].iloc[:train_end].std().replace(0, 1)
    master_norm = (master[feature_cols] - means) / stds

    os.makedirs(PROC_DIR, exist_ok=True)
    master_norm.to_parquet(os.path.join(PROC_DIR, "master_features.parquet"))
    master.to_parquet(os.path.join(PROC_DIR, "master_raw.parquet"))
    log.info(f"Saved processed data → {PROC_DIR}")
    return master_norm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_all()
