"""
utils/metrics.py
-----------------
Portfolio-level evaluation metrics used throughout the hedging system.

Provides:
  - compute_sharpe  : annualised Sharpe ratio
  - compute_cvar    : Conditional Value at Risk (tail expected loss)
  - compute_max_dd  : maximum drawdown
  - compute_he_variance : hedging error variance
"""

from typing import Sequence
import numpy as np


def compute_sharpe(
    returns: Sequence[float],
    risk_free: float = 0.04,
    annualise: bool = True,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns   : sequence of daily/per-step returns
    risk_free : annualised risk-free rate (default 4%)
    annualise : multiply by √252 to get annual Sharpe

    Returns
    -------
    sharpe : float (nan if std=0)
    """
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    rf_daily  = risk_free / 252.0
    excess    = arr - rf_daily
    mu, sigma = excess.mean(), excess.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    sharpe = mu / sigma
    if annualise:
        sharpe *= np.sqrt(252.0)
    return float(sharpe)


def compute_cvar(
    pnl: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """
    Historical CVaR (Expected Shortfall) at given confidence level.

    CVaR_α = -E[PnL | PnL < VaR_α]

    Parameters
    ----------
    pnl        : sequence of P&L observations
    confidence : e.g. 0.95 → 95% CVaR

    Returns
    -------
    cvar : positive float representing expected loss in tail
    """
    arr = np.array(pnl, dtype=np.float64)
    if len(arr) < 5:
        return 0.0
    quantile = 1.0 - confidence
    cutoff   = np.quantile(arr, quantile)
    tail     = arr[arr <= cutoff]
    if len(tail) == 0:
        return 0.0
    return float(-np.mean(tail))


def compute_max_dd(cumulative_pnl: Sequence[float]) -> float:
    """
    Maximum drawdown from peak.

    Parameters
    ----------
    cumulative_pnl : sequence of cumulative P&L values

    Returns
    -------
    max_dd : positive float (worst peak-to-trough drop)
    """
    arr = np.array(cumulative_pnl, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    peak   = np.maximum.accumulate(arr)
    dd     = peak - arr
    return float(dd.max())


def compute_he_variance(hedging_errors: Sequence[float]) -> float:
    """
    Variance of hedging errors.

    Parameters
    ----------
    hedging_errors : sequence of per-step hedging errors (option_pnl + hedge_pnl)

    Returns
    -------
    variance : float
    """
    arr = np.array(hedging_errors, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.var(arr, ddof=1))


def episode_summary(
    pnl_history:    Sequence[float],
    hedging_errors: Sequence[float],
    risk_free:      float = 0.04,
    confidence:     float = 0.95,
) -> dict:
    """
    Compute all evaluation metrics for one episode.

    Returns
    -------
    dict with keys: sharpe, cvar, max_drawdown, he_variance,
                    total_return, n_steps
    """
    arr  = np.array(pnl_history, dtype=np.float64)
    cumr = np.cumsum(arr)
    return {
        "sharpe":       compute_sharpe(arr),
        "cvar":         compute_cvar(arr, confidence),
        "max_drawdown": compute_max_dd(cumr),
        "he_variance":  compute_he_variance(hedging_errors),
        "total_return": float(cumr[-1]) if len(cumr) > 0 else 0.0,
        "n_steps":      len(arr),
    }
