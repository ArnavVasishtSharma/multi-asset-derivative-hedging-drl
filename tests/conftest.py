"""
tests/conftest.py
------------------
Shared pytest fixtures for the Multi-Asset DRL Hedging System.

All fixtures generate synthetic data so tests run without
external API calls or pre-downloaded parquet files.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ── Synthetic parquet generators ──────────────────────────────────────────────

def _make_master_raw(n: int = 200) -> pd.DataFrame:
    """
    Generate a synthetic master_raw.parquet-compatible DataFrame.

    Column layout (49 features used by MultiAssetHedgingEnv):
      [0:7]   → spx, spy, eurusd, usdjpy, rate_1y, rate_5y, rate_10y
      [7:32]  → iv_0 .. iv_24  (flattened 5×5 IV surface)
      [32:41] → corr_0 .. corr_8  (flattened 3×3 correlation matrix)
      [41:45] → delta, gamma, theta, vega  (BS Greeks)
      [45]    → time_to_expiry
      [46:50] → pool_price, funding_rate, basis_spread, liquidity  (DeFi overlap)
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)

    data = {}
    # 7 market prices
    data["spx"]      = 4000 + np.cumsum(rng.normal(0, 10, n))
    data["spy"]      = data["spx"] / 10
    data["eurusd"]   = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    data["usdjpy"]   = 110 + np.cumsum(rng.normal(0, 0.3, n))
    data["rate_1y"]  = 0.04 + rng.normal(0, 0.001, n)
    data["rate_5y"]  = 0.035 + rng.normal(0, 0.001, n)
    data["rate_10y"] = 0.03 + rng.normal(0, 0.001, n)

    # 25 IV grid values
    for i in range(25):
        data[f"iv_{i}"] = 0.2 + rng.normal(0, 0.02, n)

    # 9 correlation values (flattened 3×3 identity-ish)
    corr_base = np.eye(3).flatten()
    for i in range(9):
        data[f"corr_{i}"] = corr_base[i] + rng.normal(0, 0.05, n)

    # 4 Greeks
    data["delta"] = -0.5 + rng.normal(0, 0.1, n)
    data["gamma"] = 0.03 + rng.normal(0, 0.005, n)
    data["theta"] = -0.02 + rng.normal(0, 0.005, n)
    data["vega"]  = 0.15 + rng.normal(0, 0.02, n)

    # 1 time_to_expiry
    data["tte"] = np.linspace(1.0, 0.01, n)

    # 4 DeFi columns
    data["pool_price"]    = 1800 + np.cumsum(rng.normal(0, 20, n))
    data["funding_rate"]  = rng.normal(0.0001, 0.0005, n)
    data["basis_spread"]  = rng.normal(0, 0.002, n)
    data["liquidity"]     = np.abs(rng.normal(500e6, 50e6, n))

    # extra: price column for bs_greeks "price"
    data["price"] = rng.normal(5, 1, n)

    df = pd.DataFrame(data, index=dates)
    return df


def _make_defi_processed(n: int = 200) -> pd.DataFrame:
    """Generate synthetic defi_processed.parquet data (4 columns)."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "pool_price":   1800 + np.cumsum(rng.normal(0, 20, n)),
        "funding_rate": rng.normal(0.0001, 0.0005, n),
        "basis_spread": rng.normal(0, 0.002, n),
        "liquidity":    np.abs(rng.normal(500e6, 50e6, n)),
    }, index=dates)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create a temp directory with synthetic parquet files."""
    with tempfile.TemporaryDirectory() as d:
        master = _make_master_raw(200)
        defi   = _make_defi_processed(200)
        master.to_parquet(os.path.join(d, "master_raw.parquet"))
        defi.to_parquet(os.path.join(d, "defi_processed.parquet"))
        yield d


@pytest.fixture(scope="session")
def master_parquet_path(tmp_data_dir):
    return os.path.join(tmp_data_dir, "master_raw.parquet")


@pytest.fixture(scope="session")
def defi_parquet_path(tmp_data_dir):
    return os.path.join(tmp_data_dir, "defi_processed.parquet")


@pytest.fixture
def tmp_checkpoint_dir():
    """Temporary directory for saving/loading model checkpoints."""
    with tempfile.TemporaryDirectory() as d:
        yield d
