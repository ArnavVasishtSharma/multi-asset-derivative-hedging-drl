"""
tests/test_data.py
-------------------
Tests for data preprocessing utilities:
  - Black-Scholes Greeks (put-call parity, boundary conditions)
  - IV Surface Builder (shape, positivity)
  - Rolling Correlation (shape, valid values)
"""

import numpy as np
import pytest

from data.preprocessor import bs_greeks, IVSurfaceBuilder, compute_rolling_correlation
import pandas as pd


class TestBSGreeks:
    """Black-Scholes Greeks correctness tests."""

    def test_put_call_parity_delta(self):
        """Δ_call - Δ_put = 1 (put-call parity for delta)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2
        call = bs_greeks(S, K, T, r, sigma, "call")
        put  = bs_greeks(S, K, T, r, sigma, "put")
        np.testing.assert_allclose(
            call["delta"] - put["delta"], 1.0, atol=1e-10,
            err_msg="Put-call parity for delta violated"
        )

    def test_atm_put_delta_near_minus05(self):
        """ATM put delta should be close to -0.5."""
        g = bs_greeks(100, 100, 0.25, 0.05, 0.2, "put")
        assert -0.6 < g["delta"] < -0.4, f"ATM put delta={g['delta']}"

    def test_gamma_positive(self):
        """Gamma is always positive for calls and puts."""
        g_call = bs_greeks(100, 100, 0.5, 0.05, 0.3, "call")
        g_put  = bs_greeks(100, 100, 0.5, 0.05, 0.3, "put")
        assert g_call["gamma"] > 0
        assert g_put["gamma"] > 0

    def test_vega_positive(self):
        """Vega is positive (higher vol → higher option price)."""
        g = bs_greeks(100, 100, 0.5, 0.05, 0.3, "call")
        assert g["vega"] > 0

    def test_zero_time_graceful(self):
        """At expiry (T=0), should not crash and return finite values."""
        g = bs_greeks(100, 100, 0.0, 0.05, 0.2, "put")
        assert np.isfinite(g["delta"])
        assert np.isfinite(g["gamma"])

    def test_call_price_positive(self):
        """ITM call should have positive price."""
        g = bs_greeks(110, 100, 0.5, 0.05, 0.3, "call")
        assert g["price"] > 0

    def test_put_price_positive(self):
        """ITM put should have positive price."""
        g = bs_greeks(90, 100, 0.5, 0.05, 0.3, "put")
        assert g["price"] > 0


class TestIVSurfaceBuilder:
    """Tests for the IV surface interpolation/construction."""

    def test_surface_shape(self):
        builder = IVSurfaceBuilder()
        surface = builder.build_surface(vix_30d=0.2, vix_90d=0.22, vvix=0.7)
        assert surface.shape == (5, 5), f"Expected (5,5), got {surface.shape}"

    def test_surface_positive(self):
        builder = IVSurfaceBuilder()
        surface = builder.build_surface(vix_30d=0.2, vix_90d=0.22, vvix=0.7)
        assert np.all(surface > 0), "IV surface should be strictly positive"

    def test_surface_reasonable_range(self):
        """IV values should be in a sensible range (1%–300%)."""
        builder = IVSurfaceBuilder()
        surface = builder.build_surface(vix_30d=0.2, vix_90d=0.22, vvix=0.7)
        assert np.all(surface >= 0.01)
        assert np.all(surface <= 3.0)

    def test_higher_vix_higher_surface(self):
        """Higher VIX inputs should produce higher surface values on average."""
        builder = IVSurfaceBuilder()
        low  = builder.build_surface(0.10, 0.12, 0.5)
        high = builder.build_surface(0.30, 0.35, 0.5)
        assert high.mean() > low.mean()


class TestRollingCorrelation:
    """Tests for the rolling correlation matrix computation."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        eq = pd.DataFrame({"spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=dates)
        fx = pd.DataFrame({"eurusd": 1.1 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))}, index=dates)
        rates = pd.DataFrame({"rate_10y": 0.03 + np.cumsum(rng.normal(0, 0.0005, n))}, index=dates)

        corr = compute_rolling_correlation(eq, fx, rates, window=60)
        # Should have n - window - 1 rows (due to diff + window rollback)
        assert len(corr) > 0
        assert len(corr.columns) == 9  # 3×3 flattened

    def test_diagonal_near_one(self):
        """Diagonal of correlation matrix (corr_0, corr_4, corr_8) should be ~1."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        eq = pd.DataFrame({"spx": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=dates)
        fx = pd.DataFrame({"eurusd": 1.1 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))}, index=dates)
        rates = pd.DataFrame({"rate_10y": 0.03 + np.cumsum(rng.normal(0, 0.0005, n))}, index=dates)

        corr = compute_rolling_correlation(eq, fx, rates, window=60)
        # Diagonal entries of correlation matrix = 1.0
        for diag_col in ["corr_0", "corr_4", "corr_8"]:
            vals = corr[diag_col].values
            np.testing.assert_allclose(vals, 1.0, atol=1e-10,
                                       err_msg=f"{diag_col} should be 1.0")
