"""
Test Suite — Portfolio, P&L Simulation, Correlation & Integration
==================================================================
End-to-end validation of the risk engine pipeline.
"""

import numpy as np
import pytest

from risk.portfolio import Portfolio
from risk.pnl import simulate_portfolio_pnl
from risk.var import var, historical_var
from risk.cvar import cvar
from risk.correlation import (
    generate_correlated_normals,
    simulate_correlated_gbm,
    _validate_correlation_matrix,
)
from pricing.black_scholes import black_scholes_price


# =====================================================================
# Portfolio Tests
# =====================================================================

class TestPortfolio:

    def test_single_position_value(self):
        """Portfolio value for a single call should match BS price × qty."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=10)

        expected = float(black_scholes_price(100, 100, 1, 0.05, 0.2, "call")) * 10
        actual = float(port.value())
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_multi_position_value(self):
        """Portfolio of calls + puts should aggregate correctly."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=10)
        port.add_position(type="put", S=100, K=90, T=1, r=0.05, sigma=0.25, qty=5)

        bs_call = float(black_scholes_price(100, 100, 1, 0.05, 0.2, "call"))
        bs_put = float(black_scholes_price(100, 90, 1, 0.05, 0.25, "put"))
        expected = bs_call * 10 + bs_put * 5

        actual = float(port.value())
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_vectorized_valuation(self):
        """Passing array of spots should return array of values."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=1)

        spots = np.array([90, 95, 100, 105, 110], dtype=float)
        values = port.value_at_spots(spots, T_offset=0.0)

        assert isinstance(values, np.ndarray)
        assert values.shape == (5,)
        # Higher spot → higher call value
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i]

    def test_short_position(self):
        """Negative qty (short) should give negative contribution."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=-5)

        val = float(port.value())
        assert val < 0

    def test_empty_portfolio_raises(self):
        port = Portfolio()
        with pytest.raises(ValueError, match="no positions"):
            port.value()

    def test_invalid_type_raises(self):
        port = Portfolio()
        with pytest.raises(ValueError, match="call.*put"):
            port.add_position(type="swap", S=100, K=100, T=1, sigma=0.2)

    def test_summary_output(self):
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=10)
        summary = port.summary()
        assert len(summary) == 1
        assert summary[0]["type"] == "call"
        assert summary[0]["qty"] == 10
        assert summary[0]["total_value"] > 0

    def test_n_positions(self):
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, sigma=0.2, qty=1)
        port.add_position(type="put", S=100, K=90, T=1, sigma=0.2, qty=1)
        assert port.n_positions == 2

    def test_time_decay(self):
        """Portfolio value should decrease as T_offset increases (time decay)."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=10)

        v_now = float(port.value(T_offset=0.0))
        v_later = float(port.value(T_offset=0.5))
        assert v_now > v_later  # Theta decay


# =====================================================================
# P&L Simulation Tests
# =====================================================================

class TestPnLSimulation:

    def test_pnl_shape(self):
        """P&L output should have correct shape."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.25, r=0.05, sigma=0.2, qty=10)

        result = simulate_portfolio_pnl(port, n_sims=10_000, horizon_days=1, seed=42)
        assert result["pnl"].shape == (10_000,)
        assert result["V_T"].shape == (10_000,)

    def test_pnl_has_losses_and_gains(self):
        """P&L should contain both positive and negative values."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.25, r=0.05, sigma=0.3, qty=10)

        result = simulate_portfolio_pnl(port, n_sims=50_000, horizon_days=1, seed=42)
        pnl = result["pnl"]
        assert np.any(pnl > 0), "No gains in P&L"
        assert np.any(pnl < 0), "No losses in P&L"

    def test_pnl_initial_value(self):
        """V_0 should be positive for long positions."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.25, r=0.05, sigma=0.2, qty=10)

        result = simulate_portfolio_pnl(port, n_sims=1000)
        assert result["V_0"] > 0

    def test_reproducibility(self):
        """Same seed should give same P&L."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.25, r=0.05, sigma=0.2, qty=10)

        r1 = simulate_portfolio_pnl(port, n_sims=10_000, seed=99)
        r2 = simulate_portfolio_pnl(port, n_sims=10_000, seed=99)
        np.testing.assert_array_equal(r1["pnl"], r2["pnl"])


# =====================================================================
# Correlation Tests
# =====================================================================

class TestCorrelation:

    def test_identity_produces_independent(self):
        """Identity correlation → independent draws."""
        rng = np.random.default_rng(42)
        corr = np.eye(3)
        Z = generate_correlated_normals(100_000, 3, corr, rng)

        # Check empirical correlations are near zero
        emp_corr = np.corrcoef(Z)
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert emp_corr[i, j] == pytest.approx(1.0, abs=0.01)
                else:
                    assert abs(emp_corr[i, j]) < 0.02

    def test_correlated_draws(self):
        """Specified correlation should be reproduced empirically."""
        rng = np.random.default_rng(42)
        target_corr = np.array([
            [1.0, 0.7, -0.3],
            [0.7, 1.0, 0.2],
            [-0.3, 0.2, 1.0],
        ])
        Z = generate_correlated_normals(500_000, 3, target_corr, rng)
        emp_corr = np.corrcoef(Z)

        # Should match within ~0.01 for 500k samples
        np.testing.assert_allclose(emp_corr, target_corr, atol=0.015)

    def test_marginal_standard_normal(self):
        """Each marginal should be approximately N(0,1)."""
        rng = np.random.default_rng(42)
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        Z = generate_correlated_normals(200_000, 2, corr, rng)

        for i in range(2):
            assert np.mean(Z[i]) == pytest.approx(0.0, abs=0.01)
            assert np.std(Z[i]) == pytest.approx(1.0, abs=0.01)

    def test_invalid_corr_matrix_not_symmetric(self):
        bad = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            _validate_correlation_matrix(bad)

    def test_invalid_corr_matrix_bad_diagonal(self):
        bad = np.array([[2.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            _validate_correlation_matrix(bad)

    def test_correlated_gbm_shape(self):
        """Correlated GBM should return correct shape."""
        rng = np.random.default_rng(42)
        spots = np.array([100.0, 50.0])
        rates = np.array([0.05, 0.05])
        sigmas = np.array([0.2, 0.3])
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])

        S_T = simulate_correlated_gbm(spots, rates, sigmas, 1/252, 10_000, corr, rng)
        assert S_T.shape == (2, 10_000)
        assert np.all(S_T > 0)  # Prices must be positive

    def test_correlated_gbm_reproduces_correlation(self):
        """Terminal log-returns should have the specified correlation."""
        rng = np.random.default_rng(42)
        spots = np.array([100.0, 100.0])
        rates = np.array([0.05, 0.05])
        sigmas = np.array([0.2, 0.2])
        rho = 0.8
        corr = np.array([[1.0, rho], [rho, 1.0]])

        S_T = simulate_correlated_gbm(spots, rates, sigmas, 1.0, 500_000, corr, rng)

        log_returns = np.log(S_T / spots[:, None])
        emp_corr = np.corrcoef(log_returns[0], log_returns[1])[0, 1]
        assert emp_corr == pytest.approx(rho, abs=0.02)


# =====================================================================
# Diversification Tests
# =====================================================================

class TestDiversification:

    def test_diversification_reduces_risk(self):
        """
        A diversified portfolio should have lower VaR than the sum
        of individual position VaRs (subadditivity for VaR holds
        approximately for normal-like distributions).
        """
        # Single concentrated position
        port_single = Portfolio()
        port_single.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=0.3, qty=20, asset="A",
        )
        r_single = simulate_portfolio_pnl(port_single, n_sims=50_000, seed=42)
        var_single = historical_var(r_single["pnl"], 0.95)

        # Diversified: split across 2 assets (uncorrelated)
        port_div = Portfolio()
        port_div.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=0.3, qty=10, asset="A",
        )
        port_div.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=0.3, qty=10, asset="B",
        )
        corr = np.eye(2)  # Uncorrelated
        r_div = simulate_portfolio_pnl(port_div, n_sims=50_000, seed=42, corr_matrix=corr)
        var_div = historical_var(r_div["pnl"], 0.95)

        # Diversified VaR should be lower
        assert var_div < var_single, (
            f"Diversification did not reduce risk: "
            f"div VaR={var_div:.2f} vs single VaR={var_single:.2f}"
        )


# =====================================================================
# Integration Tests
# =====================================================================

class TestIntegration:

    def test_full_pipeline(self):
        """
        End-to-end: Portfolio → P&L → VaR → CVaR.
        Validates the complete risk engine pipeline.
        """
        # Build portfolio
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.25, r=0.05, sigma=0.2, qty=10)
        port.add_position(type="put", S=100, K=90, T=0.25, r=0.05, sigma=0.25, qty=5)

        # Simulate P&L
        result = simulate_portfolio_pnl(port, n_sims=50_000, horizon_days=1, seed=42)
        pnl = result["pnl"]

        # Compute risk metrics
        var_val = var(pnl, 0.95, "historical")
        cvar_val = cvar(pnl, 0.95, "historical")

        # Validate
        assert var_val > 0, "VaR should be positive"
        assert cvar_val >= var_val, f"CVaR ({cvar_val}) < VaR ({var_val})"
        assert result["V_0"] > 0, "Initial portfolio value should be positive"
        assert len(pnl) == 50_000

    def test_multi_asset_pipeline(self):
        """End-to-end with correlated multi-asset portfolio."""
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=0.5, r=0.05, sigma=0.2, qty=10, asset="AAPL")
        port.add_position(type="put", S=50, K=50, T=0.5, r=0.05, sigma=0.3, qty=5, asset="MSFT")

        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        result = simulate_portfolio_pnl(port, n_sims=20_000, horizon_days=5, seed=42, corr_matrix=corr)

        pnl = result["pnl"]
        var_val = var(pnl, 0.95)
        cvar_val = cvar(pnl, 0.95)

        assert var_val > 0
        assert cvar_val >= var_val
        assert "AAPL" in result["S_T"]
        assert "MSFT" in result["S_T"]
        assert result["S_T"]["AAPL"].shape == (20_000,)
