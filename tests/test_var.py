"""
Test Suite — VaR & CVaR
========================
Validates all risk metric computations against known distributions
and mathematical properties.
"""

import numpy as np
import pytest
from scipy.stats import norm

from risk.var import historical_var, parametric_var, monte_carlo_var, var
from risk.cvar import cvar, parametric_cvar


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def normal_pnl():
    """Large sample from N(-5, 100) — mean loss of $5, std $10."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=-5, scale=10, size=500_000)


@pytest.fixture
def known_pnl():
    """Small, hand-crafted P&L array for exact verification."""
    return np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40])


# =====================================================================
# VaR Tests
# =====================================================================

class TestHistoricalVaR:

    def test_known_distribution(self, known_pnl):
        """VaR on known data should match expected quantile."""
        # 90% VaR: 10th percentile of [-50,-40,...,40] = -40
        # VaR = -(-40) = 40
        var_val = historical_var(known_pnl, confidence=0.90)
        assert var_val == pytest.approx(40.0, abs=5.0)

    def test_positive_return(self):
        """VaR should be 0 (or very small) for all-positive P&L."""
        pnl = np.ones(1000) * 10.0  # Always profit
        var_val = historical_var(pnl, confidence=0.95)
        assert var_val == pytest.approx(0.0, abs=0.1)

    def test_higher_confidence_higher_var(self, normal_pnl):
        """99% VaR should exceed 95% VaR."""
        var_95 = historical_var(normal_pnl, 0.95)
        var_99 = historical_var(normal_pnl, 0.99)
        assert var_99 > var_95


class TestParametricVaR:

    def test_matches_analytical_normal(self):
        """For normally distributed data, parametric VaR should match
        the analytical formula: VaR = -(μ + z_α · σ)."""
        rng = np.random.default_rng(123)
        mu, sigma = -2.0, 15.0
        pnl = rng.normal(mu, sigma, size=1_000_000)

        var_val = parametric_var(pnl, 0.95)
        z_alpha = norm.ppf(0.05)
        expected = -(mu + z_alpha * sigma)

        assert var_val == pytest.approx(expected, rel=0.02)

    def test_zero_vol_zero_var(self):
        """If all P&L is identical and positive, VaR should be 0."""
        pnl = np.ones(1000) * 5.0
        var_val = parametric_var(pnl, 0.95)
        assert var_val == pytest.approx(0.0, abs=0.1)


class TestMonteCarloVaR:

    def test_matches_historical(self, normal_pnl):
        """Monte Carlo VaR should equal Historical VaR (same computation)."""
        mc = monte_carlo_var(normal_pnl, 0.95)
        hist = historical_var(normal_pnl, 0.95)
        assert mc == pytest.approx(hist, rel=1e-10)


class TestVaRDispatcher:

    def test_all_methods_agree_for_normal(self, normal_pnl):
        """All 3 methods should be within ~10% for large normal samples."""
        v_h = var(normal_pnl, 0.95, "historical")
        v_p = var(normal_pnl, 0.95, "parametric")
        v_m = var(normal_pnl, 0.95, "monte_carlo")

        # Historical and parametric should agree for normal data
        assert v_h == pytest.approx(v_p, rel=0.05)
        assert v_m == pytest.approx(v_h, rel=1e-10)

    def test_invalid_method(self, normal_pnl):
        with pytest.raises(ValueError, match="method"):
            var(normal_pnl, 0.95, "invalid")

    def test_invalid_confidence(self, normal_pnl):
        with pytest.raises(ValueError, match="confidence"):
            var(normal_pnl, 1.5)

    def test_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            var(np.array([]), 0.95)


class TestVolatilitySensitivity:

    def test_higher_vol_higher_var(self):
        """Increasing volatility must increase VaR."""
        rng = np.random.default_rng(42)
        var_values = []
        for sigma in [5, 10, 20, 50]:
            pnl = rng.normal(0, sigma, size=100_000)
            var_values.append(var(pnl, 0.95, "historical"))

        # VaR should be monotonically increasing
        for i in range(len(var_values) - 1):
            assert var_values[i + 1] > var_values[i], (
                f"VaR did not increase: σ={[5,10,20,50][i]}->{[5,10,20,50][i+1]}, "
                f"VaR={var_values[i]:.2f}->{var_values[i+1]:.2f}"
            )


# =====================================================================
# CVaR Tests
# =====================================================================

class TestCVaR:

    def test_cvar_geq_var(self, normal_pnl):
        """CVaR must ALWAYS be ≥ VaR — fundamental property."""
        for confidence in [0.90, 0.95, 0.99]:
            var_val = var(normal_pnl, confidence, "historical")
            cvar_val = cvar(normal_pnl, confidence, "historical")
            assert cvar_val >= var_val * 0.999, (
                f"CVaR < VaR at {confidence:.0%}: "
                f"CVaR={cvar_val:.4f}, VaR={var_val:.4f}"
            )

    def test_cvar_geq_var_parametric(self, normal_pnl):
        """Parametric CVaR ≥ Parametric VaR."""
        var_val = parametric_var(normal_pnl, 0.95)
        cvar_val = parametric_cvar(normal_pnl, 0.95)
        assert cvar_val >= var_val * 0.999

    def test_cvar_known_tail(self, known_pnl):
        """CVaR on known data: average of losses beyond VaR."""
        # 80% VaR: 20th percentile of [-50,-40,...,40]
        # Tail: values ≤ 20th percentile
        cvar_val = cvar(known_pnl, confidence=0.80, method="historical")
        assert cvar_val > 0  # Should indicate some loss

    def test_parametric_cvar_analytical(self):
        """Parametric CVaR should match closed-form for pure normal data."""
        rng = np.random.default_rng(99)
        mu, sigma = 0.0, 10.0
        pnl = rng.normal(mu, sigma, size=1_000_000)

        cvar_val = parametric_cvar(pnl, 0.95)
        # Analytical: CVaR = σ * φ(z_α) / α for μ=0
        z_alpha = norm.ppf(0.05)
        expected = sigma * norm.pdf(z_alpha) / 0.05

        assert cvar_val == pytest.approx(expected, rel=0.02)

    def test_invalid_method(self, normal_pnl):
        with pytest.raises(ValueError, match="method"):
            cvar(normal_pnl, 0.95, "invalid")

    def test_cvar_higher_confidence_higher(self, normal_pnl):
        """99% CVaR > 95% CVaR."""
        cvar_95 = cvar(normal_pnl, 0.95)
        cvar_99 = cvar(normal_pnl, 0.99)
        assert cvar_99 > cvar_95
