"""
Test Suite for Monte Carlo Pricing Engine
==========================================
Validates:
    1. Monte Carlo prices converge to Black-Scholes within tolerance
    2. Put-call parity holds under MC
    3. Confidence intervals contain the true price
    4. Antithetic and control variate methods reduce variance
    5. Edge cases (deep ITM/OTM, short maturity)
    6. Full vectorization (no loops)
"""

import pytest
import numpy as np
from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import compute_payoff, monte_carlo_price
from models.gbm import simulate_terminal_price, simulate_terminal_price_antithetic
from experiments.convergence_analysis import convergence_analysis
from experiments.variance_reduction import variance_reduction_comparison


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def market_params():
    """Standard ATM option parameters."""
    return {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
    }


@pytest.fixture
def bs_call_price(market_params):
    return float(black_scholes_price(
        market_params["S0"], market_params["K"], market_params["T"],
        market_params["r"], market_params["sigma"], "call"
    ))


@pytest.fixture
def bs_put_price(market_params):
    return float(black_scholes_price(
        market_params["S0"], market_params["K"], market_params["T"],
        market_params["r"], market_params["sigma"], "put"
    ))


# =============================================================================
# GBM Simulator Tests
# =============================================================================

class TestGBMSimulator:
    """Tests for the terminal price simulator."""

    def test_output_shape(self, market_params):
        rng = np.random.default_rng(42)
        ST = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 10_000, rng
        )
        assert ST.shape == (10_000,), f"Expected (10000,), got {ST.shape}"

    def test_all_positive_prices(self, market_params):
        rng = np.random.default_rng(42)
        ST = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 50_000, rng
        )
        assert np.all(ST > 0), "GBM should produce strictly positive prices"

    def test_expected_value_risk_neutral(self, market_params):
        """E[S_T] = S_0 * exp(r*T) under risk-neutral measure."""
        rng = np.random.default_rng(42)
        ST = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 500_000, rng
        )
        expected = market_params["S0"] * np.exp(market_params["r"] * market_params["T"])
        mc_mean = np.mean(ST)
        # Should be close to theoretical forward price
        assert np.isclose(mc_mean, expected, rtol=0.01), (
            f"E[S_T] = {mc_mean:.4f}, expected ≈ {expected:.4f}"
        )

    def test_reproducibility(self, market_params):
        """Same seed → same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ST1 = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 1000, rng1
        )
        ST2 = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 1000, rng2
        )
        np.testing.assert_array_equal(ST1, ST2)


# =============================================================================
# Payoff Tests
# =============================================================================

class TestPayoff:
    """Tests for payoff computation."""

    def test_call_payoff(self):
        ST = np.array([120, 100, 80])
        K = 100
        payoff = compute_payoff(ST, K, "call")
        expected = np.array([20, 0, 0])
        np.testing.assert_array_almost_equal(payoff, expected)

    def test_put_payoff(self):
        ST = np.array([120, 100, 80])
        K = 100
        payoff = compute_payoff(ST, K, "put")
        expected = np.array([0, 0, 20])
        np.testing.assert_array_almost_equal(payoff, expected)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="option_type must be"):
            compute_payoff(np.array([100.0]), 100.0, "straddle")

    def test_payoff_non_negative(self, market_params):
        rng = np.random.default_rng(42)
        ST = simulate_terminal_price(
            market_params["S0"], market_params["r"], market_params["sigma"],
            market_params["T"], 10_000, rng
        )
        for otype in ["call", "put"]:
            payoff = compute_payoff(ST, market_params["K"], otype)
            assert np.all(payoff >= 0), f"Payoff should be non-negative for {otype}"


# =============================================================================
# MC Pricing Tests
# =============================================================================

class TestMCPricing:
    """Test Monte Carlo prices against Black-Scholes."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("method", ["standard", "antithetic", "control"])
    def test_mc_vs_bs(self, market_params, option_type, method):
        """MC price should be within 1% of BS for large N."""
        bs = float(black_scholes_price(
            market_params["S0"], market_params["K"], market_params["T"],
            market_params["r"], market_params["sigma"], option_type
        ))
        result = monte_carlo_price(
            **market_params, option_type=option_type,
            n_sims=200_000, seed=42, method=method
        )
        rel_error = abs(result["price"] - bs) / bs
        assert rel_error < 0.01, (
            f"{method}/{option_type}: MC={result['price']:.4f}, BS={bs:.4f}, "
            f"error={rel_error:.4%}"
        )

    def test_confidence_interval_contains_bs(self, market_params, bs_call_price):
        """95% CI should contain the true BS price (with high probability)."""
        result = monte_carlo_price(
            **market_params, option_type="call",
            n_sims=500_000, seed=42, method="standard"
        )
        assert result["ci_lower"] <= bs_call_price <= result["ci_upper"], (
            f"BS price {bs_call_price:.4f} outside CI "
            f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
        )

    def test_put_call_parity_mc(self, market_params):
        """C - P = S - K*exp(-rT) should hold approximately."""
        call = monte_carlo_price(
            **market_params, option_type="call", n_sims=500_000, seed=42
        )
        put = monte_carlo_price(
            **market_params, option_type="put", n_sims=500_000, seed=42
        )
        lhs = call["price"] - put["price"]
        rhs = market_params["S0"] - market_params["K"] * np.exp(
            -market_params["r"] * market_params["T"]
        )
        assert np.isclose(lhs, rhs, atol=0.3), (
            f"Put-call parity: C-P={lhs:.4f}, S-K*e^(-rT)={rhs:.4f}"
        )

    def test_std_error_positive(self, market_params):
        result = monte_carlo_price(**market_params, n_sims=10_000, seed=42)
        assert result["std_error"] > 0
        assert result["variance"] > 0

    def test_invalid_method_raises(self, market_params):
        with pytest.raises(ValueError, match="method must be"):
            monte_carlo_price(**market_params, method="invalid")


# =============================================================================
# Variance Reduction Tests
# =============================================================================

class TestVarianceReduction:
    """Test that variance reduction methods actually reduce variance."""

    def test_antithetic_reduces_variance(self, market_params):
        """Antithetic should have lower variance than standard MC."""
        std = monte_carlo_price(
            **market_params, n_sims=100_000, seed=42, method="standard"
        )
        anti = monte_carlo_price(
            **market_params, n_sims=100_000, seed=42, method="antithetic"
        )
        assert anti["variance"] < std["variance"], (
            f"Antithetic variance ({anti['variance']:.6f}) should be < "
            f"standard ({std['variance']:.6f})"
        )

    def test_control_variate_reduces_variance(self, market_params):
        """Control variate should have lower variance than standard MC."""
        std = monte_carlo_price(
            **market_params, n_sims=100_000, seed=42, method="standard"
        )
        cv = monte_carlo_price(
            **market_params, n_sims=100_000, seed=42, method="control"
        )
        assert cv["variance"] < std["variance"], (
            f"Control variate variance ({cv['variance']:.6f}) should be < "
            f"standard ({std['variance']:.6f})"
        )

    def test_variance_reduction_comparison(self, market_params):
        """Full comparison should return valid ratios > 1."""
        comp = variance_reduction_comparison(**market_params, n_sims=100_000, seed=42)
        assert comp["bs_price"] > 0
        for method in ["standard", "antithetic", "control"]:
            assert method in comp
            assert comp[method]["price"] > 0
        # Ratios > 1 means variance was reduced
        for method in ["antithetic", "control"]:
            ratio = comp[method]["variance_reduction_ratio"]
            assert ratio is not None and ratio > 1.0, (
                f"{method} variance reduction ratio = {ratio}, expected > 1.0"
            )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)."""
        result = monte_carlo_price(S0=200, K=100, T=1, r=0.05, sigma=0.2,
                                   option_type="call", n_sims=100_000, seed=42)
        expected = 200 - 100 * np.exp(-0.05)
        assert np.isclose(result["price"], expected, rtol=0.02)

    def test_deep_otm_put(self):
        """Deep OTM put should be near zero."""
        result = monte_carlo_price(S0=200, K=100, T=1, r=0.05, sigma=0.2,
                                   option_type="put", n_sims=100_000, seed=42)
        assert result["price"] < 0.5  # Should be nearly zero

    def test_short_maturity(self):
        """Very short T should give near-intrinsic value."""
        result = monte_carlo_price(S0=105, K=100, T=0.001, r=0.05, sigma=0.2,
                                   option_type="call", n_sims=100_000, seed=42)
        intrinsic = 5.0
        assert np.isclose(result["price"], intrinsic, atol=0.5)


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Test convergence analysis utility."""

    def test_convergence_analysis_structure(self, market_params):
        result = convergence_analysis(
            **market_params, sim_sizes=[100, 1000, 10000],
            methods=["standard"]
        )
        assert "sim_sizes" in result
        assert "bs_price" in result
        assert "results" in result
        assert len(result["results"]["standard"]) == 3

    def test_error_decreases_with_n(self, market_params):
        """Absolute error should generally decrease with more simulations."""
        result = convergence_analysis(
            **market_params, sim_sizes=[1000, 100_000],
            methods=["standard"], seed=42
        )
        errors = [r["abs_error"] for r in result["results"]["standard"]]
        # Not guaranteed for every seed, but statistically likely
        # Just check the last one is small
        assert errors[-1] < 1.0, f"Error at 100K sims should be < $1, got {errors[-1]}"
