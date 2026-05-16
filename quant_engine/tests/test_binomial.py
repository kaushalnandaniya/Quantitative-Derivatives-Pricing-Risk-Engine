"""
Test Suite for Binomial Tree Pricing Engine
=============================================
Validates:
    1. CRR parameters are correct
    2. Binomial converges to Black-Scholes for Europeans
    3. American option price ≥ European option price
    4. American call = European call (no dividends)
    5. Put-call parity for European binomial prices
    6. Edge cases (T→0, deep ITM/OTM, σ→0)
    7. Deterministic output (no randomness)
    8. Full-tree variant consistency
"""

import pytest
import numpy as np
from pricing.binomial import binomial_price, binomial_price_with_tree, _crr_params
from pricing.black_scholes import black_scholes_price


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
def bs_call(market_params):
    return float(black_scholes_price(
        market_params["S0"], market_params["K"], market_params["T"],
        market_params["r"], market_params["sigma"], "call"
    ))


@pytest.fixture
def bs_put(market_params):
    return float(black_scholes_price(
        market_params["S0"], market_params["K"], market_params["T"],
        market_params["r"], market_params["sigma"], "put"
    ))


# =============================================================================
# CRR Parameter Tests
# =============================================================================

class TestCRRParameters:
    """Validate CRR parameter computation."""

    def test_u_times_d_equals_one(self, market_params):
        """Recombining tree requires u*d = 1."""
        params = _crr_params(market_params["sigma"], market_params["r"],
                             market_params["T"], N=100)
        assert np.isclose(params["u"] * params["d"], 1.0, atol=1e-12)

    def test_risk_neutral_probability_bounds(self, market_params):
        """Risk-neutral probability must be in (0, 1)."""
        for N in [10, 50, 100, 500, 1000]:
            params = _crr_params(market_params["sigma"], market_params["r"],
                                 market_params["T"], N)
            assert 0 < params["p"] < 1, f"p={params['p']} out of bounds for N={N}"

    def test_martingale_condition(self, market_params):
        """E[S_{t+1}|S_t] = S_t * exp(r*dt) under risk-neutral."""
        params = _crr_params(market_params["sigma"], market_params["r"],
                             market_params["T"], N=100)
        u, d, p = params["u"], params["d"], params["p"]
        expected_growth = p * u + (1 - p) * d
        theoretical = np.exp(market_params["r"] * params["dt"])
        assert np.isclose(expected_growth, theoretical, atol=1e-10)


# =============================================================================
# Convergence to Black-Scholes
# =============================================================================

class TestConvergenceToBS:
    """Binomial prices should converge to BS as N → ∞."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_convergence_large_N(self, market_params, option_type):
        """Binomial(N=1000) should match BS within 0.1%."""
        bs = float(black_scholes_price(
            market_params["S0"], market_params["K"], market_params["T"],
            market_params["r"], market_params["sigma"], option_type
        ))
        result = binomial_price(
            **market_params, option_type=option_type, style="european", N=1000
        )
        rel_error = abs(result["price"] - bs) / bs
        assert rel_error < 0.001, (
            f"Binomial({option_type}, N=1000) = {result['price']:.6f}, "
            f"BS = {bs:.6f}, error = {rel_error:.4%}"
        )

    def test_convergence_improves_with_N(self, market_params, bs_call):
        """Error should generally decrease as N increases."""
        errors = []
        for N in [50, 200, 1000]:
            result = binomial_price(**market_params, option_type="call",
                                    style="european", N=N)
            errors.append(abs(result["price"] - bs_call))
        # Last error should be smallest
        assert errors[-1] < errors[0], (
            f"Error did not decrease: N=50 → {errors[0]:.6f}, N=1000 → {errors[-1]:.6f}"
        )


# =============================================================================
# American Option Tests
# =============================================================================

class TestAmericanOptions:
    """Test American option pricing and early exercise logic."""

    def test_american_put_geq_european_put(self, market_params):
        """American put ≥ European put (early exercise premium)."""
        euro = binomial_price(**market_params, option_type="put",
                              style="european", N=500)
        amer = binomial_price(**market_params, option_type="put",
                              style="american", N=500)
        assert amer["price"] >= euro["price"] - 1e-10, (
            f"American put ({amer['price']:.6f}) < European put ({euro['price']:.6f})"
        )

    def test_american_put_strictly_greater_than_european(self):
        """For typical parameters, American put should have positive premium."""
        params = {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
        euro = binomial_price(**params, option_type="put", style="european", N=500)
        amer = binomial_price(**params, option_type="put", style="american", N=500)
        premium = amer["price"] - euro["price"]
        assert premium > 0.01, (
            f"Expected positive early exercise premium, got {premium:.6f}"
        )

    def test_american_call_equals_european_call(self, market_params):
        """American call = European call when there are no dividends."""
        euro = binomial_price(**market_params, option_type="call",
                              style="european", N=500)
        amer = binomial_price(**market_params, option_type="call",
                              style="american", N=500)
        assert np.isclose(amer["price"], euro["price"], atol=1e-8), (
            f"American call ({amer['price']:.6f}) ≠ European call ({euro['price']:.6f})"
        )

    def test_american_call_geq_european_call(self, market_params):
        """American call ≥ European call (general property)."""
        euro = binomial_price(**market_params, option_type="call",
                              style="european", N=500)
        amer = binomial_price(**market_params, option_type="call",
                              style="american", N=500)
        assert amer["price"] >= euro["price"] - 1e-10

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_american_geq_intrinsic(self, market_params, option_type):
        """American option price ≥ intrinsic value."""
        result = binomial_price(**market_params, option_type=option_type,
                                style="american", N=500)
        if option_type == "call":
            intrinsic = max(market_params["S0"] - market_params["K"], 0)
        else:
            intrinsic = max(market_params["K"] - market_params["S0"], 0)
        assert result["price"] >= intrinsic - 1e-10


# =============================================================================
# Put-Call Parity (European)
# =============================================================================

class TestPutCallParity:
    """Put-call parity for European binomial prices."""

    def test_put_call_parity(self, market_params):
        """C - P = S - K*exp(-rT) for European options."""
        call = binomial_price(**market_params, option_type="call",
                              style="european", N=500)
        put = binomial_price(**market_params, option_type="put",
                             style="european", N=500)
        lhs = call["price"] - put["price"]
        rhs = market_params["S0"] - market_params["K"] * np.exp(
            -market_params["r"] * market_params["T"]
        )
        assert np.isclose(lhs, rhs, atol=0.01), (
            f"Put-call parity: C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}"
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_at_expiry(self):
        """T → 0: option = intrinsic value."""
        call = binomial_price(S0=110, K=100, T=1e-6, r=0.05, sigma=0.2,
                              option_type="call", N=10)
        assert np.isclose(call["price"], 10.0, atol=0.01)

        put = binomial_price(S0=90, K=100, T=1e-6, r=0.05, sigma=0.2,
                             option_type="put", N=10)
        assert np.isclose(put["price"], 10.0, atol=0.01)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)."""
        result = binomial_price(S0=200, K=100, T=1, r=0.05, sigma=0.2,
                                option_type="call", style="european", N=500)
        expected = 200 - 100 * np.exp(-0.05)
        assert np.isclose(result["price"], expected, rtol=0.01)

    def test_deep_otm_put(self):
        """Deep OTM put ≈ 0."""
        result = binomial_price(S0=200, K=100, T=1, r=0.05, sigma=0.2,
                                option_type="put", style="european", N=500)
        assert result["price"] < 0.5

    def test_deep_itm_put(self):
        """Deep ITM American put ≈ K - S (immediate exercise is near-optimal)."""
        result = binomial_price(S0=50, K=100, T=1, r=0.05, sigma=0.2,
                                option_type="put", style="american", N=500)
        intrinsic = 100 - 50
        assert result["price"] >= intrinsic - 0.01
        # Should be close to intrinsic for deep ITM
        assert result["price"] < intrinsic + 5.0

    def test_zero_volatility_call(self):
        """σ → 0: call ≈ max(S - K*exp(-rT), 0) for small N."""
        # Note: very low σ with large N causes numerical overflow in the tree
        # (u^N → ∞). Using N=10 keeps it stable while testing the limiting case.
        result = binomial_price(S0=100, K=95, T=1, r=0.05, sigma=1e-4,
                                option_type="call", style="european", N=10)
        expected = max(100 - 95 * np.exp(-0.05), 0)
        assert np.isclose(result["price"], expected, atol=0.5)

    def test_invalid_option_type(self, market_params):
        with pytest.raises(ValueError, match="option_type must be"):
            binomial_price(**market_params, option_type="straddle")

    def test_invalid_style(self, market_params):
        with pytest.raises(ValueError, match="style must be"):
            binomial_price(**market_params, style="bermudan")


# =============================================================================
# Determinism & Consistency
# =============================================================================

class TestDeterminism:
    """Binomial tree is deterministic — no randomness."""

    def test_same_result_every_time(self, market_params):
        """Calling twice gives identical results."""
        r1 = binomial_price(**market_params, N=200)
        r2 = binomial_price(**market_params, N=200)
        assert r1["price"] == r2["price"]

    def test_tree_variant_matches_fast(self, market_params):
        """Full-tree version should give same price as fast version."""
        N = 50
        fast = binomial_price(**market_params, option_type="put",
                              style="american", N=N)
        tree = binomial_price_with_tree(**market_params, option_type="put",
                                        style="american", N=N)
        assert np.isclose(fast["price"], tree["price"], atol=1e-10), (
            f"Fast={fast['price']:.10f}, Tree={tree['price']:.10f}"
        )


# =============================================================================
# Full Tree Tests
# =============================================================================

class TestFullTree:
    """Tests for the tree-returning variant."""

    def test_tree_shapes(self, market_params):
        """Stock and option trees should have correct shapes."""
        N = 10
        result = binomial_price_with_tree(**market_params, N=N)
        assert len(result["stock_tree"]) == N + 1
        assert len(result["option_tree"]) == N + 1
        for i in range(N + 1):
            assert len(result["stock_tree"][i]) == i + 1
            assert len(result["option_tree"][i]) == i + 1

    def test_stock_prices_positive(self, market_params):
        result = binomial_price_with_tree(**market_params, N=20)
        for level in result["stock_tree"]:
            assert np.all(level > 0)

    def test_option_values_non_negative(self, market_params):
        result = binomial_price_with_tree(**market_params, option_type="put",
                                          style="american", N=20)
        for level in result["option_tree"]:
            assert np.all(level >= -1e-10)

    def test_exercise_boundary_exists_for_american_put(self):
        """American put should have an exercise boundary."""
        result = binomial_price_with_tree(
            S0=100, K=100, T=1, r=0.05, sigma=0.2,
            option_type="put", style="american", N=50
        )
        assert result["exercise_boundary"] is not None
        assert len(result["exercise_boundary"]) > 0

    def test_no_exercise_boundary_for_european(self, market_params):
        """European options should not have an exercise boundary."""
        result = binomial_price_with_tree(**market_params, style="european", N=20)
        assert result["exercise_boundary"] is None
