"""Tests for Implied Volatility Solver."""
import pytest
import numpy as np
from pricing.implied_vol import implied_volatility, implied_volatility_newton, implied_volatility_bisection
from pricing.black_scholes import black_scholes_price


class TestImpliedVolatility:
    """Round-trip tests: BS price -> IV solver -> recovered sigma."""

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_round_trip_call(self, sigma):
        S, K, T, r = 100, 100, 1.0, 0.05
        price = float(black_scholes_price(S, K, T, r, sigma, "call"))
        iv = implied_volatility(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6, f"Expected σ={sigma}, got IV={iv}"

    @pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50])
    def test_round_trip_put(self, sigma):
        S, K, T, r = 100, 100, 1.0, 0.05
        price = float(black_scholes_price(S, K, T, r, sigma, "put"))
        iv = implied_volatility(price, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6

    def test_deep_itm_call(self):
        S, K, T, r, sigma = 150, 100, 0.5, 0.05, 0.25
        price = float(black_scholes_price(S, K, T, r, sigma, "call"))
        iv = implied_volatility(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-4

    def test_deep_otm_put(self):
        S, K, T, r, sigma = 150, 100, 0.5, 0.05, 0.25
        price = float(black_scholes_price(S, K, T, r, sigma, "put"))
        iv = implied_volatility(price, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-4

    def test_short_maturity(self):
        S, K, T, r, sigma = 100, 100, 1/252, 0.05, 0.20
        price = float(black_scholes_price(S, K, T, r, sigma, "call"))
        iv = implied_volatility(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-3

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            implied_volatility(0.0, 100, 100, 1, 0.05, "call")

    def test_below_intrinsic_raises(self):
        with pytest.raises(ValueError):
            implied_volatility(0.01, 150, 100, 1, 0.05, "call")

    def test_newton_method(self):
        price = float(black_scholes_price(100, 100, 1, 0.05, 0.2, "call"))
        iv = implied_volatility_newton(price, 100, 100, 1, 0.05, "call")
        assert iv is not None
        assert abs(iv - 0.2) < 1e-6

    def test_bisection_method(self):
        price = float(black_scholes_price(100, 100, 1, 0.05, 0.2, "call"))
        iv = implied_volatility_bisection(price, 100, 100, 1, 0.05, "call")
        assert iv is not None
        assert abs(iv - 0.2) < 1e-6
