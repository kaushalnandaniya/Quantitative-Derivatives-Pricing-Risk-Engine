import pytest
import numpy as np
from pricing.greeks import GreeksCalculator


@pytest.fixture
def market_data():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2
    }


def test_greek_consistency(market_data):
    """
    Validation: Compare Analytical vs Numerical Greeks
    """
    calc_an = GreeksCalculator(method="analytical")
    calc_num = GreeksCalculator(method="numerical", h=1e-5)
    
    g_an = calc_an.calculate(**market_data)
    g_num = calc_num.calculate(**market_data)
    
    for key in ["delta", "gamma", "vega"]:
        assert np.isclose(g_an[key], g_num[key], rtol=1e-3), f"Greek mismatch in {key}"


def test_call_delta_range(market_data):
    """Call delta should be between 0 and 1."""
    calc = GreeksCalculator(method="analytical")
    greeks = calc.calculate(**market_data, option_type="call")
    assert 0 <= greeks["delta"] <= 1, f"Call delta out of range: {greeks['delta']}"


def test_put_delta_range(market_data):
    """Put delta should be between -1 and 0."""
    calc = GreeksCalculator(method="analytical")
    greeks = calc.calculate(**market_data, option_type="put")
    assert -1 <= greeks["delta"] <= 0, f"Put delta out of range: {greeks['delta']}"


def test_gamma_positive(market_data):
    """Gamma should always be positive."""
    calc = GreeksCalculator(method="analytical")
    greeks = calc.calculate(**market_data)
    assert greeks["gamma"] > 0, f"Gamma should be positive, got {greeks['gamma']}"


def test_vega_positive(market_data):
    """Vega should always be positive."""
    calc = GreeksCalculator(method="analytical")
    greeks = calc.calculate(**market_data)
    assert greeks["vega"] > 0, f"Vega should be positive, got {greeks['vega']}"
