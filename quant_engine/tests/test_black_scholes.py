import pytest
import numpy as np
from pricing.black_scholes import black_scholes_price

# Tolerance for floating point comparisons
TOL = 1e-7

@pytest.fixture
def market_data():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2
    }

def test_put_call_parity(market_data):
    """
    Validation: C - P = S - K * exp(-rT)
    """
    S, K, T, r, sigma = market_data.values()
    
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    
    assert np.isclose(lhs, rhs, atol=TOL), f"Put-Call Parity failed: {lhs} != {rhs}"

def test_boundary_conditions():
    """
    Test T -> 0 and sigma -> 0
    """
    # At expiry (T=0), Call = max(S-K, 0)
    price_at_expiry = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
    assert price_at_expiry == 10.0
    
    # Low vol (sigma -> 0), Call = max(S - K*exp(-rt), 0)
    price_low_vol = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=1e-9, option_type="call")
    expected = 100 - 100 * np.exp(-0.05 * 1)
    assert np.isclose(price_low_vol, expected, atol=TOL)

def test_vectorization():
    """
    Ensure engine handles arrays without looping
    """
    S_arr = np.array([100, 110, 120])
    prices = black_scholes_price(S_arr, 100, 1, 0.05, 0.2)
    assert len(prices) == 3
    assert isinstance(prices, np.ndarray)
