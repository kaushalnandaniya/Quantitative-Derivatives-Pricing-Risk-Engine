import pytest
import numpy as np
from models.heston import simulate_heston_terminal_price
from models.merton_jump import simulate_merton_terminal_price
from pricing.greeks import GreeksCalculator

def test_heston_martingale():
    # S0 = E[S_T] * e^{-rT}
    S0 = 100.0
    r = 0.05
    T = 1.0
    
    ST = simulate_heston_terminal_price(
        S0=S0, v0=0.04, r=r, kappa=2.0, theta=0.04, xi=0.1, 
        rho=-0.7, T=T, n_steps=100, n_sims=10000
    )
    
    expected_mean = S0 * np.exp(r * T)
    actual_mean = np.mean(ST)
    
    # Check within 1% error due to MC variance
    assert np.abs(actual_mean - expected_mean) / expected_mean < 0.01

def test_merton_martingale():
    S0 = 100.0
    r = 0.05
    T = 1.0
    
    rng = np.random.default_rng(42)
    ST = simulate_merton_terminal_price(
        S0=S0, r=r, sigma=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.3,
        T=T, n_sims=100000, rng=rng
    )
    
    expected_mean = S0 * np.exp(r * T)
    actual_mean = np.mean(ST)
    
    # Check within 1% error
    assert np.abs(actual_mean - expected_mean) / expected_mean < 0.01

def test_cross_greeks():
    calc_analytical = GreeksCalculator(method="analytical")
    calc_numerical = GreeksCalculator(method="numerical", h=1e-4)
    
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    greeks_a = calc_analytical.calculate(S0, K, T, r, sigma, option_type="call")
    greeks_n = calc_numerical.calculate(S0, K, T, r, sigma, option_type="call")
    
    # Vanna
    assert np.isclose(greeks_a["vanna"], greeks_n["vanna"], atol=1e-3, rtol=1e-2)
    
    # Volga
    assert np.isclose(greeks_a["volga"], greeks_n["volga"], atol=1e-3, rtol=1e-2)
    
    # Charm
    assert np.isclose(greeks_a["charm"], greeks_n["charm"], atol=1e-3, rtol=1e-2)
