import pytest
import numpy as np
from models.svi import svi_variance, svi_implied_volatility
from risk.cva import compute_expected_exposure, calculate_cva

def test_svi_variance_positive():
    k = np.array([-0.5, 0.0, 0.5])
    a, b, rho, m, sigma = 0.04, 0.1, -0.4, 0.1, 0.1
    
    w = svi_variance(k, a, b, rho, m, sigma)
    assert np.all(w > 0), "Variance must be positive"
    
def test_svi_implied_vol():
    K = np.array([90, 100, 110])
    F = 100
    T = 1.0
    a, b, rho, m, sigma = 0.04, 0.1, -0.4, 0.1, 0.1
    
    iv = svi_implied_volatility(K, F, T, a, b, rho, m, sigma)
    assert np.all(iv > 0), "Implied Volatility must be positive"
    
def test_cva_expected_exposure():
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    time_grid, EE = compute_expected_exposure(
        S0, K, r, sigma, T, n_sims=1000, n_steps=10, option_type="call"
    )
    
    assert len(time_grid) == 11
    assert len(EE) == 11
    assert np.all(EE >= 0), "Expected Exposure must be non-negative"

def test_cva_calculation():
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    cva = calculate_cva(
        S0, K, r, sigma, T, hazard_rate=0.02, recovery_rate=0.4,
        n_sims=1000, n_steps=10, option_type="call"
    )
    
    assert cva >= 0.0, "CVA must be non-negative"
    assert cva < S0, "CVA should be less than the underlying value"
