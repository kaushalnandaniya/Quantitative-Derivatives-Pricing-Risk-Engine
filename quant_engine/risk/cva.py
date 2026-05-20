"""
Credit Value Adjustment (CVA) Engine
======================================
Calculates Counterparty Credit Risk using Monte Carlo Expected Exposure.
CVA = (1 - R) * sum_i [ EE(t_i) * PD(t_i, t_{i+1}) * D(t_i) ]
"""

import numpy as np
from typing import Tuple
from scipy.stats import norm

def compute_expected_exposure(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int,
    n_steps: int,
    option_type: str = "call"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Expected Exposure (EE) profile over time using Monte Carlo paths.
    
    Returns:
        time_grid: np.ndarray of time points
        EE: np.ndarray of Expected Exposures at each time point
    """
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)
    
    rng = np.random.default_rng(42)
    ST = np.zeros((n_sims, n_steps + 1))
    ST[:, 0] = S0
    
    Z = rng.standard_normal((n_sims, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)
    
    # Vectorized path generation
    for t in range(n_steps):
        ST[:, t+1] = ST[:, t] * np.exp(drift + vol_term * Z[:, t])
        
    EE = np.zeros(n_steps + 1)
    
    for i, t in enumerate(time_grid):
        if t == T:
            if option_type.lower() == "call":
                V = np.maximum(ST[:, -1] - K, 0.0)
            else:
                V = np.maximum(K - ST[:, -1], 0.0)
        else:
            tau = T - t
            S_t = ST[:, i]
            tau_safe = max(tau, 1e-10)
            
            d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
            d2 = d1 - sigma * np.sqrt(tau_safe)
            
            if option_type.lower() == "call":
                V = S_t * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
            else:
                V = K * np.exp(-r * tau) * norm.cdf(-d2) - S_t * norm.cdf(-d1)
                
        # Expected Exposure is the expected value of max(V, 0)
        EE[i] = np.mean(np.maximum(V, 0.0))
        
    return time_grid, EE


def calculate_cva(
    S0: float, K: float, r: float, sigma: float, T: float,
    hazard_rate: float, recovery_rate: float, 
    n_sims: int = 10000, n_steps: int = 100, option_type: str = "call"
) -> float:
    """
    Calculate Counterparty Credit Value Adjustment (CVA).
    Assuming constant hazard rate.
    
    CVA = (1 - R) * sum_i [ EE(t_i) * PD(t_i-1, t_i) * D(t_i) ]
    """
    time_grid, EE = compute_expected_exposure(S0, K, r, sigma, T, n_sims, n_steps, option_type)
    
    cva = 0.0
    LGD = 1.0 - recovery_rate
    
    for i in range(1, len(time_grid)):
        t_prev = time_grid[i-1]
        t_curr = time_grid[i]
        
        # Marginal probability of default between t_prev and t_curr
        pd = np.exp(-hazard_rate * t_prev) - np.exp(-hazard_rate * t_curr)
        
        discount = np.exp(-r * t_curr)
        ee_mid = 0.5 * (EE[i] + EE[i-1])
        
        cva += LGD * ee_mid * pd * discount
        
    return cva
