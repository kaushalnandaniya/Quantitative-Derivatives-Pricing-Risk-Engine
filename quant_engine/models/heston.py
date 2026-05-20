"""
Heston Stochastic Volatility Model
==================================
Simulates terminal asset prices under the Heston model using the
Euler-Maruyama discretization scheme with full truncation for the variance process.
"""

import numpy as np
from numba import njit

@njit(fastmath=True)
def simulate_heston_terminal_price(
    S0: float, 
    v0: float, 
    r: float, 
    kappa: float, 
    theta: float, 
    xi: float,
    rho: float, 
    T: float, 
    n_steps: int, 
    n_sims: int
) -> np.ndarray:
    """
    Simulates Heston model paths and returns the terminal prices.
    Uses full truncation scheme to prevent negative variance.
    
    Args:
        S0:      Initial asset price
        v0:      Initial variance
        r:       Risk-free rate
        kappa:   Mean reversion speed
        theta:   Long-term mean variance
        xi:      Volatility of volatility (vol-of-vol)
        rho:     Correlation between asset and variance brownian motions
        T:       Time to maturity
        n_steps: Number of time steps for Euler-Maruyama
        n_sims:  Number of simulated paths
        
    Returns:
        np.ndarray of terminal asset prices.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_comp = np.sqrt(1.0 - rho**2)
    
    ST = np.empty(n_sims, dtype=np.float64)
    
    for i in range(n_sims):
        S = S0
        v = v0
        for t in range(n_steps):
            Z1 = np.random.standard_normal()
            Z2_indep = np.random.standard_normal()
            Z2 = rho * Z1 + rho_comp * Z2_indep
            
            v_plus = max(v, 0.0)
            
            S = S * np.exp((r - 0.5 * v_plus) * dt + np.sqrt(v_plus) * sqrt_dt * Z1)
            v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus) * sqrt_dt * Z2
            
        ST[i] = S
        
    return ST
