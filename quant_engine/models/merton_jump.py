"""
Merton Jump-Diffusion Model
===========================
Simulates terminal asset prices under the Merton jump-diffusion model.
Combines a continuous GBM with a Poisson-driven jump process.
"""

import numpy as np

def simulate_merton_terminal_price(
    S0: float, 
    r: float, 
    sigma: float, 
    lam: float, 
    mu_j: float, 
    sigma_j: float,
    T: float, 
    n_sims: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulates Merton jump-diffusion terminal prices.
    
    The asset price is given by:
    S_T = S_0 * exp((r - 0.5*sigma^2 - lam*k)*T + sigma*sqrt(T)*Z + \sum_{i=1}^{N_T} Y_i)
    where N_T ~ Poisson(lam*T), Y_i ~ N(mu_j, sigma_j^2), and k = exp(mu_j + 0.5*sigma_j^2) - 1
    
    Args:
        S0:      Initial asset price
        r:       Risk-free rate
        sigma:   Continuous volatility
        lam:     Jump intensity (expected number of jumps per year)
        mu_j:    Mean of jump size (in log-returns)
        sigma_j: Volatility of jump size
        T:       Time to maturity
        n_sims:  Number of simulated paths
        rng:     NumPy random Generator
        
    Returns:
        np.ndarray of terminal asset prices.
    """
    # k = E[exp(Y) - 1]
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    
    # Risk-neutral drift
    drift = r - 0.5 * sigma**2 - lam * k
    
    # Continuous diffusion component
    Z = rng.standard_normal(n_sims)
    continuous_log_return = drift * T + sigma * np.sqrt(T) * Z
    
    # Jump component
    # N_T is the number of jumps in [0, T] for each path
    N_T = rng.poisson(lam * T, n_sims)
    
    # Since sum of normals is normal: \sum_{i=1}^{N_T} Y_i ~ N(N_T * mu_j, N_T * sigma_j^2)
    jump_mean = N_T * mu_j
    jump_std = np.sqrt(N_T) * sigma_j
    jump_Z = rng.standard_normal(n_sims)
    jump_log_return = jump_mean + jump_std * jump_Z
    
    # Terminal price
    ST = S0 * np.exp(continuous_log_return + jump_log_return)
    
    return ST
