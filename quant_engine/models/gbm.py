"""
Geometric Brownian Motion (GBM) Simulator
==========================================
Risk-neutral GBM terminal price simulation for European option pricing.

Core formula:
    S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z),  Z ~ N(0,1)

Features:
    - Fully vectorized (NumPy, zero Python loops)
    - Standard and antithetic variate simulation
    - Reproducible via explicit seeding
"""

import numpy as np


def simulate_terminal_price(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate terminal stock prices under risk-neutral GBM (direct formula).

    Uses the exact solution: S_T = S0 * exp((r - 0.5*σ²)*T + σ*√T*Z)
    No path discretization — single-step, numerically exact for European payoffs.

    Args:
        S0:     Initial spot price.
        r:      Risk-free rate (annualized, continuous compounding).
        sigma:  Volatility (annualized).
        T:      Time to maturity (years).
        n_sims: Number of Monte Carlo paths.
        rng:    NumPy random Generator for reproducibility.

    Returns:
        np.ndarray of shape (n_sims,) — simulated terminal prices.
    """
    Z = rng.standard_normal(n_sims)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    return ST


def simulate_terminal_price_antithetic(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Simulate terminal prices using antithetic variates for variance reduction.

    For each Z_i, we also use -Z_i, giving 2*n_sims total paths but with
    much lower variance due to negative correlation between paired paths.

    Returns a tuple (ST_pos, ST_neg) of n_sims//2 paired prices.
    """
    half = n_sims // 2
    Z = rng.standard_normal(half)
    drift = (r - 0.5 * sigma**2) * T
    vol_term = sigma * np.sqrt(T)

    ST_pos = S0 * np.exp(drift + vol_term * Z)
    ST_neg = S0 * np.exp(drift + vol_term * (-Z))

    return ST_pos, ST_neg
