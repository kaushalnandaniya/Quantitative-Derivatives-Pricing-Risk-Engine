"""
Monte Carlo Pricing Engine
==========================
Risk-neutral Monte Carlo simulation for European option pricing.

Core formula (GBM terminal price under risk-neutral measure):
    S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z),  Z ~ N(0,1)

Option price = e^{-rT} * E[payoff(S_T)]

Features:
    - Fully vectorized (NumPy, zero Python loops)
    - Confidence intervals via CLT
    - Variance reduction: antithetic variates, control variates
    - Reproducible via explicit seeding
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict

from pricing.black_scholes import black_scholes_price
from models.gbm import simulate_terminal_price, simulate_terminal_price_antithetic

# --- Logging ---
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Payoff Computation
# =============================================================================

def compute_payoff(
    ST: np.ndarray,
    K: float,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute European option payoff at maturity.

    Args:
        ST:          Array of terminal prices.
        K:           Strike price.
        option_type: 'call' or 'put'.

    Returns:
        np.ndarray of payoffs: max(ST - K, 0) for calls, max(K - ST, 0) for puts.

    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    otype = option_type.lower()
    if otype == "call":
        return np.maximum(ST - K, 0.0)
    elif otype == "put":
        return np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# =============================================================================
# 2. Monte Carlo Pricer (with Confidence Intervals)
# =============================================================================

def monte_carlo_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_sims: int = 100_000,
    seed: int = 42,
    method: str = "standard",
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Price a European option via Monte Carlo simulation.

    Args:
        S0:               Spot price.
        K:                Strike price.
        T:                Time to maturity (years).
        r:                Risk-free rate.
        sigma:            Volatility.
        option_type:      'call' or 'put'.
        n_sims:           Number of simulations.
        seed:             Random seed for reproducibility.
        method:           Variance reduction method:
                          'standard'   — plain MC
                          'antithetic' — antithetic variates
                          'control'    — control variates (uses BS as control)
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        Dictionary with keys:
            'price':     Discounted expected payoff (the MC estimate).
            'ci_lower':  Lower bound of confidence interval.
            'ci_upper':  Upper bound of confidence interval.
            'std_error': Standard error of the estimate.
            'variance':  Sample variance of discounted payoffs.
            'n_sims':    Number of simulations used.
            'method':    Variance reduction method used.
    """
    rng = np.random.default_rng(seed)
    discount = np.exp(-r * T)

    # --- z-score for CI ---
    from scipy.stats import norm as sp_norm
    alpha = 1 - confidence_level
    z = sp_norm.ppf(1 - alpha / 2)

    logger.info(f"MC pricing | method={method} | n_sims={n_sims:,} | "
                f"S0={S0} K={K} T={T} r={r} σ={sigma} {option_type}")

    if method == "standard":
        ST = simulate_terminal_price(S0, r, sigma, T, n_sims, rng)
        payoffs = compute_payoff(ST, K, option_type)
        discounted = discount * payoffs

    elif method == "antithetic":
        ST_pos, ST_neg = simulate_terminal_price_antithetic(
            S0, r, sigma, T, n_sims, rng
        )
        payoff_pos = compute_payoff(ST_pos, K, option_type)
        payoff_neg = compute_payoff(ST_neg, K, option_type)
        # Antithetic estimator: average each pair, then average across pairs
        discounted = discount * 0.5 * (payoff_pos + payoff_neg)

    elif method == "control":
        # Control variate using Black-Scholes price of the option itself
        ST = simulate_terminal_price(S0, r, sigma, T, n_sims, rng)
        payoffs = compute_payoff(ST, K, option_type)
        discounted_raw = discount * payoffs

        # Control variate: Y = S_T, E[Y] = S0 * e^{rT}
        Y = ST
        EY = S0 * np.exp(r * T)

        # Optimal β = -Cov(X, Y) / Var(Y)
        cov_XY = np.cov(discounted_raw, Y)[0, 1]
        var_Y = np.var(Y, ddof=1)
        beta = -cov_XY / var_Y if var_Y > 1e-15 else 0.0

        discounted = discounted_raw + beta * (Y - EY)

        logger.info(f"Control variate | β={beta:.6f} | "
                    f"Var(raw)={np.var(discounted_raw):.6f} | "
                    f"Var(CV)={np.var(discounted):.6f}")
    else:
        raise ValueError(f"method must be 'standard', 'antithetic', or 'control', got '{method}'")

    # --- Statistics ---
    price = np.mean(discounted)
    variance = np.var(discounted, ddof=1)
    n = len(discounted)
    std_error = np.sqrt(variance / n)
    ci_lower = price - z * std_error
    ci_upper = price + z * std_error

    logger.info(f"MC result | price={price:.6f} | SE={std_error:.6f} | "
                f"CI=[{ci_lower:.6f}, {ci_upper:.6f}]")

    return {
        "price": float(price),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_error": float(std_error),
        "variance": float(variance),
        "n_sims": n,
        "method": method,
    }
