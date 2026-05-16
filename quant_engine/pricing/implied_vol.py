"""
Implied Volatility Solver
===========================
Recovers implied volatility (IV) from observed market prices
using the Black-Scholes model.

Methods:
    1. Newton-Raphson — fast quadratic convergence using BS Vega
    2. Bisection fallback — guaranteed convergence for edge cases

IV is the σ that satisfies: BS(S, K, T, r, σ) = Market Price
"""

import logging
import numpy as np
from typing import Optional
from scipy.stats import norm

from pricing.black_scholes import black_scholes_price

logger = logging.getLogger(__name__)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes Vega (∂V/∂σ), used as Newton-Raphson derivative."""
    if T <= 1e-10 or sigma <= 1e-10:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(S * norm.pdf(d1) * sqrt_T)


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.25,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Solve for implied volatility using Newton-Raphson iteration.

    σ_{n+1} = σ_n - (BS(σ_n) - market_price) / Vega(σ_n)

    Args:
        market_price: Observed option market price.
        S:            Spot price.
        K:            Strike price.
        T:            Time to maturity (years).
        r:            Risk-free rate.
        option_type:  'call' or 'put'.
        initial_guess: Starting σ estimate.
        tol:          Convergence tolerance.
        max_iter:     Maximum iterations.

    Returns:
        Implied volatility (float), or None if convergence fails.
    """
    sigma = initial_guess

    for i in range(max_iter):
        bs_price = float(black_scholes_price(S, K, T, r, sigma, option_type))
        diff = bs_price - market_price
        vega = _bs_vega(S, K, T, r, sigma)

        if abs(diff) < tol:
            logger.debug(f"IV Newton converged in {i+1} iters: σ={sigma:.6f}")
            return float(sigma)

        if abs(vega) < 1e-12:
            # Vega too small — Newton step would explode, fall back
            logger.debug(f"IV Newton: Vega near zero at σ={sigma:.4f}, switching to bisection")
            return None

        sigma -= diff / vega

        # Keep sigma in reasonable bounds
        sigma = max(sigma, 1e-6)
        sigma = min(sigma, 10.0)

    logger.warning(f"IV Newton failed to converge after {max_iter} iters (σ={sigma:.4f})")
    return None


def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Optional[float]:
    """
    Solve for implied volatility using bisection method.

    Guaranteed convergence (slower than Newton but robust).

    Args:
        market_price: Observed option market price.
        S, K, T, r:  Standard BS parameters.
        option_type:  'call' or 'put'.
        sigma_low:    Lower bound for σ search.
        sigma_high:   Upper bound for σ search.
        tol:          Convergence tolerance.
        max_iter:     Maximum iterations.

    Returns:
        Implied volatility (float), or None if no solution in bounds.
    """
    f_low = float(black_scholes_price(S, K, T, r, sigma_low, option_type)) - market_price
    f_high = float(black_scholes_price(S, K, T, r, sigma_high, option_type)) - market_price

    # Check that the root is bracketed
    if f_low * f_high > 0:
        logger.warning(
            f"IV bisection: root not bracketed in [{sigma_low}, {sigma_high}]. "
            f"f_low={f_low:.4f}, f_high={f_high:.4f}"
        )
        return None

    for i in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        f_mid = float(black_scholes_price(S, K, T, r, sigma_mid, option_type)) - market_price

        if abs(f_mid) < tol or (sigma_high - sigma_low) < tol:
            logger.debug(f"IV bisection converged in {i+1} iters: σ={sigma_mid:.6f}")
            return float(sigma_mid)

        if f_low * f_mid < 0:
            sigma_high = sigma_mid
            f_high = f_mid
        else:
            sigma_low = sigma_mid
            f_low = f_mid

    sigma_mid = 0.5 * (sigma_low + sigma_high)
    logger.warning(f"IV bisection max iter reached: σ≈{sigma_mid:.6f}")
    return float(sigma_mid)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """
    Compute implied volatility with automatic method selection.

    Tries Newton-Raphson first (fast), falls back to bisection (robust).

    Args:
        market_price: Observed option market price.
        S, K, T, r:  Standard BS parameters.
        option_type:  'call' or 'put'.

    Returns:
        Implied volatility as a float.

    Raises:
        ValueError: If IV cannot be solved (price below intrinsic or
                    outside feasible range).
    """
    # Validate: market price must exceed intrinsic value
    if option_type.lower() == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - 1e-6:
        raise ValueError(
            f"Market price ({market_price:.4f}) is below intrinsic value "
            f"({intrinsic:.4f}) — no valid IV exists"
        )

    if market_price < 1e-10:
        raise ValueError("Market price is effectively zero — cannot solve for IV")

    if T < 1e-10:
        raise ValueError("Time to maturity is effectively zero — cannot solve for IV")

    # Try Newton-Raphson first
    iv = implied_volatility_newton(market_price, S, K, T, r, option_type)

    if iv is not None and iv > 0:
        return iv

    # Fall back to bisection
    iv = implied_volatility_bisection(market_price, S, K, T, r, option_type)

    if iv is not None and iv > 0:
        return iv

    raise ValueError(
        f"Could not solve for IV: market_price={market_price}, "
        f"S={S}, K={K}, T={T}, r={r}, type={option_type}"
    )
