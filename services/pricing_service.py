"""
Pricing Service
================
Business logic layer for option pricing.

Delegates to core pricing modules and wraps results with
timing and metadata. API routes call these functions — never
the core modules directly.
"""

import time
import logging
from typing import Dict

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import monte_carlo_price
from pricing.binomial import binomial_price

logger = logging.getLogger(__name__)


def compute_black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> Dict:
    """
    Price a European option via Black-Scholes.

    Returns:
        Dictionary with price, inputs, and computation time.
    """
    t_start = time.perf_counter()

    price = float(black_scholes_price(S, K, T, r, sigma, option_type))

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"BS price | {option_type} | S={S} K={K} T={T} σ={sigma} → ${price:.6f} "
        f"({elapsed_ms:.2f}ms)"
    )

    return {
        "model": "black-scholes",
        "price": price,
        "inputs": {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
        },
        "elapsed_ms": round(elapsed_ms, 3),
    }


def compute_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_sims: int = 100_000,
    method: str = "standard",
    seed: int = 42,
) -> Dict:
    """
    Price a European option via Monte Carlo simulation.

    Returns:
        Dictionary with price, confidence interval, std error,
        variance, inputs, and computation time.
    """
    t_start = time.perf_counter()

    result = monte_carlo_price(
        S0=S, K=K, T=T, r=r, sigma=sigma,
        option_type=option_type,
        n_sims=n_sims, seed=seed, method=method,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"MC price | {method} | {option_type} | {n_sims:,} sims → "
        f"${result['price']:.6f} ± {result['std_error']:.6f} ({elapsed_ms:.2f}ms)"
    )

    return {
        "model": "monte-carlo",
        "price": result["price"],
        "confidence_interval": {
            "lower": result["ci_lower"],
            "upper": result["ci_upper"],
        },
        "std_error": result["std_error"],
        "variance": result["variance"],
        "inputs": {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "n_sims": result["n_sims"],
            "method": method,
            "seed": seed,
        },
        "elapsed_ms": round(elapsed_ms, 3),
    }


def compute_binomial(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    style: str = "european",
    N: int = 200,
) -> Dict:
    """
    Price an option via CRR binomial tree.

    Returns:
        Dictionary with price, tree parameters, inputs,
        and computation time.
    """
    t_start = time.perf_counter()

    result = binomial_price(
        S0=S, K=K, T=T, r=r, sigma=sigma,
        option_type=option_type, style=style, N=N,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Extract CRR parameters for response (if available)
    params = result.get("params")
    tree_params = None
    if params:
        tree_params = {
            "u": round(params["u"], 6),
            "d": round(params["d"], 6),
            "p": round(params["p"], 6),
            "dt": round(params["dt"], 6),
        }

    logger.info(
        f"Binomial price | {style} {option_type} | N={N} → "
        f"${result['price']:.6f} ({elapsed_ms:.2f}ms)"
    )

    return {
        "model": "binomial",
        "price": result["price"],
        "style": style,
        "steps": N,
        "tree_parameters": tree_params,
        "inputs": {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "style": style,
            "N": N,
        },
        "elapsed_ms": round(elapsed_ms, 3),
    }
