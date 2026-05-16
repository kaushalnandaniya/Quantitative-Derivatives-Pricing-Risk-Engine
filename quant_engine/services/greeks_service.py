"""
Greeks Service
===============
Business logic layer for option Greeks computation.
"""

import time
import logging
from typing import Dict

import numpy as np

from pricing.greeks import GreeksCalculator

logger = logging.getLogger(__name__)


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    method: str = "analytical",
) -> Dict:
    """
    Compute all Greeks (Delta, Gamma, Vega, Theta) for an option.

    Args:
        S, K, T, r, sigma: Standard option parameters.
        option_type:        'call' or 'put'.
        method:             'analytical' or 'numerical'.

    Returns:
        Dictionary with Greeks values, inputs, and computation time.
    """
    t_start = time.perf_counter()

    calc = GreeksCalculator(method=method)
    greeks = calc.calculate(S, K, T, r, sigma, option_type)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Convert numpy scalars to Python floats
    result = {k: float(np.asarray(v)) for k, v in greeks.items()}

    logger.info(
        f"Greeks | {method} | {option_type} | S={S} K={K} → "
        f"Δ={result['delta']:.4f} Γ={result['gamma']:.6f} "
        f"V={result['vega']:.4f} Θ={result['theta']:.4f} ({elapsed_ms:.2f}ms)"
    )

    return {
        "greeks": result,
        "method": method,
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
