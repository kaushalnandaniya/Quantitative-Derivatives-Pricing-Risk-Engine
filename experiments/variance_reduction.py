"""
Variance Reduction Comparison
==============================
Compare Standard MC, Antithetic Variates, and Control Variates side by side.
"""

import logging
import numpy as np
from typing import Dict

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import monte_carlo_price

logger = logging.getLogger(__name__)


def variance_reduction_comparison(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_sims: int = 100_000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compare all three methods side by side for fixed n_sims.

    Returns dict mapping method name -> MC result dict.
    Also includes 'bs_price' key with the analytical reference.
    """
    bs_price = float(black_scholes_price(S0, K, T, r, sigma, option_type))
    comparison = {"bs_price": bs_price}

    for method in ["standard", "antithetic", "control"]:
        res = monte_carlo_price(
            S0, K, T, r, sigma, option_type,
            n_sims=n_sims, seed=seed, method=method
        )
        res["error"] = res["price"] - bs_price
        res["variance_reduction_ratio"] = None  # filled below
        comparison[method] = res

    # Compute variance reduction ratios relative to standard
    std_var = comparison["standard"]["variance"]
    for method in ["antithetic", "control"]:
        if std_var > 1e-15:
            comparison[method]["variance_reduction_ratio"] = (
                std_var / comparison[method]["variance"]
            )

    return comparison
