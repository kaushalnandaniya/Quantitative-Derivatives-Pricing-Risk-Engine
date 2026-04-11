"""
Convergence Analysis
=====================
Study Monte Carlo price convergence to Black-Scholes as simulation count increases.
"""

import logging
import numpy as np
from typing import Optional, Dict

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import monte_carlo_price

logger = logging.getLogger(__name__)


def convergence_analysis(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    sim_sizes: Optional[list] = None,
    seed: int = 42,
    methods: Optional[list] = None,
) -> Dict[str, list]:
    """
    Run MC pricing across increasing simulation sizes to study convergence.

    Args:
        S0, K, T, r, sigma, option_type: Standard option parameters.
        sim_sizes: List of simulation counts, e.g. [100, 1000, 10000, 100000].
        seed:      Random seed (same for each run for fair comparison).
        methods:   List of methods to compare. Default: ['standard', 'antithetic', 'control'].

    Returns:
        Dictionary with structure:
        {
            'sim_sizes': [...],
            'bs_price': float,
            'results': {
                'standard':   [{'price': ..., 'ci_lower': ..., 'ci_upper': ..., ...}, ...],
                'antithetic': [...],
                'control':    [...],
            }
        }
    """
    if sim_sizes is None:
        sim_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    if methods is None:
        methods = ["standard", "antithetic", "control"]

    # True value from Black-Scholes
    bs_price = float(black_scholes_price(S0, K, T, r, sigma, option_type))
    logger.info(f"Convergence analysis | BS reference = {bs_price:.6f}")

    results = {m: [] for m in methods}
    for method in methods:
        for n in sim_sizes:
            res = monte_carlo_price(
                S0, K, T, r, sigma, option_type,
                n_sims=n, seed=seed, method=method
            )
            res["error"] = res["price"] - bs_price
            res["abs_error"] = abs(res["error"])
            results[method].append(res)

    return {
        "sim_sizes": sim_sizes,
        "bs_price": bs_price,
        "results": results,
    }
