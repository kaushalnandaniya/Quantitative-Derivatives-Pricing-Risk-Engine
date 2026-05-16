"""
Portfolio Greeks Service
=========================
Compute aggregated Greeks across all positions in a portfolio.

Returns per-position and total portfolio Delta, Gamma, Vega, Theta, Rho.
"""

import time
import logging
from typing import Dict, List

import numpy as np

from pricing.greeks import GreeksCalculator

logger = logging.getLogger(__name__)


def compute_portfolio_greeks(
    positions: List[Dict],
    method: str = "analytical",
) -> Dict:
    """
    Compute Greeks for each position and aggregate across the portfolio.

    Args:
        positions: List of position dicts with keys:
                   type, S, K, T, r, sigma, qty
        method:    'analytical' or 'numerical'.

    Returns:
        Dictionary with:
            'positions':  Per-position Greeks (qty-weighted)
            'totals':     Aggregated portfolio Greeks
            'method':     Calculation method used
            'elapsed_ms': Computation time
    """
    t_start = time.perf_counter()

    calc = GreeksCalculator(method=method)
    position_greeks = []

    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for i, pos in enumerate(positions):
        greeks = calc.calculate(
            S=pos["S"],
            K=pos["K"],
            T=pos["T"],
            r=pos.get("r", 0.05),
            sigma=pos.get("sigma", 0.2),
            option_type=pos["type"],
        )

        qty = pos.get("qty", 1)

        # Convert numpy scalars and apply quantity weighting
        weighted = {}
        for key in ["delta", "gamma", "vega", "theta", "rho"]:
            val = float(np.asarray(greeks[key]))
            weighted[key] = val * qty
            totals[key] += weighted[key]

        position_greeks.append({
            "index": i,
            "type": pos["type"],
            "S": pos["S"],
            "K": pos["K"],
            "T": pos["T"],
            "qty": qty,
            "greeks_per_unit": {k: float(np.asarray(greeks[k])) for k in greeks},
            "greeks_weighted": weighted,
        })

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"Portfolio Greeks | {len(positions)} positions | "
        f"Δ={totals['delta']:.4f} Γ={totals['gamma']:.6f} "
        f"V={totals['vega']:.4f} Θ={totals['theta']:.4f} ρ={totals['rho']:.4f} "
        f"({elapsed_ms:.2f}ms)"
    )

    return {
        "positions": position_greeks,
        "totals": {k: round(v, 6) for k, v in totals.items()},
        "method": method,
        "n_positions": len(positions),
        "elapsed_ms": round(elapsed_ms, 3),
    }
