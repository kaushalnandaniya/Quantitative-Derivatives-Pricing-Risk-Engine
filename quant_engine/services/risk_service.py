"""
Risk Service
=============
Business logic layer for portfolio risk analysis.

Orchestrates:
    1. Portfolio construction from position inputs
    2. P&L simulation via Monte Carlo
    3. VaR and CVaR computation
    4. Summary statistics
"""

import time
import logging
from typing import Dict, List, Optional

import numpy as np

from risk.portfolio import Portfolio
from risk.pnl import simulate_portfolio_pnl
from risk.var import var as compute_var
from risk.cvar import cvar as compute_cvar

logger = logging.getLogger(__name__)


def compute_portfolio_risk(
    portfolio_positions: List[Dict],
    method: str = "historical",
    confidence: float = 0.95,
    n_sims: int = 100_000,
    horizon_days: int = 1,
    seed: int = 42,
    correlation_matrix: Optional[List[List[float]]] = None,
) -> Dict:
    """
    Full portfolio risk analysis pipeline.

    Steps:
        1. Build Portfolio from position dicts
        2. Simulate P&L distribution (Monte Carlo)
        3. Compute VaR and CVaR
        4. Gather summary statistics

    Returns:
        Dictionary with VaR, CVaR, P&L statistics, portfolio summary,
        and computation time.
    """
    t_start = time.perf_counter()

    # --- 1. Build Portfolio ---
    portfolio = Portfolio()
    for pos in portfolio_positions:
        portfolio.add_position(**pos)

    # --- 2. Convert correlation matrix ---
    corr = None
    if correlation_matrix is not None:
        corr = np.array(correlation_matrix, dtype=np.float64)

    # --- 3. Simulate P&L ---
    pnl_result = simulate_portfolio_pnl(
        portfolio,
        n_sims=n_sims,
        horizon_days=horizon_days,
        seed=seed,
        corr_matrix=corr,
    )
    pnl = pnl_result["pnl"]

    # --- 4. VaR & CVaR ---
    var_value = compute_var(pnl, confidence, method)
    cvar_value = compute_cvar(pnl, confidence, method)

    # --- 5. Summary statistics ---
    pnl_stats = {
        "mean": float(np.mean(pnl)),
        "std": float(np.std(pnl)),
        "min": float(np.min(pnl)),
        "max": float(np.max(pnl)),
        "median": float(np.median(pnl)),
        "skewness": float(_skewness(pnl)),
        "kurtosis": float(_kurtosis(pnl)),
    }

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"Portfolio risk | {portfolio.n_positions} positions | "
        f"{method} | {confidence:.0%} | VaR=${var_value:,.2f} CVaR=${cvar_value:,.2f} "
        f"({elapsed_ms:.0f}ms)"
    )

    return {
        "VaR": round(var_value, 4),
        "CVaR": round(cvar_value, 4),
        "method": method,
        "confidence": confidence,
        "pnl_statistics": pnl_stats,
        "portfolio": {
            "n_positions": portfolio.n_positions,
            "initial_value": pnl_result["V_0"],
            "positions": portfolio_positions,
        },
        "simulation": {
            "n_sims": n_sims,
            "horizon_days": horizon_days,
            "seed": seed,
        },
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _skewness(x: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - mu) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (s ** 4) - 3.0)
