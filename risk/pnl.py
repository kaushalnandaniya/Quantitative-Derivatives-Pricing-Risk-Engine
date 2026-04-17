"""
P&L Simulation Engine
=======================
Simulate portfolio profit & loss distributions via Monte Carlo.

Pipeline:
    1. Value portfolio at current spots → V_0
    2. Simulate terminal prices via GBM (single or correlated multi-asset)
    3. Re-price portfolio at simulated spots with reduced T → V_T
    4. P&L = V_T - V_0  (array of n_sims scenarios)

The full distribution — not just the mean — is stored for downstream
risk metrics (VaR, CVaR).
"""

import logging
import numpy as np
from typing import Dict, Optional

from risk.portfolio import Portfolio
from risk.correlation import simulate_correlated_gbm
from models.gbm import simulate_terminal_price

logger = logging.getLogger(__name__)


def simulate_portfolio_pnl(
    portfolio: Portfolio,
    n_sims: int = 100_000,
    horizon_days: int = 1,
    seed: int = 42,
    corr_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """
    Simulate the full P&L distribution of a portfolio.

    Args:
        portfolio:     Portfolio object with positions.
        n_sims:        Number of Monte Carlo scenarios.
        horizon_days:  Risk horizon in trading days (252 days/year).
        seed:          Random seed for reproducibility.
        corr_matrix:   (n_assets, n_assets) correlation matrix.
                       If None and portfolio has >1 asset, uses identity
                       (uncorrelated). For single-asset, ignored.

    Returns:
        Dictionary with:
            'pnl':          np.ndarray of shape (n_sims,) — P&L per scenario
            'V_0':          float — initial portfolio value
            'V_T':          np.ndarray of shape (n_sims,) — terminal values
            'S_T':          dict mapping asset -> simulated terminal prices
            'horizon_days': int
            'n_sims':       int
    """
    if portfolio.n_positions == 0:
        raise ValueError("Portfolio has no positions")

    rng = np.random.default_rng(seed)
    T_horizon = horizon_days / 252.0  # horizon in years

    assets = portfolio.get_unique_assets()
    positions = portfolio.positions

    # -----------------------------------------------------------------
    # 1. Current portfolio value (V_0)
    # -----------------------------------------------------------------
    V_0 = float(np.sum(portfolio.value()))

    logger.info(f"P&L simulation | V_0 = ${V_0:,.2f} | {n_sims:,} sims | "
                f"horizon = {horizon_days}d")

    # -----------------------------------------------------------------
    # 2. Simulate terminal prices per asset
    # -----------------------------------------------------------------
    S_T_dict = {}

    if len(assets) == 1:
        # Single asset — simple GBM
        asset = assets[0]
        asset_positions = [p for p in positions if p["asset"] == asset]
        S0 = asset_positions[0]["S"]
        r = asset_positions[0]["r"]
        sigma = asset_positions[0]["sigma"]

        S_T = simulate_terminal_price(S0, r, sigma, T_horizon, n_sims, rng)
        S_T_dict[asset] = S_T

    else:
        # Multi-asset — correlated GBM
        # Gather per-asset parameters (use first position for each asset)
        asset_params = {}
        for asset in assets:
            asset_pos = [p for p in positions if p["asset"] == asset]
            asset_params[asset] = {
                "S0": asset_pos[0]["S"],
                "r": asset_pos[0]["r"],
                "sigma": asset_pos[0]["sigma"],
            }

        spots = np.array([asset_params[a]["S0"] for a in assets])
        rates = np.array([asset_params[a]["r"] for a in assets])
        sigmas = np.array([asset_params[a]["sigma"] for a in assets])

        n_assets = len(assets)

        # Default to identity (uncorrelated) if no correlation provided
        if corr_matrix is None:
            corr_matrix = np.eye(n_assets)
            logger.info("No correlation matrix provided — using identity (uncorrelated)")

        # Simulate correlated terminal prices
        S_T_all = simulate_correlated_gbm(
            spots, rates, sigmas, T_horizon, n_sims, corr_matrix, rng
        )

        for i, asset in enumerate(assets):
            S_T_dict[asset] = S_T_all[i]

    # -----------------------------------------------------------------
    # 3. Re-price portfolio at simulated spots (with time decay)
    # -----------------------------------------------------------------
    V_T = portfolio.value(
        spot_overrides=S_T_dict,
        T_offset=T_horizon,
    )

    # Ensure array output
    V_T = np.asarray(V_T, dtype=np.float64).ravel()

    # -----------------------------------------------------------------
    # 4. P&L
    # -----------------------------------------------------------------
    pnl = V_T - V_0

    logger.info(
        f"P&L stats | mean={np.mean(pnl):+,.2f} | std={np.std(pnl):,.2f} | "
        f"min={np.min(pnl):+,.2f} | max={np.max(pnl):+,.2f}"
    )

    return {
        "pnl": pnl,
        "V_0": V_0,
        "V_T": V_T,
        "S_T": S_T_dict,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
    }
