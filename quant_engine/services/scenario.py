"""
Scenario Analysis Engine
==========================
Stress test portfolios under user-defined scenarios:
    - Spot price shifts
    - Volatility shifts (IV crush / spike)
    - Time decay (forward scenarios)
    - Interest rate changes
    - 2D heatmap grids (Spot × Vol, Spot × Time)
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional

from pricing.black_scholes import black_scholes_price
from pricing.greeks import GreeksCalculator

logger = logging.getLogger(__name__)


def stress_test(
    positions: List[Dict],
    spot_shifts: Optional[List[float]] = None,
    vol_shifts: Optional[List[float]] = None,
    time_shifts: Optional[List[float]] = None,
    rate_shifts: Optional[List[float]] = None,
) -> Dict:
    """
    Run stress test on a portfolio under multiple scenarios.

    Args:
        positions:   List of position dicts (type, S, K, T, r, sigma, qty).
        spot_shifts: Relative spot price changes (e.g., [-0.1, -0.05, 0, 0.05, 0.1]).
        vol_shifts:  Absolute vol changes (e.g., [-0.05, 0, 0.05, 0.1]).
        time_shifts: Days forward (e.g., [0, 1, 7, 30]).
        rate_shifts: Absolute rate changes (e.g., [-0.01, 0, 0.01]).

    Returns:
        Dictionary with scenario grid, base value, and per-scenario results.
    """
    t_start = time.perf_counter()

    if spot_shifts is None:
        spot_shifts = [-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20]
    if vol_shifts is None:
        vol_shifts = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    if time_shifts is None:
        time_shifts = [0, 1, 7, 14, 30]
    if rate_shifts is None:
        rate_shifts = [0.0]

    calc = GreeksCalculator(method="analytical")

    # Base portfolio value
    base_value = _portfolio_value(positions)

    scenarios = []
    for ds in spot_shifts:
        for dv in vol_shifts:
            for dt_days in time_shifts:
                for dr in rate_shifts:
                    shifted_positions = []
                    for pos in positions:
                        shifted_positions.append({
                            **pos,
                            "S": pos["S"] * (1 + ds),
                            "sigma": max(pos.get("sigma", 0.2) + dv, 0.01),
                            "T": max(pos.get("T", 1.0) - dt_days / 365.0, 1e-6),
                            "r": pos.get("r", 0.05) + dr,
                        })

                    scenario_value = _portfolio_value(shifted_positions)
                    pnl = scenario_value - base_value

                    # Compute Greeks for this scenario
                    greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
                    for sp in shifted_positions:
                        g = calc.calculate(sp["S"], sp["K"], sp["T"], sp["r"], sp["sigma"], sp["type"])
                        qty = sp.get("qty", 1)
                        for k in greeks:
                            greeks[k] += float(np.asarray(g[k])) * qty

                    scenarios.append({
                        "spot_shift": ds,
                        "vol_shift": dv,
                        "time_shift_days": dt_days,
                        "rate_shift": dr,
                        "portfolio_value": round(scenario_value, 4),
                        "pnl": round(pnl, 4),
                        "greeks": {k: round(v, 6) for k, v in greeks.items()},
                    })

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return {
        "base_value": round(base_value, 4),
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def generate_heatmap(
    positions: List[Dict],
    x_axis: str = "spot",
    y_axis: str = "vol",
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    n_points: int = 15,
) -> Dict:
    """
    Generate a 2D P&L heatmap grid.

    Args:
        positions: Portfolio positions.
        x_axis:    'spot' or 'time' (days forward).
        y_axis:    'vol' or 'time' (days forward).
        x_range:   [min, max] for x-axis. Spot: relative shifts. Time: days.
        y_range:   [min, max] for y-axis.
        n_points:  Grid resolution per axis.

    Returns:
        Dictionary with x_values, y_values, z_matrix (P&L grid).
    """
    t_start = time.perf_counter()
    base_value = _portfolio_value(positions)

    # Define axis ranges
    if x_axis == "spot":
        x_vals = np.linspace(x_range[0] if x_range else -0.15, x_range[1] if x_range else 0.15, n_points)
        x_labels = [f"{v*100:+.1f}%" for v in x_vals]
    else:  # time
        x_vals = np.linspace(x_range[0] if x_range else 0, x_range[1] if x_range else 30, n_points)
        x_labels = [f"{int(v)}d" for v in x_vals]

    if y_axis == "vol":
        y_vals = np.linspace(y_range[0] if y_range else -0.08, y_range[1] if y_range else 0.08, n_points)
        y_labels = [f"{v*100:+.1f}%" for v in y_vals]
    else:  # time
        y_vals = np.linspace(y_range[0] if y_range else 0, y_range[1] if y_range else 30, n_points)
        y_labels = [f"{int(v)}d" for v in y_vals]

    z_matrix = np.zeros((len(y_vals), len(x_vals)))

    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            shifted = []
            for pos in positions:
                s = pos["S"] * (1 + xv) if x_axis == "spot" else pos["S"]
                sigma = max(pos.get("sigma", 0.2) + (yv if y_axis == "vol" else 0), 0.01)
                T = pos.get("T", 1.0) - (xv / 365.0 if x_axis == "time" else 0) - (yv / 365.0 if y_axis == "time" else 0)
                T = max(T, 1e-6)
                shifted.append({**pos, "S": s, "sigma": sigma, "T": T})
            z_matrix[i, j] = _portfolio_value(shifted) - base_value

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return {
        "x_axis": x_axis, "y_axis": y_axis,
        "x_values": x_vals.tolist(), "y_values": y_vals.tolist(),
        "x_labels": x_labels, "y_labels": y_labels,
        "z_matrix": z_matrix.tolist(),
        "base_value": round(base_value, 4),
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _portfolio_value(positions: List[Dict]) -> float:
    """Compute total portfolio value from positions."""
    total = 0.0
    for pos in positions:
        price = float(black_scholes_price(
            pos["S"], pos["K"], pos.get("T", 1.0),
            pos.get("r", 0.05), pos.get("sigma", 0.2), pos["type"]
        ))
        total += price * pos.get("qty", 1)
    return total
