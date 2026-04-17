"""
Risk Analysis Experiment
=========================
Standalone risk analysis: VaR sensitivity, diversification
benefit, and correlation impact studies.
"""

import logging
import numpy as np
from typing import Dict, List

from risk.portfolio import Portfolio
from risk.pnl import simulate_portfolio_pnl
from risk.var import var, historical_var, parametric_var
from risk.cvar import cvar

logger = logging.getLogger(__name__)


def volatility_sensitivity(
    base_sigma: float = 0.2,
    sigmas: List[float] = None,
    n_sims: int = 100_000,
    seed: int = 42,
) -> Dict:
    """
    Measure how VaR/CVaR change with increasing volatility.

    Returns dict mapping sigma -> {var_95, cvar_95, var_99, cvar_99}.
    """
    if sigmas is None:
        sigmas = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60]

    results = {}
    for sigma in sigmas:
        port = Portfolio()
        port.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=sigma, qty=10,
        )

        sim = simulate_portfolio_pnl(port, n_sims=n_sims, horizon_days=1, seed=seed)
        pnl = sim["pnl"]

        results[sigma] = {
            "var_95": var(pnl, 0.95, "historical"),
            "cvar_95": cvar(pnl, 0.95),
            "var_99": var(pnl, 0.99, "historical"),
            "cvar_99": cvar(pnl, 0.99),
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl": float(np.std(pnl)),
            "V_0": sim["V_0"],
        }

    return results


def diversification_analysis(
    n_sims: int = 100_000,
    seed: int = 42,
) -> Dict:
    """
    Compare VaR of concentrated vs diversified portfolios.

    Tests:
        1. Single asset (20 calls on A)
        2. Two uncorrelated assets (10 each on A, B)
        3. Two correlated assets (ρ=0.8)
        4. Two anti-correlated assets (ρ=-0.5)
    """
    results = {}

    # 1. Concentrated
    port_single = Portfolio()
    port_single.add_position(
        type="call", S=100, K=100, T=0.25, r=0.05,
        sigma=0.3, qty=20, asset="A",
    )
    sim = simulate_portfolio_pnl(port_single, n_sims=n_sims, seed=seed)
    results["concentrated"] = {
        "var_95": historical_var(sim["pnl"], 0.95),
        "cvar_95": cvar(sim["pnl"], 0.95),
        "V_0": sim["V_0"],
        "description": "20 calls on single asset A",
    }

    # Helper: 2-asset portfolio
    def _two_asset_portfolio(rho, label):
        port = Portfolio()
        port.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=0.3, qty=10, asset="A",
        )
        port.add_position(
            type="call", S=100, K=100, T=0.25, r=0.05,
            sigma=0.3, qty=10, asset="B",
        )
        corr = np.array([[1.0, rho], [rho, 1.0]])
        sim = simulate_portfolio_pnl(port, n_sims=n_sims, seed=seed, corr_matrix=corr)
        return {
            "var_95": historical_var(sim["pnl"], 0.95),
            "cvar_95": cvar(sim["pnl"], 0.95),
            "V_0": sim["V_0"],
            "description": label,
            "correlation": rho,
        }

    results["uncorrelated"] = _two_asset_portfolio(0.0, "10 calls each A,B (ρ=0)")
    results["correlated"] = _two_asset_portfolio(0.8, "10 calls each A,B (ρ=0.8)")
    results["anti_correlated"] = _two_asset_portfolio(-0.5, "10 calls each A,B (ρ=-0.5)")

    return results


def run_risk_analysis():
    """Run full risk analysis and log results."""
    logger.info("=" * 70)
    logger.info("  RISK ANALYSIS EXPERIMENT")
    logger.info("=" * 70)

    # --- Volatility Sensitivity ---
    logger.info("\n--- Volatility Sensitivity ---")
    vol_results = volatility_sensitivity()

    logger.info(
        f"\n  {'σ':>6s} | {'VaR(95%)':>10s} | {'CVaR(95%)':>10s} | "
        f"{'VaR(99%)':>10s} | {'CVaR(99%)':>10s} | {'V₀':>10s}"
    )
    logger.info("  " + "-" * 70)
    for sigma, metrics in sorted(vol_results.items()):
        logger.info(
            f"  {sigma:>6.0%} | ${metrics['var_95']:>8.2f} | "
            f"${metrics['cvar_95']:>8.2f} | ${metrics['var_99']:>8.2f} | "
            f"${metrics['cvar_99']:>8.2f} | ${metrics['V_0']:>8.2f}"
        )

    # --- Diversification ---
    logger.info("\n--- Diversification Benefit ---")
    div_results = diversification_analysis()

    logger.info(
        f"\n  {'Portfolio':<20s} | {'VaR(95%)':>10s} | {'CVaR(95%)':>10s} | {'V₀':>10s}"
    )
    logger.info("  " + "-" * 60)
    for name, metrics in div_results.items():
        logger.info(
            f"  {name:<20s} | ${metrics['var_95']:>8.2f} | "
            f"${metrics['cvar_95']:>8.2f} | ${metrics['V_0']:>8.2f}"
        )

    # Diversification benefit
    conc_var = div_results["concentrated"]["var_95"]
    uncorr_var = div_results["uncorrelated"]["var_95"]
    benefit = (conc_var - uncorr_var) / conc_var * 100

    logger.info(f"\n  Diversification benefit (uncorrelated): {benefit:+.1f}% VaR reduction")

    return {"volatility": vol_results, "diversification": div_results}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_risk_analysis()
