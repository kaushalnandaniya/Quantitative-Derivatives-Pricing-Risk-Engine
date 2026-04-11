#!/usr/bin/env python3
"""
Monte Carlo Pricing Engine — Entry Point
==========================================
Runs the full analysis pipeline and generates all visualizations.

Usage:
    python -m main
    # or from project root:
    python main.py
"""

import sys
import os
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mc_demo")

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import monte_carlo_price
from experiments.convergence_analysis import convergence_analysis
from experiments.variance_reduction import variance_reduction_comparison
from pricing.visualizations import (
    plot_convergence,
    plot_payoff_distribution,
    plot_confidence_intervals,
    plot_variance_reduction,
)
from config.settings import DEFAULT_PARAMS, MC_CONFIG


def separator(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def main():
    # --- Parameters ---
    S0 = DEFAULT_PARAMS["S0"]
    K = DEFAULT_PARAMS["K"]
    T = DEFAULT_PARAMS["T"]
    r = DEFAULT_PARAMS["r"]
    sigma = DEFAULT_PARAMS["sigma"]
    N = MC_CONFIG["n_sims"]
    SEED = MC_CONFIG["seed"]

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), "output", "mc_plots")
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # 1. BLACK-SCHOLES REFERENCE
    # =====================================================================
    separator("Black-Scholes Reference Prices")
    bs_call = float(black_scholes_price(S0, K, T, r, sigma, "call"))
    bs_put = float(black_scholes_price(S0, K, T, r, sigma, "put"))
    logger.info(f"  BS Call = ${bs_call:.6f}")
    logger.info(f"  BS Put  = ${bs_put:.6f}")
    logger.info(f"  Put-Call Parity: C - P = ${bs_call - bs_put:.6f}")
    logger.info(f"  S - K*exp(-rT) = ${S0 - K * np.exp(-r * T):.6f}")

    # =====================================================================
    # 2. MONTE CARLO PRICING — ALL METHODS
    # =====================================================================
    separator("Monte Carlo Pricing (N={:,})".format(N))

    for option_type in ["call", "put"]:
        bs = float(black_scholes_price(S0, K, T, r, sigma, option_type))
        logger.info(f"\n  --- {option_type.upper()} (BS = {bs:.6f}) ---")

        for method in ["standard", "antithetic", "control"]:
            result = monte_carlo_price(
                S0, K, T, r, sigma, option_type,
                n_sims=N, seed=SEED, method=method,
            )
            error = result["price"] - bs
            logger.info(
                f"  {method:12s} | Price = ${result['price']:.6f} | "
                f"Error = {error:+.6f} | "
                f"SE = {result['std_error']:.6f} | "
                f"CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            )

    # =====================================================================
    # 3. CONVERGENCE ANALYSIS
    # =====================================================================
    separator("Convergence Analysis")
    sim_sizes = MC_CONFIG["convergence_sim_sizes"]
    conv_data = convergence_analysis(
        S0, K, T, r, sigma, "call",
        sim_sizes=sim_sizes, seed=SEED,
        methods=["standard", "antithetic", "control"],
    )

    logger.info(f"\n  {'N':>10s} | {'Standard':>12s} | {'Antithetic':>12s} | {'Control':>12s} | {'BS':>10s}")
    logger.info("  " + "-" * 68)
    for i, n in enumerate(sim_sizes):
        std_p = conv_data["results"]["standard"][i]["price"]
        ant_p = conv_data["results"]["antithetic"][i]["price"]
        cv_p = conv_data["results"]["control"][i]["price"]
        logger.info(f"  {n:>10,d} | ${std_p:>10.4f} | ${ant_p:>10.4f} | ${cv_p:>10.4f} | ${conv_data['bs_price']:>8.4f}")

    # =====================================================================
    # 4. VARIANCE REDUCTION COMPARISON
    # =====================================================================
    separator("Variance Reduction Comparison")
    comp = variance_reduction_comparison(S0, K, T, r, sigma, "call", n_sims=N, seed=SEED)

    logger.info(f"\n  {'Method':>12s} | {'Price':>10s} | {'Variance':>12s} | {'Std Error':>10s} | {'VR Ratio':>10s}")
    logger.info("  " + "-" * 62)
    for method in ["standard", "antithetic", "control"]:
        d = comp[method]
        vr = d.get("variance_reduction_ratio")
        vr_str = f"{vr:.2f}×" if vr else "1.00×"
        logger.info(
            f"  {method:>12s} | ${d['price']:>8.4f} | {d['variance']:>12.6f} | "
            f"{d['std_error']:>10.6f} | {vr_str:>10s}"
        )

    # =====================================================================
    # 5. VISUALIZATION
    # =====================================================================
    separator("Generating Plots")

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    # 5a. Convergence
    fig1 = plot_convergence(conv_data,
                            save_path=os.path.join(output_dir, "convergence.png"),
                            show=False)
    logger.info(f"  ✓ Convergence plot saved")

    # 5b. Payoff distributions
    fig2 = plot_payoff_distribution(S0, K, T, r, sigma, n_sims=N, seed=SEED,
                                    save_path=os.path.join(output_dir, "payoff_distribution.png"),
                                    show=False)
    logger.info(f"  ✓ Payoff distribution plot saved")

    # 5c. Confidence intervals
    fig3 = plot_confidence_intervals(conv_data, method="standard",
                                      save_path=os.path.join(output_dir, "confidence_intervals.png"),
                                      show=False)
    logger.info(f"  ✓ Confidence interval plot saved")

    # 5d. Variance reduction comparison
    fig4 = plot_variance_reduction(comp,
                                    save_path=os.path.join(output_dir, "variance_reduction.png"),
                                    show=False)
    logger.info(f"  ✓ Variance reduction plot saved")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    separator("COMPLETE")
    logger.info(f"  All plots saved to: {output_dir}/")
    logger.info(f"  Files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath) / 1024
        logger.info(f"    📊 {f} ({size:.0f} KB)")
    logger.info("")


if __name__ == "__main__":
    main()
