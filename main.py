#!/usr/bin/env python3
"""
Quant Pricing Engine — Entry Point
=====================================
Runs the full analysis pipeline for BS, MC, and Binomial pricing.

Usage:
    python -m main
    # or from project root:
    python main.py
"""

import sys
import os
import time
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_engine")

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import monte_carlo_price
from pricing.binomial import binomial_price, binomial_price_with_tree
from experiments.convergence_analysis import convergence_analysis
from experiments.variance_reduction import variance_reduction_comparison
from pricing.visualizations import (
    plot_convergence,
    plot_payoff_distribution,
    plot_confidence_intervals,
    plot_variance_reduction,
    plot_binomial_tree,
    plot_binomial_convergence,
    plot_early_exercise_boundary,
)
from config.settings import DEFAULT_PARAMS, MC_CONFIG, BINOMIAL_CONFIG


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

    # Create output directories
    mc_dir = os.path.join(os.path.dirname(__file__), "output", "mc_plots")
    bn_dir = os.path.join(os.path.dirname(__file__), "output", "binomial_plots")
    os.makedirs(mc_dir, exist_ok=True)
    os.makedirs(bn_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

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
    # 3. CONVERGENCE ANALYSIS (MC)
    # =====================================================================
    separator("Convergence Analysis (MC)")
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
    # 5. BINOMIAL TREE PRICING
    # =====================================================================
    separator("Binomial Tree Pricing")

    bn_steps = BINOMIAL_CONFIG["default_steps"]
    logger.info(f"  Using N = {bn_steps} steps\n")

    # European options
    for option_type in ["call", "put"]:
        bs = float(black_scholes_price(S0, K, T, r, sigma, option_type))
        euro = binomial_price(S0, K, T, r, sigma, option_type, "european", bn_steps)
        amer = binomial_price(S0, K, T, r, sigma, option_type, "american", bn_steps)
        premium = amer["price"] - euro["price"]

        logger.info(f"  {option_type.upper()}:")
        logger.info(f"    BS Reference = ${bs:.6f}")
        logger.info(f"    European     = ${euro['price']:.6f} (err = {euro['price']-bs:+.6f})")
        logger.info(f"    American     = ${amer['price']:.6f} (premium = {premium:+.6f})")
        logger.info(f"    Time         = {euro['elapsed_ms']:.2f}ms (euro) / {amer['elapsed_ms']:.2f}ms (amer)")
        logger.info("")

    # =====================================================================
    # 6. BINOMIAL CONVERGENCE
    # =====================================================================
    separator("Binomial Convergence Analysis")

    conv_steps = BINOMIAL_CONFIG["convergence_steps"]
    bn_conv = {"steps": conv_steps, "call_prices": [], "put_prices": [],
               "bs_call": bs_call, "bs_put": bs_put}

    logger.info(f"\n  {'N':>6s} | {'Call':>10s} | {'Call Err':>10s} | {'Put':>10s} | {'Put Err':>10s}")
    logger.info("  " + "-" * 55)
    for step_n in conv_steps:
        c = binomial_price(S0, K, T, r, sigma, "call", "european", step_n)
        p = binomial_price(S0, K, T, r, sigma, "put", "european", step_n)
        bn_conv["call_prices"].append(c["price"])
        bn_conv["put_prices"].append(p["price"])
        logger.info(
            f"  {step_n:>6d} | ${c['price']:>8.4f} | {c['price']-bs_call:>+10.6f} | "
            f"${p['price']:>8.4f} | {p['price']-bs_put:>+10.6f}"
        )

    # =====================================================================
    # 7. VISUALIZATION
    # =====================================================================
    separator("Generating Plots")

    # --- MC Plots ---
    fig1 = plot_convergence(conv_data,
                            save_path=os.path.join(mc_dir, "convergence.png"),
                            show=False)
    logger.info(f"  ✓ MC convergence plot saved")

    fig2 = plot_payoff_distribution(S0, K, T, r, sigma, n_sims=N, seed=SEED,
                                    save_path=os.path.join(mc_dir, "payoff_distribution.png"),
                                    show=False)
    logger.info(f"  ✓ Payoff distribution plot saved")

    fig3 = plot_confidence_intervals(conv_data, method="standard",
                                      save_path=os.path.join(mc_dir, "confidence_intervals.png"),
                                      show=False)
    logger.info(f"  ✓ Confidence interval plot saved")

    fig4 = plot_variance_reduction(comp,
                                    save_path=os.path.join(mc_dir, "variance_reduction.png"),
                                    show=False)
    logger.info(f"  ✓ Variance reduction plot saved")

    # --- Binomial Plots ---
    fig5 = plot_binomial_convergence(bn_conv,
                                     save_path=os.path.join(bn_dir, "convergence.png"),
                                     show=False)
    logger.info(f"  ✓ Binomial convergence plot saved")

    # Tree visualization (small N for readability)
    tree_data = binomial_price_with_tree(S0, K, T, r, sigma, "put", "american", N=8)
    fig6 = plot_binomial_tree(tree_data, max_display_steps=8,
                              save_path=os.path.join(bn_dir, "tree_american_put.png"),
                              show=False)
    logger.info(f"  ✓ Binomial tree plot saved")

    # Early exercise boundary (larger N for accuracy)
    boundary_data = binomial_price_with_tree(S0, K, T, r, sigma, "put", "american", N=200)
    fig7 = plot_early_exercise_boundary(boundary_data,
                                         save_path=os.path.join(bn_dir, "exercise_boundary.png"),
                                         show=False)
    logger.info(f"  ✓ Exercise boundary plot saved")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    separator("COMPLETE")
    for out_dir, label in [(mc_dir, "Monte Carlo"), (bn_dir, "Binomial")]:
        logger.info(f"  {label} plots saved to: {out_dir}/")
        for f in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, f)
            size = os.path.getsize(fpath) / 1024
            logger.info(f"    📊 {f} ({size:.0f} KB)")
    logger.info("")


if __name__ == "__main__":
    main()

