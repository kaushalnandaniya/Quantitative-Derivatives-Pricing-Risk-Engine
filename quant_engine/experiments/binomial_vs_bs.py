"""
Binomial vs Black-Scholes Comparison
======================================
Complete analysis script comparing Binomial Tree, Black-Scholes, and Monte Carlo.

Generates:
    1. Convergence plot  — Binomial price vs N steps
    2. Error analysis    — |Binomial - BS| vs N
    3. Speed benchmark   — Timing all three models
    4. American vs Euro  — Early exercise premium
    5. Exercise boundary — Critical stock price for American puts

Usage:
    python -m experiments.binomial_vs_bs
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pricing.black_scholes import black_scholes_price
from pricing.binomial import binomial_price, binomial_price_with_tree
from pricing.monte_carlo import monte_carlo_price
from config.settings import DEFAULT_PARAMS, PLOT_STYLE, COLORS, BINOMIAL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("binomial_analysis")


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def separator(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


# =============================================================================
# 1. Convergence Analysis
# =============================================================================

def convergence_analysis(S0, K, T, r, sigma, output_dir):
    """Binomial price vs N steps, showing convergence to BS."""
    separator("Convergence Analysis — Binomial → Black-Scholes")

    steps = BINOMIAL_CONFIG["convergence_steps"]
    bs_call = float(black_scholes_price(S0, K, T, r, sigma, "call"))
    bs_put = float(black_scholes_price(S0, K, T, r, sigma, "put"))

    call_prices = []
    put_prices = []
    for N in steps:
        c = binomial_price(S0, K, T, r, sigma, "call", "european", N)
        p = binomial_price(S0, K, T, r, sigma, "put", "european", N)
        call_prices.append(c["price"])
        put_prices.append(p["price"])
        logger.info(
            f"  N={N:>5d} | Call = {c['price']:.6f} (err={c['price']-bs_call:+.6f}) | "
            f"Put = {p['price']:.6f} (err={p['price']-bs_put:+.6f})"
        )

    # --- Plot ---
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Price vs N
    ax1.plot(steps, call_prices, 'o-', color=COLORS["binomial"], linewidth=2,
             markersize=5, label="Binomial Call", alpha=0.9)
    ax1.plot(steps, put_prices, 's-', color=COLORS["binomial_put"], linewidth=2,
             markersize=5, label="Binomial Put", alpha=0.9)
    ax1.axhline(y=bs_call, color=COLORS["bs_ref"], linestyle="--", linewidth=2,
                label=f"BS Call = {bs_call:.4f}", alpha=0.8)
    ax1.axhline(y=bs_put, color=COLORS["antithetic"], linestyle="--", linewidth=2,
                label=f"BS Put = {bs_put:.4f}", alpha=0.8)
    ax1.set_xlabel("Number of Steps (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Option Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title("Binomial Tree Convergence", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: Absolute error vs N
    call_errors = [abs(cp - bs_call) for cp in call_prices]
    put_errors = [abs(pp - bs_put) for pp in put_prices]
    ax2.plot(steps, call_errors, 'o-', color=COLORS["binomial"], linewidth=2,
             markersize=5, label="Call Error", alpha=0.9)
    ax2.plot(steps, put_errors, 's-', color=COLORS["binomial_put"], linewidth=2,
             markersize=5, label="Put Error", alpha=0.9)

    # O(1/N) reference
    ref_err = call_errors[0]
    ref_n = steps[0]
    theoretical = [ref_err * ref_n / n for n in steps]
    ax2.plot(steps, theoretical, ':', color="#8b949e", linewidth=1.5,
             label=r"$O(1/N)$ reference", alpha=0.7)

    ax2.set_xlabel("Number of Steps (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Absolute Error ($)", fontsize=12, fontweight="bold")
    ax2.set_title("Error Convergence", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Binomial Tree — Convergence to Black-Scholes",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "binomial_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path}")

    return {"steps": steps, "call_prices": call_prices, "put_prices": put_prices,
            "bs_call": bs_call, "bs_put": bs_put}


# =============================================================================
# 2. Speed Benchmark — BS vs MC vs Binomial
# =============================================================================

def speed_benchmark(S0, K, T, r, sigma, output_dir):
    """Time all three pricing models and compare speed vs accuracy."""
    separator("Speed Benchmark — BS vs MC vs Binomial")

    bs_call = float(black_scholes_price(S0, K, T, r, sigma, "call"))

    results = []

    # Black-Scholes
    t0 = time.perf_counter()
    for _ in range(1000):
        black_scholes_price(S0, K, T, r, sigma, "call")
    bs_time = (time.perf_counter() - t0) / 1000 * 1000  # ms
    results.append(("Black-Scholes", bs_call, 0.0, bs_time))

    # Monte Carlo — various N
    for n_sims in [10_000, 100_000, 500_000]:
        t0 = time.perf_counter()
        mc = monte_carlo_price(S0, K, T, r, sigma, "call", n_sims=n_sims, seed=42)
        mc_time = (time.perf_counter() - t0) * 1000
        mc_err = abs(mc["price"] - bs_call)
        results.append((f"MC (N={n_sims:,})", mc["price"], mc_err, mc_time))

    # Binomial — various steps
    for N in [50, 200, 500, 1000, 2000]:
        t0 = time.perf_counter()
        bn = binomial_price(S0, K, T, r, sigma, "call", "european", N)
        bn_time = (time.perf_counter() - t0) * 1000
        bn_err = abs(bn["price"] - bs_call)
        results.append((f"Binomial (N={N})", bn["price"], bn_err, bn_time))

    # Print table
    logger.info(f"\n  {'Model':<22s} | {'Price':>10s} | {'Error':>10s} | {'Time (ms)':>10s}")
    logger.info("  " + "-" * 60)
    for name, price, err, t in results:
        logger.info(f"  {name:<22s} | ${price:>8.4f} | {err:>10.6f} | {t:>10.3f}")

    # --- Plot: Time vs Accuracy ---
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model type
    mc_data = [(r[3], r[2]) for r in results if r[0].startswith("MC")]
    bn_data = [(r[3], r[2]) for r in results if r[0].startswith("Binomial")]

    if mc_data:
        mc_times, mc_errs = zip(*mc_data)
        ax.scatter(mc_times, mc_errs, color=COLORS["standard"], s=100, zorder=5,
                   label="Monte Carlo", edgecolors="white", linewidths=1.5)
        ax.plot(mc_times, mc_errs, '--', color=COLORS["standard"], alpha=0.5)
        for i, r in enumerate([r for r in results if r[0].startswith("MC")]):
            ax.annotate(r[0].split("=")[1].rstrip(")"),
                        (mc_times[i], mc_errs[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color="#c9d1d9")

    if bn_data:
        bn_times, bn_errs = zip(*bn_data)
        ax.scatter(bn_times, bn_errs, color=COLORS["binomial"], s=100, zorder=5,
                   label="Binomial Tree", marker="^", edgecolors="white", linewidths=1.5)
        ax.plot(bn_times, bn_errs, '--', color=COLORS["binomial"], alpha=0.5)
        for i, r in enumerate([r for r in results if r[0].startswith("Binomial")]):
            ax.annotate(f"N={r[0].split('=')[1].rstrip(')')}",
                        (bn_times[i], bn_errs[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color="#c9d1d9")

    # BS point
    ax.scatter([results[0][3]], [1e-15], color=COLORS["bs_ref"], s=150, zorder=5,
               label="Black-Scholes (exact)", marker="*", edgecolors="white", linewidths=1.5)

    ax.set_xlabel("Computation Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Absolute Error ($)", fontsize=12, fontweight="bold")
    ax.set_title("Speed vs Accuracy — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.tight_layout()
    path = os.path.join(output_dir, "speed_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path}")

    return results


# =============================================================================
# 3. American vs European Comparison
# =============================================================================

def american_vs_european(S0, K, T, r, sigma, output_dir):
    """Compare American and European option prices, show early exercise premium."""
    separator("American vs European — Early Exercise Premium")

    steps_list = BINOMIAL_CONFIG["convergence_steps"]

    euro_calls, amer_calls = [], []
    euro_puts, amer_puts = [], []

    for N in steps_list:
        ec = binomial_price(S0, K, T, r, sigma, "call", "european", N)
        ac = binomial_price(S0, K, T, r, sigma, "call", "american", N)
        ep = binomial_price(S0, K, T, r, sigma, "put", "european", N)
        ap = binomial_price(S0, K, T, r, sigma, "put", "american", N)
        euro_calls.append(ec["price"])
        amer_calls.append(ac["price"])
        euro_puts.append(ep["price"])
        amer_puts.append(ap["price"])

    # Log results for largest N
    N_max = steps_list[-1]
    logger.info(f"  N = {N_max}")
    logger.info(f"  Call: Euro = {euro_calls[-1]:.6f}, Amer = {amer_calls[-1]:.6f}, "
                f"Premium = {amer_calls[-1]-euro_calls[-1]:.6f}")
    logger.info(f"  Put:  Euro = {euro_puts[-1]:.6f}, Amer = {amer_puts[-1]:.6f}, "
                f"Premium = {amer_puts[-1]-euro_puts[-1]:.6f}")

    # Verify key properties
    assert np.isclose(amer_calls[-1], euro_calls[-1], atol=1e-6), \
        "American call should equal European call (no dividends)!"
    assert amer_puts[-1] > euro_puts[-1], \
        "American put should be strictly greater than European put!"
    logger.info("  ✓ American call = European call (no dividends)")
    logger.info("  ✓ American put > European put (early exercise premium)")

    # --- Plot ---
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Calls
    ax1.plot(steps_list, euro_calls, 'o-', color=COLORS["standard"], linewidth=2,
             markersize=5, label="European Call", alpha=0.9)
    ax1.plot(steps_list, amer_calls, 's--', color=COLORS["binomial"], linewidth=2,
             markersize=5, label="American Call", alpha=0.9)
    ax1.set_xlabel("Steps (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title("Call Options — American = European", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: Puts + premium
    ax2.plot(steps_list, euro_puts, 'o-', color=COLORS["antithetic"], linewidth=2,
             markersize=5, label="European Put", alpha=0.9)
    ax2.plot(steps_list, amer_puts, 's--', color=COLORS["binomial_put"], linewidth=2,
             markersize=5, label="American Put", alpha=0.9)

    # Shade the premium
    ax2.fill_between(steps_list, euro_puts, amer_puts,
                     color=COLORS["binomial"], alpha=0.15, label="Early Exercise Premium")

    ax2.set_xlabel("Steps (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
    ax2.set_title("Put Options — American > European", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("American vs European Options — Early Exercise Analysis",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "american_vs_european.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path}")


# =============================================================================
# 4. Early Exercise Boundary
# =============================================================================

def exercise_boundary_analysis(S0, K, T, r, sigma, output_dir):
    """Map the early exercise boundary for American puts."""
    separator("Early Exercise Boundary — American Put")

    result = binomial_price_with_tree(
        S0, K, T, r, sigma, option_type="put", style="american", N=200
    )

    boundary = result["exercise_boundary"]
    if not boundary:
        logger.warning("  No exercise boundary found!")
        return

    times, prices = zip(*sorted(boundary))
    logger.info(f"  Found {len(boundary)} boundary points")
    logger.info(f"  Boundary range: S* ∈ [{min(prices):.2f}, {max(prices):.2f}]")
    logger.info(f"  At T=0: exercise if S < {prices[0]:.2f}")

    # --- Plot ---
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, prices, '-', color=COLORS["binomial"], linewidth=2.5, alpha=0.9,
            label="Exercise Boundary S*(t)")
    ax.fill_between(times, 0, prices, color=COLORS["binomial"], alpha=0.1,
                    label="Early Exercise Region")
    ax.axhline(y=K, color=COLORS["bs_ref"], linestyle="--", linewidth=1.5,
               label=f"Strike K = {K:.0f}", alpha=0.7)
    ax.axhline(y=S0, color=COLORS["antithetic"], linestyle=":", linewidth=1.5,
               label=f"Spot S₀ = {S0:.0f}", alpha=0.7)

    ax.set_xlabel("Time to Maturity (years)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Critical Stock Price S*(t)", fontsize=12, fontweight="bold")
    ax.set_title("American Put — Early Exercise Boundary",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(bottom=0)

    # Add annotation
    ax.annotate("Exercise optimal\n(S below boundary)",
                xy=(times[len(times)//2], prices[len(prices)//2] * 0.7),
                fontsize=10, color="#7ee787", ha="center",
                fontweight="bold", alpha=0.8)
    ax.annotate("Hold option\n(S above boundary)",
                xy=(times[len(times)//2], min(K, prices[len(prices)//2] * 1.2)),
                fontsize=10, color=COLORS["antithetic"], ha="center",
                fontweight="bold", alpha=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "exercise_boundary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path}")


# =============================================================================
# 5. Model Comparison Summary
# =============================================================================

def model_comparison_summary(S0, K, T, r, sigma, output_dir):
    """Create a comprehensive model comparison bar chart."""
    separator("Model Comparison Summary")

    bs_call = float(black_scholes_price(S0, K, T, r, sigma, "call"))

    models = {}

    # Black-Scholes
    t0 = time.perf_counter()
    for _ in range(1000):
        black_scholes_price(S0, K, T, r, sigma, "call")
    models["Black-\nScholes"] = {
        "price": bs_call,
        "error": 0.0,
        "time_ms": (time.perf_counter() - t0) / 1000 * 1000,
        "color": COLORS["bs_ref"],
    }

    # Monte Carlo
    t0 = time.perf_counter()
    mc = monte_carlo_price(S0, K, T, r, sigma, "call", n_sims=100_000, seed=42)
    models["Monte\nCarlo"] = {
        "price": mc["price"],
        "error": abs(mc["price"] - bs_call),
        "time_ms": (time.perf_counter() - t0) * 1000,
        "color": COLORS["standard"],
    }

    # Binomial
    t0 = time.perf_counter()
    bn = binomial_price(S0, K, T, r, sigma, "call", "european", N=500)
    models["Binomial\nTree"] = {
        "price": bn["price"],
        "error": abs(bn["price"] - bs_call),
        "time_ms": (time.perf_counter() - t0) * 1000,
        "color": COLORS["binomial"],
    }

    # --- Plot ---
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    names = list(models.keys())
    colors = [models[n]["color"] for n in names]

    # 1. Price
    ax = axes[0]
    prices = [models[n]["price"] for n in names]
    bars = ax.bar(names, prices, color=colors, alpha=0.85, edgecolor="none", width=0.5)
    ax.axhline(y=bs_call, color="#8b949e", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("Price ($)", fontsize=11, fontweight="bold")
    ax.set_title("Option Price", fontsize=13, fontweight="bold")
    for bar, p in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"${p:.4f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # 2. Error
    ax = axes[1]
    errors = [max(models[n]["error"], 1e-15) for n in names]
    bars = ax.bar(names, errors, color=colors, alpha=0.85, edgecolor="none", width=0.5)
    ax.set_ylabel("Absolute Error ($)", fontsize=11, fontweight="bold")
    ax.set_title("Error vs Black-Scholes", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    for bar, e in zip(bars, errors):
        label = "Exact" if e < 1e-10 else f"{e:.6f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                label, ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # 3. Speed
    ax = axes[2]
    times = [models[n]["time_ms"] for n in names]
    bars = ax.bar(names, times, color=colors, alpha=0.85, edgecolor="none", width=0.5)
    ax.set_ylabel("Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Computation Time", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                f"{t:.2f}ms", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    fig.suptitle("Pricing Model Comparison — BS vs MC vs Binomial",
                 fontsize=15, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    S0 = DEFAULT_PARAMS["S0"]
    K = DEFAULT_PARAMS["K"]
    T = DEFAULT_PARAMS["T"]
    r = DEFAULT_PARAMS["r"]
    sigma = DEFAULT_PARAMS["sigma"]

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "output", "binomial_plots")
    os.makedirs(output_dir, exist_ok=True)

    convergence_analysis(S0, K, T, r, sigma, output_dir)
    speed_benchmark(S0, K, T, r, sigma, output_dir)
    american_vs_european(S0, K, T, r, sigma, output_dir)
    exercise_boundary_analysis(S0, K, T, r, sigma, output_dir)
    model_comparison_summary(S0, K, T, r, sigma, output_dir)

    separator("COMPLETE")
    logger.info(f"  All plots saved to: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath) / 1024
        logger.info(f"    📊 {f} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
