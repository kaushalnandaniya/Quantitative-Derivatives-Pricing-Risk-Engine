"""
Monte Carlo Visualization Module
=================================
Production-quality plots for Monte Carlo pricing analysis.

Plots:
    1. Convergence plot (MC price vs n_sims, with BS reference)
    2. Payoff distribution (histogram of terminal payoffs)
    3. Confidence intervals (CI width vs n_sims)
    4. Variance reduction comparison (bar chart)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
from pathlib import Path
from typing import Dict, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PLOT_STYLE, COLORS


def _apply_style():
    """Apply dark theme to current plot."""
    plt.rcParams.update(PLOT_STYLE)


def plot_convergence(
    convergence_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Monte Carlo price convergence to Black-Scholes as N increases.

    Args:
        convergence_data: Output from convergence_analysis().
        save_path:        If provided, save figure to this path.
        show:             Whether to display the figure.

    Returns:
        matplotlib Figure object.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sim_sizes = convergence_data["sim_sizes"]
    bs_price = convergence_data["bs_price"]
    results = convergence_data["results"]

    # --- Left: Price vs N ---
    for method, color in [("standard", COLORS["standard"]),
                          ("antithetic", COLORS["antithetic"]),
                          ("control", COLORS["control"])]:
        if method not in results:
            continue
        prices = [r["price"] for r in results[method]]
        ci_lo = [r["ci_lower"] for r in results[method]]
        ci_hi = [r["ci_upper"] for r in results[method]]

        ax1.plot(sim_sizes, prices, 'o-', color=color, label=method.capitalize(),
                 linewidth=2, markersize=5, alpha=0.9)
        ax1.fill_between(sim_sizes, ci_lo, ci_hi, color=color, alpha=0.1)

    ax1.axhline(y=bs_price, color=COLORS["bs_ref"], linestyle="--",
                linewidth=2, label=f"Black-Scholes = {bs_price:.4f}", alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Simulations (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Option Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title("Monte Carlo Convergence", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Right: Absolute Error vs N ---
    for method, color in [("standard", COLORS["standard"]),
                          ("antithetic", COLORS["antithetic"]),
                          ("control", COLORS["control"])]:
        if method not in results:
            continue
        errors = [r["abs_error"] for r in results[method]]
        ax2.plot(sim_sizes, errors, 'o-', color=color, label=method.capitalize(),
                 linewidth=2, markersize=5, alpha=0.9)

    # Theoretical O(1/√N) line
    ref_err = results["standard"][0]["abs_error"] if "standard" in results else 1.0
    ref_n = sim_sizes[0]
    theoretical = [ref_err * np.sqrt(ref_n / n) for n in sim_sizes]
    ax2.plot(sim_sizes, theoretical, ':', color="#8b949e", linewidth=1.5,
             label=r"$O(1/\sqrt{N})$ reference", alpha=0.7)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Simulations (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Absolute Error ($)", fontsize=12, fontweight="bold")
    ax2.set_title("Error Convergence", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Monte Carlo Pricing Engine — Convergence Analysis",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()

    return fig


def plot_payoff_distribution(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 100_000,
    seed: int = 42,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of simulated terminal prices and payoff distributions.
    """
    from models.gbm import simulate_terminal_price
    from pricing.monte_carlo import compute_payoff

    _apply_style()
    rng = np.random.default_rng(seed)
    ST = simulate_terminal_price(S0, r, sigma, T, n_sims, rng)

    call_payoff = compute_payoff(ST, K, "call")
    put_payoff = compute_payoff(ST, K, "put")
    discount = np.exp(-r * T)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. Terminal price distribution
    ax = axes[0]
    ax.hist(ST, bins=100, color=COLORS["standard"], alpha=0.7, edgecolor="none",
            density=True)
    ax.axvline(S0, color=COLORS["bs_ref"], linestyle="--", linewidth=2,
               label=f"S₀ = {S0:.0f}")
    ax.axvline(K, color=COLORS["antithetic"], linestyle="--", linewidth=2,
               label=f"K = {K:.0f}")
    ax.axvline(np.mean(ST), color="#7ee787", linestyle="-", linewidth=2,
               label=f"E[S_T] = {np.mean(ST):.2f}")
    ax.set_xlabel("Terminal Price (S_T)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Terminal Price Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # 2. Call payoff distribution
    ax = axes[1]
    nonzero_call = call_payoff[call_payoff > 0]
    ax.hist(nonzero_call, bins=80, color=COLORS["hist_call"], alpha=0.7,
            edgecolor="none", density=True)
    ax.axvline(np.mean(call_payoff), color=COLORS["bs_ref"], linestyle="--",
               linewidth=2, label=f"E[payoff] = {np.mean(call_payoff):.2f}")
    itm_pct = 100 * len(nonzero_call) / n_sims
    ax.set_xlabel("Call Payoff max(S_T − K, 0)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title(f"Call Payoff | ITM = {itm_pct:.1f}%", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # 3. Put payoff distribution
    ax = axes[2]
    nonzero_put = put_payoff[put_payoff > 0]
    ax.hist(nonzero_put, bins=80, color=COLORS["hist_put"], alpha=0.7,
            edgecolor="none", density=True)
    ax.axvline(np.mean(put_payoff), color=COLORS["bs_ref"], linestyle="--",
               linewidth=2, label=f"E[payoff] = {np.mean(put_payoff):.2f}")
    itm_pct_put = 100 * len(nonzero_put) / n_sims
    ax.set_xlabel("Put Payoff max(K − S_T, 0)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title(f"Put Payoff | ITM = {itm_pct_put:.1f}%", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(f"Payoff Distributions | S₀={S0}, K={K}, T={T}, σ={sigma}, r={r} | N={n_sims:,}",
                 fontsize=14, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_confidence_intervals(
    convergence_data: Dict,
    method: str = "standard",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot confidence interval width shrinking with increasing N.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sim_sizes = convergence_data["sim_sizes"]
    bs_price = convergence_data["bs_price"]
    data = convergence_data["results"][method]

    prices = [r["price"] for r in data]
    ci_lo = [r["ci_lower"] for r in data]
    ci_hi = [r["ci_upper"] for r in data]
    ci_widths = [r["ci_upper"] - r["ci_lower"] for r in data]
    std_errors = [r["std_error"] for r in data]

    color = COLORS.get(method, COLORS["standard"])

    # Left: Price with CI bars
    ax1.errorbar(sim_sizes, prices,
                 yerr=[(p - lo) for p, lo in zip(prices, ci_lo)],
                 fmt='o-', color=color, ecolor=color, elinewidth=1.5,
                 capsize=4, linewidth=2, markersize=6, alpha=0.9,
                 label=f"{method.capitalize()} MC")
    ax1.axhline(y=bs_price, color=COLORS["bs_ref"], linestyle="--",
                linewidth=2, label=f"BS = {bs_price:.4f}", alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Simulations", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Option Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title(f"95% Confidence Intervals ({method.capitalize()})",
                  fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: CI width and SE
    ax2.plot(sim_sizes, ci_widths, 'o-', color=color, linewidth=2,
             markersize=6, label="CI Width (95%)")
    ax2.plot(sim_sizes, std_errors, 's--', color=COLORS["antithetic"],
             linewidth=2, markersize=5, label="Standard Error", alpha=0.8)

    # Theoretical 1/√N scaling
    ref_w = ci_widths[0]
    ref_n = sim_sizes[0]
    theoretical_w = [ref_w * np.sqrt(ref_n / n) for n in sim_sizes]
    ax2.plot(sim_sizes, theoretical_w, ':', color="#8b949e", linewidth=1.5,
             label=r"$O(1/\sqrt{N})$", alpha=0.7)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Simulations", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Width / Error ($)", fontsize=12, fontweight="bold")
    ax2.set_title("CI Width & Standard Error", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Confidence Interval Analysis",
                 fontsize=15, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_variance_reduction(
    comparison_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing Standard MC, Antithetic, and Control Variate methods.
    Shows price accuracy, variance, standard error, and variance reduction ratio.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    methods = ["standard", "antithetic", "control"]
    labels = ["Standard MC", "Antithetic", "Control Variate"]
    colors = [COLORS["standard"], COLORS["antithetic"], COLORS["control"]]
    bs_price = comparison_data["bs_price"]

    # 1. Price comparison
    ax = axes[0]
    prices = [comparison_data[m]["price"] for m in methods]
    bars = ax.bar(labels, prices, color=colors, alpha=0.85, edgecolor="none",
                  width=0.6)
    ax.axhline(y=bs_price, color=COLORS["bs_ref"], linestyle="--", linewidth=2,
               label=f"BS = {bs_price:.4f}")
    ax.set_ylabel("Price ($)", fontsize=11, fontweight="bold")
    ax.set_title("MC Price", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    # Add value labels
    for bar, p in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{p:.4f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # 2. Variance
    ax = axes[1]
    variances = [comparison_data[m]["variance"] for m in methods]
    bars = ax.bar(labels, variances, color=colors, alpha=0.85, edgecolor="none",
                  width=0.6)
    ax.set_ylabel("Variance", fontsize=11, fontweight="bold")
    ax.set_title("Sample Variance", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # 3. Standard Error
    ax = axes[2]
    se = [comparison_data[m]["std_error"] for m in methods]
    bars = ax.bar(labels, se, color=colors, alpha=0.85, edgecolor="none",
                  width=0.6)
    ax.set_ylabel("Std Error ($)", fontsize=11, fontweight="bold")
    ax.set_title("Standard Error", fontsize=13, fontweight="bold")
    for bar, s in zip(bars, se):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{s:.6f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # 4. Variance Reduction Ratio
    ax = axes[3]
    ratios = [1.0]  # Standard baseline
    for m in ["antithetic", "control"]:
        vr = comparison_data[m].get("variance_reduction_ratio")
        ratios.append(vr if vr is not None else 0)
    bars = ax.bar(labels, ratios, color=colors, alpha=0.85, edgecolor="none",
                  width=0.6)
    ax.axhline(y=1.0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("Ratio (× improvement)", fontsize=11, fontweight="bold")
    ax.set_title("Variance Reduction Ratio", fontsize=13, fontweight="bold")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10,
                color="#c9d1d9", fontweight="bold")

    fig.suptitle("Variance Reduction Comparison",
                 fontsize=15, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# =============================================================================
# Binomial Tree Visualizations
# =============================================================================

def plot_binomial_tree(
    tree_data: Dict,
    max_display_steps: int = 8,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Draw the binomial tree with nodes colored by early exercise decisions.

    Args:
        tree_data:          Output from binomial_price_with_tree().
        max_display_steps:  Max steps to display (truncates for readability).
        save_path:          If provided, save to this path.
        show:               Whether to display.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    N = min(tree_data["N"], max_display_steps)
    stock_tree = tree_data["stock_tree"][:N + 1]
    option_tree = tree_data["option_tree"][:N + 1]
    exercise_map = tree_data["early_exercise_map"][:N + 1]
    style = tree_data["style"]
    otype = tree_data["option_type"]

    fig, ax = plt.subplots(figsize=(max(14, N * 1.8), max(8, N * 0.8)))

    # Draw edges and nodes
    for i in range(N + 1):
        for j in range(i + 1):
            x = i
            y = j - i / 2  # Center vertically

            # Node color
            if exercise_map[i] is not None and j < len(exercise_map[i]) and exercise_map[i][j]:
                color = COLORS["exercise"]
                marker_size = 120
            else:
                color = COLORS["hold"] if option_tree[i][j] > 0 else "#8b949e"
                marker_size = 80

            ax.scatter(x, y, c=color, s=marker_size, zorder=5,
                       edgecolors="white", linewidths=0.5)

            # Labels (stock price / option value)
            s_val = stock_tree[i][j]
            v_val = option_tree[i][j]
            if N <= 6:
                ax.annotate(f"S={s_val:.1f}\nV={v_val:.2f}",
                            (x, y), textcoords="offset points",
                            xytext=(0, 12), fontsize=6, ha="center",
                            color="#c9d1d9")
            elif N <= 10:
                ax.annotate(f"{s_val:.0f}", (x, y),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=5, ha="center", color="#c9d1d9")

            # Draw edges to next level
            if i < N:
                # Up edge
                x_next = i + 1
                y_up = (j + 1) - (i + 1) / 2
                y_down = j - (i + 1) / 2
                ax.plot([x, x_next], [y, y_up], '-', color="#30363d",
                        linewidth=0.8, alpha=0.6, zorder=1)
                ax.plot([x, x_next], [y, y_down], '-', color="#30363d",
                        linewidth=0.8, alpha=0.6, zorder=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["hold"], label="Hold (V > 0)"),
        Patch(facecolor="#8b949e", label="Worthless (V = 0)"),
    ]
    if style == "american":
        legend_elements.insert(0, Patch(facecolor=COLORS["exercise"],
                                        label="Early Exercise"))

    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Node Position", fontsize=12, fontweight="bold")
    ax.set_title(f"Binomial Tree — {style.capitalize()} {otype.capitalize()} "
                 f"(N={tree_data['N']}, showing {N} steps)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(range(N + 1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_binomial_convergence(
    convergence_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot binomial price convergence to BS as N increases.

    Args:
        convergence_data: Dict with 'steps', 'call_prices', 'put_prices',
                         'bs_call', 'bs_put'.
        save_path:        Optional save path.
        show:             Whether to display.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    steps = convergence_data["steps"]
    call_prices = convergence_data["call_prices"]
    put_prices = convergence_data["put_prices"]
    bs_call = convergence_data["bs_call"]
    bs_put = convergence_data["bs_put"]

    # Left: Prices
    ax1.plot(steps, call_prices, 'o-', color=COLORS["binomial"], linewidth=2,
             markersize=5, label="Binomial Call", alpha=0.9)
    ax1.plot(steps, put_prices, 's-', color=COLORS["binomial_put"], linewidth=2,
             markersize=5, label="Binomial Put", alpha=0.9)
    ax1.axhline(y=bs_call, color=COLORS["bs_ref"], linestyle="--", linewidth=2,
                label=f"BS Call = {bs_call:.4f}", alpha=0.8)
    ax1.axhline(y=bs_put, color=COLORS["antithetic"], linestyle="--", linewidth=2,
                label=f"BS Put = {bs_put:.4f}", alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Steps (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Option Price ($)", fontsize=12, fontweight="bold")
    ax1.set_title("Binomial Convergence to BS", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: Errors
    call_errors = [abs(c - bs_call) for c in call_prices]
    put_errors = [abs(p - bs_put) for p in put_prices]
    ax2.plot(steps, call_errors, 'o-', color=COLORS["binomial"], linewidth=2,
             markersize=5, label="Call Error", alpha=0.9)
    ax2.plot(steps, put_errors, 's-', color=COLORS["binomial_put"], linewidth=2,
             markersize=5, label="Put Error", alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Steps (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Absolute Error ($)", fontsize=12, fontweight="bold")
    ax2.set_title("Error vs Black-Scholes", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Binomial Tree — Convergence Analysis",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_early_exercise_boundary(
    tree_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the early exercise boundary for an American option.

    Args:
        tree_data: Output from binomial_price_with_tree() with style='american'.
        save_path: Optional save path.
        show:      Whether to display.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    boundary = tree_data.get("exercise_boundary", [])
    if not boundary:
        ax.text(0.5, 0.5, "No early exercise boundary found",
                transform=ax.transAxes, ha="center", fontsize=14, color="#c9d1d9")
        return fig

    times, prices = zip(*sorted(boundary))
    K = tree_data["params"]["dt"] * tree_data["N"]  # reconstruct T for strike display

    ax.plot(times, prices, '-', color=COLORS["binomial"], linewidth=2.5,
            label="Exercise Boundary S*(t)", alpha=0.9)
    ax.fill_between(times, 0, prices, color=COLORS["binomial"], alpha=0.1,
                    label="Exercise Region")

    ax.set_xlabel("Time (years)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Critical Stock Price S*(t)", fontsize=12, fontweight="bold")
    ax.set_title(f"American {tree_data['option_type'].capitalize()} — "
                 f"Early Exercise Boundary (N={tree_data['N']})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_model_comparison(
    comparison_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing BS, MC, and Binomial across price, error, and speed.

    Args:
        comparison_data: Dict with model names as keys, each having
                        'price', 'error', 'time_ms', 'color'.
        save_path:       Optional save path.
        show:            Whether to display.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    names = list(comparison_data.keys())
    colors = [comparison_data[n]["color"] for n in names]

    # Price
    ax = axes[0]
    prices = [comparison_data[n]["price"] for n in names]
    bars = ax.bar(names, prices, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Price ($)", fontsize=11, fontweight="bold")
    ax.set_title("Option Price", fontsize=13, fontweight="bold")
    for bar, p in zip(bars, prices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${p:.4f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # Error
    ax = axes[1]
    errors = [max(comparison_data[n]["error"], 1e-15) for n in names]
    bars = ax.bar(names, errors, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Error ($)", fontsize=11, fontweight="bold")
    ax.set_title("Absolute Error", fontsize=13, fontweight="bold")
    ax.set_yscale("log")

    # Speed
    ax = axes[2]
    times = [comparison_data[n]["time_ms"] for n in names]
    bars = ax.bar(names, times, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Compute Time", fontsize=13, fontweight="bold")
    ax.set_yscale("log")

    fig.suptitle("Model Comparison — BS vs MC vs Binomial",
                 fontsize=15, fontweight="bold", color="#f0f6fc", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig
