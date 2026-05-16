"""
Configuration Settings
=======================
Central configuration for the quant engine.
"""

# Default market parameters (ATM European option)
DEFAULT_PARAMS = {
    "S0": 24050.60,      # Spot price
    "K": 24000.0,        # Strike price
    "T": 2/252,          # Time to maturity (years)
    "r": 0.069,         # Risk-free rate (5%)
    "sigma": 0.2,      # Volatility (20%)
}

# Monte Carlo configuration
MC_CONFIG = {
    "n_sims": 500_000,          # Default number of simulations
    "seed": 42,                  # Default random seed
    "confidence_level": 0.95,    # Default confidence level
    "methods": ["standard", "antithetic", "control"],
    "convergence_sim_sizes": [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
}

# Binomial tree configuration
BINOMIAL_CONFIG = {
    "default_steps": 200,        # Default number of time steps
    "convergence_steps": [10, 25, 50, 100, 200, 500, 1000, 2000],
}

# Risk engine configuration
RISK_CONFIG = {
    "n_sims": 100_000,           # MC simulations for risk
    "confidence_level": 0.95,    # VaR/CVaR confidence
    "horizon_days": 1,           # Risk horizon in trading days
    "seed": 42,                  # Random seed
}

# Plot style (dark professional theme)
PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 10,
}

COLORS = {
    "standard": "#58a6ff",
    "antithetic": "#f78166",
    "control": "#7ee787",
    "bs_ref": "#ffd700",
    "ci_fill": "#58a6ff",
    "hist_call": "#58a6ff",
    "hist_put": "#f78166",
    "binomial": "#bc8cff",
    "binomial_put": "#ff6eb4",
    "exercise": "#ff4444",
    "hold": "#7ee787",
    # Risk engine colors
    "pnl_hist": "#58a6ff",
    "var_line": "#f85149",
    "cvar_line": "#ff6eb4",
    "cvar_region": "#f8514960",
    "tail_loss": "#ff6eb4",
}

