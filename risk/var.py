"""
Value at Risk (VaR)
====================
Three methods for computing VaR from a P&L distribution.

Convention:
    - P&L array: positive = profit, negative = loss
    - VaR is returned as a POSITIVE number representing loss magnitude
    - "95% VaR = $X" means: "there is a 5% chance of losing more than $X"

Methods:
    1. Historical VaR     — empirical quantile of P&L
    2. Parametric VaR     — assumes normal P&L, uses μ + z·σ
    3. Monte Carlo VaR    — percentile of simulated P&L distribution
"""

import logging
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def historical_var(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Historical (empirical) Value at Risk.

    Computes the (1 - confidence) quantile of the P&L distribution.
    No distributional assumptions — purely data-driven.

    Args:
        pnl:        Array of P&L values (negative = loss).
        confidence: Confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        VaR as a positive number (loss magnitude).
    """
    pnl = np.asarray(pnl)
    alpha = 1.0 - confidence
    quantile = np.percentile(pnl, alpha * 100)
    var_val = -quantile  # Flip sign: losses become positive
    return float(max(var_val, 0.0))


def parametric_var(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) Value at Risk.

    Assumes P&L is normally distributed with parameters estimated
    from the data. VaR = -(μ + z_α · σ) where z_α is the critical
    value at the (1 - confidence) level.

    Args:
        pnl:        Array of P&L values.
        confidence: Confidence level.

    Returns:
        VaR as a positive number (loss magnitude).
    """
    pnl = np.asarray(pnl)
    mu = np.mean(pnl)
    sigma = np.std(pnl, ddof=1)
    alpha = 1.0 - confidence
    z_alpha = norm.ppf(alpha)  # Negative for α < 0.5
    var_val = -(mu + z_alpha * sigma)
    return float(max(var_val, 0.0))


def monte_carlo_var(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Monte Carlo Value at Risk.

    Identical to historical VaR in computation, but semantically
    distinguished: this is applied to simulated (not observed) P&L
    distributions from Monte Carlo engines.

    Args:
        pnl:        Array of simulated P&L values.
        confidence: Confidence level.

    Returns:
        VaR as a positive number (loss magnitude).
    """
    # Same computation as historical — the distinction is in the data source
    return historical_var(pnl, confidence)


def var(
    pnl: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Unified VaR dispatcher.

    Args:
        pnl:        Array of P&L values.
        confidence: Confidence level (0 < c < 1).
        method:     'historical', 'parametric', or 'monte_carlo'.

    Returns:
        VaR as a positive number (loss magnitude).

    Raises:
        ValueError: If confidence is out of range or method is unknown.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    pnl = np.asarray(pnl)
    if pnl.size == 0:
        raise ValueError("P&L array is empty")

    method = method.lower()
    dispatch = {
        "historical": historical_var,
        "parametric": parametric_var,
        "monte_carlo": monte_carlo_var,
    }

    if method not in dispatch:
        raise ValueError(
            f"method must be one of {list(dispatch.keys())}, got '{method}'"
        )

    result = dispatch[method](pnl, confidence)

    logger.info(
        f"VaR({method}, {confidence:.0%}) = ${result:,.2f}"
    )

    return result
