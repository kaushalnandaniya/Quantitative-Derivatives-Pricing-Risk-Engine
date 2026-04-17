"""
Expected Shortfall (Conditional VaR / CVaR)
=============================================
Average loss in the tail beyond VaR.

CVaR is a COHERENT risk measure (unlike VaR) — it satisfies:
    subadditivity, monotonicity, positive homogeneity, translation invariance.

CVaR answers: "If things go bad (beyond VaR), how bad on average?"

Convention:
    - Returns positive number (loss magnitude)
    - CVaR ≥ VaR always (by definition)
"""

import logging
import numpy as np
from scipy.stats import norm

from risk.var import historical_var, parametric_var

logger = logging.getLogger(__name__)


def cvar(
    pnl: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Expected Shortfall (Conditional VaR).

    Computes E[loss | loss > VaR] — the average loss in the worst
    (1 - confidence) fraction of scenarios.

    Args:
        pnl:        Array of P&L values (negative = loss).
        confidence: Confidence level (e.g., 0.95).
        method:     'historical' or 'parametric'.

    Returns:
        CVaR as a positive number (loss magnitude).
        Guaranteed: CVaR ≥ VaR.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    pnl = np.asarray(pnl)
    if pnl.size == 0:
        raise ValueError("P&L array is empty")

    method = method.lower()

    if method == "parametric":
        return parametric_cvar(pnl, confidence)
    elif method in ("historical", "monte_carlo"):
        return _empirical_cvar(pnl, confidence)
    else:
        raise ValueError(f"method must be 'historical', 'monte_carlo', or 'parametric', got '{method}'")


def _empirical_cvar(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Empirical CVaR: average of losses beyond the VaR threshold.

    Efficient implementation:
        1. Compute VaR threshold (percentile)
        2. Boolean-mask the tail: pnl <= -VaR
        3. Average the tail losses
    No repeated sorting — single pass after VaR.
    """
    alpha = 1.0 - confidence
    var_threshold = np.percentile(pnl, alpha * 100)

    # Select tail losses (at or below the VaR quantile)
    tail_mask = pnl <= var_threshold
    tail_losses = pnl[tail_mask]

    if len(tail_losses) == 0:
        # Edge case: return VaR if no observations in tail
        return float(-var_threshold)

    cvar_val = -np.mean(tail_losses)

    logger.info(
        f"CVaR(empirical, {confidence:.0%}) = ${cvar_val:,.2f} | "
        f"tail size = {len(tail_losses):,} / {len(pnl):,}"
    )

    return float(max(cvar_val, 0.0))


def parametric_cvar(
    pnl: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) CVaR.

    Closed-form under normality assumption:
        CVaR = -(μ - σ * φ(z_α) / (1 - confidence))

    Where φ is the standard normal PDF and z_α = Φ⁻¹(1 - confidence).

    Args:
        pnl:        Array of P&L values.
        confidence: Confidence level.

    Returns:
        CVaR as a positive number.
    """
    pnl = np.asarray(pnl)
    mu = np.mean(pnl)
    sigma = np.std(pnl, ddof=1)
    alpha = 1.0 - confidence
    z_alpha = norm.ppf(alpha)

    # Closed-form: E[X | X < VaR_quantile] for normal
    # = μ - σ * φ(z_α) / α
    cvar_val = -(mu - sigma * norm.pdf(z_alpha) / alpha)

    logger.info(
        f"CVaR(parametric, {confidence:.0%}) = ${cvar_val:,.2f}"
    )

    return float(max(cvar_val, 0.0))
