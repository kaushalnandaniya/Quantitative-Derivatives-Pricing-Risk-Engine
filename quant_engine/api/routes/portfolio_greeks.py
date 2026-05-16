"""
Portfolio Greeks Routes
========================
API endpoint for portfolio-level Greeks:
    POST /greeks/portfolio
"""

import logging

from fastapi import APIRouter

from schemas.risk import PortfolioRiskInput
from services.portfolio_greeks import compute_portfolio_greeks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/greeks", tags=["Greeks"])


@router.post(
    "/portfolio",
    summary="Portfolio Greeks",
    description="Compute aggregated Greeks (Delta, Gamma, Vega, Theta, Rho) across all portfolio positions.",
)
def portfolio_greeks(data: PortfolioRiskInput):
    """Compute per-position and total portfolio Greeks."""
    positions = [pos.model_dump() for pos in data.portfolio]
    return compute_portfolio_greeks(positions=positions, method="analytical")
