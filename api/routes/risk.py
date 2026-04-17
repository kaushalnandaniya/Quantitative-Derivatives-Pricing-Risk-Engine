"""
Risk Routes
============
API endpoints for portfolio risk analysis:
    POST /risk/portfolio

Validates input, delegates to risk_service.
"""

import logging

from fastapi import APIRouter

from schemas.risk import PortfolioRiskInput
from services.risk_service import compute_portfolio_risk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


@router.post(
    "/portfolio",
    summary="Portfolio Risk Analysis",
    description=(
        "Compute VaR, CVaR, and P&L statistics for a portfolio of option positions. "
        "Supports historical, parametric, and Monte Carlo methods. "
        "Multi-asset portfolios can include a correlation matrix."
    ),
)
def portfolio_risk(data: PortfolioRiskInput):
    """Full portfolio risk analysis: VaR, CVaR, and P&L distribution statistics."""
    # Convert Pydantic models to dicts for the service layer
    positions = [pos.model_dump() for pos in data.portfolio]

    return compute_portfolio_risk(
        portfolio_positions=positions,
        method=data.method,
        confidence=data.confidence,
        n_sims=data.n_sims,
        horizon_days=data.horizon_days,
        seed=data.seed,
        correlation_matrix=data.correlation_matrix,
    )
