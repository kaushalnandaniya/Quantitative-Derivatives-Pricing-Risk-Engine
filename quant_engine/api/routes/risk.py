"""
Risk Routes
============
API endpoints for portfolio risk analysis:
    POST /risk/portfolio

Validates input, delegates to risk_service.
"""

import logging

from fastapi import APIRouter

from schemas.risk import PortfolioRiskInput, CVAInput
from services.risk_service import compute_portfolio_risk
from risk.cva import calculate_cva

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


@router.post(
    "/cva",
    summary="Credit Value Adjustment (CVA)",
    description="Calculate Counterparty Credit Risk for an option using Monte Carlo Expected Exposure."
)
def cva_analysis(data: CVAInput):
    """Calculate CVA using Expected Exposure profiles and PD."""
    cva = calculate_cva(
        S0=data.S, K=data.K, r=data.r, sigma=data.sigma, T=data.T,
        hazard_rate=data.hazard_rate, recovery_rate=data.recovery_rate,
        n_sims=data.n_sims, n_steps=data.n_steps, option_type=data.option_type
    )
    
    return {
        "cva": cva,
        "inputs": data.model_dump()
    }
