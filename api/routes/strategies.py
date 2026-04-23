"""
Strategy Routes
================
API endpoints for option strategy simulation:
    GET  /strategies/list
    POST /strategies/simulate
"""

import logging

from fastapi import APIRouter

from schemas.strategies import StrategySimulateInput
from services.strategies import list_strategies, simulate_strategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.get("/list", summary="List Available Strategies")
def get_strategies():
    """Return all available pre-built strategy templates."""
    return {"strategies": list_strategies()}


@router.post("/simulate", summary="Simulate Strategy P&L")
def simulate(data: StrategySimulateInput):
    """Simulate P&L profile, breakevens, max profit/loss, and net Greeks for a strategy."""
    return simulate_strategy(
        strategy_id=data.strategy_id,
        S=data.S, K=data.K, T=data.T,
        r=data.r, sigma=data.sigma,
        lot_size=data.lot_size,
    )
