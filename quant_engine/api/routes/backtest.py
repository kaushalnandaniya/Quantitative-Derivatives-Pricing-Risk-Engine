"""
Backtest Routes
================
API endpoint for strategy backtesting:
    POST /backtest/run
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from services.backtest import run_backtest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtest"])


from typing import Optional, List, Dict

class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str = "NIFTY"
    lookback_weeks: int = 12
    sigma: float = 0.15
    lot_size: int = 1
    custom_legs: Optional[List[Dict]] = None


@router.post("/run", summary="Run Strategy Backtest")
def backtest_run(req: BacktestRequest):
    """Backtest a strategy against historical weekly expiries."""
    return run_backtest(
        strategy_id=req.strategy_id,
        symbol=req.symbol,
        lookback_weeks=req.lookback_weeks,
        sigma=req.sigma,
        lot_size=req.lot_size,
        custom_legs=req.custom_legs,
    )
