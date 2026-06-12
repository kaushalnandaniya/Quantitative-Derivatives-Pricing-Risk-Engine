"""
Backtest Routes
================
API endpoints for Pine Script backtesting:
    GET  /backtest/search         — Search stock symbols
    POST /backtest/run            — Run strategy backtest (legacy)
    POST /backtest/run-pine       — Run Pine Script backtest
    GET  /backtest/strategies     — List user's saved strategies
    POST /backtest/strategies     — Save a new strategy
    PUT  /backtest/strategies/{id} — Update a strategy
    DELETE /backtest/strategies/{id} — Delete a strategy
"""

import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, SavedStrategy
from api.middleware.auth import get_current_user
from services.backtest import run_backtest
from services.pine_executor import run_pine_backtest
from services.historical_data import get_historical_data, search_symbols

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtest"])


# =============================================================================
# Schemas
# =============================================================================

class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str = "NIFTY"
    lookback_weeks: int = 12
    sigma: float = 0.15
    lot_size: int = 1
    custom_legs: Optional[List[Dict]] = None


class PineBacktestRequest(BaseModel):
    pine_script: str = Field(..., description="Pine Script source code")
    symbol: str = Field("RELIANCE", description="Stock symbol to backtest against")
    period_days: int = Field(365, ge=30, le=3650, description="Lookback period in days")
    interval: str = Field("day", description="Data interval: day, 60minute, 15minute")


class SaveStrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    pine_script: str = Field(..., min_length=10)
    description: Optional[str] = None


class UpdateStrategyRequest(BaseModel):
    name: Optional[str] = None
    pine_script: Optional[str] = None
    description: Optional[str] = None


# =============================================================================
# Search
# =============================================================================

@router.get("/search", summary="Search Stock Symbols")
def search_stocks(q: str = Query("", description="Search query")):
    """Search for stock symbols by name or ticker."""
    results = search_symbols(q)
    return {"results": results, "count": len(results)}


# =============================================================================
# Legacy Backtest
# =============================================================================

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


# =============================================================================
# Pine Script Backtest
# =============================================================================

@router.post("/run-pine", summary="Run Pine Script Backtest")
def backtest_pine(req: PineBacktestRequest):
    """Execute a Pine Script strategy against historical OHLCV data."""
    try:
        df = get_historical_data(req.symbol, req.period_days, req.interval)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch data for {req.symbol}: {e}")

    if df is None or df.empty:
        raise HTTPException(404, f"No historical data available for {req.symbol}")

    try:
        results = run_pine_backtest(req.pine_script, df)
    except SyntaxError as e:
        raise HTTPException(400, f"Pine Script syntax error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Backtest execution error: {e}")

    results["symbol"] = req.symbol
    results["period_days"] = req.period_days
    results["data_points"] = len(df)
    
    # Add OHLCV data for the frontend chart
    # Reset index to get dates as a column, fill NaN with None for JSON serialization
    chart_df = df.reset_index().rename(columns={"index": "time", "Date": "time"})
    chart_df = chart_df.where(pd.notnull(chart_df), None)
    
    # Format time to string format YYYY-MM-DD
    if "time" in chart_df.columns:
        chart_df["time"] = chart_df["time"].dt.strftime("%Y-%m-%d")
        
    results["ohlcv"] = chart_df.to_dict(orient="records")
    
    return results


# =============================================================================
# Strategy CRUD (requires auth)
# =============================================================================

@router.get("/strategies", summary="List Saved Strategies")
def list_strategies(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all saved strategies for the authenticated user."""
    strategies = (
        db.query(SavedStrategy)
        .filter(SavedStrategy.user_id == user.id)
        .order_by(SavedStrategy.updated_at.desc())
        .all()
    )
    return {
        "strategies": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "pine_script": s.pine_script,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in strategies
        ],
        "count": len(strategies),
    }


@router.post("/strategies", status_code=201, summary="Save Strategy")
def save_strategy(
    req: SaveStrategyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a new Pine Script strategy."""
    strategy = SavedStrategy(
        user_id=user.id,
        name=req.name,
        pine_script=req.pine_script,
        description=req.description,
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return {
        "id": strategy.id,
        "name": strategy.name,
        "created_at": strategy.created_at.isoformat(),
    }


@router.put("/strategies/{strategy_id}", summary="Update Strategy")
def update_strategy(
    strategy_id: str,
    req: UpdateStrategyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an existing saved strategy."""
    strategy = (
        db.query(SavedStrategy)
        .filter(SavedStrategy.id == strategy_id, SavedStrategy.user_id == user.id)
        .first()
    )
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    if req.name is not None:
        strategy.name = req.name
    if req.pine_script is not None:
        strategy.pine_script = req.pine_script
    if req.description is not None:
        strategy.description = req.description

    db.commit()
    db.refresh(strategy)
    return {"id": strategy.id, "name": strategy.name, "updated_at": strategy.updated_at.isoformat()}


@router.delete("/strategies/{strategy_id}", summary="Delete Strategy")
def delete_strategy(
    strategy_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a saved strategy."""
    strategy = (
        db.query(SavedStrategy)
        .filter(SavedStrategy.id == strategy_id, SavedStrategy.user_id == user.id)
        .first()
    )
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    db.delete(strategy)
    db.commit()
    return {"deleted": True, "id": strategy_id}
