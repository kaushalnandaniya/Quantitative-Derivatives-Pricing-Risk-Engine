"""
Strategy Schemas
=================
Pydantic models for strategy simulation API endpoints.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class StrategySimulateInput(BaseModel):
    """Input for strategy P&L simulation."""

    strategy_id: str = Field(..., description="Strategy template ID (e.g., 'straddle', 'iron_condor')")
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Base ATM strike price")
    T: float = Field(..., gt=0, le=30.0, description="Time to maturity (years)")
    r: float = Field(0.05, ge=-0.1, le=1.0, description="Risk-free rate")
    sigma: float = Field(0.2, gt=0, le=5.0, description="Volatility")
    lot_size: int = Field(1, ge=1, le=10000, description="Number of lots")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "strategy_id": "straddle",
                "S": 24000, "K": 24000, "T": 0.08, "r": 0.069, "sigma": 0.14,
                "lot_size": 1,
            }]
        }
    }
