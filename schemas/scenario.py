"""
Scenario Analysis Schemas
===========================
Pydantic models for scenario analysis API endpoints.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ScenarioPositionInput(BaseModel):
    type: Literal["call", "put"] = Field(..., description="Option type")
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, le=30.0, description="Time to maturity (years)")
    r: float = Field(0.05, ge=-0.1, le=1.0, description="Risk-free rate")
    sigma: float = Field(0.2, gt=0, le=5.0, description="Volatility")
    qty: int = Field(1, description="Quantity (positive=long, negative=short)")


class StressTestInput(BaseModel):
    """Input for multi-scenario stress testing."""
    positions: List[ScenarioPositionInput] = Field(..., min_length=1, description="Portfolio positions")
    spot_shifts: Optional[List[float]] = Field(None, description="Relative spot shifts (e.g., [-0.1, 0, 0.1])")
    vol_shifts: Optional[List[float]] = Field(None, description="Absolute vol shifts (e.g., [-0.05, 0, 0.05])")
    time_shifts: Optional[List[float]] = Field(None, description="Days forward (e.g., [0, 7, 30])")
    rate_shifts: Optional[List[float]] = Field(None, description="Absolute rate shifts")


class HeatmapInput(BaseModel):
    """Input for 2D P&L heatmap generation."""
    positions: List[ScenarioPositionInput] = Field(..., min_length=1, description="Portfolio positions")
    x_axis: Literal["spot", "time"] = Field("spot", description="X-axis variable")
    y_axis: Literal["vol", "time"] = Field("vol", description="Y-axis variable")
    x_range: Optional[List[float]] = Field(None, description="[min, max] for x-axis")
    y_range: Optional[List[float]] = Field(None, description="[min, max] for y-axis")
    n_points: int = Field(15, ge=5, le=50, description="Grid resolution per axis")
