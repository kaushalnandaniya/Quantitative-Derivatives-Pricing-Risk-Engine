"""
Risk Schemas
=============
Pydantic models for portfolio risk API endpoints.

Validates:
    - Portfolio positions have valid parameters
    - Confidence levels are in (0.5, 1.0)
    - Correlation matrices (if provided) are well-formed
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class PositionInput(BaseModel):
    """A single option position within a portfolio."""

    type: Literal["call", "put"] = Field(..., description="Option type")
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, le=30.0, description="Time to maturity (years)")
    r: float = Field(0.05, ge=-0.1, le=1.0, description="Risk-free rate")
    sigma: float = Field(0.2, gt=0, le=5.0, description="Volatility")
    qty: int = Field(1, description="Quantity (positive=long, negative=short)")
    asset: str = Field("default", description="Asset identifier for multi-asset portfolios")


class PortfolioRiskInput(BaseModel):
    """Input for portfolio-level risk analysis."""

    portfolio: List[PositionInput] = Field(
        ...,
        min_length=1,
        description="List of option positions in the portfolio"
    )
    method: Literal["historical", "parametric", "monte_carlo"] = Field(
        "historical",
        description="VaR/CVaR calculation method"
    )
    confidence: float = Field(
        0.95,
        gt=0.5,
        lt=1.0,
        description="Confidence level for VaR/CVaR (e.g., 0.95 for 95%)"
    )
    n_sims: int = Field(
        100_000,
        ge=1_000,
        le=1_000_000,
        description="Number of Monte Carlo simulations for P&L"
    )
    horizon_days: int = Field(
        1,
        ge=1,
        le=252,
        description="Risk horizon in trading days"
    )
    seed: int = Field(42, ge=0, description="Random seed")
    correlation_matrix: Optional[List[List[float]]] = Field(
        None,
        description="Correlation matrix for multi-asset portfolios (n×n). "
                     "If None, assets are assumed uncorrelated."
    )

    @model_validator(mode="after")
    def validate_correlation_matrix(self):
        """Validate correlation matrix dimensions match unique assets."""
        if self.correlation_matrix is not None:
            assets = sorted(set(p.asset for p in self.portfolio))
            n = len(assets)
            matrix = self.correlation_matrix

            if len(matrix) != n:
                raise ValueError(
                    f"Correlation matrix has {len(matrix)} rows but portfolio "
                    f"has {n} unique assets ({assets})"
                )
            for i, row in enumerate(matrix):
                if len(row) != n:
                    raise ValueError(
                        f"Correlation matrix row {i} has {len(row)} columns, "
                        f"expected {n}"
                    )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "portfolio": [
                        {"type": "call", "S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2, "qty": 10},
                        {"type": "put", "S": 100, "K": 95, "T": 0.25, "r": 0.05, "sigma": 0.25, "qty": 5},
                    ],
                    "method": "historical",
                    "confidence": 0.95,
                    "n_sims": 100_000,
                    "horizon_days": 1,
                    "seed": 42,
                }
            ]
        }
    }
