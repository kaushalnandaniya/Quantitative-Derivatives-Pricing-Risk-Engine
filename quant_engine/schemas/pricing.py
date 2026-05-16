"""
Pricing Schemas
================
Pydantic models for option pricing API endpoints.

Domain-aware validation ensures:
    - Prices and volatility are positive
    - Rates are bounded to realistic ranges
    - Simulation counts are within safe limits
"""

from typing import Literal
from pydantic import BaseModel, Field


class OptionInput(BaseModel):
    """Base input for all option pricing models."""

    S: float = Field(..., gt=0, description="Current spot price of the underlying asset")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, le=30.0, description="Time to maturity in years")
    r: float = Field(..., ge=-0.1, le=1.0, description="Risk-free interest rate (e.g., 0.05 for 5%)")
    sigma: float = Field(..., gt=0, le=5.0, description="Annualized volatility (e.g., 0.2 for 20%)")
    option_type: Literal["call", "put"] = Field(..., description="Option type: 'call' or 'put'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "S": 100.0,
                    "K": 100.0,
                    "T": 1.0,
                    "r": 0.05,
                    "sigma": 0.2,
                    "option_type": "call",
                }
            ]
        }
    }


class MonteCarloInput(OptionInput):
    """Input for Monte Carlo pricing with variance reduction options."""

    n_sims: int = Field(
        100_000,
        ge=1_000,
        le=5_000_000,
        description="Number of Monte Carlo simulations (1K–5M)"
    )
    method: Literal["standard", "antithetic", "control"] = Field(
        "standard",
        description="Variance reduction method"
    )
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "S": 100.0,
                    "K": 100.0,
                    "T": 1.0,
                    "r": 0.05,
                    "sigma": 0.2,
                    "option_type": "call",
                    "n_sims": 100_000,
                    "method": "antithetic",
                    "seed": 42,
                }
            ]
        }
    }


class BinomialInput(OptionInput):
    """Input for Binomial tree pricing with exercise style."""

    style: Literal["european", "american"] = Field(
        "european",
        description="Exercise style: 'european' or 'american'"
    )
    N: int = Field(
        200,
        ge=10,
        le=10_000,
        description="Number of time steps in the binomial tree (10–10K)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "S": 100.0,
                    "K": 100.0,
                    "T": 1.0,
                    "r": 0.05,
                    "sigma": 0.2,
                    "option_type": "put",
                    "style": "american",
                    "N": 500,
                }
            ]
        }
    }
