"""
Greeks Schemas
===============
Pydantic models for Greeks calculation endpoint.
"""

from typing import Literal
from pydantic import Field

from schemas.pricing import OptionInput


class GreeksInput(OptionInput):
    """Input for Greeks calculation with method selection."""

    method: Literal["analytical", "numerical"] = Field(
        "analytical",
        description="Calculation method: 'analytical' (closed-form) or 'numerical' (finite difference)"
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
                    "option_type": "call",
                    "method": "analytical",
                }
            ]
        }
    }
