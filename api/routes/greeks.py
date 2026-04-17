"""
Greeks Routes
==============
API endpoint for options Greeks calculation:
    POST /greeks/calculate
"""

import logging

from fastapi import APIRouter

from schemas.greeks import GreeksInput
from services.greeks_service import compute_greeks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/greeks", tags=["Greeks"])


@router.post(
    "/calculate",
    summary="Calculate Option Greeks",
    description=(
        "Compute Delta, Gamma, Vega, and Theta for a European option. "
        "Supports analytical (closed-form) and numerical (finite difference) methods."
    ),
)
def calculate_greeks(data: GreeksInput):
    """Compute all Greeks for the given option."""
    return compute_greeks(
        S=data.S,
        K=data.K,
        T=data.T,
        r=data.r,
        sigma=data.sigma,
        option_type=data.option_type,
        method=data.method,
    )
