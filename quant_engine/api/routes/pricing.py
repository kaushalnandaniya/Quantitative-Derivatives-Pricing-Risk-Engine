"""
Pricing Routes
===============
API endpoints for option pricing:
    POST /price/black-scholes
    POST /price/monte-carlo
    POST /price/binomial

Each endpoint validates input via Pydantic, delegates to the
pricing service, and returns structured JSON.
"""

import logging

from fastapi import APIRouter

from schemas.pricing import OptionInput, MonteCarloInput, BinomialInput
from services.pricing_service import (
    compute_black_scholes,
    compute_monte_carlo,
    compute_binomial,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/price", tags=["Pricing"])


@router.post(
    "/black-scholes",
    summary="Black-Scholes Pricing",
    description="Price a European option using the Black-Scholes closed-form formula.",
)
def price_black_scholes(data: OptionInput):
    """Compute the analytical Black-Scholes price for a European option."""
    return compute_black_scholes(
        S=data.S,
        K=data.K,
        T=data.T,
        r=data.r,
        sigma=data.sigma,
        option_type=data.option_type,
    )


@router.post(
    "/monte-carlo",
    summary="Monte Carlo Pricing",
    description=(
        "Price a European option via Monte Carlo simulation. "
        "Supports standard, antithetic, and control variate methods."
    ),
)
def price_monte_carlo(data: MonteCarloInput):
    """Compute Monte Carlo option price with confidence intervals."""
    return compute_monte_carlo(
        S=data.S,
        K=data.K,
        T=data.T,
        r=data.r,
        sigma=data.sigma,
        option_type=data.option_type,
        n_sims=data.n_sims,
        method=data.method,
        seed=data.seed,
    )


@router.post(
    "/binomial",
    summary="Binomial Tree Pricing",
    description=(
        "Price an option using the CRR (Cox-Ross-Rubinstein) binomial tree. "
        "Supports European and American exercise styles."
    ),
)
def price_binomial(data: BinomialInput):
    """Compute binomial tree option price (European or American)."""
    return compute_binomial(
        S=data.S,
        K=data.K,
        T=data.T,
        r=data.r,
        sigma=data.sigma,
        option_type=data.option_type,
        style=data.style,
        N=data.N,
    )
