"""
Quant Engine — FastAPI Application
=====================================
Production-grade REST API exposing the quant pricing, risk,
and Greeks engines.

Architecture:
    Client → API Routes → Service Layer → Core Modules

Run:
    uvicorn api.app:app --reload

Docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)
"""

import time
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.pricing import router as pricing_router
from api.routes.risk import router as risk_router
from api.routes.greeks import router as greeks_router
from api.routes.market import router as market_router
from api.routes.strategies import router as strategies_router
from api.routes.scenario import router as scenario_router
from api.routes.portfolio_greeks import router as portfolio_greeks_router

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_engine.api")

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Quant Engine API",
    version="1.0.0",
    description=(
        "Production-grade quantitative finance API.\n\n"
        "**Pricing Models:**\n"
        "- Black-Scholes (analytical)\n"
        "- Monte Carlo (standard, antithetic, control variate)\n"
        "- Binomial Tree (European & American)\n\n"
        "**Risk Analytics:**\n"
        "- Portfolio VaR (Historical, Parametric, Monte Carlo)\n"
        "- Expected Shortfall (CVaR)\n"
        "- Multi-asset correlated portfolios\n\n"
        "**Greeks:**\n"
        "- Delta, Gamma, Vega, Theta (analytical & numerical)\n"
    ),
    contact={
        "name": "Quant Engine",
    },
    license_info={
        "name": "MIT",
    },
)

# =============================================================================
# CORS Middleware (ready for Week 6 dashboard)
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Exception Handler
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all error handler.

    Returns structured JSON errors instead of raw 500 tracebacks.
    Logs the full traceback for debugging.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}"
    )
    logger.debug(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
        },
    )


# =============================================================================
# Request Timing Middleware
# =============================================================================


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log request timing for performance monitoring."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed_ms:.1f}ms)"
    )
    return response


# =============================================================================
# Routes
# =============================================================================

app.include_router(pricing_router)
app.include_router(risk_router)
app.include_router(greeks_router)
app.include_router(market_router)
app.include_router(strategies_router)
app.include_router(scenario_router)
app.include_router(portfolio_greeks_router)

# =============================================================================
# Dashboard Mounting
# =============================================================================

from pathlib import Path
dashboard_path = Path(__file__).parent.parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")

# =============================================================================
# Health Endpoint
# =============================================================================


@app.get(
    "/",
    tags=["System"],
    summary="Redirect to Dashboard",
    include_in_schema=False,
)
def root():
    """Redirects the root URL to the interactive dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Returns the API status and version.",
)
def health():
    """Health check — confirms the API is running."""
    return {
        "status": "running",
        "version": "2.0.0",
        "engine": "Quant Engine",
        "endpoints": {
            "pricing": ["/price/black-scholes", "/price/monte-carlo", "/price/binomial"],
            "risk": ["/risk/portfolio"],
            "greeks": ["/greeks/calculate", "/greeks/portfolio"],
            "market": ["/market/status", "/market/quote/{symbol}", "/market/option-chain/{symbol}"],
            "strategies": ["/strategies/list", "/strategies/simulate"],
            "scenario": ["/scenario/stress-test", "/scenario/heatmap"],
            "docs": ["/docs", "/redoc"],
        },
    }
