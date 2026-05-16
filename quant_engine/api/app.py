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

import os
import time
import logging
import traceback

from dotenv import load_dotenv
load_dotenv()  # Load .env before any config reads

from fastapi import FastAPI, Request, WebSocket
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
from api.routes.auth import router as auth_router
from api.routes.portfolios import router as portfolios_router
from api.routes.trades import router as trades_router
from api.routes.reports import router as reports_router
from api.routes.alerts import router as alerts_router
from api.routes.orders import router as orders_router
from api.routes.admin import router as admin_router
from api.routes.regulatory import router as regulatory_router

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
    title="Quant Engine Platform",
    version="3.0.0",
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

app.include_router(auth_router)
app.include_router(pricing_router)
app.include_router(risk_router)
app.include_router(greeks_router)
app.include_router(market_router)
app.include_router(strategies_router)
app.include_router(scenario_router)
app.include_router(portfolio_greeks_router)
app.include_router(portfolios_router)
app.include_router(trades_router)
app.include_router(reports_router)
app.include_router(alerts_router)
app.include_router(orders_router)
app.include_router(admin_router)
app.include_router(regulatory_router)

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
        "version": "3.0.0",
        "tier": "3",
        "engine": "Quant Engine Platform",
        "endpoints": {
            "auth": ["/auth/send-otp", "/auth/register", "/auth/login", "/auth/refresh", "/auth/me"],
            "pricing": ["/price/black-scholes", "/price/monte-carlo", "/price/binomial"],
            "risk": ["/risk/portfolio"],
            "greeks": ["/greeks/calculate", "/greeks/portfolio"],
            "market": ["/market/status", "/market/quote/{symbol}", "/market/option-chain/{symbol}"],
            "strategies": ["/strategies/list", "/strategies/simulate"],
            "scenario": ["/scenario/stress-test", "/scenario/heatmap"],
            "portfolios": ["/portfolios", "/portfolios/{id}", "/portfolios/{id}/calculate-risk"],
            "trades": ["/trades", "/trades/{id}", "/trades/{id}/close", "/trades/positions"],
            "reports": ["/reports/trades", "/reports/portfolios", "/reports/risk"],
            "alerts": ["/alerts", "/alerts/{id}", "/alerts/evaluate"],
            "orders": ["/orders", "/orders/{id}", "/orders/{id}/fill", "/orders/risk-check", "/orders/executions/history"],
            "admin": ["/admin/tenants", "/admin/users", "/admin/audit-log", "/admin/system"],
            "regulatory": ["/regulatory/var", "/regulatory/stressed-var", "/regulatory/capital-charge", "/regulatory/leverage", "/regulatory/concentration", "/regulatory/full-report"],
            "monitoring": ["/metrics", "/health/deep"],
            "websocket": ["ws://localhost:8000/ws/market/{symbol}"],
            "docs": ["/docs", "/redoc"],
        },
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

from api.websockets.market_feed import market_feed_handler

@app.websocket("/ws/market/{symbol}")
async def ws_market(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data."""
    await market_feed_handler(websocket, symbol)


# =============================================================================
# Monitoring Endpoints
# =============================================================================

from fastapi.responses import Response as RawResponse

@app.get("/metrics", tags=["Monitoring"], summary="Prometheus Metrics")
def metrics():
    """Expose Prometheus metrics endpoint."""
    from services.monitoring import get_metrics_response
    body, content_type = get_metrics_response()
    return RawResponse(content=body, media_type=content_type)

@app.get("/health/deep", tags=["Monitoring"], summary="Deep Health Check")
def deep_health():
    """Deep health check with system metrics."""
    from services.monitoring import system_health
    return system_health()


# =============================================================================
# Database Initialization on Startup
# =============================================================================

@app.on_event("startup")
def on_startup():
    from db.database import init_db, SessionLocal
    init_db()

    # Initialize Bloom filter with existing registered emails
    try:
        from services.bloom_filter import load_existing_emails
        db = SessionLocal()
        count = load_existing_emails(db)
        db.close()
        logger.info(f"Bloom filter loaded {count} emails on startup")
    except Exception as e:
        logger.warning(f"Bloom filter initialization skipped: {e}")

    logger.info("Quant Engine Platform v3.0.0 (Tier 3) started")
