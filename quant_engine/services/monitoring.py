"""
Monitoring & Observability Service
=====================================
Prometheus metrics, health checks, and performance monitoring.
Falls back to basic stats if prometheus packages aren't installed.
"""

import logging
import time
import os
import psutil

logger = logging.getLogger(__name__)

# Try to import prometheus
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True

    # Custom metrics
    REQUEST_COUNT = Counter(
        "quant_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "quant_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    PRICING_LATENCY = Histogram(
        "quant_pricing_duration_seconds",
        "Pricing engine latency",
        ["model"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
    )
    ACTIVE_WEBSOCKETS = Gauge(
        "quant_websocket_connections",
        "Active WebSocket connections",
    )
    DB_POOL_SIZE = Gauge(
        "quant_db_pool_size",
        "Database connection pool size",
    )
    CACHE_HITS = Counter("quant_cache_hits_total", "Cache hits")
    CACHE_MISSES = Counter("quant_cache_misses_total", "Cache misses")

    APP_INFO = Info("quant_app", "Application info")
    APP_INFO.info({
        "version": "3.0.0",
        "tier": "3",
        "engine": "quant_engine",
    })

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics disabled")


# =============================================================================
# Metrics Collection
# =============================================================================

def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record an HTTP request metric."""
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_pricing(model: str, duration: float):
    """Record pricing engine latency."""
    if PROMETHEUS_AVAILABLE:
        PRICING_LATENCY.labels(model=model).observe(duration)


def record_cache_hit():
    if PROMETHEUS_AVAILABLE:
        CACHE_HITS.inc()


def record_cache_miss():
    if PROMETHEUS_AVAILABLE:
        CACHE_MISSES.inc()


def set_websocket_count(count: int):
    if PROMETHEUS_AVAILABLE:
        ACTIVE_WEBSOCKETS.set(count)


# =============================================================================
# Prometheus Endpoint
# =============================================================================

def get_metrics_response():
    """Generate Prometheus metrics response."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"# Prometheus not available\n", "text/plain"


# =============================================================================
# System Health
# =============================================================================

def system_health() -> dict:
    """Comprehensive system health check."""
    try:
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        cpu_pct = process.cpu_percent(interval=0.1)
    except Exception:
        memory = None
        cpu_pct = None

    health = {
        "status": "healthy",
        "version": "3.0.0",
        "tier": "3",
        "process": {
            "pid": os.getpid(),
            "memory_mb": round(memory.rss / 1024 / 1024, 2) if memory else None,
            "cpu_percent": cpu_pct,
        },
        "system": {
            "cpu_count": os.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "available_memory_gb": round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2),
        },
        "prometheus_enabled": PROMETHEUS_AVAILABLE,
    }

    # Check Redis
    try:
        from services.cache_service import get_cache, cache_stats
        stats = cache_stats()
        health["cache"] = stats
    except Exception:
        health["cache"] = {"status": "unavailable"}

    # Check Celery
    try:
        from services.tasks import CELERY_AVAILABLE
        health["celery"] = {"available": CELERY_AVAILABLE}
    except Exception:
        health["celery"] = {"available": False}

    return health
