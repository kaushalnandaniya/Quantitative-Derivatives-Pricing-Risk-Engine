"""
Celery Task Definitions
=========================
Async task queue for heavy computations.
Falls back to synchronous execution if Celery is not available.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Try to import Celery, fall back to synchronous execution
try:
    from celery import Celery
    import os
    CELERY_AVAILABLE = True

    celery_app = Celery(
        "quant_engine",
        broker=os.getenv("REDIS_URL", os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")),
        backend=os.getenv("REDIS_URL", os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")),
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_routes={
            "tasks.compute_*": {"queue": "compute"},
            "tasks.report_*": {"queue": "reports"},
            "tasks.batch_*": {"queue": "batch"},
        },
    )
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None
    logger.warning("celery package not installed — tasks will run synchronously")


# =============================================================================
# Synchronous Fallback
# =============================================================================

class SyncTaskResult:
    """Mimics Celery AsyncResult for synchronous execution."""

    def __init__(self, result):
        self.result = result
        self.id = "sync"
        self.status = "SUCCESS"

    def get(self, timeout=None):
        return self.result

    @property
    def ready(self):
        return True


def sync_task(func):
    """Decorator that makes a function behave like a Celery task when Celery isn't available."""
    class SyncWrapper:
        def __init__(self, fn):
            self.fn = fn
            self.__name__ = fn.__name__

        def delay(self, *args, **kwargs):
            result = self.fn(*args, **kwargs)
            return SyncTaskResult(result)

        def apply_async(self, args=None, kwargs=None, **opts):
            result = self.fn(*(args or []), **(kwargs or {}))
            return SyncTaskResult(result)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    return SyncWrapper(func)


# Choose decorator based on availability
task = celery_app.task if CELERY_AVAILABLE else sync_task


# =============================================================================
# Compute Tasks
# =============================================================================

@task
def compute_monte_carlo_price(S, K, T, r, sigma, option_type, n_sims=100000, method="standard"):
    """Run Monte Carlo pricing as an async task."""
    from services.pricing_service import price_option
    start = time.perf_counter()
    result = price_option("monte_carlo", S=S, K=K, T=T, r=r, sigma=sigma,
                          option_type=option_type, n_sims=n_sims, method=method)
    elapsed = (time.perf_counter() - start) * 1000
    result["async"] = True
    result["task_elapsed_ms"] = round(elapsed, 2)
    logger.info(f"Async MC price computed: {result.get('price', 'N/A')} ({elapsed:.1f}ms)")
    return result


@task
def compute_portfolio_risk_async(portfolio, method="monte_carlo", confidence=0.95, n_sims=100000):
    """Run portfolio risk calculation as an async task."""
    from services.risk_service import compute_portfolio_risk
    start = time.perf_counter()
    result = compute_portfolio_risk(portfolio, method=method, confidence=confidence, n_sims=n_sims)
    elapsed = (time.perf_counter() - start) * 1000
    result["async"] = True
    result["task_elapsed_ms"] = round(elapsed, 2)
    logger.info(f"Async risk computed: VaR={result.get('VaR', 'N/A')} ({elapsed:.1f}ms)")
    return result


@task
def compute_scenario_heatmap_async(positions, x_axis="spot", y_axis="vol", n_points=15):
    """Run scenario heatmap as an async task."""
    from services.scenario import compute_heatmap
    start = time.perf_counter()
    result = compute_heatmap(positions, x_axis=x_axis, y_axis=y_axis, n_points=n_points)
    elapsed = (time.perf_counter() - start) * 1000
    result["async"] = True
    result["task_elapsed_ms"] = round(elapsed, 2)
    logger.info(f"Async heatmap computed: {n_points}x{n_points} ({elapsed:.1f}ms)")
    return result


@task
def compute_regulatory_report_async(portfolio):
    """Run full regulatory report as an async task."""
    from services.regulatory_service import full_regulatory_report
    start = time.perf_counter()
    result = full_regulatory_report(portfolio)
    elapsed = (time.perf_counter() - start) * 1000
    result["async"] = True
    result["task_elapsed_ms"] = round(elapsed, 2)
    logger.info(f"Async regulatory report generated ({elapsed:.1f}ms)")
    return result


# =============================================================================
# Report Tasks
# =============================================================================

@task
def generate_eod_report(user_id):
    """Generate end-of-day risk report."""
    logger.info(f"EOD report task started for user={user_id}")
    return {"user_id": user_id, "report": "eod", "status": "generated"}


# =============================================================================
# Task Status Helper
# =============================================================================

def get_task_status(task_id: str) -> dict:
    """Get status of an async task."""
    if not CELERY_AVAILABLE:
        return {"task_id": task_id, "status": "SYNC_MODE", "note": "Celery not available, tasks run synchronously"}

    result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
    }
    if result.ready():
        try:
            response["result"] = result.get(timeout=1)
        except Exception as e:
            response["error"] = str(e)
    return response
