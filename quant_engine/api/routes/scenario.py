"""
Scenario Analysis Routes
==========================
API endpoints for scenario analysis:
    POST /scenario/stress-test
    POST /scenario/heatmap
"""

import logging

from fastapi import APIRouter

from schemas.scenario import StressTestInput, HeatmapInput
from services.scenario import stress_test, generate_heatmap

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scenario", tags=["Scenario Analysis"])


@router.post("/stress-test", summary="Portfolio Stress Test")
def run_stress_test(data: StressTestInput):
    """Run stress test under multiple spot/vol/time/rate scenarios."""
    positions = [pos.model_dump() for pos in data.positions]
    return stress_test(
        positions=positions,
        spot_shifts=data.spot_shifts,
        vol_shifts=data.vol_shifts,
        time_shifts=data.time_shifts,
        rate_shifts=data.rate_shifts,
    )


@router.post("/heatmap", summary="P&L Heatmap")
def run_heatmap(data: HeatmapInput):
    """Generate a 2D P&L heatmap grid (spot×vol or spot×time)."""
    positions = [pos.model_dump() for pos in data.positions]
    return generate_heatmap(
        positions=positions,
        x_axis=data.x_axis, y_axis=data.y_axis,
        x_range=data.x_range, y_range=data.y_range,
        n_points=data.n_points,
    )
