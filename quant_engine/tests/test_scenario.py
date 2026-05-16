"""Tests for Scenario Analysis Engine."""
import pytest
import numpy as np
from services.scenario import stress_test, generate_heatmap


SAMPLE_POSITIONS = [
    {"type": "call", "S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.2, "qty": 10},
    {"type": "put", "S": 100, "K": 95, "T": 1.0, "r": 0.05, "sigma": 0.25, "qty": 5},
]


class TestStressTest:
    def test_basic_stress_test(self):
        result = stress_test(SAMPLE_POSITIONS, spot_shifts=[-0.05, 0, 0.05], vol_shifts=[0.0])
        assert "scenarios" in result
        assert result["n_scenarios"] == 3 * 1 * 5 * 1  # spot * vol * time_defaults * rate_defaults

    def test_custom_shifts(self):
        result = stress_test(
            SAMPLE_POSITIONS,
            spot_shifts=[0.0], vol_shifts=[0.0], time_shifts=[0], rate_shifts=[0.0]
        )
        assert result["n_scenarios"] == 1
        # Base case: PnL should be ~0
        assert abs(result["scenarios"][0]["pnl"]) < 0.01

    def test_spot_up_call_value_increases(self):
        result = stress_test(
            [{"type": "call", "S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "qty": 1}],
            spot_shifts=[-0.10, 0.0, 0.10],
            vol_shifts=[0.0], time_shifts=[0], rate_shifts=[0.0],
        )
        pnls = [s["pnl"] for s in result["scenarios"]]
        # Spot up -> call value up -> positive PnL
        assert pnls[2] > pnls[1] > pnls[0]

    def test_greeks_in_scenarios(self):
        result = stress_test(SAMPLE_POSITIONS, spot_shifts=[0.0], vol_shifts=[0.0],
                             time_shifts=[0], rate_shifts=[0.0])
        g = result["scenarios"][0]["greeks"]
        assert "delta" in g and "gamma" in g and "rho" in g


class TestHeatmap:
    def test_basic_heatmap(self):
        result = generate_heatmap(SAMPLE_POSITIONS, n_points=5)
        assert "z_matrix" in result
        z = np.array(result["z_matrix"])
        assert z.shape == (5, 5)

    def test_heatmap_spot_vol(self):
        result = generate_heatmap(SAMPLE_POSITIONS, x_axis="spot", y_axis="vol", n_points=8)
        assert len(result["x_values"]) == 8
        assert len(result["y_values"]) == 8

    def test_heatmap_center_near_zero(self):
        result = generate_heatmap(SAMPLE_POSITIONS, n_points=7)
        z = np.array(result["z_matrix"])
        # Center of grid (no shift) should be near zero PnL
        center = z[3, 3]
        assert abs(center) < 5.0  # roughly zero with small numerical noise

    def test_base_value_positive(self):
        result = generate_heatmap(SAMPLE_POSITIONS, n_points=5)
        assert result["base_value"] > 0
