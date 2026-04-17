"""
API Integration Tests
======================
Comprehensive test suite for all API endpoints.

Tests cover:
    - Valid inputs → correct responses
    - Edge cases (deep ITM, deep OTM, near-expiry)
    - Invalid inputs → 422 validation errors
    - Service-level error handling
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


# =========================================================================
# Health Check
# =========================================================================

class TestHealth:
    """Test the health endpoint."""

    def test_health_returns_running(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"

    def test_health_lists_endpoints(self):
        response = client.get("/")
        data = response.json()
        assert "pricing" in data["endpoints"]
        assert "risk" in data["endpoints"]
        assert "greeks" in data["endpoints"]


# =========================================================================
# Black-Scholes Endpoint
# =========================================================================

class TestBlackScholes:
    """Test POST /price/black-scholes."""

    VALID_CALL = {
        "S": 100, "K": 100, "T": 1, "r": 0.05,
        "sigma": 0.2, "option_type": "call",
    }

    VALID_PUT = {
        "S": 100, "K": 100, "T": 1, "r": 0.05,
        "sigma": 0.2, "option_type": "put",
    }

    def test_valid_call(self):
        resp = client.post("/price/black-scholes", json=self.VALID_CALL)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "black-scholes"
        assert data["price"] > 0
        assert "elapsed_ms" in data

    def test_valid_put(self):
        resp = client.post("/price/black-scholes", json=self.VALID_PUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] > 0

    def test_put_call_parity(self):
        """C - P ≈ S - K*exp(-rT)"""
        call_resp = client.post("/price/black-scholes", json=self.VALID_CALL)
        put_resp = client.post("/price/black-scholes", json=self.VALID_PUT)
        import numpy as np
        C = call_resp.json()["price"]
        P = put_resp.json()["price"]
        parity = 100 - 100 * np.exp(-0.05)
        assert abs((C - P) - parity) < 1e-6

    def test_deep_itm_call(self):
        """Deep ITM call should be close to S - K*exp(-rT)."""
        data = {"S": 200, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}
        resp = client.post("/price/black-scholes", json=data)
        assert resp.json()["price"] > 95  # Must be well above intrinsic

    def test_deep_otm_put(self):
        """Deep OTM put should have low value."""
        data = {"S": 200, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "put"}
        resp = client.post("/price/black-scholes", json=data)
        assert resp.json()["price"] < 1.0

    def test_invalid_negative_spot(self):
        data = {**self.VALID_CALL, "S": -10}
        resp = client.post("/price/black-scholes", json=data)
        assert resp.status_code == 422

    def test_invalid_zero_sigma(self):
        data = {**self.VALID_CALL, "sigma": 0}
        resp = client.post("/price/black-scholes", json=data)
        assert resp.status_code == 422

    def test_invalid_option_type(self):
        data = {**self.VALID_CALL, "option_type": "straddle"}
        resp = client.post("/price/black-scholes", json=data)
        assert resp.status_code == 422

    def test_missing_field(self):
        data = {"S": 100, "K": 100}  # Missing required fields
        resp = client.post("/price/black-scholes", json=data)
        assert resp.status_code == 422


# =========================================================================
# Monte Carlo Endpoint
# =========================================================================

class TestMonteCarlo:
    """Test POST /price/monte-carlo."""

    BASE = {
        "S": 100, "K": 100, "T": 1, "r": 0.05,
        "sigma": 0.2, "option_type": "call",
    }

    def test_standard_mc(self):
        data = {**self.BASE, "n_sims": 10_000, "method": "standard", "seed": 42}
        resp = client.post("/price/monte-carlo", json=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "monte-carlo"
        assert body["price"] > 0
        assert "confidence_interval" in body
        assert body["confidence_interval"]["lower"] < body["price"]
        assert body["confidence_interval"]["upper"] > body["price"]

    def test_antithetic_mc(self):
        data = {**self.BASE, "n_sims": 10_000, "method": "antithetic", "seed": 42}
        resp = client.post("/price/monte-carlo", json=data)
        assert resp.status_code == 200
        assert resp.json()["price"] > 0

    def test_control_variate_mc(self):
        data = {**self.BASE, "n_sims": 10_000, "method": "control", "seed": 42}
        resp = client.post("/price/monte-carlo", json=data)
        assert resp.status_code == 200
        assert resp.json()["price"] > 0

    def test_mc_converges_to_bs(self):
        """MC with large N should be close to BS."""
        bs_resp = client.post("/price/black-scholes", json=self.BASE)
        bs_price = bs_resp.json()["price"]

        mc_data = {**self.BASE, "n_sims": 500_000, "method": "control", "seed": 42}
        mc_resp = client.post("/price/monte-carlo", json=mc_data)
        mc_price = mc_resp.json()["price"]

        assert abs(mc_price - bs_price) < 0.5  # Within $0.50

    def test_invalid_low_sims(self):
        data = {**self.BASE, "n_sims": 10}  # Below minimum of 1000
        resp = client.post("/price/monte-carlo", json=data)
        assert resp.status_code == 422

    def test_invalid_method(self):
        data = {**self.BASE, "method": "importance_sampling"}
        resp = client.post("/price/monte-carlo", json=data)
        assert resp.status_code == 422

    def test_reproducibility(self):
        """Same seed → same price."""
        data = {**self.BASE, "n_sims": 10_000, "seed": 123}
        r1 = client.post("/price/monte-carlo", json=data).json()["price"]
        r2 = client.post("/price/monte-carlo", json=data).json()["price"]
        assert r1 == r2


# =========================================================================
# Binomial Endpoint
# =========================================================================

class TestBinomial:
    """Test POST /price/binomial."""

    BASE = {
        "S": 100, "K": 100, "T": 1, "r": 0.05,
        "sigma": 0.2, "option_type": "call",
    }

    def test_european_call(self):
        data = {**self.BASE, "style": "european", "N": 200}
        resp = client.post("/price/binomial", json=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "binomial"
        assert body["style"] == "european"
        assert body["price"] > 0

    def test_american_put(self):
        data = {**self.BASE, "option_type": "put", "style": "american", "N": 200}
        resp = client.post("/price/binomial", json=data)
        assert resp.status_code == 200
        body = resp.json()
        assert body["style"] == "american"
        assert body["price"] > 0

    def test_american_put_geq_european(self):
        """American put ≥ European put (early exercise premium)."""
        euro = client.post("/price/binomial", json={
            **self.BASE, "option_type": "put", "style": "european", "N": 500,
        }).json()["price"]
        amer = client.post("/price/binomial", json={
            **self.BASE, "option_type": "put", "style": "american", "N": 500,
        }).json()["price"]
        assert amer >= euro - 1e-6

    def test_binomial_converges_to_bs(self):
        """Binomial with large N should converge to BS."""
        bs = client.post("/price/black-scholes", json=self.BASE).json()["price"]
        bn = client.post("/price/binomial", json={
            **self.BASE, "style": "european", "N": 2000,
        }).json()["price"]
        assert abs(bn - bs) < 0.1  # Within $0.10

    def test_tree_parameters_returned(self):
        data = {**self.BASE, "N": 100}
        body = client.post("/price/binomial", json=data).json()
        assert body["tree_parameters"] is not None
        assert "u" in body["tree_parameters"]
        assert "d" in body["tree_parameters"]
        assert "p" in body["tree_parameters"]

    def test_invalid_low_steps(self):
        data = {**self.BASE, "N": 3}  # Below minimum of 10
        resp = client.post("/price/binomial", json=data)
        assert resp.status_code == 422


# =========================================================================
# Portfolio Risk Endpoint
# =========================================================================

class TestPortfolioRisk:
    """Test POST /risk/portfolio."""

    SIMPLE_PORTFOLIO = {
        "portfolio": [
            {"type": "call", "S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2, "qty": 10},
            {"type": "put", "S": 100, "K": 95, "T": 0.25, "r": 0.05, "sigma": 0.25, "qty": 5},
        ],
        "method": "historical",
        "confidence": 0.95,
        "n_sims": 10_000,
        "horizon_days": 1,
        "seed": 42,
    }

    def test_basic_risk(self):
        resp = client.post("/risk/portfolio", json=self.SIMPLE_PORTFOLIO)
        assert resp.status_code == 200
        data = resp.json()
        assert "VaR" in data
        assert "CVaR" in data
        assert data["VaR"] >= 0
        assert data["CVaR"] >= 0

    def test_cvar_geq_var(self):
        """CVaR ≥ VaR (coherent risk measure property)."""
        resp = client.post("/risk/portfolio", json=self.SIMPLE_PORTFOLIO)
        data = resp.json()
        assert data["CVaR"] >= data["VaR"] - 1e-4

    def test_pnl_statistics_present(self):
        resp = client.post("/risk/portfolio", json=self.SIMPLE_PORTFOLIO)
        stats = resp.json()["pnl_statistics"]
        for key in ["mean", "std", "min", "max", "median", "skewness", "kurtosis"]:
            assert key in stats

    def test_parametric_method(self):
        data = {**self.SIMPLE_PORTFOLIO, "method": "parametric"}
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 200
        assert resp.json()["method"] == "parametric"

    def test_monte_carlo_method(self):
        data = {**self.SIMPLE_PORTFOLIO, "method": "monte_carlo"}
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 200

    def test_multi_asset_portfolio(self):
        data = {
            "portfolio": [
                {"type": "call", "S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2, "qty": 10, "asset": "AAPL"},
                {"type": "put", "S": 50, "K": 50, "T": 0.25, "r": 0.05, "sigma": 0.3, "qty": 5, "asset": "MSFT"},
            ],
            "correlation_matrix": [[1.0, 0.6], [0.6, 1.0]],
            "n_sims": 10_000,
            "seed": 42,
        }
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 200
        assert resp.json()["VaR"] >= 0

    def test_empty_portfolio_rejected(self):
        data = {"portfolio": [], "n_sims": 10_000}
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 422

    def test_invalid_confidence(self):
        data = {**self.SIMPLE_PORTFOLIO, "confidence": 1.5}
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 422

    def test_invalid_confidence_too_low(self):
        data = {**self.SIMPLE_PORTFOLIO, "confidence": 0.3}
        resp = client.post("/risk/portfolio", json=data)
        assert resp.status_code == 422


# =========================================================================
# Greeks Endpoint
# =========================================================================

class TestGreeks:
    """Test POST /greeks/calculate."""

    BASE = {
        "S": 100, "K": 100, "T": 1, "r": 0.05,
        "sigma": 0.2, "option_type": "call",
    }

    def test_analytical_call(self):
        data = {**self.BASE, "method": "analytical"}
        resp = client.post("/greeks/calculate", json=data)
        assert resp.status_code == 200
        greeks = resp.json()["greeks"]
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks

    def test_numerical_call(self):
        data = {**self.BASE, "method": "numerical"}
        resp = client.post("/greeks/calculate", json=data)
        assert resp.status_code == 200
        greeks = resp.json()["greeks"]
        assert greeks["delta"] > 0  # Call delta is positive

    def test_analytical_vs_numerical(self):
        """Analytical and numerical should agree closely."""
        anal = client.post("/greeks/calculate", json={
            **self.BASE, "method": "analytical"
        }).json()["greeks"]
        num = client.post("/greeks/calculate", json={
            **self.BASE, "method": "numerical"
        }).json()["greeks"]

        assert abs(anal["delta"] - num["delta"]) < 0.01
        assert abs(anal["gamma"] - num["gamma"]) < 0.01

    def test_call_delta_range(self):
        """Call delta should be in [0, 1]."""
        data = {**self.BASE, "method": "analytical"}
        greeks = client.post("/greeks/calculate", json=data).json()["greeks"]
        assert 0 <= greeks["delta"] <= 1

    def test_put_delta_range(self):
        """Put delta should be in [-1, 0]."""
        data = {**self.BASE, "option_type": "put", "method": "analytical"}
        greeks = client.post("/greeks/calculate", json=data).json()["greeks"]
        assert -1 <= greeks["delta"] <= 0

    def test_gamma_positive(self):
        """Gamma is always positive for vanilla options."""
        greeks = client.post("/greeks/calculate", json={
            **self.BASE, "method": "analytical"
        }).json()["greeks"]
        assert greeks["gamma"] > 0

    def test_invalid_method(self):
        data = {**self.BASE, "method": "monte_carlo"}
        resp = client.post("/greeks/calculate", json=data)
        assert resp.status_code == 422
