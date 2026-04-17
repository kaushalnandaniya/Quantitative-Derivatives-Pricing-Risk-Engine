# Quant Engine

A production-grade quantitative finance system with a REST API, pricing engines, risk analytics, and Greeks computation.

## 🚀 Key Features

- **3 Pricing Models**: Black-Scholes (analytical), Monte Carlo (3 variance reduction methods), Binomial Tree (European & American)
- **Risk Engine**: Portfolio VaR, CVaR, multi-asset correlated P&L simulation
- **Greeks**: Delta, Gamma, Vega, Theta (analytical & numerical)
- **REST API**: FastAPI with Pydantic validation, Swagger docs, CORS support
- **Production Architecture**: API → Service → Core separation, structured logging, error handling
- **141 Tests**: Full coverage across core modules + API integration tests

## 📁 Project Structure
```
quant_engine/
├── api/                        # REST API layer
│   ├── app.py                  # FastAPI app, middleware, error handling
│   └── routes/
│       ├── pricing.py          # /price/* endpoints
│       ├── risk.py             # /risk/* endpoints
│       └── greeks.py           # /greeks/* endpoints
├── schemas/                    # Pydantic input validation
│   ├── pricing.py              # OptionInput, MonteCarloInput, BinomialInput
│   ├── risk.py                 # PortfolioRiskInput, PositionInput
│   └── greeks.py               # GreeksInput
├── services/                   # Business logic orchestration
│   ├── pricing_service.py      # Delegates to pricing modules
│   ├── risk_service.py         # Portfolio risk pipeline
│   └── greeks_service.py       # Greeks computation
├── pricing/                    # Core pricing engines
│   ├── black_scholes.py        # BS analytical pricing
│   ├── monte_carlo.py          # MC simulation (standard/antithetic/control)
│   ├── binomial.py             # CRR binomial tree (European/American)
│   ├── greeks.py               # Greeks calculator
│   └── visualizations.py       # Production-quality plots
├── risk/                       # Core risk engine
│   ├── portfolio.py            # Portfolio representation & valuation
│   ├── pnl.py                  # P&L simulation (single & multi-asset)
│   ├── var.py                  # VaR (historical, parametric, MC)
│   ├── cvar.py                 # Expected Shortfall / CVaR
│   └── correlation.py          # Cholesky-based correlated GBM
├── models/                     # Mathematical models
│   └── gbm.py                  # GBM terminal price simulator
├── config/                     # Configuration
│   └── settings.py             # Market params, MC/risk config, plot styles
├── experiments/                # Analysis & research
├── tests/                      # Test suite (141 tests)
│   ├── test_api.py             # API integration tests (40 tests)
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   ├── test_binomial.py
│   ├── test_greeks.py
│   ├── test_portfolio.py
│   └── test_var.py
├── main.py                     # CLI entry point (full analysis pipeline)
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/price/black-scholes` | Black-Scholes pricing |
| `POST` | `/price/monte-carlo` | Monte Carlo pricing (standard/antithetic/control) |
| `POST` | `/price/binomial` | Binomial tree pricing (European/American) |
| `POST` | `/risk/portfolio` | Portfolio VaR, CVaR & P&L analysis |
| `POST` | `/greeks/calculate` | Delta, Gamma, Vega, Theta |

### Example: Price an Option
```bash
curl -X POST http://localhost:8000/price/black-scholes \
  -H "Content-Type: application/json" \
  -d '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}'
```

### Example: Portfolio Risk
```bash
curl -X POST http://localhost:8000/risk/portfolio \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": [
      {"type": "call", "S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2, "qty": 10},
      {"type": "put", "S": 100, "K": 95, "T": 0.25, "r": 0.05, "sigma": 0.25, "qty": 5}
    ],
    "confidence": 0.95,
    "n_sims": 100000
  }'
```

## 🛠 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run API Server
```bash
cd quant_engine
uvicorn api.app:app --reload
```
Then open **http://localhost:8000/docs** for interactive Swagger UI.

### Run Tests
```bash
cd quant_engine
pytest tests/ -v
```

### Run Full Analysis Pipeline (CLI)
```bash
cd quant_engine
python main.py
```

## 🏗️ Architecture

```
Client (Swagger / Dashboard / curl)
        ↓
API Layer (FastAPI routes — zero business logic)
        ↓
Service Layer (orchestration, timing, error handling)
        ↓
Core Modules (pricing, risk, models)
```

**Design Principles:**
- **API routes** validate input and delegate — nothing else
- **Services** orchestrate core modules and add metadata
- **Core modules** are pure computational engines — no HTTP awareness
- **Pydantic schemas** enforce domain constraints at the boundary

## 📊 The Black-Scholes Formula
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

**Assumptions:** GBM log-returns, no arbitrage, constant risk-free rate, no transaction costs.