# End-to-End Quantitative Pricing, Simulation, and Risk Engine

A production-grade quantitative finance system featuring robust pricing engines, stochastic risk analytics, and an interactive frontend dashboard.

## 🚀 Key Features

- **3 Pricing Models**: Black-Scholes (analytical), Monte Carlo (3 variance reduction methods), Binomial Tree (European & American)
- **Risk Engine**: Portfolio VaR, CVaR, multi-asset correlated P&L simulation
- **Greeks**: Delta, Gamma, Vega, Theta (analytical & numerical)
- **Interactive Dashboard**: Premium HTML/CSS/JS frontend served directly by FastAPI with interactive Plotly.js charts
- **REST API**: FastAPI with Pydantic validation, Swagger docs, CORS support
- **Production Architecture**: UI → API → Service → Core separation, structured logging, error handling
- **141 Tests**: Full coverage across core modules + API integration tests

## 💻 Tech Stack
- **Core Engine Engine**: Python 3.12, NumPy, SciPy
- **Backend API**: FastAPI, Pydantic, Uvicorn
- **Frontend Dashboard**: HTML5, Vanilla JS, CSS3 (Glassmorphism), Plotly.js
- **Testing**: Pytest (141 tests, 100% CI pass rate)

## 📁 Project Structure
```
quant_engine/
├── api/                        # REST API layer
│   ├── app.py                  # FastAPI app (serves dashboard and API route)
│   └── routes/                 # Endpoint routers
├── dashboard/                  # Frontend SPA
│   ├── index.html              # HTML shell & UI layout
│   ├── styles.css              # Premium dark theme & glassmorphism
│   └── app.js                  # API integration & Plotly.js charting
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

### Run API Server & Dashboard
```bash
cd quant_engine
uvicorn api.app:app --reload
```
Then open:
- **http://localhost:8000/dashboard** for the Interactive Dashboard.
- **http://localhost:8000/docs** for the interactive Swagger UI.

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

## 🚀 Deployment & Demo

**Try it Live!**  
*(Add your deployed link here: `https://your-app.onrender.com`)*

![Dashboard Preview](dashboard/screenshot.png) *(Note: Add a screenshot of your dashboard to the project root and link it here)*

### Deployment Options:
This project is container-ready. Because FastAPI serves *both* the backend API and frontend Dashboard concurrently, you only need to run a single web service on **Render**, **Railway**, or **Heroku**:
- **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`

### Demo Video / Portfolio:
If you are viewing this on GitHub, check out the [2-minute Demo Video](#) *(Add link here)* which showcases:
1. **Pricing Lab**: Dynamic payoff charts for Black-Scholes and Monte Carlo simulations.
2. **Risk Engine**: Formulating portfolios and analyzing CVaR via empirical histogram mapping.
3. **Greeks Explorer**: Visualizing option sensitivity dynamically.

## 🏗️ Architecture

```
Client (Dashboard / Swagger / curl)
        ↓
API Layer (FastAPI serving routes & static files)
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