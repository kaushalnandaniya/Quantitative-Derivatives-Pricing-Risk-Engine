# End-to-End Quantitative Pricing, Simulation, and Risk Engine

A production-grade quantitative finance system featuring robust pricing engines, stochastic risk analytics, strategy simulation, live market data integration, and an interactive 7-page frontend dashboard.

## 🚀 Key Features
hi
- **3 Pricing Models**: Black-Scholes (analytical), Monte Carlo (3 variance reduction methods), Binomial Tree (European & American)
- **Risk Engine**: Portfolio VaR, CVaR, multi-asset correlated P&L simulation
- **Greeks**: Delta, Gamma, Vega, Theta, Rho (analytical & numerical) — per-option and portfolio-level aggregation
- **Implied Volatility Solver**: Newton-Raphson with bisection fallback for robust IV recovery
- **Strategy Simulator**: 8 pre-built multi-leg strategies (Straddle, Strangle, Iron Condor, Butterfly, Spreads) with P&L profiling, breakeven calculation, and net Greeks
- **Scenario Analysis**: Multi-dimensional stress testing (spot/vol/time/rate shifts) with 2D P&L heatmap generation
- **Market Data Integration**: Dual-mode provider — mock (NIFTY/BANKNIFTY/RELIANCE with IV smile) and live (Zerodha Kite API)
- **Interactive Dashboard**: Premium 7-page HTML/CSS/JS frontend with Plotly.js charts, option chain tables, and heatmap visualizations
- **REST API**: FastAPI with Pydantic validation, Swagger docs, CORS support — 14 endpoints
- **Production Architecture**: UI → API → Service → Core separation, structured logging, error handling
- **174 Tests**: Full coverage across core modules, new features, and API integration tests

## 💻 Tech Stack
- **Core Engine**: Python 3.12, NumPy, SciPy
- **Backend API**: FastAPI, Pydantic, Uvicorn
- **Frontend Dashboard**: HTML5, Vanilla JS, CSS3 (Glassmorphism), Plotly.js
- **Market Data**: Zerodha Kite Connect SDK (optional, mock mode available)
- **Testing**: Pytest (174 tests, 100% pass rate)

## 📁 Project Structure
```
quant_engine/
├── api/                           # REST API layer
│   ├── app.py                     # FastAPI app (serves dashboard + API)
│   └── routes/                    # Endpoint routers
│       ├── pricing.py             # BS, MC, Binomial endpoints
│       ├── risk.py                # Portfolio risk endpoint
│       ├── greeks.py              # Greeks calculation endpoint
│       ├── portfolio_greeks.py    # Aggregated portfolio Greeks
│       ├── strategies.py          # Strategy simulation endpoints
│       ├── scenario.py            # Stress test & heatmap endpoints
│       └── market.py              # Market data endpoints
├── dashboard/                     # Frontend SPA (7 pages)
│   ├── index.html                 # HTML shell & UI layout
│   ├── styles.css                 # Premium dark theme & glassmorphism
│   └── app.js                     # API integration & Plotly.js charting
├── schemas/                       # Pydantic input validation
│   ├── pricing.py                 # OptionInput, MonteCarloInput, BinomialInput
│   ├── risk.py                    # PortfolioRiskInput, PositionInput
│   ├── greeks.py                  # GreeksInput
│   ├── strategies.py              # StrategySimulateInput
│   ├── scenario.py                # StressTestInput, HeatmapInput
│   └── market.py                  # QuoteRequest, OptionChainRequest
├── services/                      # Business logic orchestration
│   ├── pricing_service.py         # Delegates to pricing modules
│   ├── risk_service.py            # Portfolio risk pipeline
│   ├── greeks_service.py          # Greeks computation
│   ├── portfolio_greeks.py        # Aggregated portfolio Greeks
│   ├── strategies.py              # Strategy builder & P&L simulator
│   ├── scenario.py                # Stress test & heatmap engine
│   └── market_data.py             # Mock + Kite market data providers
├── pricing/                       # Core pricing engines
│   ├── black_scholes.py           # BS analytical pricing
│   ├── monte_carlo.py             # MC simulation (standard/antithetic/control)
│   ├── binomial.py                # CRR binomial tree (European/American)
│   ├── greeks.py                  # Greeks calculator (Δ, Γ, V, Θ, ρ)
│   ├── implied_vol.py             # IV solver (Newton-Raphson + bisection)
│   └── visualizations.py          # Production-quality plots
├── risk/                          # Core risk engine
│   ├── portfolio.py               # Portfolio representation & valuation
│   ├── pnl.py                     # P&L simulation (single & multi-asset)
│   ├── var.py                     # VaR (historical, parametric, MC)
│   ├── cvar.py                    # Expected Shortfall / CVaR
│   └── correlation.py             # Cholesky-based correlated GBM
├── models/                        # Mathematical models
│   └── gbm.py                     # GBM terminal price simulator
├── config/                        # Configuration
│   └── settings.py                # Market params, MC/risk config, plot styles
├── experiments/                   # Analysis & research
├── tests/                         # Test suite (174 tests)
│   ├── test_api.py                # API integration tests (40 tests)
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   ├── test_binomial.py
│   ├── test_greeks.py
│   ├── test_portfolio.py
│   ├── test_var.py
│   ├── test_implied_vol.py        # IV solver tests (16 tests)
│   ├── test_strategies.py         # Strategy simulation tests (9 tests)
│   └── test_scenario.py           # Scenario analysis tests (8 tests)
├── main.py                        # CLI entry point (full analysis pipeline)
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check & endpoint listing |
| `POST` | `/price/black-scholes` | Black-Scholes pricing |
| `POST` | `/price/monte-carlo` | Monte Carlo pricing (standard/antithetic/control) |
| `POST` | `/price/binomial` | Binomial tree pricing (European/American) |
| `POST` | `/risk/portfolio` | Portfolio VaR, CVaR & P&L analysis |
| `POST` | `/greeks/calculate` | Delta, Gamma, Vega, Theta, Rho |
| `POST` | `/greeks/portfolio` | Aggregated portfolio-level Greeks |
| `GET`  | `/strategies/list` | Available strategy templates |
| `POST` | `/strategies/simulate` | Strategy P&L simulation with breakevens |
| `POST` | `/scenario/stress-test` | Multi-scenario portfolio stress test |
| `POST` | `/scenario/heatmap` | 2D P&L heatmap (spot×vol, spot×time) |
| `GET`  | `/market/status` | Market data provider status |
| `GET`  | `/market/quote/{symbol}` | Live/mock quote (NIFTY, BANKNIFTY, RELIANCE) |
| `GET`  | `/market/option-chain/{symbol}` | Full option chain with IV, Greeks, OI |

### Example: Price an Option
```bash
curl -X POST http://localhost:8000/price/black-scholes \
  -H "Content-Type: application/json" \
  -d '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}'
```

### Example: Simulate a Straddle Strategy
```bash
curl -X POST http://localhost:8000/strategies/simulate \
  -H "Content-Type: application/json" \
  -d '{"strategy_id": "straddle", "S": 24000, "K": 24000, "T": 0.08, "r": 0.069, "sigma": 0.14}'
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

### Example: P&L Heatmap
```bash
curl -X POST http://localhost:8000/scenario/heatmap \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [
      {"type": "call", "S": 24000, "K": 24000, "T": 0.08, "r": 0.069, "sigma": 0.14, "qty": 10}
    ],
    "x_axis": "spot", "y_axis": "vol", "n_points": 15
  }'
```

### Example: Market Quote
```bash
curl http://localhost:8000/market/quote/NIFTY
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
- **http://localhost:8000/dashboard** for the Interactive Dashboard (7 pages).
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

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | System health, API status, engine capabilities summary |
| **Pricing Lab** | Multi-model selector (BS/MC/Binomial) with interactive payoff charts |
| **Greeks Explorer** | Compute and visualize Delta, Gamma, Vega, Theta, Rho with delta profile chart |
| **Risk Engine** | Portfolio builder, VaR/CVaR calculation, P&L distribution histogram |
| **Strategy Sim** | 8 pre-built strategies with P&L payoff profiles, breakevens, net Greeks |
| **Scenarios** | Portfolio stress testing with 2D P&L heatmaps (spot×vol, spot×time) |
| **Market Data** | NIFTY/BANKNIFTY/RELIANCE quotes, full option chain with IV smile, OI, Greeks |

## 🎯 Strategy Templates

| Strategy | Description | Legs |
|----------|-------------|------|
| Long Call | Bullish — unlimited upside | 1 |
| Long Put | Bearish — profit on price drop | 1 |
| Bull Call Spread | Moderate bull — capped profit/loss | 2 |
| Bear Put Spread | Moderate bear — capped profit/loss | 2 |
| Long Straddle | Volatility play — profit on large moves | 2 |
| Long Strangle | Cheaper vol play with OTM options | 2 |
| Iron Condor | Sell volatility — profit in range | 4 |
| Butterfly | Profit near strike at expiry | 3 |

## 🚀 Deployment & Demo

**Try it Live!**  
*(Add your deployed link here: `https://your-app.onrender.com`)*

### Deployment Options:
This project is container-ready. Because FastAPI serves *both* the backend API and frontend Dashboard concurrently, you only need to run a single web service on **Render**, **Railway**, or **Heroku**:
- **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`

### Demo Video / Portfolio:
If you are viewing this on GitHub, check out the [2-minute Demo Video](#) *(Add link here)* which showcases:
1. **Pricing Lab**: Dynamic payoff charts for Black-Scholes and Monte Carlo simulations.
2. **Strategy Simulator**: Building multi-leg strategies and visualizing P&L profiles with breakevens.
3. **Risk Engine**: Formulating portfolios and analyzing CVaR via empirical histogram mapping.
4. **Scenario Analysis**: 2D P&L heatmaps showing portfolio sensitivity to spot and volatility changes.
5. **Market Data**: Real-time option chain with IV smile and Greeks via Zerodha Kite API integration.
6. **Greeks Explorer**: Visualizing all 5 option sensitivities (Delta, Gamma, Vega, Theta, Rho) dynamically.

## 🏗️ Architecture

```
Client (Dashboard / Swagger / curl)
        ↓
API Layer (FastAPI — 14 endpoints, CORS, logging)
        ↓
Service Layer (orchestration, timing, error handling)
        ↓
Core Modules (pricing, risk, models, market data)
```

**Design Principles:**
- **API routes** validate input and delegate — nothing else
- **Services** orchestrate core modules and add metadata
- **Core modules** are pure computational engines — no HTTP awareness
- **Pydantic schemas** enforce domain constraints at the boundary
- **Market data** uses a provider pattern — swap mock for live with one config change

## 📊 The Black-Scholes Formula
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

**Assumptions:** GBM log-returns, no arbitrage, constant risk-free rate, no transaction costs.

## 📈 Implied Volatility Solver
$$\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - V_{market}}{\text{Vega}(\sigma_n)}$$

Newton-Raphson iteration with bisection fallback. Converges within 5-10 iterations for typical options.