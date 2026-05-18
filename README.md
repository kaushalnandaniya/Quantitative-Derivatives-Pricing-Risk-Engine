<div align="center">

# ⚡ Quantitative Derivatives Pricing & Risk Engine

### An institutional-grade options analytics, strategy simulation, and risk management platform

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-quant--engine--beta.vercel.app-2962ff?style=for-the-badge)](https://quant-engine-beta.vercel.app)
[![API Docs](https://img.shields.io/badge/📄_API_Docs-Swagger_UI-089981?style=for-the-badge)](https://quant-backend-qgay.onrender.com/docs)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js_16-000?logo=next.js&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37814A?logo=celery&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

</div>

---

## 🎯 Overview

A **full-stack quantitative finance platform** built for derivative traders, risk managers, and quant enthusiasts. It combines institutional-grade pricing models with a modern, interactive UI inspired by platforms like [Sensibull](https://sensibull.com).

**What sets this apart:**
- 🧮 **Three pricing engines** — Black-Scholes, Monte Carlo (with variance reduction), and Binomial Trees
- 📊 **Sensibull-style Strategy Builder** — Real option chain data, click-to-add legs, target-date payoff curves
- 📈 **Live market integration** — Zerodha Kite Connect API for real broker execution
- 🏦 **Basel III/IV regulatory engine** — VaR, Stressed VaR, capital charges, concentration risk
- ⚡ **Async architecture** — Celery + Redis for heavy computations, WebSocket-ready

---

## ✨ Features

### 🔧 Strategy Builder (Sensibull-style)
| Capability | Description |
|------------|-------------|
| **Stock Selection** | NIFTY, BANKNIFTY, RELIANCE with real-time spot prices |
| **Option Chain** | Full chain with strikes, IVs, prices from BS pricing engine |
| **Ready-made Strategies** | Buy Call, Bull Call Spread, Straddle, Iron Condor, Butterfly, and more |
| **Custom Leg Builder** | Click any CE/PE price to add a leg, toggle Buy/Sell, delete individually |
| **Payoff Visualization** | Interactive chart with profit/loss color gradient split at breakeven |
| **Target Date Pricing** | Slider to see P&L at any date before expiry using Black-Scholes |
| **Open Interest Chart** | Call/Put OI bar chart with Put-Call Ratio (PCR) |

### 📐 Pricing & Greeks
- **Black-Scholes-Merton** — Closed-form European option pricing
- **Monte Carlo** — 100K+ path simulation with antithetic variance reduction
- **Binomial Trees** — N-step CRR model for American options
- **Full Greeks Suite** — Delta, Gamma, Vega, Theta, Rho (analytical + numerical)

### 📊 Risk Management
- **Value at Risk (VaR)** — Historical simulation & parametric (99% / 95%)
- **Conditional VaR** — Expected Shortfall for tail risk
- **Scenario Analysis** — Stress test against historical crashes (2008, COVID, etc.)
- **Basel III/IV** — Regulatory VaR, Stressed VaR, capital charges, concentration limits

### 💹 Market Data & Trading
- **Zerodha Kite Connect** — Live order placement (Market, Limit, SL)
- **Option Chain API** — Real-time chain with IV skew simulation
- **Historical Data** — Yahoo Finance integration for charting
- **Trade Blotter** — Full OMS with position tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 16)                     │
│   Dashboard │ Strategy Builder │ Risk │ Market │ Trading         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API + JWT Auth
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ Pricing  │  │  Market  │  │   Risk   │  │  Regulatory   │   │
│  │  Engine  │  │   Data   │  │  Engine  │  │    Engine     │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬────────┘   │
│       │              │             │               │             │
│  Black-Scholes   Mock/Kite     VaR/CVaR      Basel III/IV       │
│  Monte Carlo     yFinance      Scenarios     Capital Charges    │
│  Binomial Tree   Option Chain  Stress Test   Concentration      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    PostgreSQL          Redis           Celery
    (Users, Trades,   (Cache,         (Async
     Portfolios)      Sessions)       Computations)
```

---

## 📂 Project Structure

```
├── frontend/                  # Next.js 16 application
│   ├── src/app/dashboard/     # All dashboard pages
│   │   ├── strategy/          # Sensibull-style Strategy Builder
│   │   ├── pricing/           # Options pricing calculator
│   │   ├── greeks/            # Greeks analyzer
│   │   ├── risk/              # Portfolio risk dashboard
│   │   ├── market/            # Live market data & charts
│   │   ├── trades/            # Trade blotter & OMS
│   │   ├── regulatory/        # Basel III/IV reports
│   │   └── scenario/          # Scenario analysis
│   └── src/lib/               # API client, auth store
│
├── quant_engine/              # Python FastAPI backend
│   ├── api/routes/            # REST endpoints
│   ├── pricing/               # BS, MC, Binomial engines
│   ├── services/              # Market data, strategies, risk
│   └── schemas/               # Pydantic models
│
├── docker-compose.yml         # Full stack orchestration
├── render.yaml                # Render.com IaC deployment
└── prometheus.yml             # Monitoring config
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, React 19, Tailwind CSS, Recharts, Zustand |
| **Backend** | Python 3.11+, FastAPI, Pydantic, NumPy, SciPy |
| **Database** | PostgreSQL (Render), SQLAlchemy |
| **Cache/Queue** | Redis (Upstash), Celery |
| **Auth** | JWT (access + refresh tokens), bcrypt, OTP verification |
| **Market Data** | Zerodha Kite Connect, Yahoo Finance, Mock Provider |
| **DevOps** | Docker, GitHub Actions CI, Render, Vercel |

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/kaushalnandaniya/Quantitative-Derivatives-Pricing-Risk-Engine.git
cd Quantitative-Derivatives-Pricing-Risk-Engine
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |

### Manual Setup

```bash
# Backend
cd quant_engine
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend
cd frontend
npm install && npm run dev
```

---

## ☁️ Deployment

| Component | Platform | Config |
|-----------|----------|--------|
| Frontend | **Vercel** | Auto-deploy from `main` branch |
| Backend | **Render** | Defined in `render.yaml` (Web Service + PostgreSQL) |
| Redis | **Upstash** | Serverless Redis for Celery broker & caching |
| CI/CD | **GitHub Actions** | Lint → Type Check → Build on every push |

---

## 📡 API Endpoints

| Group | Endpoints | Description |
|-------|-----------|-------------|
| **Auth** | `POST /auth/register`, `/login`, `/refresh` | JWT authentication with OTP |
| **Pricing** | `POST /pricing/price`, `/greeks`, `/surface` | BS, MC, Binomial pricing |
| **Market** | `GET /market/quote/{sym}`, `/option-chain/{sym}` | Live quotes & option chains |
| **Strategies** | `POST /strategies/simulate` | Multi-leg strategy P&L simulation |
| **Risk** | `POST /risk/var`, `/risk/scenario` | VaR, CVaR, stress testing |
| **Regulatory** | `POST /regulatory/full-report` | Basel III/IV capital adequacy |
| **Trades** | `POST /trades/book`, `GET /trades/list` | Order management system |
| **Portfolios** | `POST /portfolios/create`, `GET /portfolios/list` | Portfolio CRUD |

Full interactive documentation: **[Swagger UI →](https://quant-backend-qgay.onrender.com/docs)**

---

## 🧪 Testing

```bash
# Backend tests
pytest

# Frontend lint + type check + build
cd frontend && npm run lint && npm run build
```

---

## 📝 License

This project is licensed under the **MIT License**.

---

<div align="center">

**Built by [Kaushal Nandaniya](https://github.com/kaushalnandaniya)**

</div>
