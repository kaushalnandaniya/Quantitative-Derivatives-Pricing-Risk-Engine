# Quantitative Derivatives Pricing & Risk Engine

![Live Status](https://img.shields.io/badge/Status-Live-success)
![Frontend](https://img.shields.io/badge/Frontend-Next.js-black?logo=next.js)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Database](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql)

A high-performance, full-stack quantitative finance platform designed for options pricing, portfolio risk management, and live algorithmic trading. 

🌐 **Live Demo:** [https://quant-engine-beta.vercel.app](https://quant-engine-beta.vercel.app)

---

## 🚀 Key Features

### 1. Options Pricing & Analytics
*   **Pricing Models:** Implementation of Black-Scholes-Merton, Monte Carlo Simulations (with variance reduction), and Binomial Tree models.
*   **The Greeks:** Real-time calculation of Delta, Gamma, Vega, Theta, and Rho for individual options and complex multi-leg strategies.
*   **Volatility Surface:** Dynamic implied volatility surface generation and charting.

### 2. Portfolio Risk Management
*   **Value at Risk (VaR):** Historical and Parametric VaR calculations.
*   **Conditional VaR (CVaR / Expected Shortfall):** Tail risk estimation for complex portfolios.
*   **Scenario Analysis:** Stress testing portfolios against historical market crashes and extreme volatility shifts.

### 3. Live Market Data & Trading Integration
*   **Live Broker Connectivity:** Integrated with **Zerodha Kite Connect API** for live market execution (Market, Limit, SL orders).
*   **Historical Data:** Integrated with `yfinance` for free, robust historical charting and analysis.
*   **Interactive UI:** High-performance charting using Recharts to visualize pricing convergence, Greeks, and historical price action.

---

## 🏗️ Architecture

This repository is structured as a modern Monorepo containing both the frontend and backend services.

*   **`frontend/`**: A Next.js application providing a responsive, interactive, and aesthetic dashboard for risk managers and traders.
*   **`quant_engine/`**: A Python-based FastAPI backend handling heavy quantitative computations, orchestrated with Celery for async background tasks, and backed by a PostgreSQL database.

---

## 🛠️ Local Development Setup

The easiest way to run the entire stack locally is using Docker Compose.

### Prerequisites
*   [Docker](https://www.docker.com/) & Docker Compose installed.

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaushalnandaniya/Quantitative-Derivatives-Pricing-Risk-Engine.git
   cd Quantitative-Derivatives-Pricing-Risk-Engine
   ```

2. **Start the environment**
   ```bash
   docker-compose up --build
   ```

3. **Access the services**
   * Frontend Dashboard: `http://localhost:3000`
   * Backend API Docs (Swagger): `http://localhost:8000/docs`

---

## ☁️ Deployment

The platform is configured for modern cloud deployment architectures:
*   **Frontend (Vercel):** Automatically deployed via Vercel for highly cached, globally distributed Next.js hosting.
*   **Backend (Render):** Defined via `render.yaml` Infrastructure-as-Code. It runs the FastAPI server and Celery workers on Render Web Services, connected to a Render PostgreSQL instance and an external Upstash Redis broker.

---

## 📝 License
This project is licensed under the MIT License.
