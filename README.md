# Quant Engine: Black-Scholes & Monte Carlo Pricing Suite

A high-performance, vectorized implementation of the Black-Scholes-Merton model and Monte Carlo simulation engine for European options.

## 🚀 Key Features
- **Vectorized Engine**: Handles millions of price paths simultaneously using NumPy.
- **Dual Greeks**: Analytical (closed-form) and Numerical (Finite Difference) implementations.
- **Monte Carlo**: Standard, Antithetic Variates, and Control Variates methods.
- **Robustness**: Handles edge cases like $T \to 0$ and $\sigma \to 0$ without crashes.
- **Validation**: Full test suite verifying Put-Call Parity, convergence, and variance reduction.

## 📁 Project Structure
```
quant_engine/
├── pricing/                    # Core pricing logic
│   ├── black_scholes.py        # Black-Scholes analytical pricing
│   ├── monte_carlo.py          # Monte Carlo simulation engine
│   ├── greeks.py               # Greeks (analytical & numerical)
│   └── visualizations.py       # Production-quality plots
├── models/                     # Mathematical models
│   └── gbm.py                  # Geometric Brownian Motion simulator
├── utils/                      # Shared helpers
│   ├── math_utils.py           # Safe division, numerical utilities
│   └── random_utils.py         # RNG helpers, seeds
├── experiments/                # Analysis & research
│   ├── convergence_analysis.py # MC convergence studies
│   └── variance_reduction.py   # Variance reduction comparison
├── tests/                      # Unit tests
│   ├── test_black_scholes.py   # BS pricing tests
│   ├── test_monte_carlo.py     # MC engine tests
│   └── test_greeks.py          # Greeks validation tests
├── notebooks/                  # Jupyter notebooks
│   └── visualization.ipynb     # Interactive exploration
├── config/                     # Configuration
│   └── settings.py             # Market params, MC config, plot styles
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## 📊 The Formula
The price of a Call option is derived via:
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

### Core Assumptions
1. **Geometric Brownian Motion**: Log-returns are normally distributed.
2. **Efficient Markets**: No arbitrage opportunities and constant risk-free rate.
3. **Frictionless**: No transaction costs or taxes.

## 🛠 Usage
```python
from pricing.black_scholes import black_scholes_price

price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"Option Fair Value: {price:.2f}")
```

## 🧪 Running Tests
```bash
cd quant_engine
pytest tests/ -v
```

## 🚀 Running the Full Pipeline
```bash
cd quant_engine
python main.py
```