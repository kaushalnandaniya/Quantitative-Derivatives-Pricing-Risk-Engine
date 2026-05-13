"""
Regulatory Reporting Service (Basel III/IV)
=============================================
Computes regulatory risk metrics:
    - Regulatory VaR (10-day, 99%)
    - Stressed VaR (crisis period)
    - Expected Shortfall
    - Capital Charge
    - Leverage Ratio
    - Concentration Risk
"""

import math
import logging
import numpy as np
from typing import List, Dict

from services.risk_service import compute_portfolio_risk

logger = logging.getLogger(__name__)


# =============================================================================
# Regulatory VaR (10-day, 99%)
# =============================================================================

def regulatory_var(portfolio: List[Dict], confidence: float = 0.99) -> dict:
    """
    Compute 10-day regulatory VaR using sqrt(10) scaling rule.
    Basel requires 10-day holding period at 99% confidence.
    """
    # Compute 1-day VaR
    result_1d = compute_portfolio_risk(portfolio, method="monte_carlo", confidence=confidence, n_sims=100000, seed=42)
    var_1d = abs(result_1d.get("VaR", 0))
    cvar_1d = abs(result_1d.get("CVaR", 0))

    # Scale to 10-day using sqrt(10) rule
    sqrt_10 = math.sqrt(10)
    var_10d = var_1d * sqrt_10
    cvar_10d = cvar_1d * sqrt_10

    return {
        "var_1d": round(var_1d, 4),
        "var_10d": round(var_10d, 4),
        "cvar_1d": round(cvar_1d, 4),
        "cvar_10d": round(cvar_10d, 4),
        "confidence": confidence,
        "holding_period_days": 10,
        "scaling_method": "sqrt_time",
        "method": "monte_carlo",
        "portfolio_value": result_1d.get("portfolio_value", 0),
    }


# =============================================================================
# Stressed VaR (Crisis Period)
# =============================================================================

def stressed_var(portfolio: List[Dict], stress_factor: float = 2.0) -> dict:
    """
    Compute Stressed VaR using elevated volatility scenario.
    Simulates crisis conditions (2008/2020-style) by scaling vol by stress_factor.
    """
    # Create stressed portfolio with elevated vol
    stressed_portfolio = []
    for pos in portfolio:
        stressed_pos = pos.copy()
        stressed_pos["sigma"] = pos["sigma"] * stress_factor
        stressed_portfolio.append(stressed_pos)

    result = compute_portfolio_risk(stressed_portfolio, method="monte_carlo", confidence=0.99, n_sims=100000, seed=42)
    svar = abs(result.get("VaR", 0)) * math.sqrt(10)

    return {
        "stressed_var_10d": round(svar, 4),
        "stress_factor": stress_factor,
        "stress_scenario": f"Vol × {stress_factor}",
        "original_portfolio_value": result.get("portfolio_value", 0),
    }


# =============================================================================
# Capital Charge
# =============================================================================

def capital_charge(portfolio: List[Dict], avg_var_60d: float = None) -> dict:
    """
    Regulatory capital charge = max(VaR, k × avg_VaR_60d) + sVaR
    where k = multiplier (minimum 3, typically 3-4).
    """
    k = 3.0  # Basel multiplier

    reg_var = regulatory_var(portfolio)
    s_var = stressed_var(portfolio)

    current_var = reg_var["var_10d"]
    svar_val = s_var["stressed_var_10d"]

    # If no historical average, use current VaR as proxy
    if avg_var_60d is None:
        avg_var_60d = current_var

    var_component = max(current_var, k * avg_var_60d)
    total_charge = var_component + svar_val

    return {
        "current_var_10d": round(current_var, 4),
        "avg_var_60d": round(avg_var_60d, 4),
        "multiplier": k,
        "var_component": round(var_component, 4),
        "stressed_var_10d": round(svar_val, 4),
        "total_capital_charge": round(total_charge, 4),
        "portfolio_value": reg_var["portfolio_value"],
        "capital_ratio": round(total_charge / max(reg_var["portfolio_value"], 1), 4),
    }


# =============================================================================
# Leverage Ratio
# =============================================================================

def leverage_ratio(portfolio: List[Dict], tier1_capital: float = None) -> dict:
    """
    Leverage ratio = Tier 1 Capital / Total Exposure.
    Basel III minimum: 3%.
    """
    total_exposure = 0
    for pos in portfolio:
        notional = abs(pos.get("qty", 1)) * pos.get("S", 0)
        total_exposure += notional

    # If Tier 1 not specified, estimate from portfolio value
    if tier1_capital is None:
        result = compute_portfolio_risk(portfolio)
        tier1_capital = abs(result.get("portfolio_value", 0))

    ratio = tier1_capital / max(total_exposure, 1)
    min_ratio = 0.03  # Basel III minimum

    return {
        "tier1_capital": round(tier1_capital, 4),
        "total_exposure": round(total_exposure, 4),
        "leverage_ratio": round(ratio, 6),
        "minimum_requirement": min_ratio,
        "adequate": ratio >= min_ratio,
        "surplus_deficit": round(ratio - min_ratio, 6),
    }


# =============================================================================
# Concentration Risk
# =============================================================================

def concentration_risk(portfolio: List[Dict]) -> dict:
    """
    Analyze single-name/strike concentration in the portfolio.
    Flags positions exceeding 25% of total exposure.
    """
    total_notional = 0
    by_strike = {}
    by_type = {"call": 0, "put": 0}

    for pos in portfolio:
        notional = abs(pos.get("qty", 1)) * pos.get("S", 0)
        total_notional += notional
        strike = pos.get("K", 0)
        by_strike[strike] = by_strike.get(strike, 0) + notional
        by_type[pos.get("type", "call")] += notional

    concentrations = []
    for strike, notional in sorted(by_strike.items(), key=lambda x: -x[1]):
        pct = notional / max(total_notional, 1)
        concentrations.append({
            "strike": strike,
            "notional": round(notional, 2),
            "percentage": round(pct * 100, 2),
            "breach": pct > 0.25,
        })

    breaches = [c for c in concentrations if c["breach"]]

    return {
        "total_notional": round(total_notional, 2),
        "type_breakdown": {k: round(v, 2) for k, v in by_type.items()},
        "concentrations": concentrations,
        "breach_count": len(breaches),
        "breaches": breaches,
        "concentration_limit": "25%",
    }


# =============================================================================
# Full Regulatory Report
# =============================================================================

def full_regulatory_report(portfolio: List[Dict]) -> dict:
    """Generate a complete Basel III/IV regulatory report."""
    logger.info(f"Generating regulatory report for {len(portfolio)} positions")

    reg_v = regulatory_var(portfolio)
    s_var = stressed_var(portfolio)
    cap = capital_charge(portfolio)
    lev = leverage_ratio(portfolio)
    conc = concentration_risk(portfolio)

    return {
        "report_type": "basel_iii_iv_regulatory",
        "n_positions": len(portfolio),
        "regulatory_var": reg_v,
        "stressed_var": s_var,
        "capital_charge": cap,
        "leverage_ratio": lev,
        "concentration_risk": conc,
        "summary": {
            "var_10d_99": reg_v["var_10d"],
            "stressed_var_10d": s_var["stressed_var_10d"],
            "total_capital_charge": cap["total_capital_charge"],
            "leverage_adequate": lev["adequate"],
            "concentration_breaches": conc["breach_count"],
        },
    }
