"""
Binomial Tree Pricing Engine
==============================
CRR (Cox-Ross-Rubinstein) binomial lattice pricer for European and American
options.
"""

import logging
import time
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


# =============================================================================
# CRR Parameter Computation
# =============================================================================

def _crr_params(
    sigma: float,
    r: float,
    T: float,
    N: int,
) -> Dict[str, float]:

    dt = T / N

    # 🔥 FIX 1: Proper handling of near-zero volatility
    if sigma < 1e-3:
        u = np.exp(r * dt)
        d = u
        p = 1.0
        discount = np.exp(-r * dt)
        return {
            "dt": dt,
            "u": u,
            "d": d,
            "p": p,
            "discount": discount,
        }

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    return {
        "dt": dt,
        "u": u,
        "d": d,
        "p": p,
        "discount": discount,
    }


# =============================================================================
# 1. Fast Pricer — O(N) Memory
# =============================================================================

def binomial_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    style: str = "european",
    N: int = 200,
) -> Dict:

    otype = option_type.lower()
    if otype not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    ex_style = style.lower()
    if ex_style not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got '{style}'")

    t_start = time.perf_counter()

    # 🔥 FIX 2: Deterministic pricing when sigma → 0
    if sigma < 1e-8:
        if otype == "call":
            price = max(S0 - K * np.exp(-r * T), 0.0)
        else:
            price = max(K * np.exp(-r * T) - S0, 0.0)

        return {
            "price": float(price),
            "params": None,
            "N": N,
            "style": ex_style,
            "option_type": otype,
            "elapsed_ms": 0.0,
        }

    # --- CRR parameters ---
    params = _crr_params(sigma, r, T, N)
    u, d, p, discount = params["u"], params["d"], params["p"], params["discount"]

    logger.debug(
        f"Binomial | N={N} | u={u:.6f} d={d:.6f} p={p:.6f} | "
        f"{ex_style} {otype}"
    )

    # --- Terminal stock prices ---
    j = np.arange(N + 1, dtype=np.float64)
    ST = S0 * (u ** j) * (d ** (N - j))

    # --- Terminal payoffs ---
    if otype == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # --- Backward induction ---
    for i in range(N - 1, -1, -1):
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])

        if ex_style == "american":
            j_i = np.arange(i + 1, dtype=np.float64)
            S_i = S0 * (u ** j_i) * (d ** (i - j_i))

            if otype == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)

            V = np.maximum(V, exercise)

    elapsed = (time.perf_counter() - t_start) * 1000

    return {
        "price": float(V[0]),
        "params": params,
        "N": N,
        "style": ex_style,
        "option_type": otype,
        "elapsed_ms": elapsed,
    }


# =============================================================================
# 2. Full-Tree Pricer — For Visualization
# =============================================================================

def binomial_price_with_tree(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    style: str = "european",
    N: int = 50,
) -> Dict:

    otype = option_type.lower()
    if otype not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    ex_style = style.lower()
    if ex_style not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got '{style}'")

    # 🔥 FIX 3: Deterministic case
    if sigma < 1e-8:
        if otype == "call":
            price = max(S0 - K * np.exp(-r * T), 0.0)
        else:
            price = max(K * np.exp(-r * T) - S0, 0.0)

        return {
            "price": float(price),
            "stock_tree": None,
            "option_tree": None,
            "exercise_boundary": None,
            "early_exercise_map": None,
            "params": None,
            "N": N,
            "style": ex_style,
            "option_type": otype,
        }

    params = _crr_params(sigma, r, T, N)
    u, d, p, discount = params["u"], params["d"], params["p"], params["discount"]

    stock_tree = []
    for i in range(N + 1):
        j = np.arange(i + 1, dtype=np.float64)
        S_i = S0 * (u ** j) * (d ** (i - j))
        stock_tree.append(S_i)

    ST = stock_tree[N]
    if otype == "call":
        V_N = np.maximum(ST - K, 0.0)
    else:
        V_N = np.maximum(K - ST, 0.0)

    option_tree = [None] * (N + 1)
    option_tree[N] = V_N.copy()

    early_exercise_map = [None] * (N + 1)
    early_exercise_map[N] = np.zeros(N + 1, dtype=bool)

    exercise_boundary = []

    V = V_N.copy()
    for i in range(N - 1, -1, -1):
        hold = discount * (p * V[1:] + (1.0 - p) * V[:-1])

        if ex_style == "american":
            S_i = stock_tree[i]

            if otype == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)

            exercised = exercise > hold
            V = np.maximum(hold, exercise)

            early_exercise_map[i] = exercised

            if otype == "put" and np.any(exercised):
                boundary_price = float(np.max(S_i[exercised]))
                step_time = i * params["dt"]
                exercise_boundary.append((step_time, boundary_price))
            elif otype == "call" and np.any(exercised):
                boundary_price = float(np.min(S_i[exercised]))
                step_time = i * params["dt"]
                exercise_boundary.append((step_time, boundary_price))
        else:
            V = hold
            early_exercise_map[i] = np.zeros(i + 1, dtype=bool)

        option_tree[i] = V.copy()

    return {
        "price": float(V[0]),
        "stock_tree": stock_tree,
        "option_tree": option_tree,
        "exercise_boundary": exercise_boundary if ex_style == "american" else None,
        "early_exercise_map": early_exercise_map,
        "params": params,
        "N": N,
        "style": ex_style,
        "option_type": otype,
    }