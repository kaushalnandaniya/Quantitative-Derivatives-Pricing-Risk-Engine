"""
Correlated Multi-Asset Simulation
====================================
Generate correlated random variables via Cholesky decomposition
and simulate correlated GBM terminal prices.

Theory:
    Given correlation matrix Σ, compute L = cholesky(Σ).
    For independent ε ~ N(0, I), Z = L @ ε gives correlated normals
    with Cov(Z) = Σ.
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _validate_correlation_matrix(corr: np.ndarray) -> None:
    """Validate that a matrix is a valid correlation matrix."""
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {corr.shape}")

    n = corr.shape[0]

    # Check symmetry
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric")

    # Check unit diagonal
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("Correlation matrix must have unit diagonal")

    # Check positive semi-definite (all eigenvalues ≥ 0)
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues < -1e-8):
        raise ValueError(
            f"Correlation matrix must be positive semi-definite. "
            f"Min eigenvalue: {eigenvalues.min():.6e}"
        )

    # Check bounds [-1, 1]
    if np.any(corr < -1.0 - 1e-8) or np.any(corr > 1.0 + 1e-8):
        raise ValueError("Correlation values must be in [-1, 1]")


def generate_correlated_normals(
    n_sims: int,
    n_assets: int,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate correlated standard normal draws via Cholesky decomposition.

    Args:
        n_sims:      Number of simulation paths.
        n_assets:    Number of assets.
        corr_matrix: (n_assets, n_assets) correlation matrix.
        rng:         NumPy random Generator.

    Returns:
        np.ndarray of shape (n_assets, n_sims) — correlated normal draws.
        Z[i] has marginal N(0,1) and Corr(Z[i], Z[j]) ≈ corr_matrix[i,j].
    """
    corr = np.asarray(corr_matrix, dtype=np.float64)
    _validate_correlation_matrix(corr)

    if corr.shape[0] != n_assets:
        raise ValueError(
            f"corr_matrix size {corr.shape[0]} != n_assets {n_assets}"
        )

    # Cholesky decomposition: Σ = L L^T
    L = np.linalg.cholesky(corr)

    # Independent standard normals: (n_assets, n_sims)
    epsilon = rng.standard_normal((n_assets, n_sims))

    # Correlated normals: Z = L @ ε
    Z = L @ epsilon

    logger.debug(
        f"Generated {n_sims:,} correlated draws for {n_assets} assets | "
        f"empirical corr[0,1] = {np.corrcoef(Z[0], Z[1])[0,1]:.4f}"
        if n_assets >= 2 else
        f"Generated {n_sims:,} draws for 1 asset"
    )

    return Z


def simulate_correlated_gbm(
    spots: np.ndarray,
    rates: np.ndarray,
    sigmas: np.ndarray,
    T: float,
    n_sims: int,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate correlated GBM terminal prices for multiple assets.

    Uses exact solution: S_T^i = S_0^i * exp((r_i - σ_i²/2)*T + σ_i*√T*Z_i)
    where Z is a vector of correlated normals.

    Args:
        spots:       (n_assets,) array of initial spot prices.
        rates:       (n_assets,) array of risk-free rates.
        sigmas:      (n_assets,) array of volatilities.
        T:           Time to maturity (years).
        n_sims:      Number of simulations.
        corr_matrix: (n_assets, n_assets) correlation matrix.
        rng:         NumPy random Generator.

    Returns:
        np.ndarray of shape (n_assets, n_sims) — terminal prices.
        S_T[i] is the array of simulated terminal prices for asset i.
    """
    spots = np.asarray(spots, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    n_assets = len(spots)

    # Generate correlated normals
    Z = generate_correlated_normals(n_sims, n_assets, corr_matrix, rng)

    # GBM for each asset: S_T = S_0 * exp(drift + diffusion)
    drift = (rates - 0.5 * sigmas**2) * T       # (n_assets,)
    diffusion = sigmas * np.sqrt(T)               # (n_assets,)

    # Broadcast: drift[:, None] → (n_assets, 1), Z → (n_assets, n_sims)
    S_T = spots[:, None] * np.exp(
        drift[:, None] + diffusion[:, None] * Z
    )

    logger.debug(
        f"Correlated GBM | {n_assets} assets | {n_sims:,} sims | "
        f"T={T:.4f}y"
    )

    return S_T
