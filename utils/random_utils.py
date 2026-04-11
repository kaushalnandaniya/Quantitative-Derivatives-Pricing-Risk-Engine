"""
Random Number Generation Utilities
====================================
Helpers for reproducible random number generation in Monte Carlo simulations.
"""

import numpy as np
from typing import Optional


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random Generator with an optional seed.

    Args:
        seed: Random seed for reproducibility. If None, uses entropy.

    Returns:
        np.random.Generator instance.
    """
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """
    Set both NumPy legacy and new-style RNG seeds for maximum compatibility.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
