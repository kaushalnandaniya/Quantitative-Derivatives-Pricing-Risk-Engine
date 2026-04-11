import numpy as np

def safe_divide(numerator, denominator, eps=1e-15):
    """Prevents division by zero in vectorized operations."""
    return numerator / np.where(denominator == 0, eps, denominator)
