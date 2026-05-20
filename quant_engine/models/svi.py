"""
Stochastic Volatility Inspired (SVI) Parameterization
======================================================
Interpolates and extrapolates implied volatility surfaces using
the SVI model developed by Gatheral.

Formula:
    w(k) = a + b(rho(k - m) + sqrt((k - m)^2 + sigma^2))
Where:
    k     = log-strike = ln(K/F)
    w(k)  = total implied variance = Vol^2 * T
"""

import numpy as np
from typing import Union

def svi_variance(
    k: Union[float, np.ndarray], 
    a: float, 
    b: float, 
    rho: float, 
    m: float, 
    sigma: float
) -> Union[float, np.ndarray]:
    """
    Calculate the total implied variance w(k) using SVI parameterization.
    
    Args:
        k: Log-strike ln(K/F) where K is strike and F is forward price.
        a: Base variance (controls vertical shift)
        b: Controls the angle between left and right asymptotes
        rho: Controls the orientation/tilt of the smile
        m: Controls horizontal translation (at-the-money offset)
        sigma: Controls the smoothness of the vertex
        
    Returns:
        Total implied variance (w = Vol^2 * T).
    """
    # Ensure parameters are within valid ranges to prevent complex numbers
    sigma_safe = max(sigma, 1e-10)
    
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma_safe**2))


def svi_implied_volatility(
    K: Union[float, np.ndarray], 
    F: float, 
    T: float,
    a: float, 
    b: float, 
    rho: float, 
    m: float, 
    sigma: float
) -> Union[float, np.ndarray]:
    """
    Calculate the implied volatility given SVI parameters.
    
    Args:
        K: Strike price(s)
        F: Forward price
        T: Time to maturity (years)
        a, b, rho, m, sigma: SVI parameters for the given maturity T.
        
    Returns:
        Implied volatility (annualized).
    """
    k = np.log(K / F)
    w = svi_variance(k, a, b, rho, m, sigma)
    # Variance must be positive
    w_safe = np.maximum(w, 1e-10)
    return np.sqrt(w_safe / T)
