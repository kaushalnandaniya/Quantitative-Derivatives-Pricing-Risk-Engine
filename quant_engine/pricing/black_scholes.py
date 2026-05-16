import logging
import numpy as np
from scipy.stats import norm
from typing import Union, Optional

# Configure logging to track calculation flow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def black_scholes_price(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str = "call"
) -> Union[float, np.ndarray]:
    """
    Computes the Black-Scholes price for European options with numerical stability.

    Args:
        S: Current asset price.
        K: Strike price.
        T: Time to maturity in years.
        r: Risk-free interest rate (e.g., 0.05 for 5%).
        sigma: Volatility of the underlying asset.
        option_type: Type of option, either "call" or "put".

    Returns:
        The fair value price of the option under the risk-neutral measure.
    
    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    """
    logger.debug(f"Pricing {option_type} with S={S}, K={K}, T={T}")

    # Ensure inputs are arrays for vectorization
    S, K, T, r, sigma = map(np.asanyarray, [S, K, T, r, sigma])
    
    # Stability: Avoid division by zero
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0)
    else:
        logger.error(f"Invalid option type: {option_type}")
        raise ValueError("option_type must be 'call' or 'put'")

    # Return intrinsic value if T=0 (at maturity)
    return np.where(T <= 1e-10, intrinsic, price)
