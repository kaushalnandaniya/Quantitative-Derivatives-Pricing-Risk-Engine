import numpy as np
from scipy.stats import norm
from .black_scholes import black_scholes_price


class GreeksCalculator:
    def __init__(self, method="analytical", h=1e-4):
        """
        Unified interface for Greek calculations.
        method: 'analytical' or 'numerical'
        h: step size for numerical finite difference
        """
        self.method = method.lower()
        self.h = h

    def calculate(self, S, K, T, r, sigma, option_type="call"):
        if self.method == "analytical":
            return self._calculate_analytical(S, K, T, r, sigma, option_type)
        elif self.method == "numerical":
            return self._calculate_numerical(S, K, T, r, sigma, option_type)
        else:
            raise ValueError("Method must be 'analytical' or 'numerical'")

    def _calculate_analytical(self, S, K, T, r, sigma, option_type="call"):
        S, K, T, r, sigma = map(np.asanyarray, [S, K, T, r, sigma])
        T_safe = np.maximum(T, 1e-10)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
        d2 = d1 - sigma * np.sqrt(T_safe)
        
        pdf_d1 = norm.pdf(d1)
        exp_rt = np.exp(-r * T)
        
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T_safe)) - r * K * exp_rt * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T_safe)) + r * K * exp_rt * norm.cdf(-d2))
            
        gamma = pdf_d1 / (S * sigma * np.sqrt(T_safe))
        vega = S * pdf_d1 * np.sqrt(T_safe)
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    def _calculate_numerical(self, S, K, T, r, sigma, option_type="call"):
        h = self.h
        # 1. Delta & Gamma
        v_base = black_scholes_price(S, K, T, r, sigma, option_type)
        v_plus = black_scholes_price(S + h, K, T, r, sigma, option_type)
        v_minus = black_scholes_price(S - h, K, T, r, sigma, option_type)
        
        delta = (v_plus - v_minus) / (2 * h)
        gamma = (v_plus - 2 * v_base + v_minus) / (h ** 2)
        
        # 2. Vega
        v_sig_plus = black_scholes_price(S, K, T, r, sigma + h, option_type)
        v_sig_minus = black_scholes_price(S, K, T, r, np.maximum(sigma - h, 1e-10), option_type)
        vega = (v_sig_plus - v_sig_minus) / (2 * h)
        
        # 3. Theta
        v_t_minus = black_scholes_price(S, K, np.maximum(T - h, 0), r, sigma, option_type)
        theta = (v_t_minus - v_base) / h 
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}