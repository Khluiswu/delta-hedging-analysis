"""
Black-Scholes model: option pricing and Greeks.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, r, sigma, T):
    """Compute d1 component of Black-Scholes formula."""
    if T <= 0:
        return np.inf if S >= K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, sigma, T):
    """Compute d2 component of Black-Scholes formula."""
    if T <= 0:
        return np.inf if S >= K else -np.inf
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def call_price(S, K, r, sigma, T):
    """European call option price under Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0)
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def call_delta(S, K, r, sigma, T):
    """Delta of a European call: dC/dS."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, r, sigma, T))


def call_gamma(S, K, r, sigma, T):
    """Gamma: d²C/dS²  (same for call and put)."""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def call_vega(S, K, r, sigma, T):
    """Vega: dC/dσ (per 1 unit of vol, not per 1%)."""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return S * norm.pdf(D1) * np.sqrt(T)


def call_theta(S, K, r, sigma, T):
    """Theta: dC/dT (per year)."""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(D2)
    return term1 + term2
