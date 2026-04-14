"""
Geometric Brownian Motion (GBM) path simulator.
Used for synthetic Monte Carlo scenarios and stress tests.
"""

import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate GBM paths.

    Returns
    -------
    paths : np.ndarray of shape (n_paths, n_steps + 1)
        Simulated stock price paths.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return S0 * np.exp(log_paths)


def simulate_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    lam: float = 1.0,   # jumps per year
    jump_mean: float = -0.05,
    jump_std: float = 0.10,
    seed: int = None,
) -> np.ndarray:
    """
    Merton jump-diffusion model.

    Parameters
    ----------
    lam       : average number of jumps per year
    jump_mean : mean log-jump size
    jump_std  : std of log-jump size
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    # Drift correction so E[jump] doesn't inflate the mean
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1
    mu_adj = mu - lam * kappa

    Z = np.random.standard_normal((n_paths, n_steps))
    # Poisson number of jumps in each interval
    N_jumps = np.random.poisson(lam * dt, (n_paths, n_steps))
    # Jump sizes (sum of N Gaussian log-jumps)
    J = np.random.normal(jump_mean, jump_std, (n_paths, n_steps)) * N_jumps

    increments = (mu_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return S0 * np.exp(log_paths)
