"""
Delta hedging engine.

Given a price path (real or simulated), simulates the P&L of a trader who:
  1. Sells one European call at time 0.
  2. Delta-hedges the position at regular intervals.
  3. Closes out at expiry.

The hedging volatility (sigma_hedge) can differ from the realised path volatility
to model volatility misspecification.
"""

import numpy as np
from models.black_scholes import call_price, call_delta


def run_hedge(
    prices: np.ndarray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    hedge_every: int = 1,
    transaction_cost: float = 0.0,
) -> dict:
    """
    Simulate delta hedging along a single price path.

    Parameters
    ----------
    prices          : 1-D array of stock prices, length = n_steps + 1
    K               : strike price
    r               : risk-free rate (annual)
    sigma_hedge     : implied vol used for hedging (may differ from true vol)
    T               : time to expiry in years
    hedge_every     : rebalance every N steps (1 = daily, 5 = weekly, …)
    transaction_cost: cost per share traded (absolute, e.g. 0.01 = 1 cent/share)

    Returns
    -------
    dict with keys:
        pnl          : final P&L of the hedged position
        pnl_path     : cumulative P&L at each step
        delta_path   : delta held at each step
        cash_path    : cash account at each step
        hedge_costs  : total transaction costs paid
    """
    n_steps = len(prices) - 1
    dt = T / n_steps

    # ── Initialise at t = 0 ──────────────────────────────────────────────────
    S0 = prices[0]
    premium = call_price(S0, K, r, sigma_hedge, T)

    # We sell the call: receive premium, put into cash
    cash = premium
    delta_held = call_delta(S0, K, r, sigma_hedge, T)

    # Buy delta shares to hedge (deducted from cash)
    cash -= delta_held * S0
    hedge_costs = abs(delta_held) * S0 * transaction_cost

    pnl_path = np.zeros(n_steps + 1)
    delta_path = np.zeros(n_steps + 1)
    cash_path = np.zeros(n_steps + 1)

    delta_path[0] = delta_held
    cash_path[0] = cash

    # ── Step through time ────────────────────────────────────────────────────
    for i in range(1, n_steps + 1):
        S = prices[i]
        t_remaining = T - i * dt

        # Accrue interest on cash position
        cash *= np.exp(r * dt)

        # Rebalance at specified frequency (or at expiry always)
        if i % hedge_every == 0 or i == n_steps:
            new_delta = call_delta(S, K, r, sigma_hedge, max(t_remaining, 0))
            d_delta = new_delta - delta_held
            # Adjust stock position
            cash -= d_delta * S
            hedge_costs += abs(d_delta) * S * transaction_cost
            delta_held = new_delta

        delta_path[i] = delta_held
        cash_path[i] = cash

        # Mark-to-market P&L (unrealised)
        option_value = call_price(S, K, r, sigma_hedge, max(t_remaining, 0))
        pnl_path[i] = cash + delta_held * S - option_value

    # ── Settle at expiry ─────────────────────────────────────────────────────
    ST = prices[-1]
    payoff = max(ST - K, 0.0)

    # Unwind stock position
    cash_path[-1] = cash + delta_held * ST
    final_pnl = cash_path[-1] - payoff - hedge_costs

    return {
        "pnl": final_pnl,
        "pnl_path": pnl_path,
        "delta_path": delta_path,
        "cash_path": cash_path,
        "hedge_costs": hedge_costs,
        "premium": premium,
        "payoff": payoff,
    }


def run_simulation(
    paths: np.ndarray,
    K: float,
    r: float,
    sigma_hedge: float,
    T: float,
    hedge_every: int = 1,
    transaction_cost: float = 0.0,
) -> dict:
    """
    Run the hedging simulation over many paths.

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, n_steps + 1)

    Returns
    -------
    dict with aggregate statistics and per-path arrays
    """
    n_paths = paths.shape[0]
    pnls = np.zeros(n_paths)
    premiums = np.zeros(n_paths)
    payoffs = np.zeros(n_paths)
    costs = np.zeros(n_paths)

    for i, path in enumerate(paths):
        result = run_hedge(
            prices=path,
            K=K,
            r=r,
            sigma_hedge=sigma_hedge,
            T=T,
            hedge_every=hedge_every,
            transaction_cost=transaction_cost,
        )
        pnls[i] = result["pnl"]
        premiums[i] = result["premium"]
        payoffs[i] = result["payoff"]
        costs[i] = result["hedge_costs"]

    return {
        "pnls": pnls,
        "premiums": premiums,
        "payoffs": payoffs,
        "hedge_costs": costs,
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "var_95": np.percentile(pnls, 5),
        "cvar_95": np.mean(pnls[pnls <= np.percentile(pnls, 5)]),
        "skewness": _skewness(pnls),
        "kurtosis": _kurtosis(pnls),
    }


def _skewness(x):
    m = np.mean(x)
    s = np.std(x)
    return np.mean(((x - m) / s) ** 3) if s > 0 else 0.0


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    return np.mean(((x - m) / s) ** 4) - 3 if s > 0 else 0.0
