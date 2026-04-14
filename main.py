"""
main.py — Delta Hedging PnL Simulation
=======================================

Downloads real market data via yfinance, then runs 5 scenarios:
  1. Baseline          — daily hedge, correct vol
  2. Vol misspecification — hedge with 2/3 of true vol
  3. Hedge frequency   — weekly instead of daily
  4. Jump risk         — Merton jump-diffusion paths
  5. Transaction costs — 0.05% cost per share traded

Usage:
    python main.py                     # uses defaults (SPY, 1 year)
    python main.py --ticker AAPL --n_paths 2000
"""

import argparse
import sys
import os

# Allow sibling imports regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from simulation.market_data import (
    download_prices, realized_vol, rolling_vol, get_risk_free_rate, log_returns
)
from simulation.gbm import simulate_gbm, simulate_jump_diffusion
from hedging.delta_hedge import run_simulation
from analysis.pnl_analysis import plot_full_analysis, print_summary_table


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Delta Hedging PnL Simulation")
    p.add_argument("--ticker",     default="SPY",   help="Stock ticker (default: SPY)")
    p.add_argument("--start",      default="2020-01-01")
    p.add_argument("--end",        default="2024-12-31")
    p.add_argument("--T",          type=float, default=0.25,   help="Expiry in years (default: 0.25 = 3m)")
    p.add_argument("--moneyness",  type=float, default=1.0,    help="K / S0 (default: 1.0 = ATM)")
    p.add_argument("--n_paths",    type=int,   default=1000,   help="Monte Carlo paths (default: 1000)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output",     default="output/delta_hedge_analysis.png")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_paths_from_real(prices, T, n_steps, n_paths, sigma, r, seed):
    """
    Bootstrap real log-return sequences into Monte Carlo paths.
    Scales each real daily log-return series drawn (with replacement) into full paths.
    """
    rets = log_returns(prices).values
    np.random.seed(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    S0 = float(prices.iloc[-n_steps - 1] if len(prices) > n_steps else prices.iloc[0])
    paths[:, 0] = S0
    for i in range(n_paths):
        idx = np.random.choice(len(rets), size=n_steps, replace=True)
        sampled = rets[idx]
        paths[i, 1:] = S0 * np.exp(np.cumsum(sampled))
    return paths, S0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Delta Hedging PnL Simulation")
    print(f"  Ticker : {args.ticker}")
    print(f"  Period : {args.start} → {args.end}")
    print(f"  Expiry : {args.T:.2f} years  |  Paths: {args.n_paths}")
    print(f"{'='*60}\n")

    # ── 1. Download market data ───────────────────────────────────────────────
    print("[1/6] Downloading market data …")
    prices = download_prices(args.ticker, args.start, args.end)
    r = get_risk_free_rate()
    sigma_true = realized_vol(prices)
    rv = rolling_vol(prices)
    S0 = float(prices.iloc[-1])
    K = round(S0 * args.moneyness, 2)
    n_steps = int(args.T * 252)

    print(f"      S0 = {S0:.2f}   K = {K:.2f}   r = {r:.2%}   σ_true = {sigma_true:.2%}")

    # ── 2. Build bootstrapped real-data paths ─────────────────────────────────
    print("[2/6] Bootstrapping paths from real returns …")
    paths_real, _ = build_paths_from_real(
        prices, args.T, n_steps, args.n_paths, sigma_true, r, args.seed
    )

    # ── 3. Also build GBM paths (for jump comparison) ─────────────────────────
    paths_gbm = simulate_gbm(
        S0, mu=r, sigma=sigma_true, T=args.T,
        n_steps=n_steps, n_paths=args.n_paths, seed=args.seed
    )
    paths_jump = simulate_jump_diffusion(
        S0, mu=r, sigma=sigma_true, T=args.T,
        n_steps=n_steps, n_paths=args.n_paths,
        lam=4, jump_mean=-0.03, jump_std=0.06,
        seed=args.seed
    )

    # ── 4. Run all scenarios ──────────────────────────────────────────────────
    common = dict(K=K, r=r, T=args.T)

    print("[3/6] Scenario 1: Baseline (daily hedge, σ_hedge = σ_true) …")
    res_base = run_simulation(paths_real, sigma_hedge=sigma_true,
                              hedge_every=1, transaction_cost=0.0, **common)

    print("[4/6] Scenario 2: Vol misspecification (σ_hedge = 2/3 σ_true) …")
    res_vol = run_simulation(paths_real, sigma_hedge=sigma_true * 0.67,
                             hedge_every=1, transaction_cost=0.0, **common)

    print("[4/6] Scenario 3: Weekly hedge …")
    res_weekly = run_simulation(paths_real, sigma_hedge=sigma_true,
                                hedge_every=5, transaction_cost=0.0, **common)

    print("[5/6] Scenario 4: Jump-diffusion paths …")
    res_jumps = run_simulation(paths_jump, sigma_hedge=sigma_true,
                               hedge_every=1, transaction_cost=0.0, **common)

    print("[5/6] Scenario 5: Transaction costs (5bps of notional per trade) …")
    res_tcost = run_simulation(paths_real, sigma_hedge=sigma_true,
                               hedge_every=1, transaction_cost=0.0005, **common)

    # ── 5. Print summary ──────────────────────────────────────────────────────
    scenarios = {
        "[1] Baseline (daily, correct vol)": res_base,
        "[2] Vol misspecification (2/3 vol)": res_vol,
        "[3] Weekly hedge":                  res_weekly,
        "[4] Jump-diffusion paths":          res_jumps,
        "[5] With transaction costs":        res_tcost,
    }
    print_summary_table(scenarios)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    print("[6/6] Generating figures …")
    price_history = {"prices": prices, "rolling_vol": rv}
    plot_full_analysis(
        results_base=res_base,
        results_vol_stress=res_vol,
        results_weekly=res_weekly,
        results_jumps=res_jumps,
        results_tcost=res_tcost,
        price_history=price_history,
        ticker=args.ticker,
        K=K,
        T=args.T,
        sigma_true=sigma_true,
        sigma_hedge=sigma_true,
        output_path=args.output,
    )
    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()
