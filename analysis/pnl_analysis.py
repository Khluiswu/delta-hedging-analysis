"""
PnL analysis and visualisation.
Generates a professional multi-panel figure saved to output/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


# ── Style ──────────────────────────────────────────────────────────────────

COLORS = {
    "base":   "#2563EB",   # blue
    "stress": "#DC2626",   # red
    "jumps":  "#D97706",   # amber
    "tcost":  "#7C3AED",   # violet
    "weekly": "#059669",   # green
    "fill":   "#DBEAFE",   # light blue
}
BG = "#F8FAFC"
GRID = "#E2E8F0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct(x, pos):
    return f"{x:.1f}%"

def _dollar(x, pos):
    return f"${x:.2f}"

def _add_stats(ax, pnls, color):
    mu = np.mean(pnls)
    sd = np.std(pnls)
    var = np.percentile(pnls, 5)
    cvar = np.mean(pnls[pnls <= var])
    skew = np.mean(((pnls - mu) / sd) ** 3)
    txt = (
        f"μ = ${mu:.3f}\n"
        f"σ = ${sd:.3f}\n"
        f"VaR₉₅ = ${var:.3f}\n"
        f"CVaR₉₅ = ${cvar:.3f}\n"
        f"Skew = {skew:.2f}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, color=color,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.85),
    )


def _hist(ax, pnls, color, label, bins=60):
    ax.hist(pnls, bins=bins, color=color, alpha=0.75, edgecolor="white",
            linewidth=0.4, label=label, density=True)
    ax.axvline(np.mean(pnls), color=color, lw=1.5, ls="--", label=f"Mean ${np.mean(pnls):.3f}")
    ax.axvline(np.percentile(pnls, 5), color=color, lw=1.5, ls=":",
               label=f"VaR₉₅ ${np.percentile(pnls, 5):.3f}")


# ── Main plotting function ────────────────────────────────────────────────────

def plot_full_analysis(
    results_base: dict,
    results_vol_stress: dict,
    results_weekly: dict,
    results_jumps: dict,
    results_tcost: dict,
    price_history,
    ticker: str,
    K: float,
    T: float,
    sigma_true: float,
    sigma_hedge: float,
    output_path: str = "output/delta_hedge_analysis.png",
):
    """Generate the full 6-panel analysis figure."""

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Delta Hedging P&L Simulation  ·  {ticker}  ·  K={K:.0f}  T={T:.2f}y\n"
        f"σ_true≈{sigma_true:.1%}  |  σ_hedge={sigma_hedge:.1%}  |  1,000 paths",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Baseline PnL histogram ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _hist(ax1, results_base["pnls"], COLORS["base"], "Daily hedge")
    _add_stats(ax1, results_base["pnls"], COLORS["base"])
    ax1.set_title("[1] Baseline: daily hedge, correct vol", fontweight="bold")
    ax1.set_xlabel("Final P&L ($)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # ── Panel 2: Vol misspecification ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _hist(ax2, results_base["pnls"], COLORS["base"], f"σ_hedge={sigma_hedge:.0%} (base)", bins=50)
    _hist(ax2, results_vol_stress["pnls"], COLORS["stress"],
          f"σ_hedge={sigma_hedge*0.67:.0%} (under-hedged)", bins=50)
    _add_stats(ax2, results_vol_stress["pnls"], COLORS["stress"])
    ax2.set_title("[2] Vol Misspecification\n(hedging with lower vol)", fontweight="bold")
    ax2.set_xlabel("Final P&L ($)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7)

    # ── Panel 3: Hedge frequency ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _hist(ax3, results_base["pnls"], COLORS["base"], "Daily (1-day)", bins=50)
    _hist(ax3, results_weekly["pnls"], COLORS["weekly"], "Weekly (5-day)", bins=50)
    _add_stats(ax3, results_weekly["pnls"], COLORS["weekly"])
    ax3.set_title("[3] Hedge Frequency\n(daily vs weekly rebalance)", fontweight="bold")
    ax3.set_xlabel("Final P&L ($)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=7)

    # ── Panel 4: Jump risk ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _hist(ax4, results_base["pnls"], COLORS["base"], "GBM (no jumps)", bins=50)
    _hist(ax4, results_jumps["pnls"], COLORS["jumps"], "Jump-diffusion", bins=50)
    _add_stats(ax4, results_jumps["pnls"], COLORS["jumps"])
    ax4.set_title("[4] Jump Risk\n(Merton jump-diffusion)", fontweight="bold")
    ax4.set_xlabel("Final P&L ($)")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=7)

    # ── Panel 5: Transaction costs ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _hist(ax5, results_base["pnls"], COLORS["base"], "No costs", bins=50)
    _hist(ax5, results_tcost["pnls"], COLORS["tcost"], "With t-costs", bins=50)
    _add_stats(ax5, results_tcost["pnls"], COLORS["tcost"])
    ax5.set_title("[5] Transaction Costs\n(daily hedge, 5bps per trade notional)", fontweight="bold")
    ax5.set_xlabel("Final P&L ($)")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=7)

    # ── Panel 6: Real price path + rolling vol ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6b = ax6.twinx()
    price_vals = price_history["prices"]
    roll_vol = price_history["rolling_vol"].dropna()

    ax6.plot(price_vals.index, price_vals.values, color=COLORS["base"], lw=1.4,
             label=f"{ticker} Price")
    ax6.fill_between(price_vals.index, price_vals.values, alpha=0.08, color=COLORS["base"])
    ax6b.plot(roll_vol.index, roll_vol.values * 100, color=COLORS["stress"],
              lw=1.1, ls="--", label="21d Rolling Vol (%)")
    ax6b.axhline(sigma_true * 100, color="gray", lw=0.9, ls=":", label=f"Full-period σ={sigma_true:.1%}")

    ax6.set_title(f"[6] {ticker} Price & Realised Volatility", fontweight="bold")
    ax6.set_ylabel("Price ($)", color=COLORS["base"])
    ax6b.set_ylabel("Annualised Vol (%)", color=COLORS["stress"])
    ax6b.yaxis.set_major_formatter(FuncFormatter(_pct))

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure saved to {output_path}")


def print_summary_table(scenarios: dict):
    """Print a clean summary table to console."""
    header = f"{'Scenario':<30} {'Mean PnL':>10} {'Std PnL':>10} {'VaR 95%':>10} {'CVaR 95%':>10} {'Skew':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, res in scenarios.items():
        pnls = res["pnls"]
        mu = np.mean(pnls)
        sd = np.std(pnls)
        var = np.percentile(pnls, 5)
        cvar = np.mean(pnls[pnls <= var])
        sk = np.mean(((pnls - mu) / (sd + 1e-9)) ** 3)
        print(f"{name:<30} {mu:>10.4f} {sd:>10.4f} {var:>10.4f} {cvar:>10.4f} {sk:>7.2f}")
    print("=" * len(header) + "\n")
