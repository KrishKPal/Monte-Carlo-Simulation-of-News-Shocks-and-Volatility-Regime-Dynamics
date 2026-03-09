import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats as stats
from data import load_returns
from stats import (compute_summary, compute_acf_suite,
                     rolling_volatility, print_summary)
from pathlib import Path

plt.style.use("dark_background")
COLORS = {
    "nifty":   "#7ee8a2",
    "normal":  "#f0c060",
    "accent":  "#58a6ff",
    "danger":  "#f85149",
}

def run_phase1():
    r = load_returns()

    summary = compute_summary(r)
    print_summary(summary)
    acfs = compute_acf_suite(r, nlags=30)
    rv   = rolling_volatility(r, window=21)

#Plottings--------------------------------------------------
    fig = plt.figure(figsize=(16, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r, bins=120, density=True,
             color=COLORS["nifty"], alpha=0.75, label="NIFTY 50")
    x = np.linspace(r.min(), r.max(), 300)
    ax1.plot(x, stats.norm.pdf(x, r.mean(), r.std()),
             color=COLORS["normal"], lw=2, label="Normal fit")
    ax1.set_title("Return Distribution", color="white")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, s=4, color=COLORS["nifty"], alpha=0.5)
    ax2.plot(osm, slope * np.array(osm) + intercept,
             color=COLORS["normal"], lw=2)
    ax2.set_title("Q-Q Plot vs Normal", color="white")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    lags = np.arange(len(acfs["acf_returns"]))
    conf = 1.96 / np.sqrt(len(r))  # 95% CI band
    def plot_acf(ax, acf_vals, title):
        ax.bar(lags[1:], acf_vals[1:], color=COLORS["accent"],
               alpha=0.7, width=0.6)
        ax.axhline(conf,  color=COLORS["danger"], ls="--", lw=1)
        ax.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
        ax.set_title(title, color="white")
        ax.set_xlabel("Lag (days)")
    plot_acf(fig.add_subplot(gs[1, 0]),
             acfs["acf_returns"], "ACF of Returns")
    plot_acf(fig.add_subplot(gs[1, 1]),
             acfs["acf_abs"],    "ACF of |Returns| (Vol clustering)")
    plot_acf(fig.add_subplot(gs[2, 0]),
             acfs["acf_sq"],     "ACF of Squared Returns")


    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(rv.index, rv.values,
             color=COLORS["nifty"], lw=1, alpha=0.9)
    ax6.set_title("Rolling Volatility (21-day)", color="white")
    ax6.set_ylabel("Annualized Vol")
    plt.suptitle("PHASE 1 — NIFTY 50 Empirical Benchmark",
                 color="white", fontsize=16, y=0.98)

    FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURES_DIR / "phase3_newsShock.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()

if __name__ == "__main__":
    run_phase1()