import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from pathlib import Path
from data import load_returns
from stats import compute_summary, compute_acf_suite, rolling_volatility


# Output path
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load real data and extract parameters
r_real = load_returns()
mu  = r_real.mean()
sig = r_real.std()
n   = len(r_real)

# Simulate IID Gaussian returns ( drawing random num from normal distribution)
def simulate_iid(mu, sigma, n):
    return np.random.normal(loc=mu, scale=sig, size=n)

# ── Run the simulation
np.random.seed(42)
r_sim = pd.Series(simulate_iid(mu, sig, n))

# Compute diagnostics on simulated returns (Reusing the same functions from stats.py)

summary_real = compute_summary(r_real)
summary_sim  = compute_summary(r_sim)

acfs_real = compute_acf_suite(r_real, nlags=30)
acfs_sim  = compute_acf_suite(r_sim,  nlags=30)

rv_real = rolling_volatility(r_real, window=21)

#  Rolling volatility for simulated series (converting to pandas series and calling rolling_vol function)
def compute_rolling_vol_sim(r_sim, window=21):
  return rolling_volatility(pd.Series(r_sim),21)
rv_sim = compute_rolling_vol_sim(r_sim)


# Printing Data
def print_comparison(summary_real, summary_sim):
    for key in ["mean", "std", "excess_kurtosis", "skewness"]:
     print(f"{key:<22} {summary_real[key]:>8.4f} {summary_sim[key]:>8.4f}")

print_comparison(summary_real, summary_sim)

# Plotting
plt.style.use("dark_background")
COLORS = {"nifty": "#7ee8a2", "sim": "#f0c060", "danger": "#f85149", "accent": "#58a6ff"}

lags = np.arange(31)
conf = 1.96 / np.sqrt(n)

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Distribution comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(r_real, bins=120, density=True,
         color=COLORS["nifty"], alpha=0.6, label="NIFTY 50 (real)")
ax1.hist(r_sim,  bins=120, density=True,
         color=COLORS["sim"],   alpha=0.5, label="IID Normal (sim)")
ax1.set_title("Return Distribution", color="white")
ax1.legend(fontsize=8)

# ACF of |returns|: real vs simulated
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(lags[1:], acfs_real["acf_abs"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax2.bar(lags[1:] + 0.4, acfs_sim["acf_abs"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="IID Sim")
ax2.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax2.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax2.set_title("ACF of |Returns|", color="white")
ax2.legend(fontsize=8)

# ACF of squared returns
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(lags[1:], acfs_real["acf_sq"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax3.bar(lags[1:] + 0.4, acfs_sim["acf_sq"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="IID Sim")
ax3.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax3.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax3.set_title("ACF of Squared Returns", color="white")
ax3.legend(fontsize=8)

#  Rolling volatility: real
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(rv_real.values, color=COLORS["nifty"], lw=1)
ax4.set_title("Rolling Vol — Real (clusters)", color="white")
ax4.set_ylabel("Annualized Vol")

#  Rolling volatility: simulated
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(rv_sim.values, color=COLORS["sim"], lw=1)
ax5.set_title("Rolling Vol — IID Sim (flat)", color="white")
ax5.set_ylabel("Annualized Vol")

#  Q-Q plot of simulated returns
ax6 = fig.add_subplot(gs[1, 2])
(osm, osr), (slope, intercept, _) = stats.probplot(r_sim, dist="norm")
ax6.scatter(osm, osr, s=4, color=COLORS["sim"], alpha=0.5)
ax6.plot(osm, slope * np.array(osm) + intercept,
         color="white", lw=2)
ax6.set_title("Q-Q Plot — IID Sim", color="white")
ax6.set_xlabel("Theoretical Quantiles")

plt.suptitle("PHASE 2 — IID Baseline: Can Normal Dist Reproduce Stylized Facts?",
             color="white", fontsize=13, y=0.98)

fig.savefig(FIGURES_DIR / "phase2_iid_baseline.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()