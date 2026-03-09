import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from pathlib import Path

from data import load_returns
from stats import compute_summary, compute_acf_suite, rolling_volatility

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# STEP 1 — Load real data and extract parameters

r_real = load_returns()
mu     = r_real.mean()
sig    = r_real.std()
n      = len(r_real)


# Simulate the news shock model
def simulate_jump(mu, sigma, n, p, sigma_jump):
   eps =  np.random.normal(0, sigma, n)
   J  =  np.random.binomial(1, p, n)
   S  = np.random.normal(0, sigma_jump, n)
   return mu+eps+J*S

P = 0.05
SIGMA_JUMP = 3 * sig

np.random.seed(42)
r_sim = pd.Series(simulate_jump(mu, sig, n, P, SIGMA_JUMP))

#  Compute diagnostics

summary_real = compute_summary(r_real)
summary_sim  = compute_summary(r_sim)
acfs_real  = compute_acf_suite(r_real, nlags=30)
acfs_sim  = compute_acf_suite(r_sim,  nlags=30)
rv_real  = rolling_volatility(r_real, window=21)
rv_sim  = rolling_volatility(r_sim,  window=21)

#  Print comparison

def print_comparison(summary_real, summary_sim, p, sigma_jump):

        print(f"  {'METRIC':<22} {'REAL':>8} {'JUMP SIM':>8}")
        print()
        for key in ["mean", "std", "skewness", "excess_kurtosis"]:
            print(f"  {key:<22} {summary_real[key]:>8.4f} {summary_sim[key]:>8.4f}")
        print()
        print(f"  Parameters used:  p={p}  sigma_jump={sigma_jump:.4f}")
        print(f"sig={sig:.6f}  sigma_jump={SIGMA_JUMP:.6f}  max_jump={3 * SIGMA_JUMP:.6f}")
print_comparison(summary_real, summary_sim, P, SIGMA_JUMP)

#  Plot

plt.style.use("dark_background")
COLORS = {"nifty": "#7ee8a2", "sim": "#c084fc",
          "danger": "#f85149", "accent": "#58a6ff"}

lags = np.arange(31)
conf = 1.96 / np.sqrt(n)

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Distribution comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(r_real, bins=120, density=True,
         color=COLORS["nifty"], alpha=0.6, label="NIFTY 50 (real)")
ax1.hist(r_sim,  bins=120, density=True,
         color=COLORS["sim"],   alpha=0.5, label="Jump model (sim)")
ax1.set_title("Return Distribution", color="white")
ax1.legend(fontsize=8)

# Panel 2 — ACF of |returns|
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(lags[1:], acfs_real["acf_abs"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax2.bar(lags[1:] + 0.4, acfs_sim["acf_abs"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="Jump Sim")
ax2.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax2.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax2.set_title("ACF of |Returns|", color="white")
ax2.legend(fontsize=8)

# Panel 3 — ACF of squared returns
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(lags[1:], acfs_real["acf_sq"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax3.bar(lags[1:] + 0.4, acfs_sim["acf_sq"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="Jump Sim")
ax3.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax3.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax3.set_title("ACF of Squared Returns", color="white")
ax3.legend(fontsize=8)

# Panel 4 — Rolling vol real
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(rv_real.values, color=COLORS["nifty"], lw=1)
ax4.set_title("Rolling Vol — Real", color="white")
ax4.set_ylabel("Annualized Vol")

# Panel 5 — Rolling vol simulated
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(rv_sim.values, color=COLORS["sim"], lw=1)
ax5.set_title("Rolling Vol — Jump Sim ", color="white")
ax5.set_ylabel("Annualized Vol")

# Panel 6 — Q-Q plot
ax6 = fig.add_subplot(gs[1, 2])
(osm, osr), (slope, intercept, _) = stats.probplot(r_sim, dist="norm")
ax6.scatter(osm, osr, s=4, color=COLORS["sim"], alpha=0.5)
ax6.plot(osm, slope * np.array(osm) + intercept,
         color="white", lw=2)
ax6.set_title("Q-Q Plot — Jump Sim ", color="white")
ax6.set_xlabel("Theoretical Quantiles")

plt.suptitle(f"PHASE 3 — News Shock Model  |  p={P}  σ_jump={SIGMA_JUMP:.4f}",
             color="white", fontsize=13, y=0.98)

fig.savefig(FIGURES_DIR / "phase3_jump_model.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
