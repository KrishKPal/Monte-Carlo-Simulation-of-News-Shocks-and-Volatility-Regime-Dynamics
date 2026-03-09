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


r_real = load_returns()
mu     = r_real.mean()
sig    = r_real.std()
n      = len(r_real)

# Simulate ARCH(1) volatility persistence
# This model CANNOT be vectorised like Phase 3
# because σ_t depends on r_{t-1}

def simulate_arch(n, omega, alpha, mu):
 returns = np.zeros(n)
 variances = np.zeros(n)
 variances[0]=omega/(1-alpha)
 for t in range(1, n):
        sigma_t = np.sqrt(variances[t - 1])
        eps=np.random.normal(0,1)
        returns[t] = mu + sigma_t * eps
        variances[t] = omega + alpha * returns[t-1]**2
 return returns, variances
ALPHA = 0.3
OMEGA = sig**2 * (1 - ALPHA)   # calibrated to match real vol

np.random.seed(42)
r_sim_arr, variances = simulate_arch(n, OMEGA, ALPHA, mu)
r_sim = pd.Series(r_sim_arr)

# STEP 3 — Diagnostics (identical pattern to Phase 3)

summary_real = compute_summary(r_real)
summary_sim  = compute_summary(r_sim)
acfs_real    = compute_acf_suite(r_real, nlags=30)
acfs_sim     = compute_acf_suite(r_sim,  nlags=30)
rv_real      = rolling_volatility(r_real, window=21)
rv_sim       = rolling_volatility(r_sim,  window=21)

long_run_vol = np.sqrt(OMEGA / (1 -ALPHA)) * np.sqrt(252)
real_ann_vol = summary_real["std"] * np.sqrt(252)
# STEP 4 — Print comparison

def print_comparison(summary_real, summary_sim, alpha, omega):
    print(f"  {'METRIC':<22} {'REAL':>8} {'ARCHSIM':>8}")
    print()
    for key in ["mean", "std", "skewness", "excess_kurtosis"]:
        print(f"  {key:<22} {summary_real[key]:>8.4f} {summary_sim[key]:>8.4f}")
    print()
    print(f"  Alpha:                {alpha}")
    print(f"  Omega:                {omega:.6f}")
    print(f"  Model long-run vol:   {long_run_vol:.4f}")
    print(f"  Real annualized vol:  {real_ann_vol:.4f}")
print_comparison(summary_real, summary_sim, ALPHA, OMEGA)

#  Plot (already written)


plt.style.use("dark_background")
COLORS = {"nifty": "#7ee8a2", "sim": "#f97316",
          "danger": "#f85149"}

lags = np.arange(31)
conf = 1.96 / np.sqrt(n)

fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(r_real, bins=120, density=True,
         color=COLORS["nifty"], alpha=0.6, label="NIFTY 50 (real)")
ax1.hist(r_sim,  bins=120, density=True,
         color=COLORS["sim"],   alpha=0.5, label="ARCH(1) sim")
ax1.set_title("Return Distribution", color="white")
ax1.legend(fontsize=8)

# Panel 2 — Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(r_sim, dist="norm")
ax2.scatter(osm, osr, s=4, color=COLORS["sim"], alpha=0.5)
ax2.plot(osm, slope * np.array(osm) + intercept,
         color="white", lw=2)
ax2.set_title("Q-Q Plot — ARCH(1)", color="white")
ax2.set_xlabel("Theoretical Quantiles")

# Panel 3 — ACF of |returns|
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(lags[1:], acfs_real["acf_abs"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax3.bar(lags[1:] + 0.4, acfs_sim["acf_abs"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="ARCH Sim")
ax3.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax3.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax3.set_title("ACF of |Returns|", color="white")
ax3.legend(fontsize=8)

# Panel 4 — ACF of squared returns
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(lags[1:], acfs_real["acf_sq"][1:],
        color=COLORS["nifty"], alpha=0.7, width=0.4, label="Real")
ax4.bar(lags[1:] + 0.4, acfs_sim["acf_sq"][1:],
        color=COLORS["sim"],   alpha=0.7, width=0.4, label="ARCH Sim")
ax4.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax4.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax4.set_title("ACF of Squared Returns", color="white")
ax4.legend(fontsize=8)

# Panel 5 — Rolling vol real
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(rv_real.values, color=COLORS["nifty"], lw=1)
ax5.set_title("Rolling Vol — Real ", color="white")
ax5.set_ylabel("Annualized Vol")

# Panel 6 — Rolling vol simulated
# Also plot the simulated sigma_t path directly
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(rv_sim.values, color=COLORS["sim"], lw=1, label="Rolling vol")
ax6.plot(np.sqrt(variances) * np.sqrt(252),
         color="white", lw=0.8, alpha=0.5, label="σ_t path")
ax6.set_title("Rolling Vol — ARCH(1) ", color="white")
ax6.set_ylabel("Annualized Vol")
ax6.legend(fontsize=8)

plt.suptitle(f"PHASE 4 — ARCH(1) Volatility Persistence  |  α={ALPHA}  ω={OMEGA:.6f}",
             color="white", fontsize=13, y=0.98)

fig.savefig(FIGURES_DIR / "phase4_arch_model.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
