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

#  Load real data


r_real = load_returns()
mu     = r_real.mean()
sig    = r_real.std()
n      = len(r_real)

#  Combined model simulation

ALPHA = 0.08
BETA  = 0.784475
OMEGA = sig**2 * (1 - ALPHA - BETA)
P          = 0.05
SIGMA_JUMP = 5 * sig

def simulate_combined(n, omega, alpha, beta, mu, p, sigma_jump):
    returns   = np.zeros(n)
    variances = np.zeros(n)
    variances[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma_t      = np.sqrt(variances[t - 1])
        eps          = np.random.standard_t(df=5)      # fat-tailed shocks
        J            = np.random.binomial(1, p)         # correct Bernoulli
        S            = np.random.normal(0, sigma_jump)
        returns[t]   = mu + sigma_t * (eps + J * S)
        variances[t] = omega + alpha * returns[t-1]**2 + beta * variances[t-1]
    return returns, variances

np.random.seed(42)
r_sim_arr, variances = simulate_combined(n, OMEGA, ALPHA, BETA, mu, P, SIGMA_JUMP)
r_sim = pd.Series(r_sim_arr)


# STEP 3 — Diagnostics


summary_real = compute_summary(r_real)
summary_sim  = compute_summary(r_sim)
acfs_real    = compute_acf_suite(r_real, nlags=30)
acfs_sim     = compute_acf_suite(r_sim,  nlags=30)
rv_real      = rolling_volatility(r_real, window=21)
rv_sim       = rolling_volatility(r_sim,  window=21)


# STEP 4 — Print comparison

def print_comparison(summary_real, summary_sim):
    print(f"  {'METRIC':<22} {'REAL':>8} {'COMBINED':>8}")
    print()
    for key in ["mean", "std", "skewness", "excess_kurtosis"]:
        print(f"  {key:<22} {summary_real[key]:>8.4f} {summary_sim[key]:>8.4f}")
    print()
    print(f"  Alpha:       {ALPHA}")
    print(f"  Beta:        {BETA}")
    print(f"  Omega:       {OMEGA:.6f}")
    print(f"  P:           {P}")
    print(f"  Sigma_Jump:  {SIGMA_JUMP:.6f}")
    print(f"  Alpha+Beta:  {ALPHA + BETA:.2f}  (must be < 1)")


print_comparison(summary_real, summary_sim)


# STEP 5 — Final comparison plot


plt.style.use("dark_background")
COLORS = {
    "nifty":    "#7ee8a2",
    "combined": "#f97316",
    "danger":   "#f85149",
}

lags = np.arange(31)
conf = 1.96 / np.sqrt(n)

fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(r_real, bins=120, density=True,
         color=COLORS["nifty"],    alpha=0.6, label="NIFTY 50 (real)")
ax1.hist(r_sim,  bins=120, density=True,
         color=COLORS["combined"], alpha=0.5, label="Combined model")
ax1.set_title("Return Distribution", color="white")
ax1.legend(fontsize=8)

# Panel 2 — Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(r_sim, dist="norm")
ax2.scatter(osm, osr, s=4, color=COLORS["combined"], alpha=0.5)
ax2.plot(osm, slope * np.array(osm) + intercept, color="white", lw=2)
ax2.set_title("Q-Q Plot — Combined Model", color="white")
ax2.set_xlabel("Theoretical Quantiles")

# Panel 3 — ACF of |returns|
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(lags[1:], acfs_real["acf_abs"][1:],
        color=COLORS["nifty"],    alpha=0.7, width=0.4, label="Real")
ax3.bar(lags[1:] + 0.4, acfs_sim["acf_abs"][1:],
        color=COLORS["combined"], alpha=0.7, width=0.4, label="Combined")
ax3.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax3.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax3.set_title("ACF of |Returns|", color="white")
ax3.legend(fontsize=8)

# Panel 4 — ACF of squared returns
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(lags[1:], acfs_real["acf_sq"][1:],
        color=COLORS["nifty"],    alpha=0.7, width=0.4, label="Real")
ax4.bar(lags[1:] + 0.4, acfs_sim["acf_sq"][1:],
        color=COLORS["combined"], alpha=0.7, width=0.4, label="Combined")
ax4.axhline( conf, color=COLORS["danger"], ls="--", lw=1)
ax4.axhline(-conf, color=COLORS["danger"], ls="--", lw=1)
ax4.set_title("ACF of Squared Returns", color="white")
ax4.legend(fontsize=8)

# Panel 5 — Rolling vol real
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(rv_real.values, color=COLORS["nifty"], lw=1)
ax5.set_title("Rolling Vol — Real", color="white")
ax5.set_ylabel("Annualized Vol")

# Panel 6 — Rolling vol combined
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(rv_sim.values,               color=COLORS["combined"], lw=1,   label="Rolling vol")
ax6.plot(np.sqrt(variances) * np.sqrt(252),
         color="white", lw=0.8, alpha=0.5, label="σ_t path")
ax6.set_title("Rolling Vol — Combined Model", color="white")
ax6.set_ylabel("Annualized Vol")
ax6.legend(fontsize=8)

plt.suptitle(
    f"PHASE 5 — Combined Model  |  α={ALPHA}  p={P}  σ_jump={SIGMA_JUMP:.4f}",
    color="white", fontsize=13, y=0.98)

fig.savefig(FIGURES_DIR / "phase5_combined_model.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()