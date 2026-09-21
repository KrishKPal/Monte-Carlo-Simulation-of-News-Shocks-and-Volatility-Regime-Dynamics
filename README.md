# Monte Carlo Simulation of News Shocks and Volatility Regime Dynamics

### Modeling Volatility Clustering, Fat Tails, and Exogenous News Shocks in NIFTY 50 Returns

A Monte Carlo study of whether increasingly realistic return-generating models can reproduce key characteristics of NIFTY 50 returns, particularly **fat tails, volatility clustering, and occasional exogenous shocks**.

> NIFTY 50 → IID Gaussian → News Shocks → ARCH → Combined Model

---

## Model Progression

| Phase | Model | Main Feature |
|---|---|---|
| 1 | NIFTY 50 | Empirical benchmark |
| 2 | IID Gaussian | Baseline |
| 3 | News Shock Model | Exogenous jumps |
| 4 | ARCH(1) | Volatility persistence |
| 5 | Combined Model | Persistent volatility + fat tails + news shocks |

Each stage is compared against the historical NIFTY 50 return series using the same set of diagnostics.

---

## Final Model

The final model combines:

- GARCH-style persistent volatility
- Student-t innovations for fat-tailed returns
- Discrete exogenous news shocks

The volatility component follows:

$$
h_t = \omega + \alpha(r_{t-1}-\mu)^2 + \beta h_{t-1}
$$

with returns generated from the persistent volatility component and an additive news-jump term.

### Parameters

```text
α = 0.08
β = 0.784475
α + β = 0.864475
Student-t df = 5
News jump probability = 0.05
Jump-scale parameter = 5σ
```

The Student-t innovations are standardized to unit variance, and the model uses a fixed random seed (`42`) for reproducibility.

---

## Diagnostics

The real and simulated returns are compared using:

- Return distribution and Q-Q plots
- Skewness and excess kurtosis
- ACF of absolute returns
- ACF of squared returns
- 21-day rolling volatility
- Conditional volatility path

---

## Data

Historical NIFTY 50 data:

```text
Ticker: ^NSEI
Period: 2010–2024
Frequency: Daily
```

Processed into prices and daily log returns.

---

## Project Structure

```text
├── data/
├── results/
│   └── figures/
├── volatility-clustering-study/
│   ├── data/
│   └── source/
│       ├── config.py
│       ├── data.py
│       ├── stats.py
│       ├── main.py
│       ├── simulation_IID.py
│       ├── News_Shock_Sim.py
│       ├── simulation_vol_persist.py
│       └── simulate_final.py
└── requirements.txt
```
---

## Scope

This is a **Monte Carlo/statistical modelling study**, not a trading strategy or forecasting system.

The goal is to study how volatility persistence, heavy-tailed innovations, and exogenous shocks affect simulated market return dynamics.
