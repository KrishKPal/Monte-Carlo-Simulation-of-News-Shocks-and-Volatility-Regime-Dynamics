import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
def compute_summary(r: pd.Series) -> dict:
    n = len(r)
    summary = {
        "n_obs":          n,
        "mean":           r.mean(),
        "std":            r.std(),
        "skewness":       stats.skew(r),

        "excess_kurtosis": stats.kurtosis(r, fisher=True),
        "min":            r.min(),
        "max":            r.max(),
        "annualized_vol": r.std() * np.sqrt(252),
    }
    return summary

def compute_acf_suite(r: pd.Series, nlags: int = 30) -> dict:
    r_arr = r.values
    return {
        "acf_returns":   acf(r_arr,        nlags=nlags, fft=True),
        "acf_abs":       acf(np.abs(r_arr), nlags=nlags, fft=True),
        "acf_sq":        acf(r_arr**2,      nlags=nlags, fft=True),
    }
def rolling_volatility(r: pd.Series, window: int = 21) -> pd.Series:

    return r.rolling(window).std() * np.sqrt(252)
def print_summary(summary: dict) -> None:

    print("  EMPIRICAL SUMMARY STATISTICS")
    print()
    for k, v in summary.items():
        print(f"  {k:<22} {v:>12.6f}")
    print()
    print(f"  NOTE: Normal dist has excess kurtosis = 0")
