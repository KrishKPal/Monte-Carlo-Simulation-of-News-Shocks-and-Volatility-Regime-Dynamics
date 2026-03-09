import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

TICKER   = "^NSEI"
START    = "2010-01-01"
END      = "2024-12-31"
DATA_DIR = Path("data")
def download_and_process() -> pd.Series:
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading {TICKER} from {START} to {END}...")
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True)
    prices = raw["Close"].dropna()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"

    prices.to_csv(DATA_DIR / "raw_prices.csv")
    log_returns.to_csv(DATA_DIR / "returns.csv")
    return log_returns

def load_returns() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    return df.iloc[:, 0]

if __name__ == "__main__":
    r = download_and_process()
    print(r.describe())