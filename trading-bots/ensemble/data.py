"""Daily closes for the four funds, cached locally as CSV.

First run downloads everything since `history_start`. Later runs download only
the last few days and append them. If Yahoo has re-adjusted old prices (after a
dividend, adjusted history shifts), the overlap won't match and the full history
is downloaded again so the series stays consistent.
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd

from common.market import download_daily, now_ny
from common.settings import ROOT


def _cache_path(cache_dir, symbol) -> Path:
    d = Path(cache_dir)
    if not d.is_absolute():
        d = ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.csv"


def _read(path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_symbol(symbol, start, cache_dir, refresh=True, log=None) -> pd.Series:
    path = _cache_path(cache_dir, symbol)
    say = log.info if log else (lambda *a: None)

    if path.exists():
        cached = _read(path)
        if not refresh:
            return cached
        last = cached.index.max()
        if last.date() >= (now_ny() - timedelta(days=1)).date():
            return cached  # already current
        fresh = download_daily(symbol, start=(last - timedelta(days=10)).date())["Close"].dropna()
        overlap = cached.index.intersection(fresh.index)
        if len(overlap) and ((fresh[overlap] / cached[overlap] - 1).abs().max() > 0.005):
            say("%s: old prices were re-adjusted by Yahoo, re-downloading full history", symbol)
            full = download_daily(symbol, start=start)["Close"].dropna()
        else:
            full = pd.concat([cached[cached.index < fresh.index.min()], fresh])
            say("%s: +%d new day(s) from cache update", symbol, int((fresh.index > last).sum()))
    else:
        say("%s: downloading full history since %s", symbol, start)
        full = download_daily(symbol, start=start)["Close"].dropna()

    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.rename(symbol).to_csv(path, header=True)
    return full


def load_prices(symbols, start, cache_dir, refresh=True, log=None) -> pd.DataFrame:
    """Close prices, one column per symbol, only dates where every fund traded."""
    cols = {s: load_symbol(s, start, cache_dir, refresh, log) for s in symbols}
    return pd.concat(cols, axis=1).dropna(how="any")
