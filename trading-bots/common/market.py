"""Market-time helpers and daily price downloads from Yahoo Finance."""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


def now_ny() -> datetime:
    return datetime.now(NY)


def parse_hhmm(text: str) -> time:
    h, m = text.strip().split(":")
    return time(int(h), int(m))


def in_run_window(start: str, end: str, now: datetime | None = None) -> bool:
    """True when New York time is inside [start, end].

    The cloud scheduler fires at two UTC times, an hour apart, so one of them
    is near the open in both summer and winter. Runs outside the window (e.g.
    8:35 ET in winter) exit quietly.
    """
    now = now or now_ny()
    return parse_hhmm(start) <= now.time() <= parse_hhmm(end)


def drop_unfinished_day(frame: pd.DataFrame | pd.Series, now: datetime | None = None):
    """Remove today's bar if the market hasn't closed yet.

    At 9:30 ET Yahoo already shows a 'today' row holding only the opening price.
    Signals must use completed days only, or they change during the day.
    """
    now = now or now_ny()
    if frame is None or len(frame) == 0:
        return frame
    if now.time() < time(16, 10):
        today = pd.Timestamp(now.date())
        idx = pd.to_datetime(frame.index)
        if idx.tz is not None:
            idx = idx.tz_convert(NY).tz_localize(None)
        frame = frame[idx.normalize() < today]
    return frame


def download_daily(symbols, start, end=None) -> pd.DataFrame:
    """Daily adjusted bars for one or more symbols. Columns: Open, High, Low, Close, Volume.

    One symbol -> flat columns. Several symbols -> MultiIndex (field, symbol).
    """
    import yfinance as yf

    single = isinstance(symbols, str)
    df = yf.download(
        symbols, start=str(start), end=None if end is None else str(end),
        auto_adjust=True, progress=False, threads=True,
        multi_level_index=not single,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbols}.")
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    return df


def start_date_for(trading_days: int, now: datetime | None = None):
    """A calendar start date that comfortably covers N trading days."""
    now = now or now_ny()
    return (now - timedelta(days=int(trading_days * 1.6) + 10)).date()
