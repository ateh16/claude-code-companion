"""Moving average crossover bot.

Each weekday morning at the open:
  1. fetch the last `lookback_days` completed daily bars from Yahoo Finance
  2. compute the short and long simple moving averages
  3. a real crossover = the short MA was below (above) the long MA yesterday
     and is above (below) it today -> BUY (SELL)
  4. place a market order on the Alpaca paper account
  5. log everything to logs/ma_crossover.log

Run:  python -m ma_crossover.bot            (normal, scheduled run)
      python -m ma_crossover.bot --dry-run  (no orders, just show the signal)
      python -m ma_crossover.bot --force    (ignore the 9:25-15:45 ET window)
"""
import argparse
import math

import pandas as pd

from common.broker import Broker, DuplicateOrder
from common.log import get_logger
from common.market import download_daily, drop_unfinished_day, in_run_window, now_ny, start_date_for
from common.settings import load_conf, load_credentials

CONF_PATH = "ma_crossover/bot.conf"


def fetch_closes(symbol: str, lookback_days: int, now=None) -> pd.Series:
    df = download_daily(symbol, start=start_date_for(lookback_days, now))
    closes = drop_unfinished_day(df["Close"].dropna(), now)
    return closes.tail(lookback_days)


def detect_crossover(closes: pd.Series, short_window: int, long_window: int):
    """Return ('BUY' | 'SELL' | None, details) using the last two completed days."""
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    short_ma = closes.rolling(short_window).mean()
    long_ma = closes.rolling(long_window).mean()
    if len(closes) < long_window + 1 or any(math.isnan(v) for v in (short_ma.iloc[-2], long_ma.iloc[-2])):
        raise ValueError(f"Need at least {long_window + 1} days of prices, got {len(closes)}.")

    prev_s, prev_l = float(short_ma.iloc[-2]), float(long_ma.iloc[-2])
    cur_s, cur_l = float(short_ma.iloc[-1]), float(long_ma.iloc[-1])
    details = {
        "date": closes.index[-1].date(),
        "close": float(closes.iloc[-1]),
        "prev_short": prev_s, "prev_long": prev_l,
        "short": cur_s, "long": cur_l,
    }
    if prev_s <= prev_l and cur_s > cur_l:
        return "BUY", details
    if prev_s >= prev_l and cur_s < cur_l:
        return "SELL", details
    return None, details


def run(dry_run=False, force=False, broker=None, now=None) -> str:
    conf = load_conf(CONF_PATH)
    symbol = conf.get("strategy", "symbol").upper()
    short_w = conf.getint("strategy", "short_window")
    long_w = conf.getint("strategy", "long_window")
    lookback = conf.getint("strategy", "lookback_days")
    size = conf.getfloat("trading", "trade_size")
    allow_short = conf.getboolean("trading", "allow_short")
    log = get_logger("ma_crossover", conf.get("logging", "log_file"))
    now = now or now_ny()

    log.info("=== Run start: %s %d/%d-day crossover, %s share(s)%s ===",
             symbol, short_w, long_w, f"{size:g}", " [DRY RUN]" if dry_run else "")

    if not force and not in_run_window(conf.get("schedule", "run_window_start"),
                                       conf.get("schedule", "run_window_end"), now):
        log.info("Outside the run window (%s ET). Nothing to do.", now.strftime("%H:%M"))
        return "skipped-window"

    if broker is None and not dry_run:
        broker = Broker(load_credentials())
    if broker is not None:
        log.info("Account: %s", "PAPER" if broker.paper else "LIVE MONEY")
        if not broker.market_is_open():
            log.info("Market is closed today (weekend or holiday). No trade.")
            return "skipped-closed"

    closes = fetch_closes(symbol, lookback, now)
    signal, d = detect_crossover(closes, short_w, long_w)
    log.info("Data through %s | close %.2f | %d-day MA %.2f (was %.2f) | %d-day MA %.2f (was %.2f)",
             d["date"], d["close"], short_w, d["short"], d["prev_short"], long_w, d["long"], d["prev_long"])

    if signal is None:
        trend = "above" if d["short"] > d["long"] else "below"
        log.info("No crossover. Short MA stays %s long MA. No trade.", trend)
        return "no-signal"

    log.info("SIGNAL: %s (%d-day MA crossed %s the %d-day MA)",
             signal, short_w, "above" if signal == "BUY" else "below", long_w)

    if dry_run:
        log.info("Dry run: would place a %s order for %s share(s) of %s.", signal, f"{size:g}", symbol)
        return f"dry-{signal.lower()}"

    qty = size
    if signal == "SELL" and not allow_short:
        held = broker.position_qty(symbol)
        qty = min(size, held)
        if qty <= 0:
            log.info("Sell signal, but the account holds no %s. Skipped (short selling is off).", symbol)
            return "sell-skipped"

    order_id = f"ma-{symbol}-{d['date']}-{signal.lower()}"
    try:
        order = broker.market_order(symbol, signal, qty=qty, client_order_id=order_id)
    except DuplicateOrder:
        log.info("This %s order was already placed today (%s). Not repeating it.", signal, order_id)
        return "duplicate"
    except Exception as e:  # keep a clear record of broker errors
        log.error("Order failed: %s", e)
        raise
    log.info("ORDER PLACED: %s %s %s | id %s | status %s",
             signal, f"{qty:g}", symbol, getattr(order, "id", "?"), getattr(order, "status", "?"))
    return signal.lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the signal, place no order")
    ap.add_argument("--force", action="store_true", help="run even outside the morning window")
    a = ap.parse_args()
    run(dry_run=a.dry_run, force=a.force)


if __name__ == "__main__":
    main()
