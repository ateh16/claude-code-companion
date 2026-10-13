"""Daily live run of the four-program ensemble.

  1. update the cached prices (only the missing days are downloaded)
  2. every program votes; Program 4 retrains from scratch on all history
  3. votes are blended into one target portfolio
  4. compare with what the Alpaca account holds and trade only the differences

Run:  python -m ensemble.run            (scheduled run, places orders)
      python -m ensemble.run --dry-run  (show votes and planned trades only)
      python -m ensemble.run --force    (ignore the morning time window)
"""
import argparse
import math

import pandas as pd

from common.broker import Broker, DuplicateOrder
from common.log import get_logger
from common.market import drop_unfinished_day, in_run_window, now_ny
from common.settings import load_conf, load_credentials
from ensemble.data import load_prices
from ensemble.programs import all_votes, blend

CONF_PATH = "ensemble/ensemble.conf"


def plan_trades(target: pd.Series, equity: float, holdings: dict, prices: pd.Series,
                min_trade_pct: float, fractional: bool, held_qty: dict | None = None) -> list:
    """Orders that move the account from `holdings` ($ per symbol) to `target` (weights)."""
    held_qty = held_qty or {}
    orders = []
    for sym, w in target.items():
        want = w * equity
        have = holdings.get(sym, 0.0)
        diff = want - have
        if abs(diff) < min_trade_pct * equity:
            continue
        price = float(prices[sym])
        if diff < 0:
            if w <= 1e-6:
                orders.append({"symbol": sym, "side": "sell", "close_all": True, "amount": have})
            elif fractional:
                orders.append({"symbol": sym, "side": "sell", "notional": -diff, "amount": -diff})
            else:
                qty = min(math.floor(-diff / price), int(held_qty.get(sym, 0)))
                if qty > 0:
                    orders.append({"symbol": sym, "side": "sell", "qty": qty, "amount": qty * price})
        else:
            if fractional:
                orders.append({"symbol": sym, "side": "buy", "notional": diff, "amount": diff})
            else:
                qty = math.floor(diff / price)
                if qty > 0:
                    orders.append({"symbol": sym, "side": "buy", "qty": qty, "amount": qty * price})
    return sorted(orders, key=lambda o: o["side"] != "sell")   # sells first, to free cash


def run(dry_run=False, force=False, broker=None, now=None, px=None):
    conf = load_conf(CONF_PATH)
    a = conf["assets"]
    syms = [a["sp500"], a["nasdaq"], a["international"], a["bonds"]]
    log = get_logger("ensemble", conf.get("logging", "log_file"))
    now = now or now_ny()
    log.info("=== Ensemble run start%s ===", " [DRY RUN]" if dry_run else "")

    if not force and not in_run_window(conf.get("schedule", "run_window_start"),
                                       conf.get("schedule", "run_window_end"), now):
        log.info("Outside the run window (%s ET). Nothing to do.", now.strftime("%H:%M"))
        return None

    if broker is None and not dry_run:
        broker = Broker(load_credentials())
    if broker is not None:
        log.info("Account: %s", "PAPER" if broker.paper else "LIVE MONEY")
        if not dry_run and not broker.market_is_open():
            log.info("Market is closed today. No trades.")
            return None

    if px is None:
        px = load_prices(syms, a.get("history_start"), a.get("cache_dir"), refresh=True, log=log)
    px = drop_unfinished_day(px, now)
    log.info("Prices through %s (%d days of history)", px.index[-1].date(), len(px))

    votes, prob = all_votes(px, conf, live=True)
    for name, v in votes.items():
        row = v.iloc[-1]
        held = ", ".join(f"{s} {w:.0%}" for s, w in row.items() if w > 0.001) or "cash"
        extra = f" (model: {prob.iloc[-1]:.0%} chance of up in 5 days)" if name == "p4_ml" and not pd.isna(prob.iloc[-1]) else ""
        log.info("Vote %-12s weight %.0f%% -> %s%s", name, conf.getfloat("weights", name) * 100, held, extra)

    target = blend(votes, conf).iloc[-1]
    log.info("TARGET: %s | cash %.0f%%",
             ", ".join(f"{s} {w:.1%}" for s, w in target.items()), (1 - target.sum()) * 100)

    if broker is None:
        log.info("Dry run without an account: no trades planned.")
        return target

    equity = broker.equity()
    holdings = broker.position_values()
    others = [s for s in holdings if s not in syms]
    if others:
        log.info("Leaving other positions untouched: %s", ", ".join(others))
    log.info("Account equity $%s | holding: %s", f"{equity:,.2f}",
             ", ".join(f"{s} ${v:,.0f}" for s, v in holdings.items() if s in syms) or "nothing")

    fractional = conf.getboolean("trading", "fractional")
    held_qty = {s: broker.position_qty(s) for s in syms} if not fractional else None
    orders = plan_trades(target, equity, holdings, px.iloc[-1],
                         conf.getfloat("trading", "min_trade_pct"), fractional, held_qty)
    if not orders:
        log.info("Account already matches the target. No trades.")
        return target

    day = px.index[-1].date()
    for o in orders:
        desc = f"{o['side'].upper()} {o['symbol']} ~${o['amount']:,.2f}"
        if dry_run:
            log.info("Dry run: would %s", desc)
            continue
        try:
            if o.get("close_all"):
                res = broker.close_position(o["symbol"])
            else:
                res = broker.market_order(o["symbol"], o["side"], qty=o.get("qty"), notional=o.get("notional"),
                                          client_order_id=f"ens-{o['symbol']}-{day}-{o['side']}")
            log.info("ORDER PLACED: %s | id %s", desc, getattr(res, "id", "?"))
        except DuplicateOrder:
            log.info("Already placed today, not repeating: %s", desc)
        except Exception as e:
            log.error("Order failed (%s): %s", desc, e)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    broker = None
    if a.dry_run:
        try:
            broker = Broker(load_credentials())   # read the account, but don't trade
        except SystemExit:
            broker = None
    run(dry_run=a.dry_run, force=a.force, broker=broker)


if __name__ == "__main__":
    main()
