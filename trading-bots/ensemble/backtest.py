"""Replay history through the four programs and report how the blend would have done.

Rules: a decision made with day t's closing prices is held during day t+1.
Program 4 is walk-forward: each model only learns from outcomes already known.
The last `out_of_sample_years` are reported separately.

Run:  python -m ensemble.backtest              (updates the data cache first)
      python -m ensemble.backtest --no-refresh (use cached data only)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.settings import ROOT, load_conf
from ensemble.data import load_prices
from ensemble.programs import all_votes, blend

CONF_PATH = "ensemble/ensemble.conf"


def stats(r: pd.Series) -> dict:
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    return {"total": eq.iloc[-1] - 1, "cagr": cagr, "vol": vol,
            "sharpe": (r.mean() * 252) / vol if vol > 0 else np.nan, "max_dd": dd.min()}


def worst_drawdown_window(r: pd.Series):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    return peak, trough, dd.min()


def returns_from_weights(W, rets, cost_bps):
    held = W.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    return gross - turnover * cost_bps / 1e4


def run_backtest(px: pd.DataFrame, conf, out=print) -> dict:
    a = conf["assets"]
    bt = conf["backtest"]
    start = pd.Timestamp(bt.get("start"))
    cost = bt.getfloat("cost_bps")
    oos_years = bt.getint("out_of_sample_years")

    out("Computing votes (Program 4 retrains every %d days in the backtest)..."
        % conf.getint("p4_ml", "backtest_retrain_every"))
    votes, prob = all_votes(px, conf, live=False)
    W = blend(votes, conf)
    rets = px.pct_change().fillna(0.0)

    strat = returns_from_weights(W, rets, cost)[start:]
    bench = rets[a["sp500"]][start:]
    if len(strat) < 252:
        raise SystemExit("Not enough history after the backtest start date.")

    # each program on its own, plus Program 4 NOT reversed, for comparison
    singles = {k: returns_from_weights(v, rets, cost)[start:] for k, v in votes.items()}
    straight = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    straight[a["sp500"]] = (prob >= conf.getfloat("p4_ml", "pessimistic_below")).astype(float).where(prob.notna(), 0.0)
    singles["p4_ml_not_reversed"] = returns_from_weights(straight, rets, cost)[start:]

    yearly = pd.DataFrame({
        "bot": (1 + strat).groupby(strat.index.year).prod() - 1,
        "sp500": (1 + bench).groupby(bench.index.year).prod() - 1,
    })
    oos_start = strat.index[-1] - pd.DateOffset(years=oos_years)
    peak, trough, bench_dd = worst_drawdown_window(bench)
    eq = (1 + strat).cumprod()
    bot_in_crash = eq[trough] / eq[peak] - 1

    s_all, b_all = stats(strat), stats(bench)
    s_oos, b_oos = stats(strat[oos_start:]), stats(bench[oos_start:])
    pct = lambda x: f"{x:+.1%}"

    lines = [
        "",
        f"BACKTEST {strat.index[0].date()} to {strat.index[-1].date()}  ({len(yearly)} calendar years)",
        f"{'':28}{'Bot':>10}{'S&P 500':>10}",
        f"{'Yearly growth (CAGR)':28}{pct(s_all['cagr']):>10}{pct(b_all['cagr']):>10}",
        f"{'Worst drop (max drawdown)':28}{pct(s_all['max_dd']):>10}{pct(b_all['max_dd']):>10}",
        f"{'Volatility':28}{s_all['vol']:>10.1%}{b_all['vol']:>10.1%}",
        f"{'Sharpe ratio':28}{s_all['sharpe']:>10.2f}{b_all['sharpe']:>10.2f}",
        f"Years with a gain: bot {(yearly['bot'] > 0).sum()} of {len(yearly)}, "
        f"S&P 500 {(yearly['sp500'] > 0).sum()} of {len(yearly)}",
        f"Worst S&P 500 crash in the test: {pct(bench_dd)} ({peak.date()} to {trough.date()}); "
        f"bot over the same dates: {pct(bot_in_crash)}",
        "",
        f"LAST {oos_years} YEARS ONLY (from {oos_start.date()})",
        f"{'Yearly growth (CAGR)':28}{pct(s_oos['cagr']):>10}{pct(b_oos['cagr']):>10}",
        f"{'Worst drop (max drawdown)':28}{pct(s_oos['max_dd']):>10}{pct(b_oos['max_dd']):>10}",
        "",
        "EACH PROGRAM ALONE (CAGR / worst drop)",
    ]
    for k, r in singles.items():
        st = stats(r)
        lines.append(f"  {k:24}{pct(st['cagr']):>10}{pct(st['max_dd']):>10}")
    lines += ["", "Year-by-year:", (yearly * 100).round(1).to_string(), ""]
    report = "\n".join(lines)
    out(report)

    res_dir = Path(bt.get("results_dir"))
    res_dir = res_dir if res_dir.is_absolute() else ROOT / res_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bot": (1 + strat).cumprod(), "sp500": (1 + bench).cumprod()}).to_csv(res_dir / "equity_curve.csv")
    yearly.to_csv(res_dir / "yearly_returns.csv")
    W[start:].to_csv(res_dir / "daily_target_weights.csv")
    (res_dir / "report.txt").write_text(report, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot((1 + strat).cumprod(), label="Bot", linewidth=1.6)
        ax.plot((1 + bench).cumprod(), label="S&P 500 buy and hold", linewidth=1.1, alpha=0.8)
        ax.axvspan(oos_start, strat.index[-1], color="grey", alpha=0.12, label=f"Last {oos_years} years")
        ax.set_yscale("log"); ax.set_ylabel("Growth of $1 (log scale)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Four-program ensemble vs S&P 500")
        fig.tight_layout(); fig.savefig(res_dir / "equity_curve.png", dpi=130); plt.close(fig)
    except ImportError:
        pass
    out(f"Saved results to {res_dir}")
    return {"all": s_all, "bench": b_all, "oos": s_oos, "yearly": yearly,
            "crash": (peak, trough, bench_dd, bot_in_crash), "weights": W}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-refresh", action="store_true", help="use cached prices only")
    args = ap.parse_args()
    conf = load_conf(CONF_PATH)
    a = conf["assets"]
    syms = [a["sp500"], a["nasdaq"], a["international"], a["bonds"]]
    px = load_prices(syms, a.get("history_start"), a.get("cache_dir"), refresh=not args.no_refresh)
    run_backtest(px, conf)


if __name__ == "__main__":
    main()
