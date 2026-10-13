"""Mid-cap growth momentum stock picker.

  1. Screen: US stocks worth $2-10B, price above $10, revenue growing 15%+.
  2. Momentum: 12-month return (skipping the last month), 6-month return,
     price above its 50-day average and the 50-day above the 200-day.
  3. Fundamentals check on the strongest names (revenue and earnings growth).
  4. Rank on momentum + growth and print the top 3 with a one-line thesis.

This is a screen, not advice: it ranks what the numbers show today.

Run:  python -m stock_picker.picker
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from common.settings import ROOT, load_conf

CONF_PATH = "stock_picker/picker.conf"


# ------------------------------------------------------------ 1. universe
def screen_universe(u, yf=None) -> pd.DataFrame:
    """Mid-cap growth names from Yahoo's screener, or universe.txt as a fallback."""
    import yfinance as _yf
    yf = yf or _yf
    q = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("btwn", ["intradaymarketcap", u.getfloat("min_market_cap"), u.getfloat("max_market_cap")]),
        yf.EquityQuery("gt", ["intradayprice", u.getfloat("min_price")]),
        yf.EquityQuery("gt", ["quarterlyrevenuegrowth.quarterly", u.getfloat("min_quarterly_revenue_growth_pct")]),
    ])
    rows, offset, limit = [], 0, u.getint("max_candidates")
    try:
        while offset < limit:
            res = yf.screen(q, offset=offset, size=min(250, limit - offset),
                            sortField="intradaymarketcap", sortAsc=False)
            quotes = res.get("quotes", [])
            rows += [{"symbol": x.get("symbol"),
                      "name": x.get("shortName") or x.get("longName") or x.get("symbol"),
                      "market_cap": x.get("marketCap")} for x in quotes]
            if len(quotes) < 250:
                break
            offset += len(quotes)
    except Exception as e:
        print(f"Yahoo screener unavailable ({e}); trying {u.get('universe_file')}")
        rows = []
    if rows:
        df = pd.DataFrame(rows).dropna(subset=["symbol"])
        # skip share-class duplicates and odd symbols (e.g. BRK.B style, warrants)
        return df[~df["symbol"].str.contains(r"[.\-^=]")].drop_duplicates("symbol")

    path = Path(u.get("universe_file"))
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise SystemExit("No universe: the screener failed and no universe.txt was found.")
    syms = [s.strip().upper() for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")]
    return pd.DataFrame({"symbol": syms, "name": syms, "market_cap": np.nan})


# ------------------------------------------------------------ 2. momentum
def momentum_table(close: pd.DataFrame, volume: pd.DataFrame, m) -> pd.DataFrame:
    long_d, skip_d, med_d = m.getint("long_days"), m.getint("skip_days"), m.getint("medium_days")
    rows = []
    for sym in close.columns:
        p = close[sym].dropna()
        if len(p) < long_d + 1:
            continue
        v = volume[sym].reindex(p.index).fillna(0)
        last = p.iloc[-1]
        sma50, sma200 = p.tail(50).mean(), p.tail(200).mean()
        rows.append({
            "symbol": sym,
            "price": last,
            "mom_12_1": p.iloc[-1 - skip_d] / p.iloc[-1 - long_d] - 1,
            "mom_6m": last / p.iloc[-1 - med_d] - 1,
            "vs_200d": last / sma200 - 1,
            "trend_ok": bool(last > sma50 > sma200),
            "dollar_vol": float((p.tail(20) * v.tail(20)).mean()),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t[t["trend_ok"] & (t["dollar_vol"] >= m.getfloat("min_avg_dollar_volume"))]
    t["mom_score"] = t["mom_12_1"].rank(pct=True) * 0.6 + t["mom_6m"].rank(pct=True) * 0.4
    return t.sort_values("mom_score", ascending=False)


# ------------------------------------------------------------ 3. fundamentals
def add_fundamentals(t: pd.DataFrame, yf=None) -> pd.DataFrame:
    import yfinance as _yf
    yf = yf or _yf
    info_rows = []
    for sym in t["symbol"]:
        try:
            i = yf.Ticker(sym).info or {}
        except Exception:
            i = {}
        info_rows.append({"symbol": sym, "rev_growth": i.get("revenueGrowth"),
                          "eps_growth": i.get("earningsGrowth"),
                          "sector": i.get("sector") or "", "industry": i.get("industry") or "",
                          "long_name": i.get("shortName") or i.get("longName")})
    return t.merge(pd.DataFrame(info_rows), on="symbol", how="left")


def thesis(r) -> str:
    who = f"{r['symbol']} ({r['name']}{', ' + r['industry'] if r.get('industry') else ''})"
    growth = f"revenue up {r['rev_growth']:.0%} y/y" if pd.notna(r.get("rev_growth")) else "fast revenue growth"
    if pd.notna(r.get("eps_growth")) and r["eps_growth"] > 0:
        growth += f", earnings up {r['eps_growth']:.0%}"
    return (f"{who}: {growth}; shares +{r['mom_12_1']:.0%} over the past year (ex last month) "
            f"and {r['vs_200d']:.0%} above the 200-day average.")


def pick(conf, yf=None, out=print) -> pd.DataFrame:
    import yfinance as _yf
    yf = yf or _yf
    u, m, o = conf["universe"], conf["momentum"], conf["output"]

    uni = screen_universe(u, yf)
    out(f"Universe: {len(uni)} mid-cap growth stocks")
    data = yf.download(list(uni["symbol"]), period="15mo", auto_adjust=True, progress=False,
                       threads=True, group_by="column")
    close, volume = data["Close"], data["Volume"]
    if isinstance(close, pd.Series):          # single ticker edge case
        close, volume = close.to_frame(uni["symbol"].iloc[0]), volume.to_frame(uni["symbol"].iloc[0])

    mom = momentum_table(close, volume, m)
    out(f"In an uptrend with enough trading volume: {len(mom)}")
    short = add_fundamentals(mom.head(m.getint("shortlist")), yf)
    short = short.merge(uni[["symbol", "name", "market_cap"]], on="symbol", how="left")
    short["name"] = short["long_name"].fillna(short["name"])
    short = short[short["rev_growth"].isna() | (short["rev_growth"] > 0)]

    growth_rank = short["rev_growth"].rank(pct=True).fillna(0.5)
    short["score"] = short["mom_score"].rank(pct=True) * 0.6 + growth_rank * 0.4
    short = short.sort_values("score", ascending=False).reset_index(drop=True)

    top = short.head(o.getint("top_n"))
    out(f"\nTOP {len(top)} MID-CAP GROWTH MOMENTUM NAMES ({date.today()})")
    for n, (_, r) in enumerate(top.iterrows(), 1):
        out(f"{n}. {thesis(r)}")
    out("\nScreen output, not investment advice. Check the news and filings before acting.")

    rep = Path(o.get("report_dir"))
    rep = rep if rep.is_absolute() else ROOT / rep
    rep.mkdir(parents=True, exist_ok=True)
    f = rep / f"picks_{date.today()}.csv"
    short.to_csv(f, index=False)
    out(f"Full ranked shortlist saved to {f}")
    return top


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    pick(load_conf(CONF_PATH))


if __name__ == "__main__":
    main()
