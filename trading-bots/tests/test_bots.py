"""Offline tests on simulated prices (no internet, no real orders).

Run:  python -m pytest -q tests
"""
import configparser
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from common.settings import load_conf

NY = ZoneInfo("America/New_York")
MORNING = datetime(2025, 6, 10, 9, 31, tzinfo=NY)   # a Tuesday, inside the run window


@pytest.fixture(scope="session", autouse=True)
def _logs_to_temp(tmp_path_factory):
    """Keep test runs out of the real logs/ files the bots write."""
    import common.log
    import ensemble.run
    import ma_crossover.bot
    real, log_dir = common.log.get_logger, tmp_path_factory.mktemp("logs")
    with pytest.MonkeyPatch.context() as mp:
        for mod in (ma_crossover.bot, ensemble.run):
            mp.setattr(mod, "get_logger", lambda name, f: real(f"test-{name}", log_dir / f"{name}.log"))
        yield


# ------------------------------------------------------------------ helpers
def synthetic_prices(start="2000-01-03", end="2025-06-09", seed=7) -> pd.DataFrame:
    """Four fake funds with bull runs, crashes and recoveries."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    n = len(idx)
    regime = np.where((idx.year == 2008) | ((idx.year == 2020) & (idx.month <= 3)) | (idx.year == 2022), -0.0012, 0.0005)
    market = regime + rng.normal(0, 0.011, n)
    out = {
        "SPY": market,
        "QQQ": 1.3 * market + rng.normal(0, 0.006, n),
        "EFA": 0.8 * market + rng.normal(0, 0.007, n) - 0.0001,
        "IEF": -0.15 * market + rng.normal(0.0001, 0.003, n),
    }
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in out.items()}, index=idx)


class FakeBroker:
    paper = True

    def __init__(self, equity=100_000.0, holdings=None, qty=None, is_open=True):
        self._equity, self._holdings, self._qty, self._open = equity, holdings or {}, qty or {}, is_open
        self.orders, self.closed = [], []

    def market_is_open(self): return self._open
    def equity(self): return self._equity
    def position_values(self): return dict(self._holdings)
    def position_qty(self, s): return self._qty.get(s, 0.0)

    def market_order(self, symbol, side, qty=None, notional=None, client_order_id=None):
        from common.broker import DuplicateOrder
        if any(o["id"] == client_order_id for o in self.orders):
            raise DuplicateOrder(client_order_id)
        self.orders.append({"symbol": symbol, "side": side, "qty": qty, "notional": notional, "id": client_order_id})
        return SimpleNamespace(id=f"fake-{len(self.orders)}", status="accepted")

    def close_position(self, symbol):
        self.closed.append(symbol)
        return SimpleNamespace(id="fake-close")


# ------------------------------------------------------------------ MA crossover
def crossing_series(direction="up"):
    """Prices that fall then rise (or the reverse) so the 20-day crosses the 50-day."""
    idx = pd.bdate_range("2025-01-02", periods=100)
    if direction == "up":
        p = np.r_[np.linspace(200, 150, 70), np.linspace(151, 215, 30)]
    else:
        p = np.r_[np.linspace(150, 200, 70), np.linspace(199, 135, 30)]
    return pd.Series(p, index=idx)


def first_cross(series, direction):
    from ma_crossover.bot import detect_crossover
    want = "BUY" if direction == "up" else "SELL"
    for n in range(52, len(series) + 1):
        sig, _ = detect_crossover(series.iloc[:n], 20, 50)
        if sig is not None:
            return sig, n
    return None, None


@pytest.mark.parametrize("direction,expected", [("up", "BUY"), ("down", "SELL")])
def test_crossover_fires_once_on_the_crossing_day(direction, expected):
    from ma_crossover.bot import detect_crossover
    s = crossing_series(direction)
    sig, n = first_cross(s, direction)
    assert sig == expected
    short, long_ = s.iloc[:n].rolling(20).mean(), s.iloc[:n].rolling(50).mean()
    assert (short.iloc[-2] - long_.iloc[-2]) * (short.iloc[-1] - long_.iloc[-1]) <= 0   # sign actually flipped
    assert detect_crossover(s.iloc[: n + 1], 20, 50)[0] is None                       # not again the next day


def test_crossover_needs_enough_data():
    from ma_crossover.bot import detect_crossover
    with pytest.raises(ValueError):
        detect_crossover(crossing_series().iloc[:40], 20, 50)


def test_today_partial_bar_is_dropped():
    from common.market import drop_unfinished_day
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-06-06", "2025-06-09", "2025-06-10"]))
    assert list(drop_unfinished_day(s, MORNING).index.strftime("%m-%d")) == ["06-06", "06-09"]


def _patch_download(monkeypatch, series):
    import ma_crossover.bot as bot
    frame = pd.DataFrame({"Close": series})
    monkeypatch.setattr(bot, "download_daily", lambda *a, **k: frame)


def test_bot_buys_on_signal_and_never_twice(monkeypatch):
    import ma_crossover.bot as bot
    s = crossing_series("up")
    _, n = first_cross(s, "up")
    _patch_download(monkeypatch, s.iloc[:n])
    b = FakeBroker()
    assert bot.run(broker=b, now=MORNING) == "buy"
    assert b.orders[0]["symbol"] == "AAPL" and b.orders[0]["side"] == "BUY" and b.orders[0]["qty"] == 1
    assert bot.run(broker=b, now=MORNING) == "duplicate"        # second scheduler run same day
    assert len(b.orders) == 1


def test_bot_does_not_short_without_shares(monkeypatch):
    import ma_crossover.bot as bot
    s = crossing_series("down")
    _, n = first_cross(s, "down")
    _patch_download(monkeypatch, s.iloc[:n])
    assert bot.run(broker=FakeBroker(qty={}), now=MORNING) == "sell-skipped"
    b = FakeBroker(qty={"AAPL": 3})
    assert bot.run(broker=b, now=MORNING) == "sell" and b.orders[0]["qty"] == 1


def test_bot_respects_window_and_holidays(monkeypatch):
    import ma_crossover.bot as bot
    _patch_download(monkeypatch, crossing_series())
    assert bot.run(broker=FakeBroker(), now=MORNING.replace(hour=8, minute=30)) == "skipped-window"
    assert bot.run(broker=FakeBroker(), now=MORNING.replace(hour=15, minute=50)) == "skipped-window"
    assert bot.run(broker=FakeBroker(), now=MORNING.replace(hour=11, minute=5)) != "skipped-window"  # GitHub ran late
    assert bot.run(broker=FakeBroker(), now=MORNING.replace(hour=15, minute=30)) != "skipped-window"  # hours late
    assert bot.run(broker=FakeBroker(is_open=False), now=MORNING) == "skipped-closed"


# ------------------------------------------------------------------ Ensemble
@pytest.fixture(scope="module")
def conf(tmp_path_factory):
    c = load_conf("ensemble/ensemble.conf")
    c["backtest"]["results_dir"] = str(tmp_path_factory.mktemp("bt"))
    return c


@pytest.fixture(scope="module")
def px():
    return synthetic_prices()


def test_votes_are_valid_weights(px, conf):
    from ensemble.programs import all_votes, blend
    votes, prob = all_votes(px, conf, live=False)
    for name, v in votes.items():
        assert ((v >= -1e-9) & (v <= 1 + 1e-9)).all().all(), name
        assert (v.sum(axis=1) <= 1 + 1e-9).all(), name
    W = blend(votes, conf)
    assert (W.sum(axis=1) <= 1 + 1e-9).all()
    assert prob.notna().sum() > 3000


def test_no_lookahead(px, conf):
    """Changing future prices must not change any past decision."""
    from ensemble.programs import all_votes, blend
    cut = px.index[4000]
    future_changed = px.copy()
    future_changed.loc[future_changed.index > cut] *= np.linspace(1, 0.3, (px.index > cut).sum())[:, None]
    W1 = blend(all_votes(px, conf)[0], conf)
    W2 = blend(all_votes(future_changed, conf)[0], conf)
    pd.testing.assert_frame_equal(W1.loc[:cut], W2.loc[:cut])


def test_backtest_runs_and_reports(px, conf):
    from ensemble.backtest import run_backtest
    lines = []
    res = run_backtest(px, conf, out=lines.append)
    text = "\n".join(lines)
    assert "Years with a gain" in text and "LAST 6 YEARS" in text and "p4_ml_not_reversed" in text
    assert -1 < res["all"]["max_dd"] <= 0
    assert len(res["yearly"]) >= 20


def test_plan_trades_only_moves_the_difference():
    from ensemble.run import plan_trades
    target = pd.Series({"SPY": 0.5, "QQQ": 0.3, "EFA": 0.0, "IEF": 0.1})
    prices = pd.Series({"SPY": 500.0, "QQQ": 400.0, "EFA": 80.0, "IEF": 95.0})
    holdings = {"SPY": 49_500.0, "QQQ": 10_000.0, "EFA": 5_000.0}      # SPY within 1%: no trade
    orders = plan_trades(target, 100_000, holdings, prices, 0.01, True)
    by = {o["symbol"]: o for o in orders}
    assert "SPY" not in by
    assert by["EFA"]["close_all"] and by["EFA"]["side"] == "sell"
    assert by["QQQ"]["side"] == "buy" and abs(by["QQQ"]["notional"] - 20_000) < 1e-6
    assert by["IEF"]["side"] == "buy" and abs(by["IEF"]["notional"] - 10_000) < 1e-6
    assert orders[0]["side"] == "sell"                                   # sells go first


def test_live_run_trades_against_account(px, conf, monkeypatch):
    import ensemble.run as er
    monkeypatch.setattr(er, "load_conf", lambda p: conf)
    b = FakeBroker(equity=100_000, holdings={"EFA": 40_000.0, "TSLA": 5_000.0})
    target = er.run(broker=b, now=MORNING, px=px)
    assert target is not None and target.sum() <= 1 + 1e-9
    assert "TSLA" not in [o["symbol"] for o in b.orders] + b.closed      # untouched
    again = len(b.orders)
    er.run(broker=b, now=MORNING, px=px)                                  # second run same day
    assert len(b.orders) == again


def test_live_run_only_spends_its_own_budget(px, conf, monkeypatch):
    """$100k account with $70k in AAPL: the ensemble may only use the other $30k."""
    import ensemble.run as er
    monkeypatch.setattr(er, "load_conf", lambda p: conf)
    monkeypatch.setattr(er, "blend", lambda votes, c: pd.DataFrame(      # force a fully invested target
        {"SPY": 0.5, "QQQ": 0.5, "EFA": 0.0, "IEF": 0.0}, index=px.index))
    b = FakeBroker(equity=100_000, holdings={"AAPL": 70_000.0})
    er.run(broker=b, now=MORNING, px=px)
    bought = sum(o["notional"] for o in b.orders if o["side"] == "buy")
    assert abs(bought - 30_000) < 1e-6
    assert "AAPL" not in [o["symbol"] for o in b.orders] + b.closed


# ------------------------------------------------------------------ Stock picker
class FakeYF:
    def __init__(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2024-03-01", periods=320)
        self.syms = [f"MID{i}" for i in range(12)]
        drift = np.linspace(-0.001, 0.003, 12)
        close = {s: 50 * np.cumprod(1 + d + rng.normal(0, 0.012, len(idx))) for s, d in zip(self.syms, drift)}
        self.close = pd.DataFrame(close, index=idx)
        self.vol = pd.DataFrame(1_000_000, index=idx, columns=self.syms)

    class EquityQuery:
        def __init__(self, *a): self.a = a

    def screen(self, q, **k):
        return {"quotes": [{"symbol": s, "shortName": f"{s} Corp", "marketCap": 5e9} for s in self.syms] + [{"symbol": "BRK.B"}]}

    def download(self, syms, **k):
        return pd.concat({"Close": self.close[syms], "Volume": self.vol[syms]}, axis=1)

    industries = ("Software", "Semiconductors", "Biotechnology")

    def Ticker(self, s):
        g = 0.10 + 0.02 * int(s[3:])
        return SimpleNamespace(info={"revenueGrowth": g, "earningsGrowth": g * 1.5,
                                     "industry": self.industries[int(s[3:]) % len(self.industries)],
                                     "sector": "Technology", "shortName": f"{s} Inc."})


def test_picker_returns_top3_with_thesis(tmp_path):
    from stock_picker.picker import pick, thesis
    c = load_conf("stock_picker/picker.conf")
    c["output"]["report_dir"] = str(tmp_path)
    lines = []
    top = pick(c, yf=FakeYF(), out=lines.append)
    assert len(top) == 3
    assert all(top["trend_ok"])
    assert all(s.startswith("MID") for s in top["symbol"])
    t = thesis(top.iloc[0])
    assert "revenue up" in t and "above the 200-day" in t
    assert any("not investment advice" in l for l in lines)
    assert top["industry"].nunique() == 3                            # one per industry


def test_picker_caps_each_industry():
    from stock_picker.picker import pick_diversified
    ranked = pd.DataFrame({"symbol": ["ECO", "INSW", "IOVA", "STNG", "XYZ"],
                           "industry": ["Shipping", "Shipping", "Biotech", "Shipping", ""]})
    assert list(pick_diversified(ranked, 3, 1)["symbol"]) == ["ECO", "IOVA", "XYZ"]
    assert list(pick_diversified(ranked, 3, 2)["symbol"]) == ["ECO", "INSW", "IOVA"]


def test_picker_skips_stretched_names(tmp_path):
    from stock_picker.picker import pick
    c = load_conf("stock_picker/picker.conf")
    c["output"]["report_dir"] = str(tmp_path)
    c["momentum"]["max_above_200d_pct"] = "0"                        # everything in an uptrend is "stretched"
    lines = []
    top = pick(c, yf=FakeYF(), out=lines.append)
    assert top.empty and any(l.startswith("Skipped ") for l in lines)


def test_live_run_skips_programs_with_zero_weight(px, conf):
    from ensemble.programs import all_votes, blend
    c = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    c.read_dict(conf)
    c["weights"].update({"p1_trend": "0.6", "p2_momentum": "0.4", "p3_dip": "0", "p4_ml": "0"})
    votes, prob = all_votes(px, c, live=True)
    assert set(votes) == {"p1_trend", "p2_momentum"} and prob.isna().all()   # model not trained
    assert abs(blend(votes, c).iloc[-1].sum()) <= 1 + 1e-9
    assert set(all_votes(px.iloc[-1500:], c, live=False)[0]) == {"p1_trend", "p2_momentum", "p3_dip", "p4_ml"}


def test_picker_skips_names_that_move_together():
    """Two tanker stocks under different Yahoo labels: correlation catches them."""
    from stock_picker.picker import pick_diversified
    rng = np.random.default_rng(5)
    oil = rng.normal(0, 0.02, 126)
    rets = pd.DataFrame({"ECO": oil + rng.normal(0, 0.005, 126), "INSW": oil + rng.normal(0, 0.005, 126),
                         "XMTR": rng.normal(0, 0.02, 126), "ABC": rng.normal(0, 0.02, 126)})
    ranked = pd.DataFrame({"symbol": ["ECO", "INSW", "XMTR", "ABC"],
                           "industry": ["Marine Shipping", "Oil & Gas Midstream", "Industrial Distribution", "Software"]})
    lines = []
    top = pick_diversified(ranked, 3, 1, rets, 0.7, out=lines.append)
    assert list(top["symbol"]) == ["ECO", "XMTR", "ABC"]
    assert any("INSW" in l and "ECO" in l for l in lines)


def test_picker_gives_up_on_a_stalled_lookup():
    import time
    from stock_picker.picker import add_fundamentals

    class SlowYF(FakeYF):
        def Ticker(self, s):
            if s == "MID1":
                time.sleep(5)                                # Yahoo throttling this one
            return super().Ticker(s)

    lines, t0 = [], time.time()
    df = add_fundamentals(pd.DataFrame({"symbol": ["MID0", "MID1", "MID2"]}), SlowYF(), timeout=0.5, out=lines.append)
    assert time.time() - t0 < 3
    assert df.set_index("symbol")["rev_growth"].isna().tolist() == [False, True, False]
    assert any("Checking fundamentals 1/3" in l for l in lines) and any("MID1" in l for l in lines)
