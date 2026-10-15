"""The four voting programs and the blend.

Every program returns a DataFrame of target weights: one row per trading day,
one column per fund, values 0..1 (the rest is cash). The row for day t uses only
prices up to the close of day t, and is what to hold on day t+1.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _zeros(px):
    return pd.DataFrame(0.0, index=px.index, columns=px.columns)


# ---------------------------------------------------------------- Program 1
def p1_trend(px, a, c):
    """NASDAQ above its 200-day average -> own it, scaled down when volatile."""
    q = px[a["nasdaq"]]
    sma = q.rolling(c.getint("sma_days")).mean()
    vol = q.pct_change().rolling(c.getint("vol_days")).std() * np.sqrt(TRADING_DAYS)
    size = (c.getfloat("target_vol") / vol).clip(upper=1.0)
    w = _zeros(px)
    w[a["nasdaq"]] = ((q > sma) * size).fillna(0.0)
    return w


# ---------------------------------------------------------------- Program 2
def p2_momentum(px, a, c):
    """Once a month: hold US or international stocks, whichever did better over a
    year. If both lost money, hold bonds."""
    lb = c.getint("lookback_days")
    us, intl, bonds = a["sp500"], a["international"], a["bonds"]
    r = px[[us, intl]].pct_change(lb)

    pick = pd.Series(np.where(r[us] >= r[intl], us, intl), index=px.index)
    pick[(r[us] <= 0) & (r[intl] <= 0)] = bonds
    pick[r.isna().any(axis=1)] = None

    # decide on the last trading day of each month, hold through the next month
    dates = px.index.to_series()
    month_end = dates.dt.to_period("M") != dates.shift(-1).dt.to_period("M")
    # last row: it closes a month only if the next business day is in a new month
    last = px.index[-1]
    month_end.iloc[-1] = (last + pd.offsets.BDay(1)).month != last.month
    decided = pick.where(month_end).ffill()

    w = _zeros(px)
    for sym in (us, intl, bonds):
        w[sym] = (decided == sym).astype(float)
    return w


# ---------------------------------------------------------------- Program 3
def p3_dip(px, a, c):
    """Buy a sharp 2-day drop while the long trend is still up; sell on the bounce
    (price back to where it was before the drop), or after max_hold_days."""
    s = px[a["sp500"]]
    n = c.getint("drop_days")
    drop = s.pct_change(n)
    sma = s.rolling(c.getint("trend_sma_days")).mean()
    entry = (drop <= -c.getfloat("drop_pct")) & (s > sma)
    max_hold = c.getint("max_hold_days")

    held = np.zeros(len(s))
    in_pos, ref, days = False, 0.0, 0
    vals, ent = s.to_numpy(), entry.to_numpy()
    for i in range(len(s)):
        if not in_pos:
            if ent[i]:
                in_pos, ref, days = True, vals[i - n], 0
        else:
            days += 1
            if vals[i] >= ref or days >= max_hold:
                in_pos = False
        held[i] = 1.0 if in_pos else 0.0

    w = _zeros(px)
    w[a["sp500"]] = held
    return w


# ---------------------------------------------------------------- Program 4
def make_features(px, a) -> pd.DataFrame:
    """14 features describing the S&P 500's recent behavior."""
    s = px[a["sp500"]]
    b = px[a["bonds"]]
    r = s.pct_change()
    f = pd.DataFrame(index=px.index)
    for d in (1, 2, 3, 5, 10, 20, 60):                       # 7 return windows
        f[f"ret_{d}d"] = s.pct_change(d)
    for d in (20, 50, 200):                                   # 3 distance-from-average
        f[f"vs_sma{d}"] = s / s.rolling(d).mean() - 1
    f["vol_10d"] = r.rolling(10).std() * np.sqrt(TRADING_DAYS)   # 2 volatility
    f["vol_20d"] = r.rolling(20).std() * np.sqrt(TRADING_DAYS)
    gain = r.clip(lower=0).rolling(14).mean()                   # 1 RSI
    loss = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["bonds_ret_20d"] = b.pct_change(20)                       # 1 bond trend
    return f


def _model(c):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=c.getint("max_iter"), max_depth=c.getint("max_depth"),
        learning_rate=c.getfloat("learning_rate"), random_state=42)


def p4_probabilities(px, a, c, retrain_every=21, last_only=False) -> pd.Series:
    """Walk-forward 'chance the S&P 500 is up in 5 days', retrained from scratch.

    At day i the model only trains on rows whose 5-day outcome was already known
    at day i (rows up to i - horizon), so it never sees the future.
    last_only=True (live runs): one fresh model trained on all known history,
    predicting the latest day only.
    """
    h = c.getint("horizon_days")
    X = make_features(px, a)
    s = px[a["sp500"]]
    y = (s.shift(-h) / s - 1 > 0).astype(float)
    y[s.shift(-h).isna()] = np.nan

    ok = X.notna().all(axis=1).to_numpy()
    Xv, yv = X.to_numpy(), y.to_numpy()
    prob = pd.Series(np.nan, index=px.index)
    first = c.getint("min_train_days")
    valid_idx = np.flatnonzero(ok)
    if len(valid_idx) <= first:
        return prob
    start = valid_idx[first]

    i = len(px) - 1 if last_only else start
    while i < len(px):
        end_train = i - h                           # labels known up to here
        tr = ok[: end_train + 1] & ~np.isnan(yv[: end_train + 1])
        model = _model(c).fit(Xv[: end_train + 1][tr], yv[: end_train + 1][tr])
        stop = min(i + retrain_every, len(px))
        rows = np.arange(i, stop)
        rows = rows[ok[rows]]
        if len(rows):
            prob.iloc[rows] = model.predict_proba(Xv[rows])[:, 1]
        i = stop
    return prob


def p4_contrarian(px, a, c, retrain_every=21, last_only=False, prob=None):
    """Wired backwards on purpose: when the model is pessimistic, buy."""
    if prob is None:
        prob = p4_probabilities(px, a, c, retrain_every, last_only)
    w = _zeros(px)
    w[a["sp500"]] = (prob < c.getfloat("pessimistic_below")).astype(float).where(prob.notna(), 0.0)
    return w, prob


# ---------------------------------------------------------------- Blend
def all_votes(px, conf, live=False):
    """live=True: Program 4 trains once on everything and votes for today only, and
    programs with a weight of 0 are skipped (they can't change the target).
    live=False (backtest): every program runs, so each can be reported on its own;
    Program 4 is walk-forward, retraining every `backtest_retrain_every` days."""
    a = conf["assets"]
    used = lambda k: not live or conf.getfloat("weights", k) > 0
    votes = {}
    for name, fn in (("p1_trend", p1_trend), ("p2_momentum", p2_momentum), ("p3_dip", p3_dip)):
        if used(name):
            votes[name] = fn(px, a, conf[name])
    prob = pd.Series(np.nan, index=px.index)
    if used("p4_ml"):
        every = conf.getint("p4_ml", "backtest_retrain_every")
        votes["p4_ml"], prob = p4_contrarian(px, a, conf["p4_ml"], every, last_only=live)
    return votes, prob


def blend(votes, conf) -> pd.DataFrame:
    wts = {k: conf.getfloat("weights", k) for k in votes}
    total = sum(wts.values())
    if abs(total - 1) > 1e-6:
        raise SystemExit(f"[weights] must add up to 1.0, they add up to {total:.3f}")
    out = sum(votes[k] * w for k, w in wts.items())
    return out.clip(lower=0)
