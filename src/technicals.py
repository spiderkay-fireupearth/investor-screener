"""Technical indicators, computed per-market on that market's own bar series.

Critical detail for a five-market app: SGX, HKEX, SET, IDX and NYSE keep five
different holiday calendars — Chinese New Year, Songkran, Hari Raya, Golden
Week, Thanksgiving. Never reindex one market's bars onto another's calendar or
every moving average silently shifts. Each series here is computed on its own
trading days, and relative strength is measured against that market's OWN index
(an Indonesian name measured against the S&P tells you about the rupiah, not
about the company).
"""
from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


def _last(s: pd.Series) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    The zero-loss case matters more than it looks. A window with no down days
    makes avg_loss 0, and a naive implementation divides by zero and returns
    NaN — which propagates as "no RSI" and, under strict unknown-handling,
    silently fails the RSI band test on exactly the strongest momentum names.
    Zero loss with positive gain is RSI 100; zero of both is a flat series, 50.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    warmed = avg_gain.notna() & avg_loss.notna()
    out = out.mask(warmed & (avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask(warmed & (avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0.0))
    return (direction * df["Volume"].fillna(0.0)).cumsum()


def compute(df: pd.DataFrame,
            index_df: Optional[pd.DataFrame] = None,
            fx_to_usd: Optional[float] = None) -> Dict[str, Any]:
    """Full indicator set for one instrument. `df` is that market's own bars."""
    out: Dict[str, Any] = {}
    if df is None or df.empty or len(df) < 30:
        out["insufficient_history"] = True
        return out

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df else pd.Series(dtype=float)
    n = len(close)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    price = _last(close)
    out["price"] = price
    out["sma20"] = _last(sma20)
    out["sma50"] = _last(sma50)
    out["sma200"] = _last(sma200)
    out["ema20"] = _last(ema20)

    out["price_above_sma200"] = (
        1 if (price and out["sma200"] and price > out["sma200"]) else 0
    ) if out["sma200"] is not None else None
    out["sma50_above_sma200"] = (
        1 if (out["sma50"] and out["sma200"] and out["sma50"] > out["sma200"]) else 0
    ) if (out["sma50"] is not None and out["sma200"] is not None) else None

    # Golden / death cross within the last 60 sessions
    if n >= 220:
        cross = (sma50 > sma200).astype(int).diff()
        recent = cross.iloc[-60:]
        out["golden_cross_recent"] = int((recent == 1).any())
        out["death_cross_recent"] = int((recent == -1).any())
    else:
        out["golden_cross_recent"] = None
        out["death_cross_recent"] = None

    r = rsi(close, 14)
    out["rsi_14"] = _last(r)

    line, sig, hist = macd(close)
    out["macd_line"] = _last(line)
    out["macd_signal"] = _last(sig)
    out["macd_histogram"] = _last(hist)

    # Bollinger
    bb_mid = sma20
    bb_std = close.rolling(20).std()
    out["bb_upper"] = _last(bb_mid + 2 * bb_std)
    out["bb_lower"] = _last(bb_mid - 2 * bb_std)
    if price and out["bb_upper"] and out["bb_lower"] and out["bb_upper"] != out["bb_lower"]:
        out["bb_pct_b"] = (price - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    else:
        out["bb_pct_b"] = None

    if "High" in df and "Low" in df:
        a = atr(df, 14)
        out["atr_14"] = _last(a)
        atr_pct = (a / close).dropna()
        out["atr_pct"] = _last(atr_pct)
        if len(atr_pct) > 60:
            cur = atr_pct.iloc[-1]
            out["atr_pct_percentile"] = float((atr_pct <= cur).mean())
        else:
            out["atr_pct_percentile"] = None
    else:
        out["atr_14"] = out["atr_pct"] = out["atr_pct_percentile"] = None

    # 52-week and 5-year position
    w52 = close.iloc[-252:] if n >= 252 else close
    out["high_52w"] = float(w52.max())
    out["low_52w"] = float(w52.min())
    out["pct_below_52w_high"] = (
        (out["high_52w"] - price) / out["high_52w"] if out["high_52w"] else None
    )
    y5 = close.iloc[-1260:] if n >= 1260 else close
    low5 = float(y5.min())
    out["low_5y"] = low5
    out["pct_above_5y_low"] = (price - low5) / low5 if low5 else None
    out["years_of_price_history"] = round(n / 252, 1)

    # Volume participation
    if len(vol.dropna()) >= 50:
        v20 = vol.rolling(20).mean().iloc[-1]
        v50 = vol.rolling(50).mean().iloc[-1]
        out["vol20_over_vol50"] = float(v20 / v50) if v50 and v50 > 0 else None
        med_turnover = float((close * vol).iloc[-60:].median())
        out["median_turnover"] = med_turnover
        out["median_turnover_usd"] = med_turnover * fx_to_usd if fx_to_usd else None
    else:
        out["vol20_over_vol50"] = out["median_turnover"] = out["median_turnover_usd"] = None

    o = obv(df) if "Volume" in df else None
    if o is not None and len(o) > 50:
        out["obv_slope_50d"] = float(np.polyfit(range(50), o.iloc[-50:].values, 1)[0])
    else:
        out["obv_slope_50d"] = None

    # Returns
    for label, days in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
        if n > days:
            out[f"return_{label}"] = float(close.iloc[-1] / close.iloc[-1 - days] - 1)
        else:
            out[f"return_{label}"] = None

    # Relative strength vs this market's own index, aligned on shared dates only.
    out["rs_vs_market_index_6m"] = None
    out["rs_vs_market_index_12m"] = None
    if index_df is not None and not index_df.empty:
        idx = index_df["Close"].astype(float)
        joined = pd.concat([close.rename("stock"), idx.rename("index")],
                           axis=1, join="inner").dropna()
        for label, days in (("6m", 126), ("12m", 252)):
            if len(joined) > days:
                s_ret = joined["stock"].iloc[-1] / joined["stock"].iloc[-1 - days] - 1
                i_ret = joined["index"].iloc[-1] / joined["index"].iloc[-1 - days] - 1
                out[f"rs_vs_market_index_{label}"] = float(s_ret - i_ret)

    # ---- Marks: survival, and asymmetry ----------------------------------
    # "It's not enough to survive 'on average'; you have to survive on the
    # worst days." Peak-to-trough decline over five years is the closest
    # observable to that — it is what an owner actually had to sit through,
    # not a standard deviation.
    out["max_drawdown_5y"] = None
    win = close.iloc[-1260:] if len(close) > 1260 else close
    if len(win) >= 250:
        dd = win / win.cummax() - 1.0
        out["max_drawdown_5y"] = float(-dd.min())

    # Soros stage CD — "Doubts arise, but the trend survives... the trend
    # waivers but reasserts itself. Such testing may be repeated several
    # times." A trend that has been tested and held is a different object from
    # one that has never been tested, and the one-year drawdown is how you
    # tell them apart.
    out["max_drawdown_1y"] = None
    w1 = close.iloc[-252:]
    if len(w1) >= 200:
        out["max_drawdown_1y"] = float(-(w1 / w1.cummax() - 1.0).min())      # reported positive

    # "If we avoid the losers, the winners will take care of themselves."
    # Downside capture: how much of the index's bad days this name takes.
    # Below 1.0 means it falls less than the market when the market falls.
    # Deliberately measured on DOWN periods only — an all-days beta blends
    # the upside in and hides exactly the asymmetry Marks cares about.
    out["downside_capture"] = None
    out["upside_capture"] = None
    out["capture_ratio"] = None
    if index_df is not None and not index_df.empty:
        j = pd.concat([close.rename("s"), index_df["Close"].astype(float)
                       .rename("i")], axis=1, join="inner").dropna()
        if len(j) >= 260:
            r = j.pct_change().dropna().iloc[-756:]
            down, up = r[r["i"] < 0], r[r["i"] > 0]
            # Need a real sample of down periods; a handful of days would make
            # the ratio an artefact of two or three sessions.
            if len(down) >= 40 and down["i"].mean() != 0:
                out["downside_capture"] = float(
                    down["s"].mean() / down["i"].mean())
            if len(up) >= 40 and up["i"].mean() != 0:
                out["upside_capture"] = float(up["s"].mean() / up["i"].mean())
        if out["downside_capture"] and out["upside_capture"]:
            # The number Marks's risk/return diagram is really about: are you
            # being paid more upside than the downside you accept?
            out["capture_ratio"] = (out["upside_capture"]
                                    / out["downside_capture"])

    out["last_bar_date"] = df.index[-1].date().isoformat()
    out["insufficient_history"] = False
    return out
