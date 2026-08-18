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


SPARK_POINTS = 100
SPARK_YEARS = 2


def sparkline(df: pd.DataFrame, points: int = SPARK_POINTS,
              years: int = SPARK_YEARS) -> Optional[Dict[str, Any]]:
    """A compact price series with its 50- and 200-day averages.

    Three decisions here are about PAYLOAD, not about analysis, and they are
    worth stating because they are the difference between a page that loads on
    a phone and one that does not:

      * Sampled to ~80 points over two years, not 500 daily bars. At the size
        this renders — a strip a few centimetres wide — the extra resolution is
        invisible and costs six times the bytes.
      * Normalised to 100 at the start and rounded to one decimal. The chart is
        about SHAPE; the axis labels carry the real prices.
      * The 200-day average is sampled from the DAILY series at the same dates,
        not recomputed from the sampled points. A 200-day mean of 80 weekly
        samples would be a different and wrong line.

    Returns None where there is not enough history to draw anything honest.
    """
    if df is None or "Close" not in df or len(df) < 60:
        return None
    close = df["Close"].astype(float).dropna()
    if len(close) < 60:
        return None
    window = close.iloc[-(years * 252):]
    sma50 = close.rolling(50).mean().iloc[-(years * 252):]
    sma200 = close.rolling(200).mean().iloc[-(years * 252):]
    # Ceiling, not floor: a floored step overshoots the cap (504 bars // 100 is
    # a step of 5, which yields 101 points, not 100). The cap is a payload
    # budget, so it has to be one.
    step = max(1, -(-len(window) // points))
    idx = list(range(len(window) - 1, -1, -step))[::-1]
    base = float(window.iloc[idx[0]])
    if not base:
        return None

    def _norm(series):
        out = []
        for i in idx:
            v = series.iloc[i] if i < len(series) else None
            out.append(round(float(v) / base * 100.0, 1)
                       if v is not None and v == v else None)
        return out

    px = [round(float(window.iloc[i]) / base * 100.0, 1) for i in idx]
    dates = []
    for i in idx:
        try:
            dates.append(str(window.index[i])[:10])
        except Exception:                             # noqa: BLE001
            dates.append("")
    return {
        "px": px, "ma50": _norm(sma50), "ma": _norm(sma200),
        # Only three date labels are kept, not one hundred. The axis shows
        # three ticks, and a hover tooltip reads its date from these plus the
        # index — carrying every date would double the series payload to
        # populate a label the chart never draws.
        "d0": dates[0] if dates else "",
        "dmid": dates[len(dates) // 2] if dates else "",
        "d1": dates[-1] if dates else "",
        "first": round(base, 4), "last": round(float(window.iloc[-1]), 4),
        "lo": round(float(window.min()), 4), "hi": round(float(window.max()), 4),
        "points": len(px),
        "years": round(len(window) / 252.0, 1),
    }


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
    _regime = rsi_regime(price, out.get("sma50"), out.get("sma200"))
    _div = rsi_divergence(close, r)
    _read = rsi_reading(out["rsi_14"], _regime, _div)
    out["rsi_regime"] = _regime
    out["rsi_label"] = _read.get("label")
    out["rsi_note"] = _read.get("note")
    out["rsi_divergence"] = _read.get("divergence")
    out["rsi_divergence_note"] = _read.get("divergence_note")

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
    out["spark"] = sparkline(df)
    # The price five years ago, for Buffett's one-dollar test: what the market
    # capitalisation was then is half of "did a dollar retained become a dollar
    # of value?" and nothing else on the page carries it.
    out["price_5y_ago"] = float(close.iloc[-1260]) if n > 1260 else None
    out["return_5y"] = (float(price / out["price_5y_ago"] - 1)
                        if out["price_5y_ago"] else None)

    # Schloss's false-bottom check. A stock down from 125 to 60 looks like a
    # bargain until you notice it traded at 20 three years ago: the 52-week
    # chart shows a collapse, the 10-year chart shows the collapse is not over.
    # The ten-year window is the one that tells you which you are looking at.
    y10 = close.iloc[-2520:] if n >= 2520 else close
    low10 = float(y10.min())
    high10 = float(y10.max())
    out["low_10y"] = low10
    out["high_10y"] = high10
    out["pct_above_10y_low"] = (price - low10) / low10 if low10 else None
    # Where the price sits inside its own decade: 0 is the floor, 1 the ceiling.
    out["price_in_10y_range"] = ((price - low10) / (high10 - low10)
                                 if high10 > low10 else None)

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
    # The worst single month inside the last six. A 30% fall concentrated in
    # one month is a cascade; the same fall spread over six is a re-rating,
    # and telling them apart is most of the dislocation question.
    out["worst_month_in_6m"] = None
    w6 = close.iloc[-126:]
    if len(w6) >= 60:
        roll = w6 / w6.shift(21) - 1.0
        mn = roll.dropna().min()
        if mn == mn:
            out["worst_month_in_6m"] = float(-mn) if mn < 0 else 0.0

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


# ---------------------------------------------------------------------------
# Reading an RSI number, which depends on what the market is doing.
#
# The textbook 30/70 bands are a RANGING-market rule. In a strong trend they
# are actively misleading: RSI rarely reaches 30 in a bull run, so a reading of
# 45 there is a pullback to buy rather than the "bearish zone" a static table
# would call it — and in a bear market RSI can sit under 30 for weeks while the
# price keeps falling, so "oversold" is not a buy signal, it is a description.
#
# The app already knows the regime from the moving averages, so the reading is
# made against that rather than against a fixed table.
# ---------------------------------------------------------------------------

def rsi_regime(price, sma50, sma200) -> str:
    """uptrend / downtrend / ranging, from the moving-average structure."""
    if not all(isinstance(x, (int, float)) and x == x
               for x in (price, sma50, sma200) if x is not None):
        return "ranging"
    if price is None or sma200 is None:
        return "ranging"
    if sma50 is not None:
        if price > sma200 and sma50 > sma200:
            return "uptrend"
        if price < sma200 and sma50 < sma200:
            return "downtrend"
    return "ranging"


def rsi_divergence(close: pd.Series, rsi_series: pd.Series,
                   window: int = 60) -> Optional[str]:
    """Bullish or bearish divergence over the recent window.

    Bullish: price makes a LOWER low while RSI makes a HIGHER low — selling
    momentum is weakening beneath a falling price. Bearish is the mirror.

    Compares the two halves of the window rather than hunting for swing pivots,
    because pivot detection needs parameters that would themselves need
    justifying, and the halves version is transparent about what it measured.
    """
    c = close.dropna().astype(float)
    r = rsi_series.dropna().astype(float)
    j = pd.concat([c.rename("p"), r.rename("r")], axis=1, join="inner").dropna()
    if len(j) < window:
        return None
    w = j.iloc[-window:]
    half = window // 2
    a, b = w.iloc[:half], w.iloc[half:]
    lo_a, lo_b = a["p"].idxmin(), b["p"].idxmin()
    hi_a, hi_b = a["p"].idxmax(), b["p"].idxmax()
    if (b["p"].loc[lo_b] < a["p"].loc[lo_a]
            and b["r"].loc[lo_b] > a["r"].loc[lo_a] + 3):
        return "bullish"
    if (b["p"].loc[hi_b] > a["p"].loc[hi_a]
            and b["r"].loc[hi_b] < a["r"].loc[hi_a] - 3):
        return "bearish"
    return None


def rsi_reading(rsi: Optional[float], regime: str = "ranging",
                divergence: Optional[str] = None) -> Dict[str, Any]:
    """A label and a sentence for one RSI value, in context."""
    if rsi is None or rsi != rsi:
        return {"label": None, "note": "RSI unavailable"}

    if regime == "uptrend":
        # "In a strong uptrend RSI typically stays between 40 and 80."
        if rsi >= 80:
            lab, note = "extended", ("very high even for an uptrend — the move "
                                     "is stretched, though a strong trend can "
                                     "hold this for weeks")
        elif rsi >= 70:
            lab, note = ("overbought — normal in an uptrend",
                         "above 70, but a rising trend spends much of its life "
                         "here; not a sell signal on its own")
        elif rsi >= 50:
            lab, note = "healthy uptrend", "buyers in control, momentum intact"
        elif rsi >= 40:
            lab, note = ("pullback within an uptrend",
                         "the 40-50 zone is where uptrends usually find "
                         "support — a continuation area rather than weakness")
        else:
            lab, note = ("unusually weak for an uptrend",
                         "RSI rarely goes below 40 in a healthy uptrend; treat "
                         "this as the trend being questioned")
    elif regime == "downtrend":
        # "In a strong downtrend RSI typically stays between 20 and 60."
        if rsi <= 20:
            lab, note = ("deeply oversold",
                         "extended, but a falling trend can stay oversold for "
                         "weeks — this describes the fall, it does not end it")
        elif rsi <= 30:
            lab, note = ("oversold — normal in a downtrend",
                         "below 30, which a downtrend does routinely; wait for "
                         "a cross back ABOVE 30 before reading it as a turn")
        elif rsi <= 50:
            lab, note = "downtrend intact", "sellers still in control"
        elif rsi <= 60:
            lab, note = ("rally into resistance",
                         "the 50-60 zone is where downtrend rallies usually "
                         "stall")
        else:
            lab, note = ("unusually strong for a downtrend",
                         "above 60 in a downtrend is rare and may be the first "
                         "sign the trend is changing")
    else:
        # Ranging: the classic bands, and the classic caution with them.
        if rsi >= 70:
            lab, note = "overbought", ("risk of a pullback; the signal is the "
                                       "cross back BELOW 70, not the 70 itself")
        elif rsi >= 50:
            lab, note = "bullish zone", "upward momentum has the upper hand"
        elif rsi > 30:
            lab, note = "bearish zone", "downward momentum, but moderating"
        else:
            lab, note = "oversold", ("potential rebound; the signal is the "
                                     "cross back ABOVE 30, not the 30 itself")

    out = {"label": lab, "note": note, "regime": regime}
    if divergence == "bullish":
        out["divergence"] = "bullish divergence"
        out["divergence_note"] = ("price made a lower low while RSI made a "
                                  "higher low — selling momentum is weakening")
    elif divergence == "bearish":
        out["divergence"] = "bearish divergence"
        out["divergence_note"] = ("price made a higher high while RSI made a "
                                  "lower high — buying momentum is fading")
    return out
