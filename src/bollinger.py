"""Bollinger Bands, with the regime gate that makes them mean anything.

A 20-period SMA with a ±2σ envelope is four lines of arithmetic. The reason
most implementations of it are worse than useless is that they skip the three
rules the method actually rests on, and this module is mostly those three:

  1. BAND PENETRATION IS NOT A SIGNAL. Closing outside a band is a statement
     about MOMENTUM, not an instruction to fade it. A screener that emits "sell
     — price touched the upper band" is not implementing Bollinger Bands, it is
     shorting strength. Here a band tag never produces a verdict on its own; it
     is read through the regime it happened in.

  2. THE REGIME DECIDES THE STRATEGY, not the other way round. Mean reversion
     is a RANGE tactic. In a strong trend price rides the band for weeks, and
     fading each tag is a sequence of stop-outs. So the regime is classified
     FIRST, and only then does one of the three strategies apply. The same
     upper-band tag is a sell in a range and a hold in an uptrend — and that
     is not a contradiction, it is the whole method.

  3. SQUEEZE PREDICTS MAGNITUDE, NOT DIRECTION. A multi-month low in BandWidth
     says a large move is coming. It says nothing whatever about which way. A
     squeeze therefore never emits buy or sell by itself — it emits "wait, and
     here is the level that resolves it".

The 200-period SMA sits over all of it as a bias filter: longs only above it,
shorts only below, and an open idea is invalidated when price crosses back.

BANDWIDTH is (upper - lower) / middle. %B is (price - lower) / (upper - lower):
0 at the lower band, 1 at the upper, and outside [0,1] when price closes beyond
the envelope.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULTS = {
    "period": 20,               # the middle band, per the standard settings
    "k": 2.0,                   # standard deviations for the outer bands
    "trend_sma": 200,           # the macro bias filter
    "squeeze_lookback": 126,    # ~6 months of sessions for "multi-month low"
    "squeeze_pct": 0.15,        # bandwidth inside its lowest 15% => squeeze
    "ride_bars": 10,            # window used to detect price riding a band
    "ride_frac": 0.5,           # what share of it must be beyond 0.8/0.2 %B
    "trend_slope": 0.02,        # 20-SMA move over the ride window to call trend
    "reject_frac": 0.35,        # a rejection candle gives back this much range
    "min_bars": 60,             # below this the bands are not worth computing
}


def _sig(x, digits: int = 4):
    """Round to significant figures — the series file carries these per row."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    if v == 0:
        return 0.0
    import math
    return round(v, -int(math.floor(math.log10(abs(v)))) + (digits - 1))


def compute(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """The bands, the derived measures, and enough history to draw them."""
    c = {**DEFAULTS, **(cfg or {})}
    if df is None or "Close" not in df:
        return {"available": False, "reason": "no price series"}
    close = df["Close"].astype(float).dropna()
    n = int(c["period"])
    if len(close) < max(n + 5, int(c["min_bars"])):
        return {"available": False,
                "reason": (f"needs at least {max(n + 5, int(c['min_bars']))} "
                           f"daily closes, have {len(close)}")}

    mid = close.rolling(n).mean()
    # Population standard deviation (ddof=0), which is what Bollinger specifies
    # and what every charting package draws. The sample version (ddof=1, numpy
    # and pandas' own default) gives visibly wider bands on a 20-period window
    # — about 2.6% wider — so the choice is not cosmetic.
    sd = close.rolling(n).std(ddof=0)
    upper, lower = mid + c["k"] * sd, mid - c["k"] * sd

    width = (upper - lower) / mid.replace(0, np.nan)
    rng = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / rng

    sma_t = close.rolling(int(c["trend_sma"])).mean()

    out: Dict[str, Any] = {
        "available": True,
        "period": n, "k": c["k"],
        "close": float(close.iloc[-1]),
        "upper": float(upper.iloc[-1]) if upper.notna().iloc[-1] else None,
        "middle": float(mid.iloc[-1]) if mid.notna().iloc[-1] else None,
        "lower": float(lower.iloc[-1]) if lower.notna().iloc[-1] else None,
        "bandwidth": float(width.iloc[-1]) if width.notna().iloc[-1] else None,
        "percent_b": float(pct_b.iloc[-1]) if pct_b.notna().iloc[-1] else None,
        "sma_trend": (float(sma_t.iloc[-1]) if sma_t.notna().iloc[-1] else None),
        "trend_sma_period": int(c["trend_sma"]),
    }

    # Squeeze: where does today's bandwidth sit inside its own recent range?
    # Expressed as a percentile of the trailing window rather than an absolute
    # number, because a utility and a biotech have permanently different
    # volatility and a fixed threshold would only ever fire for one of them.
    w = width.dropna()
    look = int(c["squeeze_lookback"])
    if len(w) >= 20:
        recent = w.iloc[-look:]
        rank = float((recent <= w.iloc[-1]).sum()) / len(recent)
        out["bandwidth_pctile"] = rank
        out["bandwidth_low"] = float(recent.min())
        out["bandwidth_high"] = float(recent.max())
        out["squeeze"] = bool(rank <= c["squeeze_pct"])
        out["squeeze_window"] = len(recent)
    else:
        out["bandwidth_pctile"] = None
        out["squeeze"] = False
        out["squeeze_window"] = 0

    out["series"] = _series(close, upper, mid, lower, width, sma_t, c)
    out.update(_regime(close, mid, pct_b, sma_t, c))
    return out


def _series(close, upper, mid, lower, width, sma_t, c) -> Dict[str, Any]:
    """The last N bars of every line, for the chart."""
    bars = int(c.get("chart_bars", 60))

    def arr(s):
        tail = s.iloc[-bars:]
        return [(_sig(v) if v == v else None) for v in tail]

    idx = close.index[-bars:]
    return {
        "d0": str(idx[0])[:10] if len(idx) else "",
        "dx": _offsets(idx),
        "c": arr(close), "u": arr(upper), "m": arr(mid), "l": arr(lower),
        "w": [(_sig(v, 3) if v == v else None) for v in width.iloc[-bars:]],
        "t": arr(sma_t),
    }


def _offsets(idx) -> List[int]:
    """Dates as day offsets from the first, as elsewhere in the payload."""
    import datetime as dt
    try:
        base = dt.date.fromisoformat(str(idx[0])[:10])
        return [(dt.date.fromisoformat(str(x)[:10]) - base).days for x in idx]
    except Exception:                                  # noqa: BLE001
        return list(range(len(idx)))


def _regime(close, mid, pct_b, sma_t, c) -> Dict[str, Any]:
    """Which of the three market states we are in. Everything hangs on this.

    Riding a band is tested first and wins outright, including over a squeeze:
    price pinned to one edge of the envelope is directional by definition, and
    a quiet steady trend can post a bandwidth low without coiling at all.
    After that a trend beats a range, because fading a trend is the single most
    expensive mistake this indicator invites.
    """
    k = int(c["ride_bars"])
    pb = pct_b.dropna()
    if len(pb) < k:
        return {"regime": "unknown", "regime_why": "not enough history",
                "riding": None, "bias": None}

    recent = pb.iloc[-k:]
    ride_up = float((recent >= 0.8).sum()) / k
    ride_dn = float((recent <= 0.2).sum()) / k

    m = mid.dropna()
    slope = 0.0
    if len(m) > k and m.iloc[-k - 1]:
        slope = float((m.iloc[-1] - m.iloc[-k - 1]) / abs(m.iloc[-k - 1]))

    # The macro bias filter: longs only above the long SMA, shorts only below.
    price = float(close.iloc[-1])
    bias = None
    if sma_t.notna().iloc[-1]:
        t = float(sma_t.iloc[-1])
        bias = "long only" if price > t else "short only"

    # "Riding a band" needs the mean to be genuinely advancing, not merely
    # non-negative. Accepting any positive slope let a stock that touched the
    # upper band five times while going nowhere read as a strong uptrend.
    riding = None
    if ride_up >= c["ride_frac"] and slope >= c["trend_slope"]:
        riding = "upper"
    elif ride_dn >= c["ride_frac"] and slope <= -c["trend_slope"]:
        riding = "lower"

    if riding == "upper":
        return {"regime": "strong uptrend", "riding": "upper", "bias": bias,
                "regime_why": (
                    f"price has closed in the top fifth of the envelope on "
                    f"{int(ride_up * k)} of the last {k} sessions while the "
                    f"20-day average rose {slope * 100:.1f}% — it is RIDING "
                    "the upper band, which is trend behaviour, not an "
                    "overbought reading")}
    if riding == "lower":
        return {"regime": "strong downtrend", "riding": "lower", "bias": bias,
                "regime_why": (
                    f"price has closed in the bottom fifth of the envelope on "
                    f"{int(ride_dn * k)} of the last {k} sessions while the "
                    f"20-day average fell {abs(slope) * 100:.1f}% — it is "
                    "riding the lower band, and every mean-reversion buy into "
                    "this is a stop-out")}
    if abs(slope) >= c["trend_slope"]:
        d = "up" if slope > 0 else "down"
        return {"regime": f"trending {d}", "riding": None, "bias": bias,
                "regime_why": (
                    f"the 20-day average has moved {slope * 100:+.1f}% over "
                    f"{k} sessions without price pinning either band")}
    return {"regime": "range-bound", "riding": None, "bias": bias,
            "regime_why": (
                f"the 20-day average has moved only {slope * 100:+.1f}% over "
                f"{k} sessions and price is oscillating inside the envelope — "
                "the one regime where mean reversion is the right tactic")}


# ---------------------------------------------------------------------------
# The three strategies. Each declares the regime it applies to, and NONE of
# them is consulted outside it — which is the entire point.
# ---------------------------------------------------------------------------

def evaluate(bb: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """buy / hold / sell / wait, with the reasoning and the invalidation level.

    "hold" is a real answer here rather than a shrug. Riding the upper band in
    an uptrend is the case where the naive reading says sell and the method
    says stay in — so the verdict has to be able to say exactly that.
    """
    c = {**DEFAULTS, **(cfg or {})}
    if not bb.get("available"):
        return {"available": False, "reason": bb.get("reason")}

    pb, price = bb.get("percent_b"), bb.get("close")
    mid, up, lo = bb.get("middle"), bb.get("upper"), bb.get("lower")
    t = bb.get("sma_trend")
    regime, riding = bb.get("regime"), bb.get("riding")
    above_trend = (t is not None and price is not None and price > t)

    notes: List[str] = []
    if bb.get("bandwidth") is not None:
        notes.append(f"BandWidth {bb['bandwidth'] * 100:.1f}% of the middle band")
    if pb is not None:
        notes.append(f"%B {pb:.2f}")

    # ---- 1. Squeeze. Magnitude without direction: never a buy or a sell.
    #
    # Checked BEFORE the strategies but AFTER riding, because a squeeze is a
    # consolidation and riding a band is the precise opposite of consolidating.
    # A steady low-volatility grind up the upper band scores as a bandwidth low
    # and is not coiling at all — it is trending quietly, and calling it a
    # squeeze would have told the reader to stand aside from the cleanest
    # trend on the page.
    if bb.get("squeeze") and not riding:
        pct = (bb.get("bandwidth_pctile") or 0) * 100
        return _verdict(
            "wait", "The Volatility Squeeze", regime, bb,
            why=(f"BandWidth is in the lowest {pct:.0f}% of its last "
                 f"{bb.get('squeeze_window')} sessions — the bands have "
                 "coiled, and a squeeze reliably precedes a large move. It "
                 "says nothing about the DIRECTION of that move, so there is "
                 "no trade here until price resolves it."),
            action_note=(
                f"The break decides it: a close above {_f(up)} or below "
                f"{_f(lo)}. Take the break in the direction the 200-day "
                "average already allows, then trail the stop along the "
                "20-day middle band."),
            stop=mid, notes=notes)

    # ---- 2. Trend riding. The case the naive reading gets backwards.
    if riding == "upper":
        if not above_trend:
            return _verdict(
                "wait", "Trend Riding & Scaling", regime, bb,
                why=("Price is riding the upper band, but it is BELOW the "
                     f"{bb.get('trend_sma_period')}-day average, so the macro "
                     "filter does not permit a long. This is strength inside a "
                     "larger downtrend — the setup and the bias disagree."),
                stop=t, notes=notes)
        return _verdict(
            "hold", "Trend Riding & Scaling", regime, bb,
            why=("Price is riding the upper band in an uptrend and above the "
                 f"{bb.get('trend_sma_period')}-day average. This is the "
                 "reading most often got backwards: tagging the upper band is "
                 "momentum, NOT an overbought sell. Existing positions stay."),
            action_note=(
                f"Add on pullbacks to the 20-day middle band at {_f(mid)} "
                f"rather than chasing here. Exit on a daily close back below "
                f"it; the position is invalid outright below {_f(t)}."),
            stop=mid, notes=notes)

    if riding == "lower":
        return _verdict(
            "sell", "Trend Riding & Scaling", regime, bb,
            why=("Price is riding the lower band in a downtrend. The band tag "
                 "is not a bargain — it is the trend working. Buying each tag "
                 "here is the classic way to be stopped out repeatedly."),
            action_note=(
                f"For longs this is an exit, not an entry. A reversal is not "
                f"credible until a close back above the 20-day middle band at "
                f"{_f(mid)}"
                + (f", and the {bb.get('trend_sma_period')}-day average at "
                   f"{_f(t)} above that." if t else ".")),
            stop=mid, notes=notes)

    # ---- 3. Mean reversion. ONLY in a range.
    if regime == "range-bound" and pb is not None:
        if pb <= 0.05:
            if not above_trend:
                return _verdict(
                    "wait", "Mean Reversion (Range)", regime, bb,
                    why=(f"Price is at the lower band in a range, but below the "
                         f"{bb.get('trend_sma_period')}-day average. The macro "
                         "filter allows no long here, and a range beneath a "
                         "falling long-term average often becomes the next leg "
                         "down rather than a floor."),
                    stop=lo, notes=notes)
            return _verdict(
                "buy", "Mean Reversion (Range)", regime, bb,
                why=("Price has tagged the LOWER band inside a range, and a "
                     "range is the one regime where mean reversion is the "
                     f"right tactic. It is above the "
                     f"{bb.get('trend_sma_period')}-day average, so the macro "
                     "filter permits the long."),
                action_note=(
                    f"The target is the 20-day middle band at {_f(mid)}, and "
                    f"the upper band at {_f(up)} beyond it. The idea is wrong "
                    f"on a close below the lower band at {_f(lo)}. Wait for a "
                    "rejection candle rather than buying the tag itself."),
                stop=lo, notes=notes)
        if pb >= 0.95:
            return _verdict(
                "sell", "Mean Reversion (Range)", regime, bb,
                why=("Price has tagged the UPPER band inside a range. In a "
                     "range — and ONLY in a range — that is the mean-reversion "
                     "sell. If this were a trend the same tag would read as "
                     "strength, which is why the regime is checked first."),
                action_note=(
                    f"The target is the 20-day middle band at {_f(mid)}, and "
                    f"the lower band at {_f(lo)} beyond it. The idea is wrong "
                    f"on a close above the upper band at {_f(up)}."),
                stop=up, notes=notes)
        return _verdict(
            "hold", "Mean Reversion (Range)", regime, bb,
            why=("Range-bound, with price mid-envelope. Mean reversion needs a "
                 "tag of one of the outer bands to work from, and there isn't "
                 "one — the middle of a range is the worst place to act."),
            action_note=(f"Watch the lower band at {_f(lo)} for a long and the "
                         f"upper at {_f(up)} for an exit."),
            stop=mid, notes=notes)

    # ---- Trending, but not pinning a band.
    d = "up" if "up" in (regime or "") else "down"
    if d == "up":
        return _verdict(
            "hold" if above_trend else "wait", "Trend Riding & Scaling",
            regime, bb,
            why=(f"The 20-day average is rising and price sits mid-envelope. "
                 "There is no band signal to act on"
                 + ("." if above_trend else
                    f", and price is below the "
                    f"{bb.get('trend_sma_period')}-day average, so the macro "
                    "filter blocks a long in any case.")),
            action_note=(f"A pullback to the middle band at {_f(mid)} is the "
                         "entry this strategy waits for."),
            stop=mid, notes=notes)
    return _verdict(
        "sell" if not above_trend else "hold", "Trend Riding & Scaling",
        regime, bb,
        why=("The 20-day average is falling"
             + (" and price is below the "
                f"{bb.get('trend_sma_period')}-day average — the macro filter "
                "permits no long." if not above_trend else
                ", though price is still above the "
                f"{bb.get('trend_sma_period')}-day average.")),
        action_note=(f"A close back above the middle band at {_f(mid)} is the "
                     "first thing that would change this."),
        stop=mid, notes=notes)


def _f(v) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _verdict(action, strategy, regime, bb, why, stop, notes,
             action_note: str = "") -> Dict[str, Any]:
    return {
        "available": True,
        "action": action,
        "strategy": strategy,
        "regime": regime,
        "regime_why": bb.get("regime_why"),
        "why": why,
        "action_note": action_note,
        "stop": stop,
        "bias": bb.get("bias"),
        "notes": " · ".join(notes),
        "squeeze": bool(bb.get("squeeze")),
        "riding": bb.get("riding"),
        "caveat": (
            "Bollinger Bands measure where price sits relative to its own "
            "recent volatility. They say nothing about what a business is "
            "worth, and a band tag is never a reason to own or not own a "
            "company — only a way to time an entry into one the frameworks "
            "already justify. Avoid acting on a squeeze immediately before a "
            "scheduled earnings or macro release: the move it predicts is "
            "then a coin toss on the announcement rather than on the chart."),
    }
