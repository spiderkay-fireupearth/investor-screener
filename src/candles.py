"""Candlestick patterns, read the way the book actually specifies them.

From *Simple Candlestick Patterns*. Three things in that book get dropped by
almost every screener that implements it, and all three are load-bearing:

  1. TREND CONTEXT IS PART OF THE PATTERN. "Hammer is a single candlestick
     pattern that is formed AT THE END OF A DOWNTREND". The same shape in the
     middle of an uptrend is not a hammer signal, it is a Tuesday. Every
     pattern here declares the prior trend it requires and is not reported
     without it.

  2. CONFIRMATION IS A SEPARATE BAR. "Traders can enter a long position IF NEXT
     DAY a bullish candle is formed." A pattern printed on the most recent bar
     has not been confirmed yet and cannot have been — the confirming bar does
     not exist. So every signal carries a state: confirmed, unconfirmed, or
     failed. Reporting the three as one thing is how a pattern scanner turns
     into a random-number generator.

  3. THE STOP IS PART OF THE SIGNAL. "...and can place a stop-loss at the low
     of Hammer." A pattern without its invalidation level is a suggestion with
     no way to be wrong, so the level travels with every signal.

WHAT THIS CANNOT SEE. Prices here are split- and dividend-ADJUSTED, which is
correct for every other screen in this app and quietly wrong for gap patterns:
adjustment redistributes the gap a dividend creates. Window, Tasuki and
On-Neck patterns all key off gaps, so they are detected but flagged, and the
panel says so rather than letting an artefact of adjustment read as a signal.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Shape thresholds. The book gives these in words — "more than twice the real
# body", "small real body" — and these are those words as numbers, exposed in
# config so a market with different volatility can be tuned rather than argued
# with.
DEFAULTS = {
    "long_body_atr": 0.6,        # a "long" body, in units of 14-day ATR
    "small_body_atr": 0.3,       # a "small" real body
    "shadow_to_body": 2.0,       # "more than twice the real body"
    # "no or little upper shadow" — measured against the candle's RANGE, not
    # its body. Against the body it is unusable: these patterns all require a
    # SMALL body, so dividing by it makes the ratio explode and a shadow worth
    # 8% of the day's range reads as 40% of "the body". The range is the stable
    # denominator and the one the eye actually uses.
    "tiny_shadow_range": 0.15,
    "doji_body_frac": 0.05,      # body under 5% of the day's range
    "trend_lookback": 10,        # bars used to establish the prior trend
    "trend_move": 0.04,          # how far the prior move must have gone
    "scan_bars": 60,             # how far back to report signals
    "near_close_frac": 0.15,     # "closes near the prior close", in body units
}

# Patterns that key off a GAP, and are therefore sensitive to the price
# adjustment every other screen in this app depends on.
GAP_SENSITIVE = {"Piercing", "On-Neck", "Rising Window", "Falling Window",
                 "Downside Tasuki Gap", "Upside Tasuki Gap"}


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and x == x


class Bar:
    """One candle, with the vocabulary the book uses."""

    __slots__ = ("o", "h", "l", "c", "date")

    def __init__(self, o, h, l, c, date=""):
        self.o, self.h, self.l, self.c, self.date = (
            float(o), float(h), float(l), float(c), date)

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def range(self) -> float:
        return max(self.h - self.l, 1e-12)

    @property
    def bullish(self) -> bool:
        return self.c > self.o

    @property
    def bearish(self) -> bool:
        return self.c < self.o

    @property
    def upper(self) -> float:
        return self.h - max(self.o, self.c)

    @property
    def lower(self) -> float:
        return min(self.o, self.c) - self.l

    @property
    def top(self) -> float:
        return max(self.o, self.c)

    @property
    def bottom(self) -> float:
        return min(self.o, self.c)


def _bars(df: pd.DataFrame) -> List[Bar]:
    need = ("Open", "High", "Low", "Close")
    if df is None or any(c not in df for c in need):
        return []
    sub = df[list(need)].dropna()
    out = []
    for idx, r in sub.iterrows():
        out.append(Bar(r["Open"], r["High"], r["Low"], r["Close"],
                       str(idx)[:10]))
    return out


def _atr(bars: List[Bar], i: int, period: int = 14) -> float:
    """Average true range up to bar i — the yardstick for 'long' and 'small'."""
    lo = max(1, i - period + 1)
    trs = []
    for k in range(lo, i + 1):
        prev = bars[k - 1].c
        trs.append(max(bars[k].h - bars[k].l, abs(bars[k].h - prev),
                       abs(bars[k].l - prev)))
    return float(np.mean(trs)) if trs else bars[i].range


def _trend(bars: List[Bar], i: int, cfg: Dict[str, Any]) -> str:
    """The prior trend, measured on the bars BEFORE the pattern starts.

    Measured before rather than including, because a two-bar bullish pattern
    would otherwise help create the very downtrend it is supposed to be
    reversing.
    """
    n = int(cfg["trend_lookback"])
    if i - n < 0:
        return "unknown"
    start, end = bars[i - n].c, bars[i].c
    move = (end - start) / start if start else 0.0
    if move <= -cfg["trend_move"]:
        return "down"
    if move >= cfg["trend_move"]:
        return "up"
    return "flat"


# ---------------------------------------------------------------- the patterns
# Each detector receives the bar list and an index i pointing at the LAST bar
# of the pattern, plus the ATR at that point and the config. It returns None or
# a dict describing what was found. Trend context is checked by the caller so
# no detector can forget it.

def _hammer(b, i, atr, cfg):
    x = b[i]
    if x.body <= 0:
        return None
    if x.lower < cfg["shadow_to_body"] * x.body:
        return None
    if x.upper > cfg["tiny_shadow_range"] * x.range:
        return None
    if x.body > cfg["small_body_atr"] * atr:
        return None
    return {"name": "Hammer", "direction": "bullish", "needs": "down",
            "bars": 1, "stop": x.l,
            "rule": "small real body at the top, lower shadow more than twice "
                    "the body, little or no upper shadow, at the end of a "
                    "downtrend",
            "why": "sellers pushed the price down, buyers took it back before "
                   "the close — the downtrend may be ending"}


def _inverted_hammer(b, i, atr, cfg):
    x = b[i]
    if x.body <= 0 or x.body > cfg["small_body_atr"] * atr:
        return None
    if x.upper < cfg["shadow_to_body"] * x.body:
        return None
    if x.lower > cfg["tiny_shadow_range"] * x.range:
        return None
    return {"name": "Inverted Hammer", "direction": "bullish", "needs": "down",
            "bars": 1, "stop": x.l,
            "rule": "small real body at the bottom with an upper shadow more "
                    "than twice the body, at the end of a downtrend",
            "why": "buyers tried to lift it and lost the level, but the attempt "
                   "itself is the first sign of demand"}


def _hanging_man(b, i, atr, cfg):
    x = b[i]
    if x.body <= 0 or x.body > cfg["small_body_atr"] * atr:
        return None
    if x.lower < cfg["shadow_to_body"] * x.body:
        return None
    if x.upper > cfg["tiny_shadow_range"] * x.range:
        return None
    return {"name": "Hanging Man", "direction": "bearish", "needs": "up",
            "bars": 1, "stop": x.h,
            "rule": "the hammer's shape, but formed after an UPTREND — same "
                    "candle, opposite meaning",
            "why": "a sharp intraday sell-off inside an uptrend: sellers are "
                   "present where previously there were none"}


def _shooting_star(b, i, atr, cfg):
    x = b[i]
    if x.body <= 0 or x.body > cfg["small_body_atr"] * atr:
        return None
    if x.upper < cfg["shadow_to_body"] * x.body:
        return None
    if x.lower > cfg["tiny_shadow_range"] * x.range:
        return None
    return {"name": "Shooting Star", "direction": "bearish", "needs": "up",
            "bars": 1, "stop": x.h,
            "rule": "small body at the bottom, long upper shadow, after an "
                    "uptrend",
            "why": "the rally was sold into and gave everything back — supply "
                   "above the market"}


def _bullish_engulfing(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bearish and x.bullish):
        return None
    if not (x.c > a.o and x.o < a.c):
        return None
    return {"name": "Bullish Engulfing", "direction": "bullish", "needs": "down",
            "bars": 2, "stop": min(x.l, a.l),
            "rule": "a bullish candle whose body completely covers the "
                    "previous bearish body, after a downtrend",
            "why": "one session undoes the whole of the last — buyers took "
                   "control of the range"}


def _bearish_engulfing(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bullish and x.bearish):
        return None
    if not (x.o > a.c and x.c < a.o):
        return None
    return {"name": "Bearish Engulfing", "direction": "bearish", "needs": "up",
            "bars": 2, "stop": max(x.h, a.h),
            "rule": "a bearish candle whose body completely covers the previous "
                    "bullish body, after an uptrend",
            "why": "the sellers erased a full session of gains in one day"}


def _piercing(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bearish and x.bullish):
        return None
    if x.o >= a.l:                       # the book requires a gap down
        return None
    mid = (a.o + a.c) / 2.0
    if not (x.c > mid and x.c < a.o):
        return None
    return {"name": "Piercing", "direction": "bullish", "needs": "down",
            "bars": 2, "stop": min(x.l, a.l),
            "rule": "downtrend, a bearish candle, a gap DOWN, then a bullish "
                    "candle closing above the midpoint of that bearish body",
            "why": "it opened worse and still closed better than half of "
                   "yesterday's loss — the sellers could not hold the gap"}


def _dark_cloud(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bullish and x.bearish):
        return None
    if x.o <= a.h:                       # gap up
        return None
    mid = (a.o + a.c) / 2.0
    if not (x.c < mid and x.c > a.o):
        return None
    return {"name": "Dark Cloud Cover", "direction": "bearish", "needs": "up",
            "bars": 2, "stop": max(x.h, a.h),
            "rule": "uptrend, a bullish candle, a gap UP, then a bearish candle "
                    "closing more than 50% down the previous body",
            "why": "the gap up was sold all day and more than half of the prior "
                   "session was given back"}


def _three_outside_up(b, i, atr, cfg):
    if i < 2:
        return None
    a, m, x = b[i - 2], b[i - 1], b[i]
    if not (a.bearish and m.bullish and x.bullish):
        return None
    if not (m.c > a.o and m.o < a.c):    # the middle bar engulfs the first
        return None
    if x.c <= m.c:
        return None
    return {"name": "Three Outside Up", "direction": "bullish", "needs": "down",
            "bars": 3, "stop": min(a.l, m.l),
            "rule": "a short bearish candle, a large bullish candle covering "
                    "it, then a third bullish candle closing higher still",
            "why": "an engulfing pattern that has already been confirmed by a "
                   "third session — the confirmation is inside the pattern"}


def _three_outside_down(b, i, atr, cfg):
    if i < 2:
        return None
    a, m, x = b[i - 2], b[i - 1], b[i]
    if not (a.bullish and m.bearish and x.bearish):
        return None
    if not (m.o > a.c and m.c < a.o):
        return None
    if x.c >= m.c:
        return None
    return {"name": "Three Outside Down", "direction": "bearish", "needs": "up",
            "bars": 3, "stop": max(a.h, m.h),
            "rule": "a short bullish candle, a large bearish candle covering "
                    "it, then a third bearish candle closing lower still",
            "why": "a bearish engulfing already confirmed by its third session"}


def _three_inside_up(b, i, atr, cfg):
    if i < 2:
        return None
    a, m, x = b[i - 2], b[i - 1], b[i]
    if not (a.bearish and m.bullish and x.bullish):
        return None
    # Harami: the middle body sits INSIDE the first, which is the opposite of
    # the engulfing shape and the reason these two are different patterns.
    if not (m.o > a.c and m.c < a.o):
        return None
    if a.body <= m.body:
        return None
    if x.c <= a.o:
        return None
    return {"name": "Three Inside Up", "direction": "bullish", "needs": "down",
            "bars": 3, "stop": min(a.l, m.l),
            "rule": "a long bearish candle, a small bullish candle contained "
                    "inside it (a harami), then a bullish candle closing above "
                    "the first candle's open",
            "why": "selling pressure stopped, then buyers took out the level "
                   "the sellers opened from"}


def _three_inside_down(b, i, atr, cfg):
    if i < 2:
        return None
    a, m, x = b[i - 2], b[i - 1], b[i]
    if not (a.bullish and m.bearish and x.bearish):
        return None
    if not (m.c > a.o and m.o < a.c):
        return None
    if a.body <= m.body:
        return None
    if x.c >= a.o:
        return None
    return {"name": "Three Inside Down", "direction": "bearish", "needs": "up",
            "bars": 3, "stop": max(a.h, m.h),
            "rule": "a long bullish candle, a small bearish candle inside it, "
                    "then a bearish candle closing below the first candle's open",
            "why": "buying pressure stopped and sellers took out the level the "
                   "buyers opened from"}


def _on_neck(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bearish and x.bullish):
        return None
    if a.body < cfg["long_body_atr"] * atr:
        return None
    if x.o >= a.l:                       # gaps down on the open
        return None
    if abs(x.c - a.c) > cfg["near_close_frac"] * max(a.body, 1e-9):
        return None
    return {"name": "On-Neck", "direction": "bearish", "needs": "down",
            "bars": 2, "stop": x.h,
            "rule": "downtrend, a long bearish candle, then a smaller bullish "
                    "candle that gaps down and closes back AT the previous "
                    "close, forming a horizontal neckline",
            "why": "the bounce died exactly where the last session ended — a "
                   "continuation pattern, not a reversal, however green the "
                   "second candle looks"}


def _bullish_counterattack(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if not (a.bearish and x.bullish):
        return None
    if a.body < cfg["long_body_atr"] * atr or x.body < cfg["long_body_atr"] * atr:
        return None
    if abs(x.c - a.c) > cfg["near_close_frac"] * max(a.body, 1e-9):
        return None
    return {"name": "Bullish Counterattack", "direction": "bullish",
            "needs": "down", "bars": 2, "stop": min(x.l, a.l),
            "rule": "a long bearish candle then a long bullish one closing "
                    "near the first candle's close, after a strong downtrend",
            "why": "the whole of a heavy session was recovered, though the "
                   "close only returned to where it started rather than beyond"}


def _rising_window(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if x.l <= a.h:
        return None
    return {"name": "Rising Window", "direction": "bullish", "needs": "up",
            "bars": 2, "stop": a.h,
            "rule": "a gap: today's LOW is above yesterday's HIGH",
            "why": "a continuation signal — the window is expected to act as "
                   "support, and the pattern fails if it closes"}


def _falling_window(b, i, atr, cfg):
    if i < 1:
        return None
    a, x = b[i - 1], b[i]
    if x.h >= a.l:
        return None
    return {"name": "Falling Window", "direction": "bearish", "needs": "down",
            "bars": 2, "stop": a.l,
            "rule": "a gap: today's HIGH is below yesterday's LOW",
            "why": "a continuation signal — the window is expected to act as "
                   "resistance"}


def _high_wave(b, i, atr, cfg):
    x = b[i]
    if x.body > cfg["small_body_atr"] * atr:
        return None
    if x.upper < cfg["shadow_to_body"] * max(x.body, 1e-9):
        return None
    if x.lower < cfg["shadow_to_body"] * max(x.body, 1e-9):
        return None
    if x.range < cfg["long_body_atr"] * atr:
        return None
    return {"name": "High Wave", "direction": "neutral", "needs": "any",
            "bars": 1, "stop": None,
            "rule": "a small body with long shadows on BOTH sides",
            "why": "indecision, not direction. The book's own reading: the "
                   "market has lost its sense of where it is going, which is a "
                   "reason to wait rather than to act"}


DETECTORS = (
    _hammer, _inverted_hammer, _hanging_man, _shooting_star,
    _bullish_engulfing, _bearish_engulfing, _piercing, _dark_cloud,
    _three_outside_up, _three_outside_down, _three_inside_up,
    _three_inside_down, _on_neck, _bullish_counterattack,
    _rising_window, _falling_window, _high_wave,
)


# ------------------------------------------------------------------- scanning
def detect(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    """Find every pattern in the recent window, with its trend and confirmation."""
    c = {**DEFAULTS, **(cfg or {})}
    bars = _bars(df)
    need = int(c["trend_lookback"]) + 5
    if len(bars) < need:
        return {"available": False,
                "reason": f"needs at least {need} daily bars, have {len(bars)}"}

    scan_from = max(need, len(bars) - int(c["scan_bars"]))
    found: List[Dict[str, Any]] = []
    for i in range(scan_from, len(bars)):
        atr = _atr(bars, i)
        for fn in DETECTORS:
            hit = fn(bars, i, atr, c)
            if not hit:
                continue
            # Trend context, measured on the bars BEFORE the pattern begins.
            trend = _trend(bars, i - hit["bars"], c)
            if hit["needs"] not in ("any", trend):
                continue
            hit["trend_before"] = trend
            hit["date"] = bars[i].date
            hit["index"] = i
            hit["bars_ago"] = len(bars) - 1 - i
            hit["close"] = bars[i].c
            hit["gap_sensitive"] = hit["name"] in GAP_SENSITIVE
            found.append(_confirm(bars, i, hit))
    found.sort(key=lambda h: -h["index"])
    return {"available": True, "signals": found,
            "scanned_bars": len(bars) - scan_from,
            # The trend AS OF THE LAST BAR, which is not the same thing as the
            # trend before the most recent pattern. A hammer forty sessions ago
            # was preceded by a downtrend; reporting that as "the trend" today
            # would describe a market that has since moved on.
            "trend_now": _trend(bars, len(bars) - 1, c),
            "last_date": bars[-1].date, "last_close": bars[-1].c}


def _confirm(bars: List[Bar], i: int, hit: Dict[str, Any]) -> Dict[str, Any]:
    """Confirmed, unconfirmed, or failed — the book's next-day rule, applied.

    A pattern on the most recent bar CANNOT be confirmed: the bar that would
    confirm it has not traded yet. Saying so is the difference between a signal
    and a guess.
    """
    # Direction first: an indecision candle has nothing to confirm whether or
    # not a later bar exists, so asking "was it confirmed?" is the wrong
    # question rather than an unanswered one.
    if hit["direction"] == "neutral":
        hit["state"] = "n/a"
        hit["state_note"] = "an indecision candle has nothing to confirm"
        hit["entry"] = None
        return hit
    if i >= len(bars) - 1:
        hit["state"] = "unconfirmed"
        hit["state_note"] = ("the confirming session has not traded yet — the "
                             "book's rule is to enter only if the NEXT candle "
                             "agrees")
        hit["entry"] = None
        return hit
    nxt = bars[i + 1]
    agrees = nxt.bullish if hit["direction"] == "bullish" else nxt.bearish
    hit["state"] = "confirmed" if agrees else "failed"
    hit["entry"] = nxt.c if agrees else None
    hit["state_note"] = (
        f"the next session closed {'higher' if nxt.bullish else 'lower'}, "
        + ("which confirms it" if agrees else
           "which does NOT confirm it — the book's entry condition was never met"))
    return hit


# ------------------------------------------------------------------- verdict
def _collapse(live: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per pattern name, newest kept, with a count of the repeats.

    A choppy stock can print the same indecision candle six sessions running.
    Listing it six times is not six pieces of evidence — it is one, repeated,
    and it would push a genuine directional signal off a six-row list. The
    repeat count is kept because it is real information: a market that has
    hesitated five days running is saying something a single doji does not.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for s in live:                          # already newest-first
        key = s["name"]
        if key in seen:
            seen[key]["repeats"] = seen[key].get("repeats", 1) + 1
            continue
        row = dict(s)
        row["repeats"] = 1
        seen[key] = row
    return list(seen.values())


def summarise(scan: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    """Trend, and a suggested action, from the signals that are still live.

    Deliberately conservative in three ways, each of which is in the book:

      * only CONFIRMED signals vote, because an unconfirmed one has not met the
        entry condition and a failed one has already been answered;
      * signals decay — a hammer three months ago is history, not a trade;
      * an indecision candle blocks action rather than casting a vote.
    """
    c = {**DEFAULTS, **(cfg or {})}
    if not scan.get("available"):
        return {"available": False, "reason": scan.get("reason")}
    sigs = scan.get("signals") or []
    fresh = int(c.get("fresh_bars", 10))
    live = [s for s in sigs if s["bars_ago"] <= fresh]
    confirmed = [s for s in live if s["state"] == "confirmed"]
    bull = [s for s in confirmed if s["direction"] == "bullish"]
    bear = [s for s in confirmed if s["direction"] == "bearish"]
    indecision = [s for s in live if s["direction"] == "neutral"]

    trend = scan.get("trend_now", "unknown")
    latest = live[0] if live else None

    if indecision and not confirmed:
        action, why = "wait", (
            "the most recent readable candle is an indecision candle and "
            "nothing has been confirmed — the book's advice here is to let the "
            "market choose a direction first")
    elif bull and not bear:
        action, why = "buy signal", (
            f"{len(bull)} confirmed bullish pattern"
            + ("s" if len(bull) > 1 else "")
            + f" in the last {fresh} sessions, the most recent being "
            + f"{bull[0]['name']} on {bull[0]['date']}")
    elif bear and not bull:
        action, why = "sell signal", (
            f"{len(bear)} confirmed bearish pattern"
            + ("s" if len(bear) > 1 else "")
            + f" in the last {fresh} sessions, the most recent being "
            + f"{bear[0]['name']} on {bear[0]['date']}")
    elif bull and bear:
        newest = confirmed[0]
        action = ("buy signal" if newest["direction"] == "bullish"
                  else "sell signal")
        why = (f"both directions have fired in the last {fresh} sessions "
               f"({len(bull)} bullish, {len(bear)} bearish). The most recent is "
               f"{newest['name']} on {newest['date']}, and it is the only "
               "reason this reads either way — a market printing both is a "
               "market without a view")
    else:
        pending = [s for s in live if s["state"] == "unconfirmed"]
        if pending:
            action, why = "watch", (
                f"{pending[0]['name']} printed on {pending[0]['date']} but the "
                "confirming session has not traded yet")
        else:
            action, why = "no signal", (
                f"nothing confirmed in the last {fresh} sessions")

    return {
        "available": True,
        "trend": trend,
        "action": action,
        "why": why,
        "confirmed_bullish": len(bull),
        "confirmed_bearish": len(bear),
        "live": _collapse(live)[:6],
        "latest": latest,
        "stop": (latest or {}).get("stop"),
        "caveat": (
            "Candlestick patterns are a TIMING tool over days and weeks, and "
            "this app is a value screener whose other work is measured in "
            "years. Use them to choose an entry into a name the frameworks "
            "already justify — never as the reason to own it. The book's own "
            "framing is the same: every pattern here says what the last two or "
            "three sessions did, not what the business is worth."),
    }
