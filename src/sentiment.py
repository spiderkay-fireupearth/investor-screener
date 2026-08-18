"""Market psychology: what the crowd is feeling, measured rather than asserted.

Every gauge here is contrarian in application and none of them is a timing
signal on its own. The discipline this module tries to hold:

  * A sentiment reading is a STATEMENT ABOUT POSITIONING, not about value. Fear
    at 10 tells you what has already been sold, not what a business is worth.
    So each reading is reported with what it implies for behaviour — where to
    look, where to take profit — and never as an instruction.

  * COMPOSITES HIDE THEIR PARTS. The Fear & Greed replication below publishes
    every sub-score and the count of how many were actually available. A single
    number built from four of seven inputs is not the same number as one built
    from seven, and the page says which it is.

  * NOTHING IS SUBSTITUTED SILENTLY. Where a feed is missing — the put/call
    ratio and the COT report both come from outside our normal providers — the
    gauge reports itself unavailable with a reason. It never falls back to a
    proxy and calls it the real thing.

The Fear & Greed figure here is a REPLICATION of CNN's published methodology
computed on our own data, not CNN's number. It will not match theirs, and the
panel says so.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and x == x


def _clamp(x, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, x)))


def _percentile_of_last(series: pd.Series, window: int = 252) -> Optional[float]:
    """Where the latest value sits inside its own recent history, 0–1."""
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 30:
        return None
    w = s.iloc[-window:]
    return float((w <= w.iloc[-1]).mean())


# ---------------------------------------------------------------------- VIX
def vix_state(vix: pd.Series, vix3m: Optional[pd.Series] = None,
              cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fear, as priced by people actually paying for protection.

    Three readings rather than one, because a level alone is not informative:
    the absolute level, where it sits in its own year, and — where a 3-month
    series is available — the TERM STRUCTURE. Spot above 3-month (backwardation)
    is the tell that separates ordinary nervousness from a genuine panic: it
    means the market is paying more to be protected this month than next, which
    only happens when the fear is about right now.
    """
    c = cfg or {}
    s = pd.Series(vix).dropna().astype(float)
    if s.empty:
        return {"available": False, "reason": "no VIX history"}
    level = float(s.iloc[-1])
    pct = _percentile_of_last(s)
    ma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else None
    calm, panic = c.get("vix_calm", 15.0), c.get("vix_panic", 30.0)
    if level >= panic:
        state, reading = "panic", (
            f"VIX at {level:.0f} — spiking fear. Historically the zone where "
            "selling is indiscriminate and where buying has paid, though never "
            "on the first day of it")
    elif level <= calm:
        state, reading = "complacency", (
            f"VIX at {level:.0f} — complacency. Nothing is being priced for, "
            "which is when protection is cheap and profit-taking is unpunished")
    else:
        state, reading = "ordinary", f"VIX at {level:.0f} — unremarkable"
    out = {"available": True, "level": level, "percentile_1y": pct,
           "ma50": ma50, "vs_ma50": (level / ma50 - 1) if ma50 else None,
           "state": state, "reading": reading}
    if vix3m is not None:
        t = pd.Series(vix3m).dropna().astype(float)
        if not t.empty:
            ratio = level / float(t.iloc[-1])
            out["term_structure"] = ratio
            out["term_reading"] = (
                "spot above 3-month — the curve is BACKWARDATED, which is what "
                "a real panic looks like rather than ordinary nervousness"
                if ratio > 1.0 else
                "spot below 3-month — the normal shape; whatever fear there is, "
                "it is not about this week")
    return out


# ---------------------------------------------------------------------- RSI
def rsi_state(index_rsi: Optional[float],
              universe_rsis: Optional[List[float]] = None,
              cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Momentum extremes, at the index and across the names underneath it.

    The pair matters more than either half. An index at 72 while the median
    stock sits at 51 is not a euphoric market — it is a narrow one, and the
    index number is describing a handful of large constituents rather than the
    crowd.
    """
    c = cfg or {}
    hot, cold = c.get("rsi_hot", 70.0), c.get("rsi_cold", 30.0)
    med = None
    vals = [v for v in (universe_rsis or []) if _n(v)]
    if vals:
        med = float(np.median(vals))
    if not _n(index_rsi) and med is None:
        return {"available": False, "reason": "no RSI available"}
    ref = index_rsi if _n(index_rsi) else med
    if ref >= hot:
        state, reading = "euphoric", (
            f"RSI at {ref:.0f} — above {hot:.0f}. Buying that chases price "
            "rather than value; the zone to take profit in, not to start in")
    elif ref <= cold:
        state, reading = "capitulating", (
            f"RSI at {ref:.0f} — below {cold:.0f}. Selling exhaustion, where "
            "the marginal seller is out of stock to sell")
    else:
        state, reading = "neutral", f"RSI at {ref:.0f} — neither extreme"
    out = {"available": True, "index_rsi": index_rsi, "universe_median_rsi": med,
           "state": state, "reading": reading}
    if _n(index_rsi) and med is not None and abs(index_rsi - med) >= 12:
        out["divergence"] = (
            f"The index reads {index_rsi:.0f} while the median stock reads "
            f"{med:.0f}. That gap is the market's shape, not its mood: the "
            + ("index is being carried by a few large names"
               if index_rsi > med else
               "average stock is doing better than the index suggests"))
    return out


# ------------------------------------------------------------------- breadth
def advance_decline(frames: Dict[str, pd.DataFrame],
                    days: int = 250) -> Dict[str, Any]:
    """A/D line and the McClellan Oscillator, from our own universe.

    NOT the NYSE figures. The McClellan Oscillator is properly computed on
    NYSE-wide advances and declines; this is the same arithmetic over the few
    hundred index constituents we hold prices for, and it is labelled as such
    everywhere it appears. The ratio adjustment — net advances as a share of
    total issues traded — is what makes the number comparable across a changing
    universe size, and it is applied here for exactly that reason.
    """
    closes = []
    for t, df in (frames or {}).items():
        if df is None or len(df) < days + 5:
            continue
        closes.append(df["Close"].astype(float).iloc[-(days + 1):])
    if len(closes) < 20:
        return {"available": False,
                "reason": f"only {len(closes)} names with enough price history"}
    px = pd.concat(closes, axis=1).dropna(how="all")
    chg = px.diff().iloc[1:]
    adv = (chg > 0).sum(axis=1)
    dec = (chg < 0).sum(axis=1)
    total = adv + dec
    net = (adv - dec)
    rana = (net / total.replace(0, np.nan)) * 1000.0     # ratio-adjusted
    rana = rana.dropna()
    if len(rana) < 45:
        return {"available": False, "reason": "not enough trading days"}
    ad_line = net.cumsum()
    ema19 = rana.ewm(span=19, adjust=False).mean()
    ema39 = rana.ewm(span=39, adjust=False).mean()
    osc = float((ema19 - ema39).iloc[-1])
    summation = float((ema19 - ema39).cumsum().iloc[-1])
    if osc > 60:
        osc_state = "overbought — the thrust is strong but stretched"
    elif osc < -60:
        osc_state = "oversold — selling pressure at an extreme that rarely lasts"
    elif osc > 0:
        osc_state = "positive — more names rising than falling on balance"
    else:
        osc_state = "negative — distribution under the surface"
    # Is the A/D line confirming the index, or quietly disagreeing with it?
    equal_weight = px.pct_change().mean(axis=1).add(1).cumprod()
    div = None
    if len(equal_weight) > 60:
        ad_20 = float(ad_line.iloc[-1] - ad_line.iloc[-21])
        ew_20 = float(equal_weight.iloc[-1] / equal_weight.iloc[-21] - 1)
        if ad_20 < 0 < ew_20:
            div = ("the average stock is up over the month while more names "
                   "fell than rose — a rally being carried by fewer and fewer "
                   "shoulders")
        elif ad_20 > 0 > ew_20:
            div = ("more names rose than fell while the average stock fell — "
                   "the damage is concentrated in the large ones")
    return {"available": True, "names": px.shape[1], "days": len(rana),
            "advancers": int(adv.iloc[-1]), "decliners": int(dec.iloc[-1]),
            "ad_line": float(ad_line.iloc[-1]),
            "ad_change_20d": float(ad_line.iloc[-1] - ad_line.iloc[-21])
            if len(ad_line) > 21 else None,
            "mcclellan_oscillator": osc,
            "mcclellan_summation": summation,
            "oscillator_state": osc_state,
            "divergence": div,
            "caveat": ("Computed across the index constituents this app holds "
                       "prices for, not NYSE-wide. Direction and extremes are "
                       "meaningful; the absolute level is not comparable with "
                       "a published McClellan figure. Note also what the "
                       "oscillator is: the difference between a fast and a slow "
                       "average of net advances, so it measures the RATE OF "
                       "CHANGE of breadth, not its level. A market where the "
                       "same wide majority advances every day reads near zero "
                       "— that is the indicator working, not failing.")}


def participation(frames: Dict[str, pd.DataFrame],
                  index_close: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Is the rally the market, or six companies?

    The cheapest honest test: compare what the index did with what the MEDIAN
    constituent did over the same window. When the index is up and the median
    stock is not, the index is describing its largest members, and every
    breadth statistic derived from the index alone is describing them too.
    """
    rets, highs, lows = [], 0, 0
    for _t, df in (frames or {}).items():
        if df is None or len(df) < 252:
            continue
        c = df["Close"].astype(float)
        rets.append(float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) > 127 else np.nan)
        w52 = c.iloc[-252:]
        if float(c.iloc[-1]) >= float(w52.max()) * 0.999:
            highs += 1
        if float(c.iloc[-1]) <= float(w52.min()) * 1.001:
            lows += 1
    rets = [r for r in rets if _n(r)]
    if not rets:
        return {"available": False, "reason": "no constituent returns"}
    median_ret = float(np.median(rets))
    idx_ret = None
    if index_close is not None:
        ic = pd.Series(index_close).dropna().astype(float)
        if len(ic) > 127:
            idx_ret = float(ic.iloc[-1] / ic.iloc[-127] - 1)
    out = {"available": True, "names": len(rets),
           "median_6m_return": median_ret, "index_6m_return": idx_ret,
           "new_highs": highs, "new_lows": lows,
           "high_low_ratio": (highs / (highs + lows)) if (highs + lows) else None,
           "pct_positive_6m": float(np.mean([r > 0 for r in rets]))}
    if idx_ret is not None:
        gap = idx_ret - median_ret
        out["breadth_gap"] = gap
        out["reading"] = (
            f"the index is up {idx_ret:.1%} while the median constituent is "
            f"{'up' if median_ret >= 0 else 'down'} {abs(median_ret):.1%} — a "
            "rally carried by its largest members, which is what deceptive "
            "strength looks like from the inside"
            if gap > 0.05 and idx_ret > 0 else
            f"the index is {'up' if idx_ret >= 0 else 'down'} {abs(idx_ret):.1%} "
            f"and the median constituent {'up' if median_ret >= 0 else 'down'} "
            f"{abs(median_ret):.1%} — participation is broadly consistent with "
            "the headline")
    return out


# ----------------------------------------------------------- put/call ratio
def put_call_state(pcr: Optional[float], history: Optional[List[float]] = None,
                   cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hedging demand, read backwards.

    A high put/call ratio means people are paying for protection, which is a
    contrarian BULLISH reading — the fear is already expressed in positions.
    A low one means nobody thinks they need any.
    """
    c = cfg or {}
    if not _n(pcr):
        return {"available": False,
                "reason": "no put/call feed configured — set "
                          "`sentiment.put_call_url` in config/universe.yml to "
                          "a daily CBOE ratio source, or leave it off and the "
                          "composite runs on the remaining inputs"}
    fear, greed = c.get("pcr_fear", 1.0), c.get("pcr_greed", 0.60)
    if pcr >= fear:
        state, reading = "fearful", (
            f"put/call at {pcr:.2f} — above {fear:.2f}. Heavy hedging, which "
            "reads contrarian bullish: the worry is already paid for")
    elif pcr <= greed:
        state, reading = "greedy", (
            f"put/call at {pcr:.2f} — below {greed:.2f}. Almost nobody is "
            "buying protection, which is when it is cheapest to own")
    else:
        state, reading = "ordinary", f"put/call at {pcr:.2f} — unremarkable"
    out = {"available": True, "ratio": pcr, "state": state, "reading": reading}
    h = [v for v in (history or []) if _n(v)]
    if len(h) >= 20:
        out["percentile_1y"] = float(np.mean([v <= pcr for v in h]))
        out["ma5"] = float(np.mean(h[:5]))
    return out


# ------------------------------------------------------- COT / positioning
def cot_state(rows: Optional[List[Dict[str, Any]]],
              contract: str = "") -> Dict[str, Any]:
    """Where speculators are positioned, against their own three-year range.

    The level of net speculative length says almost nothing — contracts differ
    in size and open interest drifts. What is readable is the EXTREME: net
    length as a share of open interest, placed in its own history. Commercials
    (the hedgers, who handle the physical) sit on the other side, and the point
    at which speculators are most crowded is usually the point at which the
    people who actually use the commodity are most keen to sell it to them.
    """
    if not rows:
        return {"available": False,
                "reason": "no COT rows — the CFTC feed returned nothing for "
                          + (contract or "this contract")}
    hist = []
    for r in rows:
        try:
            oi = float(r.get("open_interest_all") or 0)
            nc_l = float(r.get("noncomm_positions_long_all") or 0)
            nc_s = float(r.get("noncomm_positions_short_all") or 0)
            c_l = float(r.get("comm_positions_long_all") or 0)
            c_s = float(r.get("comm_positions_short_all") or 0)
        except (TypeError, ValueError):
            continue
        if oi <= 0:
            continue
        hist.append({"date": r.get("report_date_as_yyyy_mm_dd"),
                     "spec_net_pct_oi": (nc_l - nc_s) / oi,
                     "comm_net_pct_oi": (c_l - c_s) / oi,
                     "open_interest": oi})
    if not hist:
        return {"available": False,
                "reason": "COT rows carried no usable position fields"}
    hist.sort(key=lambda x: str(x["date"]), reverse=True)
    latest = hist[0]
    vals = [h["spec_net_pct_oi"] for h in hist]
    pct = float(np.mean([v <= latest["spec_net_pct_oi"] for v in vals]))
    if pct >= 0.90:
        state, reading = "crowded long", (
            "speculative net length is in the top decile of its own history — "
            "the crowd is already positioned for the move it expects, which is "
            "when there is nobody left to buy")
    elif pct <= 0.10:
        state, reading = "crowded short", (
            "speculative net length is in the bottom decile of its own history "
            "— the crowd is positioned for a fall, and a squeeze needs no good "
            "news to start")
    else:
        state, reading = "unremarkable", (
            "speculative positioning sits inside its normal range")
    return {"available": True, "contract": contract,
            "report_date": latest["date"],
            "spec_net_pct_oi": latest["spec_net_pct_oi"],
            "comm_net_pct_oi": latest["comm_net_pct_oi"],
            "percentile": pct, "weeks_of_history": len(hist),
            "state": state, "reading": reading}


# ----------------------------------------------------------- fear and greed
SUBSCORE_LABELS = {
    "momentum": "Market momentum (index vs its 125-day average)",
    "strength": "Stock price strength (new highs vs new lows)",
    "breadth": "Stock price breadth (advancing vs declining volume)",
    "put_call": "Put/call ratio",
    "junk_demand": "Junk bond demand (high-yield spread)",
    "volatility": "Market volatility (VIX vs its 50-day average)",
    "safe_haven": "Safe-haven demand (stocks vs bonds, 20 days)",
}


def _score_from_band(value, fear_at, greed_at) -> Optional[float]:
    """Map a value onto 0–100 where 0 is extreme fear and 100 extreme greed."""
    if not _n(value):
        return None
    if fear_at == greed_at:
        return None
    return _clamp((value - fear_at) / (greed_at - fear_at) * 100.0)


def fear_greed(index_close: Optional[pd.Series] = None,
               high_low_ratio: Optional[float] = None,
               mcclellan: Optional[float] = None,
               pcr: Optional[float] = None,
               hy_spread: Optional[float] = None,
               hy_history: Optional[List[float]] = None,
               vix_level: Optional[float] = None,
               vix_ma50: Optional[float] = None,
               stock_20d: Optional[float] = None,
               bond_20d: Optional[float] = None) -> Dict[str, Any]:
    """CNN's seven factors, computed on our own data.

    This is a REPLICATION of a published methodology, not CNN's index. It will
    not match their number: their inputs, windows and normalisations are not
    all public. What it does give you — which their number does not — is every
    sub-score, so a composite of 68 can be read as "four calm inputs and one
    screaming one" rather than as a single opaque figure.

    Sub-scores that could not be computed are EXCLUDED, not defaulted to 50.
    Defaulting to neutral is the quiet way a composite drifts toward 50 as its
    feeds fail, which would make a broken gauge look like a calm market.
    """
    subs: Dict[str, Optional[float]] = {}

    if index_close is not None:
        s = pd.Series(index_close).dropna().astype(float)
        if len(s) > 125:
            ma = float(s.rolling(125).mean().iloc[-1])
            gap = float(s.iloc[-1]) / ma - 1 if ma else None
            subs["momentum"] = _score_from_band(gap, -0.08, 0.08)

    subs["strength"] = _score_from_band(high_low_ratio, 0.15, 0.85)
    subs["breadth"] = _score_from_band(mcclellan, -80.0, 80.0)
    # Inverted: a HIGH put/call ratio is fear.
    subs["put_call"] = _score_from_band(pcr, 1.20, 0.55)

    if _n(hy_spread):
        h = [v for v in (hy_history or []) if _n(v)]
        if len(h) >= 30:
            # Tight spreads relative to their own year = greed.
            pct = float(np.mean([v <= hy_spread for v in h]))
            subs["junk_demand"] = _clamp((1 - pct) * 100.0)
        else:
            subs["junk_demand"] = _score_from_band(hy_spread, 8.0, 3.0)

    if _n(vix_level) and _n(vix_ma50) and vix_ma50:
        subs["volatility"] = _score_from_band(vix_level / vix_ma50, 1.35, 0.75)
    elif _n(vix_level):
        subs["volatility"] = _score_from_band(vix_level, 35.0, 12.0)

    if _n(stock_20d) and _n(bond_20d):
        subs["safe_haven"] = _score_from_band(stock_20d - bond_20d, -0.06, 0.06)

    have = {k: v for k, v in subs.items() if _n(v)}
    missing = [SUBSCORE_LABELS[k] for k in SUBSCORE_LABELS if k not in have]
    if not have:
        return {"available": False, "reason": "none of the seven inputs "
                                              "could be computed"}
    score = float(np.mean(list(have.values())))
    if score <= 25:
        label = "extreme fear"
    elif score <= 45:
        label = "fear"
    elif score < 55:
        label = "neutral"
    elif score < 75:
        label = "greed"
    else:
        label = "extreme greed"
    return {
        "available": True, "score": score, "label": label,
        "subscores": {k: round(v, 1) for k, v in have.items()},
        "subscore_labels": SUBSCORE_LABELS,
        "inputs_used": len(have), "inputs_total": len(SUBSCORE_LABELS),
        "missing": missing,
        "caveat": (
            f"Computed on {len(have)} of {len(SUBSCORE_LABELS)} factors from "
            "this app's own data, following CNN's published methodology. It is "
            "a replication, not CNN's figure, and will not match it. Missing "
            "factors are excluded rather than scored 50, so a composite built "
            "from four inputs is labelled as such instead of quietly drifting "
            "toward neutral as feeds fail."),
    }


# ------------------------------------------------------------------ assembly
def build(vix: Dict[str, Any], rsi: Dict[str, Any], pcr: Dict[str, Any],
          fg: Dict[str, Any], cot: List[Dict[str, Any]],
          ad: Dict[str, Any], part: Dict[str, Any],
          cycle_mode: Optional[str] = None) -> Dict[str, Any]:
    """One panel, plus a note where the gauges disagree with the cycle read."""
    gauges = {"vix": vix, "rsi": rsi, "put_call": pcr, "fear_greed": fg,
              "cot": cot or [], "advance_decline": ad, "participation": part}
    available = sum(1 for g in (vix, rsi, pcr, fg, ad, part)
                    if g.get("available"))

    # A crowd verdict, but only from the gauges that actually reported.
    votes = []
    if fg.get("available"):
        votes.append(1 if fg["score"] >= 55 else (-1 if fg["score"] <= 45 else 0))
    if vix.get("available"):
        votes.append({"panic": -1, "complacency": 1}.get(vix["state"], 0))
    if rsi.get("available"):
        votes.append({"euphoric": 1, "capitulating": -1}.get(rsi["state"], 0))
    if pcr.get("available"):
        votes.append({"greedy": 1, "fearful": -1}.get(pcr["state"], 0))
    net = sum(votes)
    if net >= 2:
        crowd, action = "greedy", (
            "The contrarian reading is to take profit and raise the bar for new "
            "purchases, not to sell everything — greed can persist far longer "
            "than it can be justified")
    elif net <= -2:
        crowd, action = "fearful", (
            "The contrarian reading is that this is where buying has paid, with "
            "the caveat that fear also persists, and that a cheap price and a "
            "falling one are the same thing until they are not")
    else:
        crowd, action = "mixed", (
            "No consensus across the gauges — which is itself the common state, "
            "and the one in which sentiment tells you least")

    note = None
    if cycle_mode and crowd != "mixed":
        agree = ((crowd == "greedy" and cycle_mode == "defensive")
                 or (crowd == "fearful" and cycle_mode == "opportunistic"))
        note = ("This agrees with the Marks cycle gauge, which is reading "
                f"{cycle_mode}." if agree else
                f"This DISAGREES with the Marks cycle gauge, which is reading "
                f"{cycle_mode}. Neither is authoritative; the disagreement is "
                "the information — one of them is measuring something the other "
                "is not.")
    return {"gauges": gauges, "crowd": crowd, "vote_net": net,
            "gauges_available": available, "action": action,
            "cycle_note": note,
            "caveat": ("Sentiment describes POSITIONING, not value. It tells "
                       "you what has already been bought or sold, never what "
                       "anything is worth, and it is at its least reliable "
                       "exactly when it is most exciting to read.")}
