"""Market-wide sentiment: TRIN, breadth divergence, and a cross-asset view.

This module computes only what can be computed. That boundary is the whole
design, so it is worth stating before any of the arithmetic:

  * Every number here is derived from price and volume series this app already
    holds. Nothing is recalled, nothing is asserted from memory.
  * Where a reading cannot be produced honestly — too few names, no volume, a
    market that has not run — the result says so and says why, rather than
    returning a plausible number.
  * Nothing here says what to buy. Each indicator carries the standard reading
    of what it MEANS, which is a statement about the indicator, not advice
    about a portfolio.

THE FIDELITY CAVEAT THAT MATTERS MOST
-------------------------------------
The published TRIN ($TRIN, the Arms Index) is computed over every issue on the
NYSE — roughly 3,000 securities. This app holds a few hundred, skewed towards
index constituents and liquid names. The arithmetic below is identical; the
POPULATION is not.

That matters because the conventional thresholds — below 0.8 overbought, above
1.2 oversold — are calibrated to the NYSE-wide series. A narrower, larger-cap
population produces a different distribution: less dispersion between advancing
and declining volume, so readings cluster nearer 1.0 and the classic bands fire
less often.

So this returns BOTH: the raw reading against the conventional bands, clearly
labelled, and the reading's own percentile within this universe's trailing
history. The percentile is the one to trust for "is this unusual"; the band is
there because it is what the reader expects to see and its absence would be
more confusing than its presence with a caveat.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULTS = {
    "trin_ma": 10,              # the smoothing the reader asked for
    "trin_overbought": 0.8,     # conventional NYSE bands, see the caveat above
    "trin_oversold": 1.2,
    "trin_percentile_window": 252,
    "min_names": 30,            # below this, breadth is noise dressed as signal
    "breadth_divergence": 15.0,  # pp gap between 50d and 200d worth naming
}

# The Magnificent Seven. Hard-coded because it is a NAMED, FIXED list — it is
# not a screen, and deriving it from market cap would silently change its
# membership and make period-to-period comparison meaningless.
MAG7 = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA", "TSLA": "Tesla",
}


def _n(x) -> bool:
    try:
        return x is not None and not pd.isna(x) and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pctile(series: pd.Series, value: float) -> Optional[float]:
    s = series.dropna()
    if len(s) < 30 or not _n(value):
        return None
    return round(float((s <= value).sum()) / len(s) * 100.0, 1)


# ---------------------------------------------------------------------------
# TRIN
# ---------------------------------------------------------------------------

def trin(frames: Dict[str, pd.DataFrame],
         cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Arms Index, smoothed over N days.

        TRIN = (advancing issues / declining issues)
             / (advancing volume / declining volume)

    Read it as a RATIO OF RATIOS. Above 1 means the declining side is carrying
    more volume per name than the advancing side — selling with conviction.
    Below 1 means the opposite. It is contrarian by convention: heavy selling
    pressure is read as oversold.

    Smoothed on the RATIO, not on its components. Averaging the four raw counts
    first and dividing once would be a different and wrong statistic — it would
    let one enormous volume day dominate the whole window instead of
    contributing one day's ratio.

    A day with no decliners, or no declining volume, has an undefined ratio.
    Those days are dropped rather than clamped: substituting a large number
    would manufacture an extreme reading out of a quiet tape.
    """
    c = {**DEFAULTS, **(cfg or {})}
    usable = {t: df for t, df in (frames or {}).items()
              if df is not None and {"Close", "Volume"}.issubset(df.columns)
              and len(df) >= c["trin_ma"] + 2}
    if len(usable) < c["min_names"]:
        return {"available": False,
                "reason": (f"only {len(usable)} names have price AND volume "
                           f"history; TRIN needs at least {c['min_names']} to "
                           f"mean anything")}

    close = pd.DataFrame({t: df["Close"] for t, df in usable.items()}).sort_index()
    vol = pd.DataFrame({t: df["Volume"] for t, df in usable.items()}).sort_index()
    vol = vol.reindex(close.index)
    chg = close.diff()

    adv_n = (chg > 0).sum(axis=1).astype(float)
    dec_n = (chg < 0).sum(axis=1).astype(float)
    adv_v = vol.where(chg > 0).sum(axis=1, min_count=1)
    dec_v = vol.where(chg < 0).sum(axis=1, min_count=1)

    ok = (dec_n > 0) & (adv_n > 0) & (dec_v > 0) & (adv_v > 0)
    raw = ((adv_n / dec_n) / (adv_v / dec_v)).where(ok)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    clean = raw.dropna()
    if len(clean) < c["trin_ma"]:
        return {"available": False,
                "reason": (f"only {len(clean)} days had both advancers and "
                           f"decliners with volume — not enough for a "
                           f"{c['trin_ma']}-day average")}

    ma = clean.rolling(c["trin_ma"]).mean()
    last_ma = float(ma.iloc[-1]) if _n(ma.iloc[-1]) else None
    last_raw = float(clean.iloc[-1])
    if last_ma is None:
        return {"available": False, "reason": "the smoothed series is empty"}

    if last_ma < c["trin_overbought"]:
        band, meaning = "overbought", (
            "Advancing issues are carrying disproportionate volume — buying is "
            "broad AND heavy. Read conventionally as an overbought tape, which "
            "is a caution flag rather than a sell.")
    elif last_ma > c["trin_oversold"]:
        band, meaning = "oversold", (
            "Declining issues are carrying disproportionate volume — selling "
            "with conviction. Read conventionally as oversold, which is where "
            "contrarian buyers look, not a guarantee of a bottom.")
    else:
        band, meaning = "neutral", (
            "Volume is distributed between advancers and decliners in roughly "
            "the proportion their numbers imply. No pressure signal.")

    pct = _pctile(ma.dropna().tail(c["trin_percentile_window"]), last_ma)
    return {
        "available": True,
        "value": round(last_ma, 3),
        "raw_today": round(last_raw, 3),
        "ma_days": c["trin_ma"],
        "band": band,
        "meaning": meaning,
        "percentile": pct,
        "names": len(usable),
        "days": int(len(clean)),
        "overbought_at": c["trin_overbought"],
        "oversold_at": c["trin_oversold"],
        "advancers": int(adv_n.iloc[-1]) if _n(adv_n.iloc[-1]) else None,
        "decliners": int(dec_n.iloc[-1]) if _n(dec_n.iloc[-1]) else None,
        "dropped_days": int(len(raw) - len(clean)),
        # Stated on the panel, not buried here. The bands below are NYSE-wide
        # conventions applied to a much narrower population.
        "caveat": (
            f"Computed across {len(usable)} names this app holds, not the "
            f"~3,000 NYSE issues the published $TRIN uses. The arithmetic is "
            f"identical; the population is not. The 0.8/1.2 bands are "
            f"calibrated to the NYSE series, so treat the percentile as the "
            f"better guide to whether this reading is unusual here."),
    }


# ---------------------------------------------------------------------------
# Breadth: the 50-day / 200-day divergence
# ---------------------------------------------------------------------------

def breadth_ma(frames: Dict[str, pd.DataFrame],
               cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Share of names above their 50-day and 200-day averages, and the gap.

    Two questions, not one:

      * % above the 200-day is how many names are in a long-term uptrend. It is
        the structural reading and it moves slowly.
      * % above the 50-day is how many are in a short-term one. It swings.

    The DIVERGENCE is the point. The two normally travel together. When the
    short-term measure runs far above the long-term one, a rally is being
    carried by names that are bouncing without having repaired their trend —
    momentum outrunning structure. When it runs far below, the tape is selling
    off inside markets that are still structurally intact, which is what an
    ordinary correction inside an uptrend looks like.

    Neither is a signal on its own, and this returns the reading rather than a
    verdict.
    """
    c = {**DEFAULTS, **(cfg or {})}
    usable = {t: df for t, df in (frames or {}).items()
              if df is not None and "Close" in df and len(df) >= 200}
    if len(usable) < c["min_names"]:
        return {"available": False,
                "reason": (f"only {len(usable)} names have the 200 trading days "
                           f"a 200-day average needs; at least {c['min_names']} "
                           f"are required for a breadth reading")}

    above50 = above200 = 0
    for df in usable.values():
        close = df["Close"].astype(float).dropna()
        if len(close) < 200:
            continue
        px = float(close.iloc[-1])
        m50 = float(close.rolling(50).mean().iloc[-1])
        m200 = float(close.rolling(200).mean().iloc[-1])
        if _n(m50) and px > m50:
            above50 += 1
        if _n(m200) and px > m200:
            above200 += 1

    n = len(usable)
    p50 = round(above50 / n * 100.0, 1)
    p200 = round(above200 / n * 100.0, 1)
    gap = round(p50 - p200, 1)

    if gap >= c["breadth_divergence"]:
        state = "short-term ahead"
        note = ("Many more names are above their 50-day average than their "
                "200-day. The rally is being carried by names that have "
                "bounced without repairing their longer trend — broad "
                "participation, thinner foundation.")
    elif gap <= -c["breadth_divergence"]:
        state = "short-term behind"
        note = ("Far fewer names are above their 50-day average than their "
                "200-day. The tape is selling off inside markets whose longer "
                "trends are still intact — the shape of a correction rather "
                "than a breakdown.")
    else:
        state = "aligned"
        note = ("Short- and long-term breadth agree. Whatever the level says "
                "about direction, there is no divergence between how many "
                "names are trending and how many are merely bouncing.")

    if p200 >= 60:
        health = "broad"
    elif p200 >= 40:
        health = "mixed"
    else:
        health = "narrow"

    return {"available": True, "names": n,
            "above_50d": p50, "above_200d": p200, "gap": gap,
            "state": state, "note": note, "health": health,
            "n_above_50d": above50, "n_above_200d": above200,
            "divergence_at": c["breadth_divergence"],
            "caveat": (f"Measured across the {n} names this run holds with a "
                       f"full 200 days of history — an index-weighted "
                       f"population, not the whole market.")}


# ---------------------------------------------------------------------------
# The Magnificent Seven
# ---------------------------------------------------------------------------

def mag7(metrics_by_ticker: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The seven, with the concentration question stated rather than implied.

    Reported as a group because that is how they move and how they distort an
    index: seven names driving a cap-weighted benchmark is a different market
    from a broad advance, and the difference does not show in the index level.
    """
    rows, missing = [], []
    for t, name in MAG7.items():
        m = metrics_by_ticker.get(t) or metrics_by_ticker.get(t.upper())
        if not m:
            missing.append(t)
            continue
        rows.append({
            "ticker": t, "name": name,
            "return_3m": m.get("return_3m"),
            "return_6m": m.get("return_6m"),
            "return_12m": m.get("return_12m"),
            "rsi": m.get("rsi_14"),
            "above_200d": m.get("price_above_sma200"),
            "pct_below_high": m.get("pct_below_52w_high"),
            "pe": m.get("pe_ttm"),
            "mcap_usd": m.get("market_cap_usd"),
        })
    if not rows:
        return {"available": False,
                "reason": ("none of the seven are in this run — they are US "
                           "names, so an Asia-only refresh will not have them"),
                "missing": missing}

    r12 = [r["return_12m"] for r in rows if _n(r["return_12m"])]
    above = [r for r in rows if r.get("above_200d")]
    rows.sort(key=lambda r: -(r["mcap_usd"] or 0))
    return {
        "available": True, "rows": rows, "missing": missing,
        "n_above_200d": len(above), "n": len(rows),
        "median_return_12m": (round(float(np.median(r12)), 3) if r12 else None),
        "dispersion_12m": (round(float(max(r12) - min(r12)), 3)
                           if len(r12) > 1 else None),
        "note": ("Seven names large enough to move a cap-weighted index on "
                 "their own. When they diverge from breadth — the seven "
                 "rising while most names do not — the index is telling you "
                 "less about the market than it appears to."),
    }


# ---------------------------------------------------------------------------
# Cross-asset view
# ---------------------------------------------------------------------------

def cross_asset(stocks: Dict[str, Any], breadth: Dict[str, Any],
                trin_state: Dict[str, Any], debt: Dict[str, Any],
                commodities: Dict[str, Any]) -> Dict[str, Any]:
    """One row per asset class, each with the evidence that produced it.

    Deliberately NOT a single composite score. Three asset classes reduced to
    one number hides the thing worth knowing — which of them disagree — and
    disagreement between them is the most informative state a cross-asset table
    can show.
    """
    rows: List[Dict[str, Any]] = []

    # --- stocks ---
    ev, lean = [], 0
    if breadth.get("available"):
        ev.append(f"{breadth['above_200d']:.0f}% of names above their 200-day "
                  f"average ({breadth['health']} participation)")
        lean += 1 if breadth["above_200d"] >= 55 else (
            -1 if breadth["above_200d"] <= 35 else 0)
        if breadth["state"] != "aligned":
            ev.append(f"50d/200d breadth {breadth['state']} by "
                      f"{abs(breadth['gap']):.0f}pp")
    if trin_state.get("available"):
        ev.append(f"TRIN({trin_state['ma_days']}d) {trin_state['value']:.2f} — "
                  f"{trin_state['band']}")
        # TRIN is CONTRARIAN, so its contribution is inverted: heavy selling
        # pressure (oversold) leans constructive, heavy buying (overbought)
        # leans cautious. Leaving it out entirely was worse than either — the
        # row claims to summarise the evidence beside it, and a summary that
        # ignores half its own evidence is not one.
        if trin_state["band"] == "oversold":
            lean += 1
        elif trin_state["band"] == "overbought":
            lean -= 1
    lean = max(-2, min(2, lean))
    rows.append({"asset": "Stocks", "lean": lean, "evidence": ev,
                 "reads": _lean_word(lean)})

    # --- bonds ---
    ev, lean = [], 0
    sig = (debt or {}).get("signals") or {}
    curve = _first_num(sig, ("curve_10y2y", "curve_10y3m"))
    if curve is not None:
        if curve < 0:
            ev.append(f"yield curve inverted ({curve:+.2f}pp) — the bond market "
                      f"is pricing slower growth ahead")
            lean -= 1
        else:
            ev.append(f"yield curve positive ({curve:+.2f}pp)")
            lean += 1
    hy = _first_num(sig, ("hy_oas",))
    if hy is not None:
        if hy > 5.0:
            ev.append(f"high-yield spread {hy:.2f}pp — credit is pricing stress")
            lean -= 1
        elif hy < 3.5:
            ev.append(f"high-yield spread {hy:.2f}pp — credit is complacent")
            lean += 1
        else:
            ev.append(f"high-yield spread {hy:.2f}pp — unremarkable")
    rows.append({"asset": "Bonds & credit", "lean": lean, "evidence": ev,
                 "reads": _lean_word(lean)})

    # --- precious metals ---
    ev, lean = [], 0
    metals = [r for r in ((commodities or {}).get("rows") or [])
              if str(r.get("name", "")) in ("Gold", "Silver", "Platinum")]
    for m in metals:
        rc = m.get("return_3m")
        if _n(rc):
            ev.append(f"{m['name']} {rc * 100:+.1f}% over three months")
            lean += 1 if rc > 0.05 else (-1 if rc < -0.05 else 0)
    if not metals:
        ev.append("no metals priced this run")
    rows.append({"asset": "Precious metals", "lean": max(-1, min(1, lean)),
                 "evidence": ev, "reads": _lean_word(max(-1, min(1, lean)))})

    # The interplay, stated as mechanism rather than prediction. These are the
    # standard transmission channels, not a forecast about this week.
    links = [
        ("Bond yields rise", "Stocks", "A higher discount rate compresses the "
         "present value of distant earnings, so long-duration growth names fall "
         "hardest — which is most of the Magnificent Seven."),
        ("Bond yields rise", "Precious metals", "Gold pays no coupon, so a "
         "higher real yield raises the cost of holding it. Usually a headwind, "
         "unless the yield is rising because of inflation rather than growth."),
        ("Credit spreads widen", "Stocks", "Credit usually moves first. Widening "
         "spreads while equities hold up is one of the more reliable "
         "disagreements to pay attention to."),
        ("Breadth narrows", "Stocks", "A cap-weighted index can keep rising on "
         "a handful of names while most fall. The index level stops describing "
         "the market."),
        ("Metals rally with equities", "All three", "Not the classic safe-haven "
         "trade. It usually means a liquidity or currency story rather than a "
         "fear one — worth separating industrial demand from hedging demand."),
    ]
    disagree = len({r["lean"] for r in rows if r["evidence"]}) > 1
    return {"available": bool(rows), "rows": rows, "links": links,
            "disagree": disagree,
            "note": ("The classes disagree, which is information: one of them "
                     "is early and it is usually credit."
                     if disagree else
                     "All three lean the same way, so none of them is "
                     "contradicting the others right now.")}


def _lean_word(lean: int) -> str:
    return "constructive" if lean > 0 else ("cautious" if lean < 0 else "neutral")


def _first_num(d: Dict[str, Any], keys) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            v = v.get("value")
        if _n(v):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Deeply oversold names, with a sector-shock flag
# ---------------------------------------------------------------------------

def oversold(results: Dict[str, Any], metrics_by_ticker: Dict[str, Dict[str, Any]],
             threshold: float = 0.50,
             sector_min: int = 3, sector_share: float = 0.30) -> Dict[str, Any]:
    """Names down more than `threshold` from their 52-week high.

    Two labels are attached, and neither claims to know why a stock fell.

    ACCOUNTS INTACT reuses `dislocation.fundamentals_intact`: do the last
    published accounts explain the fall? If earnings, margins and the balance
    sheet still look like they did, the market has repriced something the
    statements do not show. That is the closest a screen can get to "the drop
    was not about the business", and it is a question about the ACCOUNTS, not a
    diagnosis of the cause.

    SECTOR SHOCK asks whether the name fell alone or in company. A systemic
    shock hits a sector; a broken business breaks by itself. When several names
    in one sector are all down more than the threshold, and they are a
    meaningful share of that sector's coverage, the fall is flagged as sector-
    wide. That is a statistical observation about co-movement — it does not
    identify the shock, and it cannot: naming the event requires reading the
    news, which this module does not do.
    """
    try:
        from . import dislocation as dis
    except Exception:                                    # noqa: BLE001
        dis = None

    hits: List[Dict[str, Any]] = []
    sector_total: Dict[str, int] = {}
    for t, r in (results or {}).items():
        sec = (r.get("sector") or "").strip() or "Unclassified"
        sector_total[sec] = sector_total.get(sec, 0) + 1
        m = metrics_by_ticker.get(t) or (r.get("metrics") or {})
        off = m.get("pct_below_52w_high")
        if not _n(off) or float(off) < threshold:
            continue
        intact = None
        why = ""
        if dis is not None:
            try:
                fi = dis.fundamentals_intact(m)
                # `intact` is False both when the accounts DO explain the fall
                # and when there was too little data to tell. Those are
                # different answers and the page must not conflate them, so
                # carry the evaluable count and name the failing tests.
                if fi.get("evaluable"):
                    intact = bool(fi.get("intact"))
                    failed = [t["label"] for t in (fi.get("tests") or [])
                              if t.get("ok") is False]
                    why = (f"{fi['passed']} of {fi['evaluable']} accounts tests "
                           f"passed"
                           + (f"; failing: {', '.join(failed[:3])}"
                              if failed else ""))
                else:
                    why = ("no accounts test could be evaluated — the "
                           "fundamentals for this name are missing, so the "
                           "fall is neither explained nor unexplained")
            except Exception:                            # noqa: BLE001
                intact = None
        hits.append({
            "ticker": t, "name": r.get("name") or t,
            "market": r.get("market"), "sector": sec,
            "off_high": round(float(off), 4),
            "return_12m": m.get("return_12m"),
            "rsi": m.get("rsi_14"),
            "pe": m.get("pe_ttm"),
            "accounts_intact": intact,
            "accounts_note": why,
        })

    # Sector co-movement, computed over the hits rather than asserted.
    by_sector: Dict[str, int] = {}
    for h in hits:
        by_sector[h["sector"]] = by_sector.get(h["sector"], 0) + 1
    shocked = {
        s for s, n in by_sector.items()
        if n >= sector_min and sector_total.get(s, 0)
        and n / sector_total[s] >= sector_share
    }
    for h in hits:
        h["sector_shock"] = h["sector"] in shocked
        h["sector_peers"] = by_sector.get(h["sector"], 0)
        h["sector_covered"] = sector_total.get(h["sector"], 0)

    hits.sort(key=lambda h: -h["off_high"])
    return {
        "available": True,
        "threshold": threshold,
        "rows": hits,
        "n": len(hits),
        "sectors_shocked": sorted(shocked),
        "sector_rule": (f"{sector_min} or more names in a sector, and at least "
                        f"{sector_share:.0%} of that sector's coverage, all "
                        f"down more than {threshold:.0%}"),
        "note": ("A stock can be down more than half and be correctly priced. "
                 "The accounts test asks whether the last published statements "
                 "explain the fall; the sector flag asks whether it fell alone. "
                 "Neither names a cause — that needs the news, which this app "
                 "does not read."),
    }


def build(frames: Dict[str, pd.DataFrame],
          metrics_by_ticker: Dict[str, Dict[str, Any]],
          results: Dict[str, Any],
          debt: Optional[Dict[str, Any]] = None,
          commodities: Optional[Dict[str, Any]] = None,
          cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the sentiment tab needs, each part failing on its own."""
    t = trin(frames, cfg)
    b = breadth_ma(frames, cfg)
    m = mag7(metrics_by_ticker)
    o = oversold(results, metrics_by_ticker)
    x = cross_asset({}, b, t, debt or {}, commodities or {})
    return {"trin": t, "breadth": b, "mag7": m, "oversold": o,
            "cross_asset": x}
