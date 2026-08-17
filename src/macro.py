"""Macro and carry-trade indicators for the five markets.

This module computes EVIDENCE, not conclusions. The Soros brief asks for the
prevailing bias, the reflexive feedback loop and a trade matrix — those are
judgments, and a program that emitted them would be generating confident prose
from arithmetic that doesn't support it. What arithmetic *can* establish is the
state of the carry trade, and that is what this produces:

  * rate differentials between the funding currency and the target
  * realised FX volatility, which is what actually kills a carry position
  * carry-to-volatility — the ratio that decides whether the trade is worth
    doing, and whose collapse marks the unwind
  * how far the HKD sits inside its convertibility band
  * whether local equities and local currency are moving together, which is the
    observable signature of a reflexive loop rather than a fundamental one

Read the numbers, then argue about what they mean. That order matters.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# FX pairs quoted as local-per-USD, the market convention for these.
FX_TICKERS = {
    "JPY": "USDJPY=X", "IDR": "USDIDR=X", "THB": "USDTHB=X",
    "HKD": "USDHKD=X", "TWD": "USDTWD=X",
}

EQUITY_INDEX = {
    "US": "^GSPC", "JP": "^N225", "HK": "^HSI",
    "ID": "^JKSE", "TH": "^SET.BK",
}

# FRED series. Several emerging-market policy rates are published irregularly
# or discontinued; every one is optional and absence is reported, not guessed.
FRED_SERIES = {
    "us_10y": "DGS10",
    "us_2y": "DGS2",
    "us_policy": "DFF",                  # effective fed funds
    "jp_10y": "IRLTLT01JPM156N",         # Japan long-term government yield
    "jp_policy": "IRSTCI01JPM156N",      # Japan immediate rate
    "us_cpi": "CPIAUCSL",
    "hy_spread": "BAMLH0A0HYM2",
}

HKD_BAND = (7.75, 7.85)                  # HKMA convertibility undertaking


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and math.isnan(x))


def realised_vol(close: pd.Series, days: int = 30) -> Optional[float]:
    s = close.dropna().astype(float)
    if len(s) < days + 2:
        return None
    r = np.log(s / s.shift(1)).dropna().iloc[-days:]
    if len(r) < days // 2:
        return None
    return float(r.std(ddof=1) * math.sqrt(252))


def _pct_change(close: pd.Series, days: int) -> Optional[float]:
    s = close.dropna().astype(float)
    if len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1)


def carry_analysis(fx_frames: Dict[str, pd.DataFrame],
                   macro: Dict[str, Any]) -> Dict[str, Any]:
    """The yen carry trade, reduced to the two numbers that decide it.

    A carry position earns the rate differential and loses on adverse FX moves.
    Carry-to-vol — differential divided by realised volatility — is therefore
    the trade's actual Sharpe-like signature. Historically the unwind does not
    begin when the differential narrows; it begins when volatility rises while
    the differential is narrowing. Both legs are reported so the distinction
    stays visible.
    """
    out: Dict[str, Any] = {}
    jpy = fx_frames.get("JPY")
    if jpy is None or jpy.empty:
        return {"error": "no USDJPY history"}

    close = jpy["Close"].astype(float)
    out["usdjpy"] = float(close.iloc[-1])
    out["usdjpy_1m"] = _pct_change(close, 21)
    out["usdjpy_3m"] = _pct_change(close, 63)
    out["usdjpy_12m"] = _pct_change(close, 252)

    v30 = realised_vol(close, 30)
    v90 = realised_vol(close, 90)
    out["jpy_vol_30d"] = v30
    out["jpy_vol_90d"] = v90
    # Vol rising faster than its own baseline is the early tell.
    out["jpy_vol_ratio"] = (v30 / v90) if (_n(v30) and _n(v90) and v90) else None

    us10, jp10 = macro.get("us_10y"), macro.get("jp_10y")
    us_p, jp_p = macro.get("us_policy"), macro.get("jp_policy")
    out["us_10y"], out["jp_10y"] = us10, jp10
    out["us_policy"], out["jp_policy"] = us_p, jp_p

    if _n(us10) and _n(jp10):
        out["long_differential"] = us10 - jp10
    if _n(us_p) and _n(jp_p):
        out["policy_differential"] = us_p - jp_p

    diff = out.get("policy_differential") or out.get("long_differential")
    if _n(diff) and _n(v30) and v30 > 0:
        # Differential in percent, vol as a decimal — put both on the same scale.
        out["carry_to_vol"] = (diff / 100.0) / v30
        c = out["carry_to_vol"]
        if c > 0.5:
            out["carry_reading"] = ("carry is well paid relative to the risk being "
                                    "taken — the trade is still attractive on its "
                                    "own terms")
        elif c > 0.25:
            out["carry_reading"] = ("carry compensation is thinning; positions "
                                    "become sensitive to any volatility shock")
        else:
            out["carry_reading"] = ("carry no longer pays for the volatility — "
                                    "this is the zone in which unwinds have "
                                    "historically begun")

    # The stress signature: differential narrowing WHILE volatility expands.
    if _n(out.get("jpy_vol_ratio")) and _n(diff):
        narrowing = _n(out.get("usdjpy_3m")) and out["usdjpy_3m"] < 0
        vol_expanding = out["jpy_vol_ratio"] > 1.15
        out["unwind_pressure"] = bool(narrowing and vol_expanding)
        out["unwind_note"] = (
            "yen strengthening into rising volatility — both legs of an unwind "
            "are present" if out["unwind_pressure"] else
            "no simultaneous yen strength and volatility expansion")
    return out


def peg_pressure(fx_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Where the HKD sits inside the HKMA band.

    The peg holds until it doesn't, and the informative number is not the level
    but the position within the band: at 7.85 the HKMA is selling USD and
    draining HKD liquidity, which tightens local rates regardless of what the
    Fed is doing. That transmission is the mechanism worth watching.
    """
    hkd = fx_frames.get("HKD")
    if hkd is None or hkd.empty:
        return {}
    spot = float(hkd["Close"].iloc[-1])
    lo, hi = HKD_BAND
    pos = (spot - lo) / (hi - lo) if hi > lo else None
    out = {"usdhkd": spot, "band_position": pos, "band": HKD_BAND}
    if _n(pos):
        if pos > 0.9:
            out["reading"] = ("pinned to the weak end — the HKMA is defending, "
                              "draining HKD liquidity and tightening local rates")
        elif pos < 0.1:
            out["reading"] = ("pinned to the strong end — inflows are forcing HKD "
                              "creation, loosening local liquidity")
        else:
            out["reading"] = "mid-band; the peg is not under active pressure"
    return out


def reflexive_coupling(equity: Dict[str, pd.DataFrame],
                       fx_frames: Dict[str, pd.DataFrame],
                       window: int = 126) -> Dict[str, Any]:
    """Correlation between a local index and its own currency.

    In Japan a weak yen mechanically lifts the Nikkei through exporter
    translation, and a rising Nikkei attracts the foreign flows that weaken the
    yen further. That is a textbook reflexive loop, and it shows up as a strong
    correlation between index and currency. When the correlation breaks, the
    loop is breaking — which is more informative than either series alone.
    """
    out: Dict[str, Any] = {}
    pairs = {"JP": ("^N225", "JPY"), "ID": ("^JKSE", "IDR"),
             "TH": ("^SET.BK", "THB"), "HK": ("^HSI", "HKD")}
    for mkt, (idx_t, ccy) in pairs.items():
        idx = equity.get(idx_t)
        fx = fx_frames.get(ccy)
        if idx is None or fx is None or idx.empty or fx.empty:
            continue
        j = pd.concat([idx["Close"].astype(float).rename("eq"),
                       fx["Close"].astype(float).rename("fx")],
                      axis=1, join="inner").dropna()
        if len(j) < window + 5:
            continue
        r = j.pct_change().dropna().iloc[-window:]
        if len(r) < window // 2:
            continue
        corr = float(r["eq"].corr(r["fx"]))
        out[mkt] = {
            "correlation": corr,
            # USDxxx rising = local currency weakening. Positive correlation
            # therefore means "weaker currency, stronger equities".
            "reading": ("weaker currency lifting equities — the classic "
                        "reflexive export loop" if corr > 0.25 else
                        "stronger currency alongside stronger equities — "
                        "domestic-demand or inflow driven" if corr < -0.25 else
                        "no strong coupling between currency and equities"),
        }
    return out


def build(yahoo, fred, store) -> Dict[str, Any]:
    """Fetch everything and assemble the macro picture."""
    fx_frames: Dict[str, pd.DataFrame] = {}
    for ccy, tk in FX_TICKERS.items():
        df = yahoo.prices(tk, period="2y")
        if df is not None and not df.empty:
            store.save_prices(tk, df)
        else:
            df = store.load_prices(tk)
        if df is not None and not df.empty:
            fx_frames[ccy] = df

    equity: Dict[str, pd.DataFrame] = {}
    for _mkt, tk in EQUITY_INDEX.items():
        df = yahoo.prices(tk, period="2y")
        if df is not None and not df.empty:
            store.save_prices(tk, df)
        else:
            df = store.load_prices(tk)
        if df is not None and not df.empty:
            equity[tk] = df

    macro = fred.snapshot(FRED_SERIES)

    out = {
        "macro": macro,
        "carry": carry_analysis(fx_frames, macro),
        "peg": peg_pressure(fx_frames),
        "coupling": reflexive_coupling(equity, fx_frames),
        "fx": {},
        "equity": {},
        "unavailable": [
            "Bank Indonesia and Bank of Thailand policy rates — no reliable "
            "free series; check the central banks directly",
            "JGB futures positioning and CFTC yen net specs — not free",
        ],
    }
    for ccy, df in fx_frames.items():
        c = df["Close"].astype(float)
        out["fx"][ccy] = {
            "spot": float(c.iloc[-1]),
            "chg_1m": _pct_change(c, 21), "chg_3m": _pct_change(c, 63),
            "chg_12m": _pct_change(c, 252),
            "vol_30d": realised_vol(c, 30),
        }
    for mkt, tk in EQUITY_INDEX.items():
        df = equity.get(tk)
        if df is None or df.empty:
            continue
        c = df["Close"].astype(float)
        from . import technicals as ta
        out["equity"][mkt] = {
            "level": float(c.iloc[-1]),
            "chg_1m": _pct_change(c, 21), "chg_3m": _pct_change(c, 63),
            "chg_12m": _pct_change(c, 252),
            "rsi_14": ta._last(ta.rsi(c, 14)),
            "vs_200dma": (float(c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1)
                          if len(c) > 200 else None),
        }
    return out
