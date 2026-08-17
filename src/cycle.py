"""Market-cycle gauge — the Howard Marks half that isn't about single names.

Marks's central claim is that you cannot predict, but you can prepare: knowing
where the cycle stands tells you how aggressive to be, even though it says
nothing about what happens next week. This module answers "where do we stand"
from three things we already hold — the primary index's own RSI, the breadth of
our universe, and the VIX — and returns one of three postures.

No vendor sentiment feed is involved. Breadth is computed from our own stored
price history, which is both free and auditable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import technicals as ta

log = logging.getLogger(__name__)

DEFENSIVE, CORE, OPPORTUNISTIC = "defensive", "core", "opportunistic"


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and np.isnan(x))


def assess(index_df: Optional[pd.DataFrame],
           breadth: Optional[Dict[str, Any]],
           vix: Optional[float],
           cfg: Dict[str, Any],
           hy_oas: Optional[float] = None,
           cape: Optional[float] = None) -> Dict[str, Any]:
    """Return the cycle posture plus the evidence behind it.

    Deliberately a VOTE across three independent signals rather than a single
    trigger. Any one of RSI, breadth or VIX can be misleading on its own; two
    of three agreeing is a weaker claim than certainty but a more honest one.
    """
    if not cfg.get("enabled", True):
        return {"mode": CORE, "reason": "cycle gauge disabled", "signals": {}}

    sig: Dict[str, Any] = {}
    votes = {DEFENSIVE: 0, CORE: 0, OPPORTUNISTIC: 0}

    # 1) Index RSI — the crowd's short-term temperature.
    rsi = None
    if index_df is not None and len(index_df) > 60:
        rsi = ta._last(ta.rsi(index_df["Close"].astype(float), 14))
    sig["index_rsi"] = rsi
    if _n(rsi):
        if rsi > cfg.get("overbought_rsi", 70):
            votes[DEFENSIVE] += 1
        elif rsi < cfg.get("oversold_rsi", 30):
            votes[OPPORTUNISTIC] += 1
        else:
            votes[CORE] += 1

    # 2) Breadth — how many names are actually participating. A narrow advance
    #    is the classic late-cycle tell that a rising index conceals.
    pct200 = (breadth or {}).get("pct_above_200dma")
    sig["pct_above_200dma"] = pct200
    if _n(pct200):
        if pct200 > cfg.get("breadth_hot", 0.75):
            votes[DEFENSIVE] += 1
        elif pct200 < cfg.get("breadth_cold", 0.35):
            votes[OPPORTUNISTIC] += 1
        else:
            votes[CORE] += 1

    # 3) VIX — the price of insurance. Cheap insurance means complacency.
    sig["vix"] = vix
    if _n(vix):
        if vix < cfg.get("vix_calm", 15):
            votes[DEFENSIVE] += 1
        elif vix > cfg.get("vix_stressed", 28):
            votes[OPPORTUNISTIC] += 1
        else:
            votes[CORE] += 1

    # 4) Credit spreads. Marks: "the markets are riskiest when there's a
    #    widespread belief that there's no risk, since this makes investors
    #    feel it's safe to do risky things." Credit priced for no defaults is
    #    that belief expressed in the market where it does the most damage.
    sig["hy_oas_bp"] = hy_oas
    if _n(hy_oas):
        if hy_oas < cfg.get("hy_oas_calm", 350):
            votes[DEFENSIVE] += 1
        elif hy_oas > cfg.get("hy_oas_stressed", 700):
            votes[OPPORTUNISTIC] += 1
        else:
            votes[CORE] += 1

    # 5) Valuation. The slowest-moving signal here and the one that says least
    #    about the next twelve months — which is why it is one vote of five and
    #    not a gate. Marks gives no number; these are calibrations.
    sig["cape"] = cape
    if _n(cape):
        if cape > cfg.get("cape_rich", 30):
            votes[DEFENSIVE] += 1
        elif cape < cfg.get("cape_cheap", 16):
            votes[OPPORTUNISTIC] += 1
        else:
            votes[CORE] += 1

    if not any(votes.values()):
        return {"mode": CORE, "reason": "no cycle signals available",
                "signals": sig, "votes": votes}

    mode = max(votes, key=lambda k: votes[k])
    parts = []
    if _n(rsi):
        parts.append(f"index RSI {rsi:.0f}")
    if _n(pct200):
        parts.append(f"{pct200:.0%} of names above their 200-day")
    if _n(vix):
        parts.append(f"VIX {vix:.1f}")
    if _n(hy_oas):
        parts.append(f"high-yield {hy_oas:.0f}bp")
    if _n(cape):
        parts.append(f"CAPE {cape:.1f}")

    narrative = {
        DEFENSIVE: ("Raise the bar. Prices embed optimism, so the same evidence "
                    "should buy you less. Favour quality, low leverage and cash; "
                    "trim the speculative end."),
        CORE: ("Neither greed nor fear dominates. Hold the long-term allocation "
               "and rotate selectively on fundamentals rather than on the tape."),
        OPPORTUNISTIC: ("Fear is doing the pricing. This is when a disciplined "
                        "buyer earns their returns — deploy into quality that has "
                        "been sold indiscriminately."),
    }[mode]

    return {
        "mode": mode,
        "votes": votes,
        "signals": sig,
        "evidence": " · ".join(parts) if parts else "no readings",
        "reason": narrative,
        # Defensive demands one more passing test; opportunistic accepts one fewer.
        "threshold_shift": {DEFENSIVE: +1, CORE: 0, OPPORTUNISTIC: -1}[mode],
    }
