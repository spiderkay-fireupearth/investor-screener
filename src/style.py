"""Value or growth — the other axis, and why it needs the whole universe.

Lynch's six categories answer "what kind of company is this". This answers a
different question: "what kind of STOCK is this" — what the market is paying
for, and therefore what has to go right for the price to work out.

Two design points that matter more than the arithmetic:

  * IT IS CROSS-SECTIONAL, NOT ABSOLUTE. "A low price-to-book" means nothing on
    its own; 1.4× is expensive for a Hong Kong industrial and cheap for a US
    software company. So every factor is scored as a PERCENTILE within the
    universe being screened, and the label is relative to that universe. Change
    the universe and the labels move — which is a property of style boxes
    generally, not a defect here, and the panel says so.

  * SECTOR IS A LEGITIMATE INPUT, NOT A CHEAT. A technology business is priced
    on what it will earn rather than on what it owns, and that is true even in
    a year when its multiples happen to look ordinary. The sector tilt is
    applied as a NAMED component with its own weight, visible in the evidence,
    rather than folded silently into the score — so a name labelled growth
    purely because of its sector can be seen to have been.

The output is a five-point scale rather than a binary, because the middle is
where most of the market actually lives and calling it "value" or "growth" on a
coin-flip would be the least useful thing this could do.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

STYLES = ("deep value", "value", "blend", "growth", "high growth")

# Sectors and industry words priced on future earnings rather than on assets in
# place. Matched against the feed's own vocabulary so a new constituent is
# classified without a code change.
GROWTH_SECTORS = {"technology", "communication services", "healthcare"}
GROWTH_INDUSTRY_WORDS = (
    "software", "semiconductor", "internet", "cloud", "saas", "e-commerce",
    "ecommerce", "digital", "data", "cyber", "biotech", "electric vehicle",
    "artificial intelligence", "platform", "media", "entertainment",
    "interactive", "gaming", "payment", "fintech",
)
# The opposite pole: businesses valued on assets in place and current cash.
VALUE_SECTORS = {"financial services", "financials", "utilities", "energy",
                 "real estate", "basic materials"}

# What the score is built from. Value factors score HIGH when cheap; growth
# factors score HIGH when fast. Weights are deliberately blunt — a style label
# that needs three decimal places of tuning is not measuring anything real.
VALUE_FACTORS = (
    ("ebit_to_ev", 1.0, "earnings yield"),
    ("fcf_yield", 1.0, "free-cash-flow yield"),
    ("book_to_price", 1.0, "book to price"),
    ("dividend_yield", 0.5, "dividend yield"),
)
GROWTH_FACTORS = (
    ("eps_cagr_lynch", 1.0, "earnings growth"),
    ("revenue_cagr_5y", 1.0, "revenue growth, five years"),
    ("revenue_growth_1y", 0.5, "revenue growth, one year"),
    ("gross_margin_ttm", 0.5, "gross margin"),
)
SECTOR_WEIGHT = 1.0


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and x == x


def _percentiles(values: Dict[str, float]) -> Dict[str, float]:
    """Rank a {ticker: value} map into 0–1 percentiles. Ties share a rank."""
    pairs = [(t, float(v)) for t, v in values.items() if _n(v)]
    if len(pairs) < 5:
        return {}
    vals = sorted(v for _t, v in pairs)
    n = len(vals)
    out = {}
    for t, v in pairs:
        # Share of the universe at or below this value.
        lo = np.searchsorted(vals, v, side="left")
        hi = np.searchsorted(vals, v, side="right")
        out[t] = ((lo + hi) / 2.0) / n
    return out


def sector_tilt(sector: Optional[str], industry: Optional[str]) -> Dict[str, Any]:
    """Growth, value or neutral, from what the business is.

    Applied as a named component with its own weight rather than folded into
    the score, so a name labelled growth purely on its sector can be seen to
    have been.
    """
    s = (sector or "").strip().lower()
    text = f"{s} {(industry or '').strip().lower()}"
    hit = next((w for w in GROWTH_INDUSTRY_WORDS if w in text), None)
    if s in GROWTH_SECTORS or hit:
        why = (f"{industry or sector} is priced on what it will earn rather "
               "than on what it owns")
        return {"direction": "growth", "score": 1.0, "why": why,
                "matched": hit or s}
    if s in VALUE_SECTORS:
        return {"direction": "value", "score": 0.0,
                "why": (f"{industry or sector} is valued on assets in place "
                        "and current cash"), "matched": s}
    return {"direction": "neutral", "score": 0.5,
            "why": f"{industry or sector or 'this sector'} carries no strong "
                   "style tilt either way", "matched": None}


def assign(records: List[Any], metrics_by_ticker: Dict[str, Dict[str, Any]],
           cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score every name on the value–growth axis and write the label onto it.

    Returns a census for the panel. The label itself lands in each metrics dict
    as `style`, `style_score` and `style_evidence`.
    """
    c = cfg or {}
    bands = c.get("bands", {"deep value": -40, "value": -15,
                            "blend": 15, "growth": 40})

    # Book-to-price rather than price-to-book: the whole point of a percentile
    # is that higher must mean "more of this factor", and an inverted ratio
    # would rank the most expensive names as the cheapest.
    for t, m in metrics_by_ticker.items():
        pb = m.get("price_to_book")
        m["book_to_price"] = (1.0 / pb) if (_n(pb) and pb > 0) else None

    pct: Dict[str, Dict[str, float]] = {}
    for key, _w, _label in VALUE_FACTORS + GROWTH_FACTORS:
        pct[key] = _percentiles({t: m.get(key)
                                 for t, m in metrics_by_ticker.items()})

    census: Dict[str, int] = {}
    thin = 0
    for rec in records:
        m = metrics_by_ticker.get(rec.ticker)
        if m is None:
            continue
        tilt = sector_tilt(getattr(rec, "sector", None),
                           getattr(rec, "industry", None))
        ev: List[str] = []
        v_num = v_den = g_num = g_den = 0.0
        for key, w, label in VALUE_FACTORS:
            p = pct.get(key, {}).get(rec.ticker)
            if p is None:
                continue
            v_num += p * w
            v_den += w
            ev.append(f"{label} {p * 100:.0f}th pct")
        for key, w, label in GROWTH_FACTORS:
            p = pct.get(key, {}).get(rec.ticker)
            if p is None:
                continue
            g_num += p * w
            g_den += w
            ev.append(f"{label} {p * 100:.0f}th pct")

        if v_den == 0 and g_den == 0:
            # Nothing to score on. Say so rather than defaulting to blend,
            # which would put a name with no data in the same bucket as one
            # that was measured and found to be in the middle.
            m["style"] = None
            m["style_score"] = None
            m["style_evidence"] = ["no valuation or growth factors available"]
            m["style_label"] = "unscored"
            thin += 1
            census["unscored"] = census.get("unscored", 0) + 1
            continue

        value_side = (v_num / v_den) if v_den else 0.5
        growth_side = (g_num / g_den) if g_den else 0.5
        # Sector enters as one more growth-side voter with its own weight,
        # named in the evidence so its effect is never invisible.
        growth_side = ((growth_side * (g_den or 1.0)
                        + tilt["score"] * SECTOR_WEIGHT)
                       / ((g_den or 1.0) + SECTOR_WEIGHT))
        ev.append(f"sector tilt {tilt['direction']}")

        score = (growth_side - value_side) * 100.0
        if score <= bands["deep value"]:
            style = "deep value"
        elif score <= bands["value"]:
            style = "value"
        elif score < bands["blend"]:
            style = "blend"
        elif score < bands["growth"]:
            style = "growth"
        else:
            style = "high growth"

        m["style"] = style
        m["style_label"] = style
        m["style_score"] = score
        m["style_value_side"] = value_side
        m["style_growth_side"] = growth_side
        m["style_sector_tilt"] = tilt["direction"]
        m["style_evidence"] = ev
        m["style_why"] = (
            f"{style} — {tilt['why']}; on the numbers it sits in the "
            f"{growth_side * 100:.0f}th percentile for growth and the "
            f"{value_side * 100:.0f}th for cheapness across this universe")
        census[style] = census.get(style, 0) + 1

    scored = sum(v for k, v in census.items() if k != "unscored")
    growthy = census.get("growth", 0) + census.get("high growth", 0)
    valuey = census.get("value", 0) + census.get("deep value", 0)
    return {
        "census": census, "scored": scored, "unscored": thin,
        "growth_names": growthy, "value_names": valuey,
        "blend_names": census.get("blend", 0),
        "caveat": (
            "Relative to THIS universe, not to the world. Every factor is "
            "scored as a percentile across the names screened, so a stock is "
            "value or growth compared with the other names in the table — "
            "change the universe and the labels move. That is how style boxes "
            "work generally; it is worth knowing rather than forgetting. "
            "Sector enters as one named voter, so a technology business reads "
            "growth even in a year its multiples look ordinary."),
    }
