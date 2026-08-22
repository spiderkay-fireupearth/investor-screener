"""Peter Lynch's six categories, and why one set of thresholds cannot serve them.

Lynch's central procedural claim in *One Up on Wall Street* is that the first
question is not "is this cheap?" but "what KIND of company is this?" — because
the same number means opposite things across categories:

  * A low P/E on a fast grower is a bargain. A low P/E on a cyclical is usually
    the sound of the cycle topping out, because the E in the denominator is at
    a peak that is about to break. Lynch: "buying a cyclical after several
    years of record earnings... is a proven method for losing half your money
    in a short period of time."
  * 25% earnings growth is the sweet spot for a fast grower and a red flag on a
    stalwart's income statement, where it usually means an acquisition.
  * A dividend yield is the whole point of a slow grower and an irrelevance on
    a turnaround, which should not be paying one at all.

So this module labels each name, and `thresholds.yml` routes the Lynch tests
through the label with a `by:`/`cases:` block. A screen that applied one growth
band to all six would systematically reject stalwarts for growing too slowly
and reward cyclicals for a growth rate that is an artefact of where the cycle
happens to sit.

The labels are inferred from data the screener already holds, not from Lynch's
own judgement, and they are stated as such in the UI. Where the classification
is uncertain the name is left `unclassified` and the default (fast-grower)
bands apply, which is the strictest route — an unclassified name is never
passed by accident.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

CATEGORIES = ("fast_grower", "stalwart", "slow_grower", "cyclical",
              "turnaround", "asset_play", "unclassified")

LABELS = {
    "fast_grower": "Fast grower",
    "stalwart": "Stalwart",
    "slow_grower": "Slow grower",
    "cyclical": "Cyclical",
    "turnaround": "Turnaround",
    "asset_play": "Asset play",
    "unclassified": "Unclassified",
}

# What each category is judged on, in Lynch's own terms. Shown in the UI so a
# category label is never a bare word.
RATIONALE = {
    "fast_grower": ("15–25% earnings growth bought at a PEG under 1. Above 30% "
                    "growth Lynch turned cautious: that pace invites competition "
                    "and rarely survives contact with the law of large numbers."),
    "stalwart": ("10–12% growth in a large, durable business. Judged on growth "
                 "PLUS dividend against the P/E, because a stalwart's return "
                 "comes from both."),
    "slow_grower": ("Bought for the dividend, not the growth. What matters is "
                    "that the payout is generous and still covered."),
    "cyclical": ("Judged on where the cycle sits, not on the P/E. A low P/E on "
                 "peak earnings is a sell signal here; depressed earnings and "
                 "an ugly P/E are what the entry point looks like."),
    "turnaround": ("Judged on survival first. Can it pay what is due before the "
                   "recovery arrives? Growth rates from a depressed base mean "
                   "nothing."),
    "asset_play": ("Judged against what it owns rather than what it earns. The "
                   "screen can only see the assets on the balance sheet — the "
                   "hidden ones Lynch hunted are, by definition, not in the "
                   "filings as value."),
    "unclassified": ("Not enough history to place it. The strictest (fast "
                     "grower) bands are applied, so nothing passes by default."),
}

# Sectors and industries where the earnings series is a cycle, not a trend.
# Deliberately matched on the feed's own vocabulary rather than a curated list
# of tickers, so a new constituent is classified without a code change.
CYCLICAL_SECTORS = {"energy", "basic materials", "materials"}
CYCLICAL_INDUSTRY_WORDS = (
    "auto", "airline", "airport", "steel", "aluminum", "aluminium", "copper",
    "mining", "metals", "chemical", "shipping", "marine", "semiconductor",
    "oil", "gas", "coal", "paper", "lumber", "homebuild", "construction",
    "travel", "hotel", "resort", "casino", "luxury", "apparel", "leisure",
)


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x


def _is_cyclical(sector: Optional[str], industry: Optional[str]) -> bool:
    s = (sector or "").strip().lower()
    if s in CYCLICAL_SECTORS:
        return True
    text = f"{s} {(industry or '').strip().lower()}"
    return any(w in text for w in CYCLICAL_INDUSTRY_WORDS)


def classify(m: Dict[str, Any], sector: Optional[str] = None,
             industry: Optional[str] = None) -> Dict[str, Any]:
    """Return {'category', 'label', 'why', 'rationale'} for one name.

    Order matters and is Lynch's own: what the business IS comes before what
    its numbers currently say, because the numbers of a cyclical or a
    turnaround are precisely the ones that lie about the category.
    """
    # Prefer the five-year window Lynch is now run on. The six-year metric was
    # the reason so many rows came back "unclassified": a company with exactly
    # five statements has no year -5 to compare against, so `eps_cagr_5y` is
    # empty for most of the Asian universe, and the classifier fell through to
    # the strictest bands for a fencepost reason rather than a real one.
    g = m.get("eps_cagr_lynch")
    if not _n(g):
        g = m.get("eps_cagr_5y")
    if not _n(g):
        g = m.get("revenue_cagr_5y")
    if not _n(g):
        # Last resort: a single year of revenue growth. Noisy, and labelled as
        # such in the reason string, but a noisy label beats a question mark
        # when the alternative is applying fast-grower bands to a company
        # nobody has looked at.
        g = m.get("revenue_growth_1y")
        if _n(g):
            m = {**m, "_growth_source": "one year of revenue growth"}
    losses = m.get("loss_years_in_10")
    recent_loss = m.get("loss_years_in_3")
    eps_now = m.get("eps_vs_5y_avg")
    ptb = m.get("price_to_tangible_book")
    ncav = m.get("ncav_to_market_cap")
    netcash = m.get("net_cash_to_market_cap")
    off_high = m.get("pct_below_52w_high")
    yld = m.get("dividend_yield")

    # 1. Turnaround — the trouble is recent, real and in the accounts. Checked
    #    before the cycle test because a cyclical that has actually lost money
    #    and collapsed in price must be judged on survival, not on the cycle.
    if (_n(recent_loss) and recent_loss >= 1
            and _n(off_high) and off_high >= 0.40):
        return _out("turnaround",
                    f"a loss in the last three years and {off_high:.0%} off the "
                    "52-week high — judged on whether it survives, not on growth")

    # 2. Cyclical — a property of the industry, and it outranks the growth rate
    #    because that growth rate is a position in the cycle wearing a trend's
    #    clothes.
    if _is_cyclical(sector, industry):
        where = ""
        if _n(eps_now):
            where = (" with earnings well below their five-year average, which "
                     "is what the trough looks like" if eps_now <= 0.8 else
                     (" with earnings above their five-year average — late-cycle "
                      "territory, where a low P/E is a warning rather than a "
                      "bargain" if eps_now >= 1.2 else " at mid-cycle earnings"))
        return _out("cyclical",
                    f"{industry or sector} is a cyclical industry{where}")

    # 3. Asset play — priced against what it owns rather than what it earns.
    if ((_n(ncav) and ncav >= 0.66)
            or (_n(netcash) and netcash >= 0.40)
            or (_n(ptb) and 0 < ptb <= 0.80)):
        bits = []
        if _n(ptb) and 0 < ptb <= 0.80:
            bits.append(f"{ptb:.2f}× tangible book")
        if _n(ncav) and ncav >= 0.66:
            bits.append(f"net current assets alone cover {ncav:.0%} of the price")
        if _n(netcash) and netcash >= 0.40:
            bits.append(f"net cash is {netcash:.0%} of the market value")
        return _out("asset_play", "priced against its assets — " + "; ".join(bits))

    # 4–6. The growth bands.
    if not _n(g):
        return _out("unclassified",
                    "no usable growth rate at all — not five years of earnings, "
                    "not five of revenue, not even one — so the strictest bands "
                    "apply")
    src = m.get("_growth_source")
    tail = f" (measured on {src})" if src else ""
    if g >= 0.20:
        return _out("fast_grower",
                    f"earnings compounding at {g:.0%} a year{tail}")
    if g >= 0.10:
        return _out("stalwart", f"earnings compounding at {g:.0%} a year{tail} — "
                                "large and durable rather than fast")
    why = f"earnings growth of {g:.0%} a year{tail}"
    if _n(yld) and yld > 0:
        why += f", paying {yld:.1%}"
    return _out("slow_grower", why)


def _out(cat: str, why: str) -> Dict[str, Any]:
    return {"category": cat, "label": LABELS[cat], "why": why,
            "rationale": RATIONALE[cat]}


def rule_of_20(metrics_by_ticker: Dict[str, Dict[str, Any]],
               inflation_pct: Optional[float],
               market: str = "US") -> Dict[str, Any]:
    """Lynch's macro sanity check: market P/E plus inflation against 20.

    The reasoning is that a high inflation rate destroys the value of a future
    earnings stream, so the multiple the market can justify falls as inflation
    rises; the two together are the thing to watch, not either alone.

    The P/E used here is the MEDIAN TRAILING P/E of this screener's own US
    universe, not a vendor's S&P 500 figure. That is a deliberate choice: it is
    computed from the same numbers every row on the page is scored on, so the
    gauge and the table can never disagree. It will not match a headline S&P
    P/E exactly, and a median runs cooler than the cap-weighted average the
    index quotes — the gap is the mega-caps, which is worth knowing rather than
    hiding.
    """
    pes = []
    for t, m in metrics_by_ticker.items():
        if market and (m.get("market") or market) != market:
            continue
        pe = m.get("pe_ttm")
        if _n(pe) and 0 < pe < 200:
            pes.append(float(pe))
    if not pes:
        return {"available": False,
                "reason": "no trailing P/E available in this universe"}
    pes.sort()
    n = len(pes)
    median = pes[n // 2] if n % 2 else (pes[n // 2 - 1] + pes[n // 2]) / 2
    if not _n(inflation_pct):
        return {"available": False, "median_pe": median,
                "reason": "no inflation reading — the sum needs both halves"}
    total = median + inflation_pct
    if total < 20:
        verdict, reading = "cheap", (
            "under 20 — on Lynch's rule the market is not expensive, and the "
            "multiple is being supported rather than squeezed by inflation")
    elif total < 23:
        verdict, reading = "fair", (
            "just above 20 — fair rather than stretched, but with little room "
            "for either the multiple or inflation to move against you")
    else:
        verdict, reading = "expensive", (
            "well above 20 — the market is paying a high multiple in an "
            "environment that historically does not support one")
    return {"available": True, "median_pe": median,
            "inflation_pct": inflation_pct, "total": total,
            "names": n, "verdict": verdict, "reading": reading}


def peak_earnings_warning(m: Dict[str, Any], category: str) -> Optional[str]:
    """The single most expensive mistake Lynch names, made checkable.

    A cyclical on a low P/E and peak earnings is not cheap: the multiple is low
    BECAUSE the E is about to fall. This is the one place where the app must
    contradict its own value screens out loud.
    """
    if category != "cyclical":
        return None
    pe, eps_v = m.get("pe_ttm"), m.get("eps_vs_5y_avg")
    if _n(pe) and pe > 0 and pe <= 12 and _n(eps_v) and eps_v >= 1.2:
        return (f"Cyclical on a P/E of {pe:.1f} with earnings {eps_v:.1f}× their "
                "five-year average. Lynch's warning applies literally: the "
                "multiple is low because the earnings are at a peak, and a "
                "cheap-looking cyclical at the top of its cycle is the "
                "classic way to lose half your money.")
    if _n(pe) and pe >= 25 and _n(eps_v) and eps_v <= 0.8:
        return (f"Cyclical on a P/E of {pe:.0f} with earnings only {eps_v:.1f}× "
                "their five-year average. On Lynch's inversion this is the "
                "interesting end of the cycle, not the dangerous one — the "
                "multiple looks awful because the earnings are depressed.")
    return None
