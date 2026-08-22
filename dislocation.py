"""Hard falls that the accounts do not explain.

The request was: find names down more than 30% in six months where the fall is
NOT caused by deteriorating operations. Half of that is computable and half is
not, and the split matters more than anything else in this module.

**Computable:** the price fell hard, and the last published accounts did not.
That divergence is arithmetic — a 6-month return against revenue, margin, cash
flow and leverage that are all still intact.

**Not computable:** WHY it fell. Nothing in a price series or a balance sheet
distinguishes a geopolitical panic from a regulatory shock from a short-seller
report. This module therefore does NOT claim a cause. It reports the divergence
and then narrows the fifteen candidate causes to the ones the price and volume
evidence is CONSISTENT with — which is a shortlist for you to check, not a
finding.

**The failure mode to keep in front of you.** The commonest reason a stock falls
30% while its last accounts look fine is not panic. It is that the accounts are
stale and the market is right. Annual statements are up to twelve months old;
the market prices tomorrow. Every name this module surfaces is, by construction,
a name where the market disagrees with the last filing — and the market wins
that argument more often than not. Treat a hit as "go and read the last three
announcements", never as "the market is wrong".

What the evidence CAN separate:

* **Idiosyncratic or market-wide.** The name's 6-month return against its own
  market index. If the index fell too, the cause is macro (flight to safety,
  forced liquidation, FX) and the name is a passenger. If the name fell alone,
  the cause is specific to it (scandal, regulation, a short report).
* **Cliff or grind.** How much of the six-month fall happened in the worst
  single month. A cascade — algorithmic stops, a liquidity vacuum, a headline —
  is concentrated. A market steadily re-rating a business is not.
* **Who was selling.** Volume through the fall against its own baseline. Heavy
  volume looks like capitulation or forced selling; light volume looks like an
  order book with no bids, which is a different problem and a different trade.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# The fifteen candidate causes, each tagged with the evidence pattern it would
# leave. `scope`: 'market' shows up index-wide, 'name' does not, 'either' can
# be both. `shape`: 'cliff' concentrates in days, 'grind' spreads out.
CAUSES: List[Dict[str, Any]] = [
    {"n": 1, "name": "Geopolitical tension or conflict spillover",
     "scope": "market", "shape": "cliff"},
    {"n": 2, "name": "Natural calamity or physical shock",
     "scope": "either", "shape": "cliff"},
    {"n": 3, "name": "Abnormal weather or seasonal disruption",
     "scope": "either", "shape": "grind"},
    {"n": 4, "name": "Systemic contagion from a single-point failure",
     "scope": "market", "shape": "cliff"},
    {"n": 5, "name": "Non-operational executive scandal",
     "scope": "name", "shape": "cliff"},
    {"n": 6, "name": "Sudden regulatory shift",
     "scope": "either", "shape": "cliff"},
    {"n": 7, "name": "Currency devaluation and debt-repayment panic",
     "scope": "market", "shape": "either"},
    {"n": 8, "name": "Flight to safety (macro risk-off)",
     "scope": "market", "shape": "grind"},
    {"n": 9, "name": "Private-asset distress spilling into public holdings",
     "scope": "market", "shape": "grind"},
    {"n": 10, "name": "Forced liquidation and margin-call contagion",
     "scope": "market", "shape": "cliff"},
    {"n": 11, "name": "Ambiguity aversion — sell first, investigate later",
     "scope": "name", "shape": "cliff"},
    {"n": 12, "name": "Algorithmic and stop-loss cascade",
     "scope": "either", "shape": "cliff"},
    {"n": 13, "name": "Narrative cascade and availability heuristic",
     "scope": "name", "shape": "either"},
    {"n": 14, "name": "Short-seller feedback loop",
     "scope": "name", "shape": "either"},
    {"n": 15, "name": "Liquidity vacuum and contrarian mispricing",
     "scope": "name", "shape": "cliff"},
]

DEFAULTS = {
    "fall_6m": -0.30,          # the trigger the request specified
    "cliff_share": 0.55,       # worst month as a share of the total fall
    "alone_gap": -0.20,        # underperformed its index by this much = its own
    "heavy_volume": 1.25,      # 20d/50d volume
    "thin_volume": 0.80,
}


def _n(x) -> bool:
    return (x is not None and isinstance(x, (int, float))
            and not (isinstance(x, float) and math.isnan(x)))


def fundamentals_intact(m: Dict[str, Any]) -> Dict[str, Any]:
    """Did the BUSINESS deteriorate, on the last published accounts?

    Deliberately a small set of blunt tests. The question is not "is this a good
    company" — the other eleven frameworks answer that — but "is there anything
    in the accounts that would justify a third of the market value going away".
    """
    tests = []

    def add(label, ok, detail):
        tests.append({"label": label, "ok": ok, "detail": detail})

    rev = m.get("revenue_growth_1y")
    add("revenue not collapsing", (rev >= -0.10) if _n(rev) else None,
        f"revenue {rev:+.1%} y/y" if _n(rev) else "revenue growth unavailable")

    eps = m.get("eps_growth_1y")
    add("earnings not collapsing", (eps >= -0.25) if _n(eps) else None,
        f"EPS {eps:+.1%} y/y" if _n(eps) else "EPS growth unavailable")

    fcf = m.get("free_cash_flow_ttm")
    add("still generating cash", (fcf > 0) if _n(fcf) else None,
        f"trailing FCF {'positive' if _n(fcf) and fcf > 0 else 'negative'}"
        if _n(fcf) else "free cash flow unavailable")

    nd = m.get("net_debt_to_ebitda")
    add("balance sheet not stressed", (nd <= 3.5) if _n(nd) else None,
        f"net debt/EBITDA {nd:.1f}" if _n(nd) else "net debt/EBITDA unavailable")

    loss = m.get("loss_years_in_10")
    add("not a chronic loss maker", (loss <= 2) if _n(loss) else None,
        f"{loss:.0f} loss years in 10" if _n(loss) else "loss history unavailable")

    acc = m.get("accruals_ratio")
    add("earnings turning into cash", (acc <= 0.15) if _n(acc) else None,
        f"accruals {acc:+.2f}" if _n(acc) else "accruals unavailable")

    scored = [t for t in tests if t["ok"] is not None]
    passed = sum(1 for t in scored if t["ok"])
    return {"tests": tests, "passed": passed, "evaluable": len(scored),
            "intact": bool(scored) and passed >= max(3, len(scored) - 1),
            "unevaluable": len(tests) - len(scored)}


def assess(m: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None
           ) -> Optional[Dict[str, Any]]:
    """Return a dislocation reading, or None if the name has not fallen enough."""
    c = dict(DEFAULTS, **(cfg or {}))
    r6 = m.get("return_6m")
    if not _n(r6) or r6 > c["fall_6m"]:
        return None

    out: Dict[str, Any] = {"return_6m": r6}
    fund = fundamentals_intact(m)
    out["fundamentals"] = fund

    # --- did the market fall with it? -------------------------------------
    # rs_vs_market_index_6m is (stock - index), so the index move falls out of
    # it without needing another feed.
    rs = m.get("rs_vs_market_index_6m")
    idx6 = (r6 - rs) if _n(rs) else None
    out["index_6m"] = idx6
    out["relative_6m"] = rs
    if _n(rs):
        if rs <= c["alone_gap"]:
            out["scope"] = "name"
            out["scope_reading"] = (
                f"fell {abs(rs):.0%} more than its own market — this is "
                f"specific to the company, not the tape")
        elif _n(idx6) and idx6 <= -0.10:
            out["scope"] = "market"
            out["scope_reading"] = (
                f"its market fell {abs(idx6):.0%} too — the name is a "
                f"passenger, and the cause is macro rather than corporate")
        else:
            out["scope"] = "mixed"
            out["scope_reading"] = ("fell roughly in line with its market — "
                                    "neither clearly idiosyncratic nor clearly "
                                    "macro")
    else:
        out["scope"] = None
        out["scope_reading"] = "no index comparison available"

    # --- cliff or grind ---------------------------------------------------
    worst = m.get("worst_month_in_6m")
    if _n(worst) and r6 < 0:
        share = worst / abs(r6) if abs(r6) > 0 else None
        out["worst_month"] = -worst
        out["cliff_share"] = share
        if _n(share) and share >= c["cliff_share"]:
            out["shape"] = "cliff"
            out["shape_reading"] = (
                f"{share:.0%} of the fall happened in a single month — the "
                f"signature of a cascade or a shock, not a re-rating")
        else:
            out["shape"] = "grind"
            out["shape_reading"] = (
                "the fall was spread across the period — markets grinding a "
                "valuation down usually know something")
    else:
        out["shape"] = None
        out["shape_reading"] = "not enough price history to tell"

    # --- who was selling --------------------------------------------------
    vol = m.get("vol20_over_vol50")
    if _n(vol):
        if vol >= c["heavy_volume"]:
            out["volume"] = "heavy"
            out["volume_reading"] = (
                f"volume running {vol:.2f}x its own baseline — capitulation or "
                f"forced selling, which at least means the sellers are done "
                f"when they are done")
        elif vol <= c["thin_volume"]:
            out["volume"] = "thin"
            out["volume_reading"] = (
                f"volume at {vol:.2f}x baseline — the price fell into an empty "
                f"order book, which is a liquidity problem rather than a "
                f"verdict, and cuts both ways on the way out")
        else:
            out["volume"] = "normal"
            out["volume_reading"] = "volume unremarkable through the fall"

    # --- narrow the fifteen ----------------------------------------------
    scope, shape = out.get("scope"), out.get("shape")
    likely = []
    for cause in CAUSES:
        if scope and cause["scope"] != "either" and cause["scope"] != scope:
            continue
        if shape and cause["shape"] != "either" and cause["shape"] != shape:
            continue
        likely.append(cause)
    if out.get("volume") == "thin":
        likely = [x for x in likely if x["n"] != 10] + \
                 [x for x in CAUSES if x["n"] == 15 and x not in likely]
    out["candidate_causes"] = likely
    out["ruled_out"] = [x for x in CAUSES if x not in likely]

    # An OBSERVED event beats an inferred shortlist. When a feed has actually
    # reported something — an 8-K item code, a quake — the fifteen collapse to
    # what was seen, and the shortlist is kept only as context.
    ev = m.get("_events") or {}
    observed = ev.get("observed_causes") or []
    if observed:
        out["observed_causes"] = [c for c in CAUSES if c["n"] in observed]
        out["evidence_grade"] = "observed"
        out["candidate_causes"] = out["observed_causes"]
        out["inferred_shortlist"] = likely
    elif ev:
        out["evidence_grade"] = "inferred"
        out["feed_note"] = ev.get("note")
    out["events"] = ev.get("events") or []

    # --- the thing that stops this being a buy list ----------------------
    # A filed 4.02 says the company's own past accounts cannot be relied on;
    # a 1.03 says it is in bankruptcy; a 3.01 says it is being delisted. In all
    # three the "intact fundamentals" this screen just measured are worthless,
    # and the name must be removed from the list rather than shown with a
    # caveat — a caveat next to a green tick is not read.
    disq = ev.get("disqualifies") if isinstance(ev, dict) else None
    out["disqualified_by_filing"] = disq
    out["qualifies"] = bool(fund["intact"]) and not disq
    out["caution"] = (
        "The accounts here are the LAST PUBLISHED ones and may be up to a year "
        "old. The commonest reason a stock falls this hard while its filings "
        "still look healthy is not panic — it is that the filings are stale "
        "and the market is right. This is a list of questions, not answers: "
        "read the last three announcements before treating any of it as a "
        "dislocation.")
    if fund["unevaluable"]:
        out["caution"] += (f" {fund['unevaluable']} of the fundamental tests "
                           f"could not be evaluated on this name at all.")
    return out


def scan(results: Dict[str, Any], metrics_by_ticker: Dict[str, Dict[str, Any]],
         cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Label every result, in place. Returns a summary for the page."""
    hits = 0
    fell = 0
    for ticker, r in (results or {}).items():
        m = metrics_by_ticker.get(ticker) or r.get("metrics") or {}
        if not m:
            continue
        a = assess(m, cfg)
        if a is None:
            continue
        fell += 1
        r["dislocation"] = a
        if a["qualifies"]:
            hits += 1
    return {"fell_30pct": fell, "fundamentals_intact": hits,
            "threshold": (cfg or DEFAULTS).get("fall_6m", DEFAULTS["fall_6m"])}
