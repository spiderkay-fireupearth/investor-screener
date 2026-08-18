"""Ranked lists for the threshold frameworks, one per exchange.

Greenblatt's Magic Formula is rank-based by construction: two numbers, two
ranks, add them. Buffett and Munger are not — they are pass/fail screens, and
a pass/fail screen gives you a set, not an order. Turning them into lists means
choosing what to rank on, and that choice is the whole argument, so it is made
explicitly here rather than buried in a sort key.

  * BUFFETT is ranked in Greenblatt's own shape, on Buffett's own numbers: a
    QUALITY axis (return on net tangible assets — "the key qualities we seek
    are... good returns on the net tangible assets required to operate the
    business") and a PRICE axis (owner-earnings yield, the cash measure from
    the 1986 letter). A wonderful business at a fair price is two questions and
    this ranks both, then adds the ranks.

  * MUNGER is ranked on return on capital and on INVERSION — how few of the
    obvious ways to lose money are present. Munger's distinctive contribution
    is not a valuation method, it is "all I want to know is where I'm going to
    die, so I'll never go there". So the second axis counts red flags rather
    than measuring cheapness, which is what makes his list different from
    Buffett's rather than a reordering of it.

Both are ranked WITHIN each exchange, not globally. That is the request, and it
is also the more useful shape: a Hong Kong list and a US list are two different
opportunity sets, and a merged one is dominated by whichever market happens to
be structurally cheaper this decade.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and x == x


# --------------------------------------------------------------- Munger
# Each entry: (metric, operator, threshold, label). "Passing" here means the
# red flag is ABSENT — the score counts dogs that did not bark.
INVERSION_CHECKS = (
    ("accruals_ratio", "lte", 0.10,
     "earnings turn into cash"),
    ("goodwill_to_assets", "lte", 0.35,
     "book value is not mostly what it paid for other companies"),
    ("debt_to_equity", "lte", 0.50,
     "not carrying dangerous leverage"),
    ("operating_margin_slope_5y", "gte", -0.01,
     "operating margin is not eroding"),
    ("eps_cv_5y", "lte", 0.35,
     "earnings are predictable enough to estimate"),
    ("share_count_change_5y", "lte", 0.0,
     "not diluting its owners"),
    ("loss_years_in_10", "lte", 0,
     "no loss-making years on the record"),
)


def munger_inversion(m: Dict[str, Any]) -> Dict[str, Any]:
    """Count the ways this could obviously go wrong that are NOT present.

    Reported as a share of the checks that could actually be evaluated, so a
    company with a four-year feed is judged on the same standard as one with
    ten rather than penalised for its provider. The evaluated count travels
    with the score, because 5 of 5 and 5 of 7 are not the same statement.
    """
    passed, evaluated, failed = 0, 0, []
    for key, op, thr, label in INVERSION_CHECKS:
        v = m.get(key)
        if not _n(v):
            continue
        evaluated += 1
        ok = (v <= thr) if op == "lte" else (v >= thr)
        if ok:
            passed += 1
        else:
            failed.append(label)
    if not evaluated:
        return {"available": False, "reason": "none of the inversion checks "
                                              "could be evaluated"}
    return {"available": True, "score": passed / evaluated,
            "passed": passed, "evaluated": evaluated,
            "total_checks": len(INVERSION_CHECKS),
            "failed": failed,
            "reading": (f"{passed} of {evaluated} ways to lose money are absent"
                        + ("; the ones present are " + "; ".join(failed[:2])
                           if failed else " — none of the checked ones are present"))}


# --------------------------------------------------------------- the builder
def _rank_desc(rows: List[Tuple[str, float]]) -> Dict[str, int]:
    ordered = sorted(rows, key=lambda r: r[1], reverse=True)
    return {t: i + 1 for i, (t, _v) in enumerate(ordered)}


def dual_rank(records: List[Any],
              metrics_by_ticker: Dict[str, Dict[str, Any]],
              cfg: Dict[str, Any],
              excluded_sectors: Optional[List[str]] = None,
              eligible_fn: Optional[Callable[[Any, Dict[str, Any]], Optional[str]]] = None
              ) -> Dict[str, Any]:
    """Rank on two factors within each market and add the ranks.

    `cfg` names the two factors and the bar; `eligible_fn` returns a REASON
    string to exclude a name, or None to keep it. Exclusions are counted and
    reported rather than silently dropped: a top-30 drawn from 40 survivors of
    a 300-name universe is a different object from one drawn from 300, and the
    panel has to be able to say which it is.
    """
    fa, fb = cfg["factor_a"], cfg["factor_b"]
    top_n = cfg.get("top_n", 30)
    min_cap = cfg.get("min_market_cap_usd", 0)
    excluded_sectors = excluded_sectors or []

    by_market: Dict[str, List[Tuple[str, float, float, Dict[str, Any]]]] = {}
    reasons: Dict[str, str] = {}
    for rec in records:
        m = metrics_by_ticker.get(rec.ticker) or {}
        mkt = getattr(rec, "market", None) or "?"
        sector = (getattr(rec, "sector", "") or "").strip().lower()
        if any(sector == e.strip().lower() for e in excluded_sectors):
            reasons[rec.ticker] = "sector excluded"
            continue
        cap = m.get("market_cap_usd")
        if _n(cap) and min_cap and cap < min_cap:
            reasons[rec.ticker] = "below the market-cap floor"
            continue
        if eligible_fn is not None:
            why = eligible_fn(rec, m)
            if why:
                reasons[rec.ticker] = why
                continue
        a, b = m.get(fa["metric"]), m.get(fb["metric"])
        if not (_n(a) and _n(b)):
            reasons[rec.ticker] = f"missing {fa['label']} or {fb['label']}"
            continue
        if fa.get("positive_only", True) and a <= 0:
            reasons[rec.ticker] = f"{fa['label']} is not positive"
            continue
        if fb.get("positive_only", True) and b <= 0:
            reasons[rec.ticker] = f"{fb['label']} is not positive"
            continue
        by_market.setdefault(mkt, []).append(
            (rec.ticker, float(a), float(b), m))

    markets: Dict[str, Any] = {}
    for mkt, rows in by_market.items():
        ra = _rank_desc([(t, a) for t, a, _b, _m in rows])
        rb = _rank_desc([(t, b) for t, _a, b, _m in rows])
        ordered = sorted(rows, key=lambda r: ra[r[0]] + rb[r[0]])
        out = []
        for pos, (t, a, b, m) in enumerate(ordered[:top_n]):
            out.append({
                "ticker": t, "rank": pos + 1,
                "a": a, "b": b,
                "a_rank": ra[t], "b_rank": rb[t],
                "score": ra[t] + rb[t],
                "b_label": m.get("buffett_b_label"),
                "style": m.get("style"),
                "category": m.get("lynch_category_label"),
            })
        markets[mkt] = {"rows": out, "eligible": len(rows),
                        "shown": len(out), "top_n": top_n}
    return {
        "markets": markets,
        "factor_a": fa, "factor_b": fb,
        "excluded": len(reasons),
        "exclusion_reasons": _count(reasons.values()),
        "total_eligible": sum(len(v) for v in by_market.values()),
        "top_n": top_n,
    }


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
