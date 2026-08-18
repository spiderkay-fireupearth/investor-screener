"""The seven frameworks, expressed as pass/fail tests driven by thresholds.yml.

Design notes worth knowing before you tune the config:

* A test whose metric is missing evaluates to UNKNOWN, not False. What happens
  next is `global.unknown_counts_as`. The default is `fail`, deliberately: a
  value screen that passes a company because its balance sheet didn't load is
  worse than one that rejects it.
* Greenblatt is rank-based, not threshold-based, so it runs across the whole
  eligible universe after every company's metrics exist. Financials, REITs and
  utilities are excluded from it and from Klarman — EV/EBIT and
  return-on-capital are meaningless when the balance sheet IS the business.
* The Soros macro gate can fail every name at once. That is intentional: it
  encodes the idea that in a credit-stress regime, single-name signals stop
  meaning what they normally mean.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from . import buffett as _bf
from . import lynch as _lynch
from . import rankings as _rank
from . import reflexivity as rfx
from . import style as _style
from . import synopsis as syn

log = logging.getLogger(__name__)

UNKNOWN = None

OPS = {
    "gt":  lambda v, t: v > t,
    "gte": lambda v, t: v >= t,
    "lt":  lambda v, t: v < t,
    "lte": lambda v, t: v <= t,
    "eq":  lambda v, t: v == t,
}

# Tests whose config shape (lookback_years + min_years_above) maps onto a
# pre-counted metric rather than a direct comparison.
#   name -> (count_metric, threshold_key, evaluated_years_metric)
COUNT_METRICS = {
    "roe_consistency": ("roe_years_above_15", "min_years_above", "roe_years_evaluated"),
    "fcf_consistency": ("fcf_years_positive", "min_years_above", "fcf_years_evaluated"),
}

# Below this many years of statements a history-dependent test is reported as
# unevaluable rather than failed. Four is what Yahoo typically returns.
MIN_HISTORY_FLOOR = 4


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


def _is_num(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and np.isnan(x))


def evaluate_test(name: str, spec: Dict[str, Any],
                  metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one test.

    `result` is True, False or None. None means either "data missing" (which
    counts against the company under the strict default) or "not enough years
    of history to judge" — flagged separately as `insufficient`, and excluded
    from the denominator rather than held against the company.

    The distinction matters because fundamentals depth is a property of the
    DATA SOURCE, not the business. US names come from EDGAR with 10+ years;
    Asian names come from Yahoo with about 4. Treating a short history as a
    failure silently makes every Asian company look worse than every American
    one, which is a statement about our plumbing, not about the companies.
    """
    history = metrics.get("history_years") or 0
    insufficient = False

    # ---- category routing --------------------------------------------------
    # Lynch's rule that one bar cannot serve six kinds of company, expressed in
    # config rather than in code. `by:` names a metric holding a LABEL (the
    # Lynch category, the value regime); `cases:` maps that label to overrides.
    # A case may replace the metric, the threshold, the operator, or declare
    # the test not applicable to that kind of company at all.
    #
    # `not_applicable` is deliberately NOT a failure. A cyclical has no
    # meaningful five-year growth band, so failing it on one would be scoring a
    # question that was never asked. It leaves the denominator instead, exactly
    # as a test with too little history does, and the bar scales down with it.
    spec_by = spec.get("by")
    routed_note = ""
    if spec_by:
        label = metrics.get(spec_by)
        case = (spec.get("cases") or {}).get(label)
        if case is None and label:
            case = (spec.get("cases") or {}).get("default")
        if isinstance(case, dict):
            if case.get("not_applicable"):
                return {"name": name, "metric": spec.get("metric"), "value": None,
                        "threshold": None, "operator": spec.get("operator", "gte"),
                        "result": UNKNOWN, "via_alt": False, "insufficient": True,
                        "not_applicable": True,
                        "note": case.get("note")
                        or f"does not apply to a {str(label).replace('_', ' ')}"}
            spec = {**spec, **{k: v for k, v in case.items() if k != "note"}}
            routed_note = case.get("note", "")

    # Tests that need a minimum window before they mean anything at all.
    need_years = spec.get("min_history_years")
    if need_years and history < need_years:
        return {"name": name, "metric": spec.get("metric"), "value": None,
                "threshold": spec.get("threshold"), "operator": spec.get("operator", "gte"),
                "result": UNKNOWN, "via_alt": False, "insufficient": True,
                "note": f"needs {need_years}y of statements, have {history}"}

    if name in COUNT_METRICS:
        metric_key, thr_key, evaluated_key = COUNT_METRICS[name]
        value = metrics.get(metric_key)
        evaluated = metrics.get(evaluated_key) or 0
        lookback = spec.get("lookback_years", 10) or 10
        required_full = spec.get(thr_key, spec.get("threshold"))
        if evaluated < MIN_HISTORY_FLOOR:
            return {"name": name, "metric": metric_key, "value": value,
                    "threshold": required_full, "operator": "gte",
                    "result": UNKNOWN, "via_alt": False, "insufficient": True,
                    "note": f"only {evaluated}y of data"}
        # Scale the bar to the window we actually have: "8 of 10" becomes
        # "4 of 4" at 80%, so the standard holds without punishing shallow data.
        threshold = max(1, _round_half_up(required_full * evaluated / lookback))
        op = "gte"
        metric_key = metric_key
    else:
        metric_key = spec.get("metric")
        value = metrics.get(metric_key)
        threshold = spec.get("threshold")
        op = spec.get("operator", "gte")

    # Some fields exist only where the feed carries them — insider and
    # institutional ownership are published for US names and patchily elsewhere.
    # Scoring their absence as a failure would mark down whole markets for a
    # gap in our plumbing, which is the one thing this screener refuses to do.
    if spec.get("skip_if_missing") and not _is_num(value):
        return {"name": name, "metric": metric_key, "value": None,
                "threshold": threshold, "operator": spec.get("operator", "gte"),
                "result": UNKNOWN, "via_alt": False, "insufficient": True,
                "not_applicable": True,
                "note": spec.get("missing_note")
                or "the feed carries no value for this field in this market"}

    result: Optional[bool]
    if not _is_num(value) or not _is_num(threshold):
        result = UNKNOWN
    else:
        result = bool(OPS[op](value, threshold))

    # Some tests offer an alternative satisfying route (Klarman's net-net).
    alt_used = False
    if result is not True and spec.get("alt_metric"):
        alt_val = metrics.get(spec["alt_metric"])
        alt_thr = spec.get("alt_threshold")
        alt_op = spec.get("alt_operator", "gte")
        if _is_num(alt_val) and _is_num(alt_thr) and OPS[alt_op](alt_val, alt_thr):
            result = True
            alt_used = True

    out = {
        "name": name,
        "metric": metric_key,
        "value": value,
        "threshold": threshold,
        "operator": op,
        "result": result,
        "via_alt": alt_used,
        "insufficient": insufficient,
    }
    if routed_note:
        out["note"] = routed_note
    return out


def run_framework(fw_name: str, cfg: Dict[str, Any],
                  metrics: Dict[str, Any],
                  unknown_counts_as: str = "fail",
                  cycle_shift: int = 0) -> Dict[str, Any]:
    tests_cfg = cfg.get("tests", {})
    results = [evaluate_test(n, s, metrics) for n, s in tests_cfg.items()]

    passed = sum(1 for r in results if r["result"] is True)
    failed = sum(1 for r in results if r["result"] is False)
    # Both leave the denominator, but they mean different things and the UI
    # must not conflate them: "we lack the years to judge" is about our data,
    # "this question is not asked of this kind of company" is about the
    # framework. Reporting both as "limited history" would have the app tell
    # the user its feed was short when the feed was fine.
    not_applicable = sum(1 for r in results if r.get("not_applicable"))
    insufficient = sum(1 for r in results if r.get("insufficient"))
    unknown = sum(1 for r in results if r["result"] is UNKNOWN
                  and not r.get("insufficient"))

    n_total = len(tests_cfg)
    base_required = cfg.get("min_tests_passed", n_total)
    # Marks: the same evidence should demand more of you when the crowd is
    # greedy and less when it is fearful. Only frameworks opting in are moved.
    if cfg.get("cycle_adjust") and cycle_shift:
        base_required = max(1, min(n_total, base_required + cycle_shift))

    # Tests we couldn't evaluate for lack of history leave the denominator, and
    # the bar drops proportionally. Missing DATA still counts against the
    # company under the strict default; missing YEARS does not, because that is
    # our provider's limitation rather than the company's.
    effective_total = n_total - insufficient
    if insufficient and effective_total > 0:
        required = max(1, _round_half_up(base_required * effective_total / n_total))
    else:
        required = base_required

    if unknown_counts_as == "skip":
        denominator = passed + failed
        if denominator > 0 and effective_total > 0:
            required = max(1, _round_half_up(required * denominator / effective_total))
        overall = denominator > 0 and passed >= required
    else:
        overall = effective_total > 0 and passed >= required

    return {
        "framework": fw_name,
        "label": cfg.get("label", fw_name),
        "passed": bool(overall),
        "n_passed": passed,
        "n_failed": failed,
        "n_unknown": unknown,
        "n_insufficient": insufficient,
        "n_not_applicable": not_applicable,
        "n_total": n_total,
        "effective_total": effective_total,
        "required": required,
        "limited_history": bool(insufficient - not_applicable > 0),
        "tests": results,
    }


# ---------------------------------------------------------------------------
def sector_excluded(sector: Optional[str], excluded: List[str]) -> bool:
    if not sector:
        return False
    s = sector.strip().lower()
    return any(s == e.strip().lower() for e in excluded)


def run_greenblatt(records: List[Any], metrics_by_ticker: Dict[str, Dict[str, Any]],
                   cfg: Dict[str, Any], excluded_sectors: List[str]) -> Dict[str, Dict[str, Any]]:
    """Magic Formula: rank by earnings yield and return on capital, combine.

    Ranked within each market by default, so one structurally cheap market
    (Hong Kong, typically) doesn't monopolise a global top-30 and crowd out
    everything else.
    """
    scope = cfg.get("rank_scope", "market")
    top_n = cfg.get("top_n", 30)
    min_cap = cfg.get("min_market_cap_usd", 100_000_000)

    eligible: List[Tuple[str, str, float, float]] = []
    ineligible: Dict[str, str] = {}

    for rec in records:
        m = metrics_by_ticker.get(rec.ticker, {})
        if sector_excluded(rec.sector, excluded_sectors):
            ineligible[rec.ticker] = "sector excluded (financials/REITs/utilities)"
            continue
        # Greenblatt's own screening parameter, and it belongs HERE rather than
        # only on the surfacing gate: a micro-cap left in the ranking pushes a
        # real candidate out of the top 30 before anyone sees either of them.
        cap = m.get("market_cap_usd")
        if _is_num(cap) and cap < min_cap:
            ineligible[rec.ticker] = (
                f"below the USD {min_cap:,.0f} market-cap floor the Magic "
                "Formula screens on")
            continue
        # Greenblatt's OWN definitions of yield and capital, not the general
        # ones: excess cash rather than all cash, and a capital base net of
        # interest-bearing current liabilities. They fall back to the general
        # metrics where the feed cannot support the stricter version, and the
        # row records which was used so no ranking is struck on a mixture
        # nobody can see.
        ey = m.get("ebit_to_ev_greenblatt")
        roc = m.get("ebit_to_invested_capital_greenblatt")
        basis = "greenblatt"
        if not (_is_num(ey) and _is_num(roc)):
            ey = m.get("ebit_to_ev") if not _is_num(ey) else ey
            roc = (m.get("ebit_to_invested_capital")
                   if not _is_num(roc) else roc)
            basis = "general"
        if not (_is_num(ey) and _is_num(roc)):
            ineligible[rec.ticker] = "missing EBIT, EV or invested capital"
            continue
        if ey <= 0 or roc <= 0:
            ineligible[rec.ticker] = "negative earnings yield or return on capital"
            continue
        eligible.append((rec.ticker, rec.market, float(ey), float(roc), basis))

    out: Dict[str, Dict[str, Any]] = {}

    def _ranks(rows):
        by_ey = sorted(rows, key=lambda r: r[2], reverse=True)
        by_roc = sorted(rows, key=lambda r: r[3], reverse=True)
        return ({r[0]: i + 1 for i, r in enumerate(by_ey)},
                {r[0]: i + 1 for i, r in enumerate(by_roc)})

    # The global ranking is ALWAYS computed, whatever the scope in use. The
    # Magic Formula as written is one list across the whole universe; ranking
    # per market is this app's deviation, made so that one structurally cheap
    # market cannot monopolise the table. Both numbers are published so the
    # deviation is visible rather than assumed.
    g_ey, g_roc = _ranks(eligible)
    g_order = sorted(eligible, key=lambda r: g_ey[r[0]] + g_roc[r[0]])
    global_pos = {r[0]: i + 1 for i, r in enumerate(g_order)}

    groups: Dict[str, List[Tuple[str, str, float, float, str]]] = {}
    for row in eligible:
        key = row[1] if scope == "market" else "_global"
        groups.setdefault(key, []).append(row)

    for key, rows in groups.items():
        ey_rank, roc_rank = _ranks(rows)
        combined = sorted(rows, key=lambda r: ey_rank[r[0]] + roc_rank[r[0]])
        for pos, r in enumerate(combined):
            t = r[0]
            out[t] = {
                "framework": "greenblatt",
                "label": cfg.get("label", "Magic Formula"),
                "passed": pos < top_n,
                "n_passed": 2 if pos < top_n else 0,
                "n_failed": 0 if pos < top_n else 2,
                "n_unknown": 0,
                "n_total": 2,
                "required": 2,
                "scope": key,
                "universe_size": len(rows),
                "basis": r[4],
                "tests": [
                    {"name": "earnings_yield",
                     "metric": ("ebit_to_ev_greenblatt" if r[4] == "greenblatt"
                                else "ebit_to_ev"),
                     "value": r[2],
                     "threshold": None, "operator": "rank",
                     "result": True, "rank": ey_rank[t], "via_alt": False,
                     "note": ("EBIT ÷ enterprise value, with EV struck on "
                              "EXCESS cash rather than all cash"
                              if r[4] == "greenblatt" else
                              "struck on the general EV — the feed lacked what "
                              "Greenblatt's stricter definition needs")},
                    {"name": "return_on_capital",
                     "metric": ("ebit_to_invested_capital_greenblatt"
                                if r[4] == "greenblatt"
                                else "ebit_to_invested_capital"),
                     "value": r[3],
                     "threshold": None, "operator": "rank",
                     "result": True, "rank": roc_rank[t], "via_alt": False,
                     "note": ("EBIT ÷ (net working capital + net fixed assets), "
                              "excluding goodwill, excess cash and "
                              "interest-bearing current liabilities"
                              if r[4] == "greenblatt" else
                              "struck on the general capital base — the feed "
                              "lacked the short-term debt split")},
                ],
                "combined_rank": pos + 1,
                "combined_rank_score": ey_rank[t] + roc_rank[t],
                "global_rank": global_pos.get(t),
                "global_rank_score": g_ey.get(t, 0) + g_roc.get(t, 0),
                "global_universe_size": len(eligible),
            }

    # The portfolio the formula actually prescribes: the top N by COMBINED
    # rank across the whole universe, which is what "buy the top 20 to 30 and
    # rebalance annually" means. Kept separate from the per-market pass flag so
    # the two are never confused.
    portfolio = []
    for r in g_order[:max(top_n, 1)]:
        t = r[0]
        portfolio.append({
            "ticker": t, "market": r[1],
            "earnings_yield": r[2], "return_on_capital": r[3],
            "ey_rank": g_ey[t], "roc_rank": g_roc[t],
            "combined_score": g_ey[t] + g_roc[t],
            "rank": global_pos[t], "basis": r[4],
        })
    out["_portfolio"] = {
        "rows": portfolio,
        "universe_size": len(eligible),
        "excluded": len(ineligible),
        "top_n": top_n,
        "scope_in_use": scope,
        "basis_mixed": len({r[4] for r in eligible}) > 1,
        "on_greenblatt_basis": sum(1 for r in eligible if r[4] == "greenblatt"),
    }

    for t, reason in ineligible.items():
        out[t] = {
            "framework": "greenblatt", "label": cfg.get("label", "Magic Formula"),
            "passed": False, "n_passed": 0, "n_failed": 0, "n_unknown": 2,
            "n_total": 2, "required": 2, "tests": [], "ineligible_reason": reason,
        }
    return out


def macro_gate_open(macro: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Soros's reflexivity brake. Returns (open?, human-readable reason)."""
    if not cfg.get("enabled", False):
        return True, "macro gate disabled"
    if not macro or not macro.get("_enabled"):
        return True, "macro data unavailable — gate skipped"

    hy = macro.get("hy_credit_spread")
    curve = macro.get("yield_curve_10y2y")
    reasons = []
    if _is_num(hy) and hy > cfg.get("hy_spread_max", 6.0):
        reasons.append(f"high-yield spread {hy:.2f}% above {cfg['hy_spread_max']}%")
    if _is_num(curve) and curve < cfg.get("yield_curve_min", -0.5):
        reasons.append(f"10y-2y curve {curve:.2f} below {cfg['yield_curve_min']}")

    if reasons:
        return False, "credit/regime stress: " + "; ".join(reasons)
    return True, "risk conditions normal"


# ---------------------------------------------------------------------------
def add_relative_value(records, metrics_by_ticker: Dict[str, Dict[str, Any]],
                       min_names: int = 20) -> None:
    """Price each name against its own market's median, in place.

    Marks: *"Superior results don't come from buying high quality assets, but
    from buying assets — regardless of quality — for less than they're worth."*
    A fixed EV/EBIT cutoff answers "is this cheap in the abstract". It does not
    answer "is this cheap given what is on offer right now", which is the
    question that actually moves with the cycle — his "sometimes there are
    plentiful opportunities... and sometimes opportunities are few".

    Scoped per market, because a Thai median and a Nasdaq median are not the
    same yardstick, and a global median would simply rank every US name as
    expensive and every Indonesian name as cheap. A market with fewer than
    `min_names` priced constituents gets no ratio at all rather than a median
    computed off a handful of names.
    """
    by_market: Dict[str, List[float]] = {}
    for rec in records or []:
        m = metrics_by_ticker.get(rec.ticker) or {}
        v = m.get("ev_to_ebit")
        # Negative EV/EBIT means negative EBIT — undefined as "cheap", and
        # including it would drag the median toward meaninglessness.
        if _is_num(v) and v > 0:
            by_market.setdefault(rec.market, []).append(float(v))

    medians: Dict[str, float] = {}
    for mkt, vals in by_market.items():
        if len(vals) >= min_names:
            medians[mkt] = float(sorted(vals)[len(vals) // 2])

    for rec in records or []:
        m = metrics_by_ticker.get(rec.ticker)
        if m is None:
            continue
        med = medians.get(rec.market)
        v = m.get("ev_to_ebit")
        m["market_median_ev_ebit"] = med
        # Below 1.0 = cheaper than the median name in its own market.
        m["ev_ebit_vs_market"] = (
            float(v) / med if (_is_num(v) and v > 0 and med) else None)


def set_value_regime(metrics_by_ticker: Dict[str, Dict[str, Any]],
                     schloss_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Is deep value on offer today, or must Schloss settle for relative value?

    Schloss bought below book. Edwin Schloss's adjustment, in markets where
    sub-book stocks had stopped existing, was to switch to relative value —
    depressed price-to-sales or P/E against normal earning power — rather than
    to stand aside for a decade. Which regime we are in is not an opinion: it
    is the share of the universe currently trading below tangible book, and it
    is measured here and written onto every row so the threshold engine can
    route the valuation test through it.
    """
    floor = schloss_cfg.get("relative_value_below_book_share", 0.05)
    vals = [m.get("price_to_tangible_book") for m in metrics_by_ticker.values()]
    scored = [v for v in vals if _is_num(v) and v > 0]
    share = (sum(1 for v in scored if v < 1.0) / len(scored)) if scored else None
    regime = ("deep_value_available" if (share is None or share >= floor)
              else "relative_value")
    for m in metrics_by_ticker.values():
        m["value_regime"] = regime
        m["below_book_share_of_universe"] = share
    return {"regime": regime, "below_book_share": share,
            "names_scored": len(scored), "floor": floor}


def add_buffett_valuation(records: List[Any],
                          metrics_by_ticker: Dict[str, Dict[str, Any]],
                          macro: Dict[str, Any],
                          cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Intrinsic value, margin of safety, and the B label — done once, here.

    This lives outside `metrics.py` because it needs the risk-free rate, which
    is a property of the world rather than of the company. Buffett's own
    practice: discount "at the rate of the long-term government bond". The rate
    is reported on the page beside the answer, because a DCF whose discount
    rate is invisible is not a valuation anyone can check.
    """
    dcf_cfg = cfg.get("intrinsic_value", {})
    rf = macro.get("us_10y")
    prem = dcf_cfg.get("equity_risk_premium", 0.045)
    discount = (rf / 100.0 + prem) if _is_num(rf) else None
    labelled = 0
    for rec in records:
        m = metrics_by_ticker.get(rec.ticker)
        if m is None:
            continue
        iv = _bf.intrinsic_value(m.get("owner_earnings_per_share"),
                                 m.get("owner_earnings_cagr_5y"),
                                 discount, dcf_cfg)
        m["intrinsic_value_detail"] = iv
        m["intrinsic_value_per_share"] = iv.get("value_per_share")
        m["discount_rate_used"] = iv.get("discount_rate")
        m["margin_of_safety"] = _bf.margin_of_safety(
            getattr(rec, "price", None), iv.get("value_per_share"))
        tenets = _bf.business_tenets(m, getattr(rec, "sector", None),
                                     getattr(rec, "industry", None),
                                     cfg.get("business_tenets", {}))
        m["business_tenets"] = tenets
        m["buffett_b_label"] = tenets["label"]
        m["buffett_tenets_summary"] = tenets["summary"]
        labelled += bool(tenets["passed"])

        # Munger's three baskets. Decided here rather than in metrics.py
        # because the circle half comes from the tenets, which need the moat,
        # which needs the whole universe to have been measured first.
        bucket = _rank.munger_bucket(m, tenets["circle"].get("ok"),
                                     cfg.get("business_tenets", {}))
        m["munger_bucket"] = bucket["bucket"]
        m["munger_bucket_label"] = bucket["label"]
        m["munger_bucket_why"] = bucket["why"]

        # Cannibalisation, completed. Retiring shares is only a virtue if the
        # shares were retired BELOW intrinsic value; above it, a buyback moves
        # money from the owners who stay to the ones who leave. The share-count
        # half was computed in metrics.py; the price half needs the DCF, which
        # has just run, so the two are joined here.
        if m.get("cannibalisation") == 1:
            mos = m.get("margin_of_safety")
            if _is_num(mos):
                below = mos > 0
                m["cannibalisation"] = 2 if below else 0
                m["cannibalisation_reading"] = (
                    (m.get("cannibalisation_reading") or "")
                    + (", and bought below the intrinsic value estimated here "
                       "— the version of a buyback that concentrates value"
                       if below else
                       ", but bought ABOVE the intrinsic value estimated here "
                       "— that moves money from the owners who stay to the "
                       "ones who leave"))
    return {"risk_free_pct": rf, "equity_risk_premium": prem,
            "discount_rate": discount, "b_labelled": labelled,
            "names": len(records)}



def build_rankings(records: List[Any],
                   metrics_by_ticker: Dict[str, Dict[str, Any]],
                   thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Buffett's and Munger's top N per exchange.

    Buffett's list is gated on the three business tenets, because a list
    carrying his name that includes businesses outside the declared circle of
    competence, without a moat, or mid-turnaround is not his list. The gate is
    dropped per-market — and SAID to have been dropped — where an exchange has
    too few names carrying the label to make thirty of anything.
    """
    excluded = thresholds.get("global", {}).get(
        "capital_intensive_excluded_sectors", [])
    out: Dict[str, Any] = {}

    b_cfg = (thresholds.get("buffett", {}) or {}).get("ranking")
    if b_cfg:
        def tenet_gate(rec, m):
            return None if m.get("buffett_b_label") else \
                "does not clear the three business tenets"
        gated = _rank.dual_rank(records, metrics_by_ticker, b_cfg,
                                excluded_sectors=[], eligible_fn=tenet_gate)
        floor = b_cfg.get("min_names_before_fallback", 8)
        # The ungated ranking is ALWAYS computed, because a market can fail the
        # gate in two different ways: too few names cleared it, or none did —
        # and a market where none did never appears in the gated result at all.
        # An earlier version only looked at markets that were present and thin,
        # so an exchange with zero tenet-clearing names silently vanished from
        # the page rather than falling back. Absent and thin are the same
        # condition here and are handled as one.
        ungated = _rank.dual_rank(records, metrics_by_ticker, b_cfg,
                                  excluded_sectors=[])
        for mk, blk in ungated["markets"].items():
            have = gated["markets"].get(mk)
            if have and have["eligible"] >= floor:
                continue
            blk = dict(blk)
            blk["fallback"] = True
            blk["gated_eligible"] = have["eligible"] if have else 0
            gated["markets"][mk] = blk
        gated["fallback_markets"] = sorted(
            mk for mk, v in gated["markets"].items() if v.get("fallback"))
        gated["total_eligible"] = sum(v["eligible"]
                                      for v in gated["markets"].values())
        gated["gate"] = "business tenets"
        gated["fallback_floor"] = floor
        out["buffett_ranking"] = gated

    m_cfg = (thresholds.get("munger", {}) or {}).get("ranking")
    if m_cfg:
        out["munger_ranking"] = _rank.dual_rank(
            records, metrics_by_ticker, m_cfg, excluded_sectors=excluded)
    return out


def screen_universe(records: List[Any],
                    metrics_by_ticker: Dict[str, Dict[str, Any]],
                    thresholds: Dict[str, Any],
                    macro: Dict[str, Any],
                    cycle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    g = thresholds.get("global", {})
    unknown_mode = g.get("unknown_counts_as", "fail")
    excluded = g.get("capital_intensive_excluded_sectors", [])
    min_cap = g.get("min_market_cap_usd", 0)
    min_turnover = g.get("min_median_daily_turnover_usd", 0)

    gate_open, gate_reason = macro_gate_open(
        macro, thresholds.get("soros", {}).get("macro_gate", {}))

    greenblatt = run_greenblatt(records, metrics_by_ticker,
                                thresholds.get("greenblatt", {}), excluded)
    # Pulled out before the per-name loop, so a ticker literally named
    # "_portfolio" could never collide with it.
    magic_formula = greenblatt.pop("_portfolio", {})

    add_relative_value(records, metrics_by_ticker)
    value_regime = set_value_regime(
        metrics_by_ticker, thresholds.get("schloss", {}))
    style_census = _style.assign(records, metrics_by_ticker,
                                 thresholds.get("style", {}))
    buffett_valuation = add_buffett_valuation(
        records, metrics_by_ticker, macro, thresholds.get("buffett", {}))

    framework_names = ["buffett", "munger", "schloss", "klarman", "lynch",
                       "templeton", "marks", "soros", "rogers", "graham"]
    # Frameworks that read a company's accounts. A fund has none of the inputs,
    # so these are marked not-applicable rather than failed — failing an ETF on
    # "missing ROE" is noise dressed as a finding.
    COMPANY_FRAMEWORKS = {"buffett", "munger", "schloss", "klarman", "lynch",
                          "templeton", "marks", "rogers", "graham"}
    results: Dict[str, Any] = {}

    for rec in records:
        m = metrics_by_ticker.get(rec.ticker, {})
        is_fund = str(getattr(rec, "quote_type", "") or "").upper() in (
            "ETF", "MUTUALFUND", "INDEX")
        gates: List[str] = []

        cap_usd = m.get("market_cap_usd")
        if _is_num(cap_usd) and cap_usd < min_cap:
            gates.append(f"market cap below USD {min_cap:,.0f}")
        turn_usd = m.get("median_turnover_usd")
        if _is_num(turn_usd) and turn_usd < min_turnover:
            gates.append(f"daily turnover below USD {min_turnover:,.0f}")

        fw: Dict[str, Any] = {}
        for name in framework_names:
            cfg = thresholds.get(name, {})
            if not cfg:
                continue
            r = run_framework(name, cfg, m, unknown_mode,
                              cycle_shift=(cycle or {}).get("threshold_shift", 0))
            # Rogers analysed the COMMODITY first and the company second:
            # "The smart investor looking into a copper company first has to
            # examine the supply-demand dynamics of copper." A name with no
            # commodity exposure has no such underlying to analyse, so the
            # framework is not-applicable rather than failed.
            # Soros's subject is a company whose share price feeds its own
            # fundamentals. A name that neither issues, retires nor acquires
            # has no such channel, and the book says the framework does not
            # apply there — so it is not-applicable, not a failure.
            if name == "soros":
                _ch = rfx.channel_open(m)
                if _ch.get("open") is False:
                    r["passed"] = False
                    r["ineligible_reason"] = (
                        "near-equilibrium — " + "; ".join(_ch["reasons"])
                        + ". Reflexivity describes prices that CHANGE "
                          "fundamentals, not prices that merely reflect them")
            if cfg.get("themes_only") and not (getattr(rec, "themes", None) or []):
                r["passed"] = False
                r["ineligible_reason"] = (
                    "no commodity theme — this framework is about a commodity's "
                    "supply cycle, and there is no underlying to analyse")
            if is_fund and name in COMPANY_FRAMEWORKS:
                r["passed"] = False
                r["ineligible_reason"] = ("fund, not an operating company — "
                                          "revenue, equity and ROE are undefined")
            if name == "soros" and not gate_open:
                r["passed"] = False
                r["macro_gate_blocked"] = gate_reason
            if name == "klarman" and sector_excluded(rec.sector, excluded):
                r["passed"] = False
                r["ineligible_reason"] = "sector excluded (financials/REITs/utilities)"
            fw[name] = r

        if is_fund:
            fw["greenblatt"] = {
                "framework": "greenblatt", "label": "Magic Formula",
                "passed": False, "n_passed": 0, "n_total": 2, "tests": [],
                "ineligible_reason": "fund, not an operating company"}
        else:
            fw["greenblatt"] = greenblatt.get(rec.ticker, {
            "framework": "greenblatt", "passed": False, "n_passed": 0,
                "n_total": 2, "tests": [], "ineligible_reason": "not evaluated"})

        tech_cfg = thresholds.get("technical", {})
        tech = run_framework("technical", tech_cfg, m, unknown_mode)

        passed_names = [k for k, v in fw.items() if v.get("passed")]
        comp = thresholds.get("composite", {})
        always = set(comp.get("always_surface_if_passed", []))
        surface = (len(passed_names) >= comp.get("min_frameworks_passed", 2)
                   or bool(always & set(passed_names)))
        if gates:
            surface = False

        # Persist the display metrics INTO the result. The published page merges
        # this region's fresh results with the other region's stored ones, and a
        # merged row has no live metrics dict behind it — without this, half the
        # table renders every key metric as "—" and the 200-day-MA field
        # contradicts the technical test card sitting right beside it.
        # The list lives in render.py, next to the code that consumes it, so
        # adding a metric to the display cannot silently fail to persist it.
        from .render import DISPLAY_METRICS
        key_metrics = {k: m.get(k) for k in DISPLAY_METRICS}

        results[rec.ticker] = {
            "ticker": rec.ticker,
            "name": rec.name,
            "market": rec.market,
            "themes": list(getattr(rec, "themes", []) or []),
            "quote_type": getattr(rec, "quote_type", None),
            "is_fund": is_fund,
            "sector": rec.sector,
            "industry": rec.industry,
            # Trimmed at write time, not at render time: the published page
            # inlines every row, and a full profile paragraph per name would
            # add megabytes to a file that has to load on a phone.
            "business_summary": syn.trim_description(
                getattr(rec, "business_summary", None)),
            "currency": rec.currency,
            "price": rec.price,
            "market_cap": rec.market_cap,
            "market_cap_usd": cap_usd,
            "metrics": key_metrics,
            "frameworks": fw,
            "technical": tech,
            "frameworks_passed": passed_names,
            "n_frameworks_passed": len(passed_names),
            "technical_passed": tech.get("passed", False),
            "surfaced": surface,
            "gates_failed": gates,
            "warnings": rec.warnings,
        }

    return {
        "results": results,
        "cycle": cycle or {},
        "macro_gate_open": gate_open,
        "macro_gate_reason": gate_reason,
        "macro": macro,
        "value_regime": value_regime,
        "buffett_valuation": buffett_valuation,
        "magic_formula": magic_formula,
        "style_census": style_census,
        **build_rankings(records, metrics_by_ticker, thresholds),
        "lynch_census": lynch_census(metrics_by_ticker),
        # Lynch's macro sanity check, computed on this screener's own US rows
        # so the gauge and the table are struck on the same numbers.
        "rule_of_20": _lynch.rule_of_20(
            {rec.ticker: metrics_by_ticker.get(rec.ticker, {})
             for rec in records if getattr(rec, "market", None) == "US"},
            macro.get("us_cpi_yoy"), market=None),
    }


def lynch_census(metrics_by_ticker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """How the universe splits across Lynch's six categories.

    Published for the same reason the reflexive census is: a classifier nobody
    can audit is a classifier nobody should trust. If nine names in ten come
    back 'cyclical', the industry word list is too greedy and the number on the
    page is what says so.
    """
    out: Dict[str, int] = {}
    for m in metrics_by_ticker.values():
        c = m.get("lynch_category")
        if c:
            out[c] = out.get(c, 0) + 1
    return out
