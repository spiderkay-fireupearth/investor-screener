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

    return {
        "name": name,
        "metric": metric_key,
        "value": value,
        "threshold": threshold,
        "operator": op,
        "result": result,
        "via_alt": alt_used,
        "insufficient": insufficient,
    }


def run_framework(fw_name: str, cfg: Dict[str, Any],
                  metrics: Dict[str, Any],
                  unknown_counts_as: str = "fail",
                  cycle_shift: int = 0) -> Dict[str, Any]:
    tests_cfg = cfg.get("tests", {})
    results = [evaluate_test(n, s, metrics) for n, s in tests_cfg.items()]

    passed = sum(1 for r in results if r["result"] is True)
    failed = sum(1 for r in results if r["result"] is False)
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
        "n_total": n_total,
        "effective_total": effective_total,
        "required": required,
        "limited_history": bool(insufficient),
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

    eligible: List[Tuple[str, str, float, float]] = []
    ineligible: Dict[str, str] = {}

    for rec in records:
        m = metrics_by_ticker.get(rec.ticker, {})
        if sector_excluded(rec.sector, excluded_sectors):
            ineligible[rec.ticker] = "sector excluded (financials/REITs/utilities)"
            continue
        ey = m.get("ebit_to_ev")
        roc = m.get("ebit_to_invested_capital")
        if not (_is_num(ey) and _is_num(roc)):
            ineligible[rec.ticker] = "missing EBIT, EV or invested capital"
            continue
        if ey <= 0 or roc <= 0:
            ineligible[rec.ticker] = "negative earnings yield or return on capital"
            continue
        eligible.append((rec.ticker, rec.market, float(ey), float(roc)))

    out: Dict[str, Dict[str, Any]] = {}

    groups: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for row in eligible:
        key = row[1] if scope == "market" else "_global"
        groups.setdefault(key, []).append(row)

    for key, rows in groups.items():
        by_ey = sorted(rows, key=lambda r: r[2], reverse=True)
        ey_rank = {r[0]: i + 1 for i, r in enumerate(by_ey)}
        by_roc = sorted(rows, key=lambda r: r[3], reverse=True)
        roc_rank = {r[0]: i + 1 for i, r in enumerate(by_roc)}

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
                "tests": [
                    {"name": "earnings_yield", "metric": "ebit_to_ev", "value": r[2],
                     "threshold": None, "operator": "rank",
                     "result": True, "rank": ey_rank[t], "via_alt": False},
                    {"name": "return_on_capital", "metric": "ebit_to_invested_capital",
                     "value": r[3], "threshold": None, "operator": "rank",
                     "result": True, "rank": roc_rank[t], "via_alt": False},
                ],
                "combined_rank": pos + 1,
                "combined_rank_score": ey_rank[t] + roc_rank[t],
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

    add_relative_value(records, metrics_by_ticker)

    framework_names = ["buffett", "munger", "schloss", "klarman", "lynch",
                       "templeton", "marks", "soros"]
    # Frameworks that read a company's accounts. A fund has none of the inputs,
    # so these are marked not-applicable rather than failed — failing an ETF on
    # "missing ROE" is noise dressed as a finding.
    COMPANY_FRAMEWORKS = {"buffett", "munger", "schloss", "klarman", "lynch",
                          "templeton", "marks"}
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
        key_metrics = {k: m.get(k) for k in (
            "pe_ttm", "price_to_tangible_book", "price_to_book", "ev_to_ebit",
            "roe_ttm", "roic_5y_avg", "debt_to_equity", "fcf_yield", "peg_ratio",
            "eps_cagr_5y", "rsi_14", "price_above_sma200", "sma50_above_sma200",
            "history_years",
            "rs_vs_market_index_6m", "pct_above_5y_low", "pct_below_52w_high",
            "net_cash_to_market_cap", "ncav_to_market_cap", "statement_currency",
            "macd_histogram", "atr_pct_percentile")}

        results[rec.ticker] = {
            "ticker": rec.ticker,
            "name": rec.name,
            "market": rec.market,
            "themes": list(getattr(rec, "themes", []) or []),
            "quote_type": getattr(rec, "quote_type", None),
            "is_fund": is_fund,
            "sector": rec.sector,
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
    }
