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
COUNT_METRICS = {
    "roe_consistency": ("roe_years_above_15", "min_years_above"),
    "fcf_consistency": ("fcf_years_positive", "min_years_above"),
}


def _is_num(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and np.isnan(x))


def evaluate_test(name: str, spec: Dict[str, Any],
                  metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one test. `result` is True, False or None (unknown)."""
    if name in COUNT_METRICS:
        metric_key, thr_key = COUNT_METRICS[name]
        value = metrics.get(metric_key)
        threshold = spec.get(thr_key, spec.get("threshold"))
        op = "gte"
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
    }


def run_framework(fw_name: str, cfg: Dict[str, Any],
                  metrics: Dict[str, Any],
                  unknown_counts_as: str = "fail") -> Dict[str, Any]:
    tests_cfg = cfg.get("tests", {})
    results = [evaluate_test(n, s, metrics) for n, s in tests_cfg.items()]

    passed = sum(1 for r in results if r["result"] is True)
    failed = sum(1 for r in results if r["result"] is False)
    unknown = sum(1 for r in results if r["result"] is UNKNOWN)

    if unknown_counts_as == "skip":
        denominator = passed + failed
        required = cfg.get("min_tests_passed", len(tests_cfg))
        # Scale the requirement down proportionally when tests were skipped.
        if denominator > 0 and len(tests_cfg) > 0:
            required = int(round(required * denominator / len(tests_cfg)))
        overall = denominator > 0 and passed >= max(required, 1)
    else:
        required = cfg.get("min_tests_passed", len(tests_cfg))
        overall = passed >= required

    return {
        "framework": fw_name,
        "label": cfg.get("label", fw_name),
        "passed": bool(overall),
        "n_passed": passed,
        "n_failed": failed,
        "n_unknown": unknown,
        "n_total": len(tests_cfg),
        "required": required,
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
def screen_universe(records: List[Any],
                    metrics_by_ticker: Dict[str, Dict[str, Any]],
                    thresholds: Dict[str, Any],
                    macro: Dict[str, Any]) -> Dict[str, Any]:
    g = thresholds.get("global", {})
    unknown_mode = g.get("unknown_counts_as", "fail")
    excluded = g.get("capital_intensive_excluded_sectors", [])
    min_cap = g.get("min_market_cap_usd", 0)
    min_turnover = g.get("min_median_daily_turnover_usd", 0)

    gate_open, gate_reason = macro_gate_open(
        macro, thresholds.get("soros", {}).get("macro_gate", {}))

    greenblatt = run_greenblatt(records, metrics_by_ticker,
                                thresholds.get("greenblatt", {}), excluded)

    framework_names = ["buffett", "munger", "schloss", "klarman", "lynch", "soros"]
    results: Dict[str, Any] = {}

    for rec in records:
        m = metrics_by_ticker.get(rec.ticker, {})
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
            r = run_framework(name, cfg, m, unknown_mode)
            if name == "soros" and not gate_open:
                r["passed"] = False
                r["macro_gate_blocked"] = gate_reason
            if name == "klarman" and sector_excluded(rec.sector, excluded):
                r["passed"] = False
                r["ineligible_reason"] = "sector excluded (financials/REITs/utilities)"
            fw[name] = r

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
            "rs_vs_market_index_6m", "pct_above_5y_low", "pct_below_52w_high",
            "net_cash_to_market_cap", "ncav_to_market_cap", "statement_currency",
            "macd_histogram", "atr_pct_percentile")}

        results[rec.ticker] = {
            "ticker": rec.ticker,
            "name": rec.name,
            "market": rec.market,
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
        "macro_gate_open": gate_open,
        "macro_gate_reason": gate_reason,
        "macro": macro,
    }
