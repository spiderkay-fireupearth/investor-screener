"""Static HTML screener output — self-contained, no build step, no server.

Pass/fail presentation as specified: a name either clears a framework's tests
or it doesn't. The detail drawer exists so a fail is never a black box — you can
always see which test failed, on what value, against what threshold. That
matters more than it sounds: most of the time a surprising result is a data
problem, not an investment insight, and this is how you tell the difference.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import synopsis as syn

log = logging.getLogger(__name__)

FRAMEWORKS = [
    ("buffett", "Buffett"), ("munger", "Munger"), ("schloss", "Schloss"),
    ("klarman", "Klarman"), ("lynch", "Lynch"), ("templeton", "Templeton"),
    ("marks", "Marks"), ("greenblatt", "Greenblatt"), ("soros", "Soros"),
    ("rogers", "Rogers"), ("graham", "Graham"),
]


# Every metric key the payload displays. This list is the CONTRACT between the
# renderer and the store: `screens.py` persists exactly these with each result,
# because a row merged in from the other region's last run has no live metrics
# dict behind it and falls back to the stored copy.
#
# It used to be hand-maintained inside screens.py, and drifted twice — the
# returns and the RSI reading were both added to the display and forgotten
# here, so every merged row rendered them as "—" while the live rows showed
# them. A silent dash is indistinguishable from missing data, which is the
# worst possible failure for a screener whose whole discipline is making
# missing data visible. There is now a test that reads build_payload's source
# and fails if it touches a key this list does not contain.
DISPLAY_METRICS = (
    "pe_ttm", "price_to_tangible_book", "price_to_book", "ev_to_ebit",
    "roe_ttm", "roic_5y_avg", "debt_to_equity", "fcf_yield", "peg_ratio",
    "eps_cagr_5y", "rsi_14", "price_above_sma200", "sma50_above_sma200",
    "history_years", "rs_vs_market_index_6m", "pct_above_5y_low",
    "pct_below_52w_high", "net_cash_to_market_cap", "ncav_to_market_cap",
    "statement_currency", "macd_histogram", "atr_pct_percentile",
    # returns and the RSI reading — the two that drifted
    "return_3m", "return_6m", "return_12m", "worst_month_in_6m",
    "rsi_label", "rsi_regime", "rsi_note", "rsi_divergence",
    "rsi_divergence_note",
    # Lynch: the category IS the framework, so it has to survive the round
    # trip through the store or a merged row loses the reason it passed.
    "lynch_category", "lynch_category_label", "lynch_category_why",
    "lynch_peak_earnings_warning",
    "peg_ratio", "pegy_ratio", "growth_plus_yield_to_pe", "eps_vs_5y_avg",
    "dividend_yield", "payout_ratio",
    "net_cash_per_share", "net_cash_share_of_price", "pe_ex_cash",
    "short_term_debt_share", "long_term_debt_to_equity",
    "cash_to_short_term_debt",
    "insider_ownership", "institutional_ownership",
    # Schloss: price history depth and the listing-age proxy for his 20-year bar
    "listing_age_years", "pct_above_10y_low", "price_in_10y_range",
    "price_to_sales", "cfo_positive_share_10y", "cfo_positive_share_5y",
    "loss_years_in_5", "statement_years_used_schloss", "goodwill_to_assets",
    "value_regime", "below_book_share_of_universe",
    "loss_years_in_10", "loss_years_in_3",
    # Buffett: owner earnings, the DCF, and the B label. `business_tenets` is
    # a nested dict rather than a scalar — it carries the evidence behind the
    # label, and a badge without its evidence is an instruction, not a finding.
    "net_margin_ttm", "gross_margin_ttm", "gross_margin_cv",
    "owner_earnings_per_share", "owner_earnings_yield",
    "owner_earnings_to_net_income", "owner_earnings_cagr_5y",
    "maintenance_capex", "debt_payoff_years", "capex_to_net_income",
    "sga_to_gross_profit", "one_dollar_premise", "current_ratio",
    "intrinsic_value_per_share", "margin_of_safety", "discount_rate_used",
    "buffett_b_label", "buffett_tenets_summary", "business_tenets",
    "return_on_net_tangible_assets", "roic_5y_avg", "eps_cv_5y",
    # Greenblatt's own definitions, kept apart from the general EV and capital
    # base so the Magic Formula panel can say which basis a row was ranked on.
    "ebit_to_ev", "ebit_to_invested_capital",
    "ebit_to_ev_greenblatt", "ebit_to_invested_capital_greenblatt",
    "enterprise_value_greenblatt", "invested_capital_greenblatt",
    "excess_cash", "greenblatt_working_capital_floored",
    "munger_inversion_score", "munger_inversion_reading",
    "munger_bucket", "munger_bucket_label", "munger_bucket_why",
    "pricing_power", "pricing_power_reading", "gross_margin_slope_5y",
    "goodwill_growth_3y", "cannibalisation", "cannibalisation_reading",
    # The value/growth axis. `style_evidence` is a list rather than a scalar —
    # it is what stops the badge being an assertion.
    "style", "style_label", "style_score", "style_why", "style_evidence",
    "style_sector_tilt", "style_value_side", "style_growth_side",
    "book_to_price", "revenue_cagr_5y", "revenue_growth_1y",
    "eps_cagr_lynch", "eps_cagr_lynch_years", "peg_ratio_lynch",
    # The price series behind the technical chart. Persisted like any other
    # display field so a merged row draws a chart too — but PRUNED from the
    # published page for rows nobody is reviewing (see build_payload), because
    # 900 series is a megabyte of JSON on a phone.
    "spark", "return_1m", "macd_histogram", "vol20_over_vol50",
    "candle_action", "candle_why", "candle_trend", "candle_stop",
    "candle_bullish", "candle_bearish", "candle_signals", "candle_caveat",
    "atr_pct", "max_drawdown_1y", "max_drawdown_5y",
)

# Retained only so an older caller importing it does not break. The cap it
# used to enforce is gone: price series are sidecar files now, so there is no
# payload budget to ration and every row gets a chart.
MAX_SPARKLINES = None

MARKET_LABELS = {"US": "US large cap", "JP": "Nikkei 225", "SG": "SGX",
                 "HK": "HKEX", "TH": "SET", "ID": "IDX",
                 "MY": "Bursa Malaysia"}


def _fmt_num(v, dp=2, pct=False, money=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    try:
        if pct:
            return f"{v * 100:.{dp}f}%"
        if money:
            for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
                if abs(v) >= div:
                    return f"{v / div:.2f}{unit}"
            return f"{v:,.0f}"
        return f"{v:,.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def _test_rows(fw: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for t in fw.get("tests", []):
        res = t.get("result")
        if t.get("insufficient"):
            # Two different "not scored" cases, both excluded from the score,
            # and the difference is not cosmetic. "Not enough YEARS" is a
            # property of our data feed; "does not apply" is a property of the
            # framework — Lynch does not ask a cyclical for a growth band. A
            # shallow feed must never masquerade as a failing business, and a
            # deliberate exclusion must never masquerade as a shallow feed.
            state = "na"
            thresh = ("not applicable" if t.get("not_applicable")
                      else "insufficient history")
        else:
            state = "pass" if res is True else ("fail" if res is False else "unknown")
            thresh = (("rank " + str(t.get("rank"))) if t.get("operator") == "rank"
                      else f"{t.get('operator', '')} {_fmt_num(t.get('threshold'), 3)}")
        rows.append({
            "name": t["name"].replace("_", " "),
            "metric": t.get("metric") or "",
            "value": "—" if t.get("insufficient") else _fmt_num(t.get("value"), 3),
            "threshold": thresh,
            "state": state,
            "alt": bool(t.get("via_alt")),
            # Why THIS yardstick. A category-routed test measures a different
            # metric than the row above it, and without the reason on the page
            # the panel looks inconsistent rather than deliberate.
            "note": t.get("note") or "",
        })
    return rows


def build_payload(results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]],
                  screened: Dict[str, Any],
                  report_tickers: Optional[set] = None) -> List[Dict[str, Any]]:
    report_tickers = report_tickers or set()
    rows = []
    # Price series are NOT embedded in this payload. They are written to
    # out/series/<MARKET>.json and fetched by the drawer when a row is opened.
    #
    # The previous design carried them inline for a capped subset of rows, and
    # it was wrong in the way that matters: the cap was invisible to the reader
    # and the fallback text told them to change a filter that could not
    # possibly help. Every row now gets a chart; the page stays small because
    # the series live beside it rather than inside it.
    for ticker, r in results.items():
        # Prefer metrics persisted in the result. Rows merged in from the other
        # region's last run have no entry in the live metrics dict.
        m = metrics.get(ticker) or r.get("metrics") or {}
        # A merged row is read back from the store, and the store wrote it with
        # whatever DISPLAY_METRICS looked like on the run that produced it. If
        # that run predates a field being added, the key is ABSENT (not None —
        # the writer always materialises every key it knows about), and the row
        # would quietly show dashes for data the app can compute perfectly
        # well. Detect the older schema and say so, rather than letting a
        # stale row impersonate missing data.
        stale_schema = (not metrics.get(ticker) and bool(m)
                        and any(k not in m for k in DISPLAY_METRICS))
        fw_state = {}
        fw_detail = {}
        for key, _label in FRAMEWORKS:
            f = r.get("frameworks", {}).get(key, {})
            if f.get("ineligible_reason"):
                fw_state[key] = "na"
            elif f.get("passed"):
                fw_state[key] = "pass"
            elif f.get("n_unknown", 0) and f.get("n_passed", 0) == 0:
                fw_state[key] = "unknown"
            else:
                fw_state[key] = "fail"
            eff = f.get("effective_total", f.get("n_total", 0))
            note = f.get("ineligible_reason") or f.get("macro_gate_blocked") or ""
            n_na = f.get("n_not_applicable", 0)
            if f.get("limited_history") and not note:
                note = (f"judged on {m.get('history_years', '?')} years of statements — "
                        f"{f.get('n_insufficient') - n_na} test(s) need a longer window "
                        f"and were excluded, with the bar scaled down to match")
            if n_na and not f.get("ineligible_reason"):
                # A test that does not apply is NOT a data gap, and saying so
                # matters: Lynch's growth band is meaningless on a cyclical, and
                # the app should say that rather than imply a short feed.
                extra = (f"{n_na} test(s) do not apply to a "
                         f"{(m.get('lynch_category_label') or 'name of this kind').lower()} "
                         "and were excluded, with the bar scaled down to match")
                note = f"{note} · {extra}" if note else extra
            fw_detail[key] = {
                "label": f.get("label", key),
                "passed": bool(f.get("passed")),
                "summary": f"{f.get('n_passed', 0)}/{eff} tests"
                           + (f", need {f.get('required')}" if f.get("required") else "")
                           + (f" · {f.get('n_insufficient')} n/a" if f.get("n_insufficient") else ""),
                "note": note,
                "limited": bool(f.get("limited_history")),
                "rank": f.get("combined_rank"),
                "tests": _test_rows(f),
            }

        tech = r.get("technical", {})
        # A sentence-level read of the row, assembled from the same numbers the
        # panels below it show. Built here rather than at screen time so a row
        # merged in from another region's run gets one too — and so the wording
        # can never lag the data it describes by a full refresh cycle.
        try:
            sy = syn.build(r, m, dict(FRAMEWORKS), len(FRAMEWORKS))
        except Exception as e:                       # noqa: BLE001
            sy = {"what": "", "what_source": "", "one_liner": "",
                  "numbers": [f"synopsis unavailable: {e}"]}
        rows.append({
            "ticker": ticker,
            "name": r.get("name") or ticker,
            "market": r.get("market"),
            "market_label": MARKET_LABELS.get(r.get("market"), r.get("market")),
            "sector": r.get("sector") or "—",
            "currency": r.get("currency"),
            "price": _fmt_num(r.get("price")),
            "mcap_usd": _fmt_num(r.get("market_cap_usd"), money=True),
            "mcap_sort": r.get("market_cap_usd") or 0,
            "fw": fw_state,
            "fw_detail": fw_detail,
            "n_passed": r.get("n_frameworks_passed", 0),
            "tech_pass": bool(r.get("technical_passed")),
            # Whether a series EXISTS, so the drawer can tell "not fetched yet"
            # from "this company has too little history to chart".
            "has_series": bool(m.get("spark")),
            # The numbers the charts are drawn from, kept as raw values rather
            # than formatted strings: a chart cannot plot "12.5%".
            "tech_raw": {
                "rsi": m.get("rsi_14"),
                "rsi_label": m.get("rsi_label"),
                "below_52w_high": m.get("pct_below_52w_high"),
                "above_5y_low": m.get("pct_above_5y_low"),
                "in_10y_range": m.get("price_in_10y_range"),
                "r1": m.get("return_1m"), "r3": m.get("return_3m"),
                "r6": m.get("return_6m"), "r12": m.get("return_12m"),
                "worst_month": m.get("worst_month_in_6m"),
                "above_sma200": m.get("price_above_sma200"),
                "golden_cross": m.get("sma50_above_sma200"),
                "macd": m.get("macd_histogram"),
                "vol": m.get("vol20_over_vol50"),
                "rs": m.get("rs_vs_market_index_6m"),
            },
            "candles": {
                "action": m.get("candle_action"),
                "why": m.get("candle_why"),
                "trend": m.get("candle_trend"),
                "stop": m.get("candle_stop"),
                "bullish": m.get("candle_bullish"),
                "bearish": m.get("candle_bearish"),
                "signals": m.get("candle_signals") or [],
                "caveat": m.get("candle_caveat"),
            },
            "tech_detail": {
                "summary": f"{tech.get('n_passed', 0)}/{tech.get('n_total', 0)} tests",
                "tests": _test_rows(tech),
            },
            "rfx": (r.get("reflexive") or {}).get("stage"),
            "rfx_label": (r.get("reflexive") or {}).get("label"),
            "rfx_note": ((r.get("reflexive") or {}).get("warning")
                         or (r.get("reflexive") or {}).get("note") or ""),
            "rfx_late": bool((r.get("reflexive") or {}).get("late")),
            "rfx_evidence": "; ".join((r.get("reflexive") or {}).get("evidence") or []),
            # The stage's practical reading, on the row rather than only in the
            # legend. A badge you have to go and look up is a badge nobody
            # reads twice.
            "rfx_rule": (r.get("reflexive") or {}).get("rule") or "",
            "rfx_action": (r.get("reflexive") or {}).get("action") or "",
            "rfx_group": (r.get("reflexive") or {}).get("group") or "",
            # Two separate facts. `dis_fell` is "dropped more than 30% in six
            # months" — the thing actually asked for. `dis` is the narrower
            # "and the accounts do not explain it". Showing only the second
            # made the first invisible, which is the wrong way round: you
            # cannot judge a shortlist without seeing what it was drawn from.
            "dis_fell": bool(r.get("dislocation")),
            "dis": bool((r.get("dislocation") or {}).get("qualifies")),
            "dis_6m": _fmt_num((r.get("dislocation") or {}).get("return_6m"),
                               1, pct=True),
            "dis_note": _dis_tip(r.get("dislocation")),
            "syn": sy,
            # Lynch's category, on the row rather than buried in his panel:
            # it changes how every other number on the line should be read.
            "cat": m.get("lynch_category"),
            "cat_label": m.get("lynch_category_label"),
            "cat_why": m.get("lynch_category_why") or "",
            "cat_warn": m.get("lynch_peak_earnings_warning") or "",
            # Value or growth: the other axis, and the one that says what has
            # to go right for the price to work out.
            "mun_bucket": m.get("munger_bucket"),
            "mun_bucket_label": m.get("munger_bucket_label") or "",
            "mun_bucket_why": m.get("munger_bucket_why") or "",
            "mun_readings": [x for x in (m.get("munger_inversion_reading"),
                                         m.get("pricing_power_reading"),
                                         m.get("cannibalisation_reading")) if x],
            "sty": m.get("style"),
            "sty_score": m.get("style_score"),
            "sty_why": m.get("style_why") or "",
            "sty_ev": m.get("style_evidence") or [],
            # Buffett's three business tenets, as a label plus its evidence.
            "b": bool(m.get("buffett_b_label")),
            "b_summary": m.get("buffett_tenets_summary") or "",
            "b_detail": m.get("business_tenets") or {},
            "surfaced": bool(r.get("surfaced")),
            "has_report": ticker in report_tickers,
            "themes": r.get("themes") or [],
            "listing": r.get("listing") or "",
            "is_fund": bool(r.get("is_fund")),
            "gates": r.get("gates_failed", []),
            "warnings": (list(r.get("warnings", []))
                         + (["some fields are blank because this row was "
                             "stored by an earlier build — re-run this "
                             "market's refresh to fill them in"]
                            if stale_schema else [])),
            "key_metrics": {
                "P/E": _fmt_num(m.get("pe_ttm")),
                "P/TB": _fmt_num(m.get("price_to_tangible_book")),
                "EV/EBIT": _fmt_num(m.get("ev_to_ebit")),
                "ROE": _fmt_num(m.get("roe_ttm"), 1, pct=True),
                "ROIC 5y": _fmt_num(m.get("roic_5y_avg"), 1, pct=True),
                "D/E": _fmt_num(m.get("debt_to_equity")),
                "FCF yield": _fmt_num(m.get("fcf_yield"), 1, pct=True),
                "PEG": _fmt_num(m.get("peg_ratio")),
                "EPS CAGR 5y": _fmt_num(m.get("eps_cagr_5y"), 1, pct=True),
                "Return 3m": _fmt_num(m.get("return_3m"), 1, pct=True),
                "Return 6m": _fmt_num(m.get("return_6m"), 1, pct=True),
                "Return 12m": _fmt_num(m.get("return_12m"), 1, pct=True),
                "Worst month in 6m": _fmt_num(m.get("worst_month_in_6m"), 1, pct=True),
                # The number alone is not readable: 45 is weakness in a
                # range and a buying opportunity in an uptrend, and the same
                # 28 that means "rebound" sideways means "this is what a
                # downtrend looks like" in a falling one.
                "RSI(14)": (f'{_fmt_num(m.get("rsi_14"), 1)} · '
                            f'{m.get("rsi_label")}'
                            if m.get("rsi_label") else _fmt_num(m.get("rsi_14"), 1)),
                "RSI context": (f'{m.get("rsi_regime", "")} — {m.get("rsi_note", "")}'
                                if m.get("rsi_note") else "—"),
                "RSI divergence": (f'{m.get("rsi_divergence")} — '
                                   f'{m.get("rsi_divergence_note")}'
                                   if m.get("rsi_divergence") else "none"),
                "vs 200d MA": "above" if m.get("price_above_sma200") else "below",
                "RS 6m vs index": _fmt_num(m.get("rs_vs_market_index_6m"), 1, pct=True),
                # Lynch
                "Lynch category": m.get("lynch_category_label") or "—",
                "Style": (m.get("style_label") or "—").title(),
                "Style score": _fmt_num(m.get("style_score"), 0),
                "PEGY": _fmt_num(m.get("pegy_ratio")),
                "Growth+yield ÷ P/E": _fmt_num(m.get("growth_plus_yield_to_pe")),
                "Dividend yield": _fmt_num(m.get("dividend_yield"), 1, pct=True),
                "Net cash/share": _fmt_num(m.get("net_cash_per_share")),
                "P/E ex-cash": _fmt_num(m.get("pe_ex_cash")),
                "EPS vs 5y avg": _fmt_num(m.get("eps_vs_5y_avg")),
                "Short-term debt share": _fmt_num(m.get("short_term_debt_share"),
                                                  0, pct=True),
                "Insider owned": _fmt_num(m.get("insider_ownership"), 1, pct=True),
                "Institutions owned": _fmt_num(m.get("institutional_ownership"),
                                               0, pct=True),
                # Schloss
                "Listing age": (f'{m.get("listing_age_years"):.0f}y'
                                if isinstance(m.get("listing_age_years"), (int, float))
                                else "—"),
                "Above 10y low": _fmt_num(m.get("pct_above_10y_low"), 0, pct=True),
                "In 10y range": _fmt_num(m.get("price_in_10y_range"), 0, pct=True),
                "P/S": _fmt_num(m.get("price_to_sales")),
                # Buffett
                "Owner earnings/share": _fmt_num(m.get("owner_earnings_per_share")),
                "Owner earnings yield": _fmt_num(m.get("owner_earnings_yield"),
                                                 1, pct=True),
                "Owner earnings ÷ NI": _fmt_num(m.get("owner_earnings_to_net_income")),
                "Intrinsic value/share": _fmt_num(m.get("intrinsic_value_per_share")),
                "Margin of safety": _fmt_num(m.get("margin_of_safety"), 0, pct=True),
                "Net margin": _fmt_num(m.get("net_margin_ttm"), 1, pct=True),
                "Gross margin": _fmt_num(m.get("gross_margin_ttm"), 1, pct=True),
                "Debt payoff (years)": _fmt_num(m.get("debt_payoff_years"), 1),
                "Capex ÷ net income": _fmt_num(m.get("capex_to_net_income"),
                                               0, pct=True),
                "SG&A ÷ gross profit": _fmt_num(m.get("sga_to_gross_profit"),
                                                0, pct=True),
                "$1 retained → $ value": _fmt_num(m.get("one_dollar_premise")),
                # Greenblatt
                "Earnings yield (EBIT/EV)": _fmt_num(
                    m.get("ebit_to_ev_greenblatt")
                    if m.get("ebit_to_ev_greenblatt") is not None
                    else m.get("ebit_to_ev"), 1, pct=True),
                "Return on capital": _fmt_num(
                    m.get("ebit_to_invested_capital_greenblatt")
                    if m.get("ebit_to_invested_capital_greenblatt") is not None
                    else m.get("ebit_to_invested_capital"), 0, pct=True),
                "Excess cash": _fmt_num(m.get("excess_cash"), money=True),
                "Inversion score": _fmt_num(m.get("munger_inversion_score"),
                                            0, pct=True),
                "Munger basket": m.get("munger_bucket_label") or "—",
                "Pricing power": ("yes" if m.get("pricing_power") == 1
                                  else ("no" if m.get("pricing_power") == 0
                                        else "—")),
                "Goodwill growth 3y": _fmt_num(m.get("goodwill_growth_3y"),
                                               1, pct=True),
                "Years CFO positive (5y)": _fmt_num(m.get("cfo_positive_share_5y"),
                                                    0, pct=True),
                "Loss years in 5": _fmt_num(m.get("loss_years_in_5"), 0),
            },
        })
    rows.sort(key=lambda x: (-x["n_passed"], -x["mcap_sort"]))
    return rows


def write_series(results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]],
                 out_dir: str) -> Dict[str, int]:
    """Price series as sidecar files, one per market.

    Sharded by market rather than written as one file because the drawer knows
    which market it needs: opening a Hong Kong row should not download the US
    series as well. Written every run for every row that has one, so there is
    no cap and no subset — the reason a chart is missing is now always "this
    company has too little price history", never "the page ran out of budget".
    """
    dest = os.path.join(out_dir, "series")
    os.makedirs(dest, exist_ok=True)
    # GitHub Pages will happily serve a directory of JSON, but a Jekyll build
    # step in front of it will not, and the symptom is a 404 on a file that is
    # plainly in the repo. One empty marker file removes the whole class of
    # failure and costs nothing if it was never going to happen.
    try:
        open(os.path.join(out_dir, ".nojekyll"), "w").close()
    except OSError:
        pass
    by_market: Dict[str, Dict[str, Any]] = {}
    for ticker, r in results.items():
        m = metrics.get(ticker) or r.get("metrics") or {}
        sp = m.get("spark")
        if not sp:
            continue
        by_market.setdefault(r.get("market") or "NA", {})[ticker] = sp
    counts = {}
    for mkt, payload in by_market.items():
        path = os.path.join(dest, f"{mkt}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), default=str)
        counts[mkt] = len(payload)
    # A market that produced no series at all still gets an empty file, so the
    # drawer's fetch returns {} rather than a 404 it has to interpret.
    for mkt in {r.get("market") for r in results.values() if r.get("market")}:
        path = os.path.join(dest, f"{mkt}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            counts.setdefault(mkt, 0)
    log.info("Wrote price series: %s",
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    return counts


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Value + Technical Screener</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2b303b;--tx:#e6e8ec;
--tx2:#a3aab8;--tx3:#6f7789;--acc:#5b9dff;--ok:#3fbf7f;--bad:#e2585e;--warn:#c9a227;
/* Chart series. Three named identities in fixed order, from a palette
   validated for colour-vision deficiency against BOTH surfaces — the dark
   steps are chosen for the dark background, not derived from the light ones
   by inversion. The same three hues the deep-dive report uses, so the two
   views of one stock never need re-learning. */
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f2f6;
--line:#dfe3ea;--tx:#12151b;--tx2:#4d5567;--tx3:#798193;--acc:#1f6feb;--warn:#8a6d10;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.55 ui-sans-serif,
-apple-system,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1400px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.02em}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--tx3);margin-bottom:8px}
.meta{color:var(--tx2);font-size:13px}
.statbar{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:110px}
.stat .v{font-size:20px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat .l{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em;margin-top:1px}
.gate{border-radius:10px;padding:11px 15px;margin:14px 0;font-size:13.5px;border:1px solid var(--line)}
.gate.open{background:rgba(63,191,127,.10);border-color:rgba(63,191,127,.35)}
.gate.closed{background:rgba(226,88,94,.10);border-color:rgba(226,88,94,.4)}
.gate.cyc-defensive{background:rgba(201,162,39,.12);border-color:rgba(201,162,39,.4)}
.gate.cyc-core{background:var(--panel);border-color:var(--line)}
.gate.cyc-opportunistic{background:rgba(63,191,127,.10);border-color:rgba(63,191,127,.35)}
.muted-ink{color:var(--tx3)}
.rfx{font-size:9.5px;font-weight:700;margin-left:5px;padding:1px 5px;border-radius:4px;
background:var(--panel2);color:var(--tx3);border:1px solid var(--line);cursor:help;
letter-spacing:.04em}
.dis{font-size:9.5px;font-weight:700;margin-left:4px;padding:1px 5px;
border-radius:4px;background:rgba(91,157,255,.16);color:var(--acc);
border:1px solid rgba(91,157,255,.45);cursor:help;letter-spacing:.03em}
.dis.expl{background:var(--panel2);color:var(--tx3);border-color:var(--line)}
.rfx.late{background:rgba(226,88,94,.16);color:var(--bad);border-color:rgba(226,88,94,.45)}
.cat{font-size:9.5px;font-weight:700;margin-left:4px;padding:1px 5px;border-radius:4px;
background:var(--panel2);color:var(--tx2);border:1px solid var(--line);cursor:help;
letter-spacing:.04em}
.cat.cyclical{background:rgba(201,162,39,.14);color:var(--warn);border-color:rgba(201,162,39,.4)}
.cat.turnaround{background:rgba(226,88,94,.14);color:var(--bad);border-color:rgba(226,88,94,.4)}
.cat.fast_grower{background:rgba(63,191,127,.13);color:var(--ok);border-color:rgba(63,191,127,.38)}
.cat.warnflag{background:rgba(226,88,94,.2);color:var(--bad);border-color:rgba(226,88,94,.5)}
.sty{font-size:9.5px;font-weight:700;margin-left:4px;padding:1px 5px;border-radius:4px;
background:var(--panel2);color:var(--tx2);border:1px solid var(--line);cursor:help;
letter-spacing:.04em}
.sty.growth,.sty.highgrowth{background:rgba(91,157,255,.16);color:var(--acc);
border-color:rgba(91,157,255,.45)}
.sty.value,.sty.deepvalue{background:rgba(63,191,127,.13);color:var(--ok);
border-color:rgba(63,191,127,.38)}
.blab{font-size:10px;font-weight:800;margin-left:4px;padding:1px 6px;border-radius:4px;
background:rgba(63,191,127,.18);color:var(--ok);border:1px solid rgba(63,191,127,.5);
cursor:help;letter-spacing:.04em}
.dalio{background:var(--panel);border:1px solid var(--line);border-radius:10px;
margin:14px 0;padding:0;overflow:hidden}
.dalio summary{list-style:none;cursor:pointer;padding:12px 15px;display:flex;
gap:12px;align-items:center;flex-wrap:wrap;font-size:13.5px}
.dalio summary::-webkit-details-marker{display:none}
.dalio summary:hover{background:var(--panel2)}
.dalio .body{padding:2px 15px 15px;border-top:1px solid var(--line)}
.stg{font-weight:700;font-size:13px;padding:3px 10px;border-radius:20px;
background:var(--panel2);border:1px solid var(--line);white-space:nowrap}
.alert{font-size:11px;font-weight:700;letter-spacing:.08em;padding:3px 9px;
border-radius:20px;text-transform:uppercase}
.alert.RED{background:rgba(226,88,94,.16);color:var(--bad);
border:1px solid rgba(226,88,94,.45)}
.alert.YELLOW{background:rgba(201,162,39,.16);color:var(--warn);
border:1px solid rgba(201,162,39,.45)}
.alert.GREEN{background:rgba(63,191,127,.14);color:var(--ok);
border:1px solid rgba(63,191,127,.4)}
.alert.GREY{background:var(--panel2);color:var(--tx3);border:1px solid var(--line)}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
gap:10px;margin-top:12px}
.dcard{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
padding:10px 12px}
.dcard h4{margin:0 0 7px;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
color:var(--tx3);font-weight:600}
.drow{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
font-size:12.5px;border-bottom:1px solid var(--line)}
.drow:last-child{border-bottom:none}
.drow .k{color:var(--tx2)}
.drow .v{text-align:right;font-variant-numeric:tabular-nums}
.pill{font-size:10.5px;padding:1px 7px;border-radius:20px;border:1px solid var(--line);
color:var(--tx3);white-space:nowrap}
.pill.hot{background:rgba(226,88,94,.14);color:var(--bad);border-color:rgba(226,88,94,.4)}
.pill.warm{background:rgba(201,162,39,.14);color:var(--warn);border-color:rgba(201,162,39,.4)}
.pill.cool{background:rgba(63,191,127,.12);color:var(--ok);border-color:rgba(63,191,127,.35)}
.stages{display:flex;gap:3px;margin:10px 0 4px;flex-wrap:wrap}
.stages i{flex:1;min-width:56px;font-style:normal;font-size:10px;text-align:center;
padding:5px 3px;border-radius:5px;background:var(--panel2);color:var(--tx3);
border:1px solid var(--line)}
.stages i.on{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:700}
.dnote{font-size:12.5px;color:var(--tx2);margin-top:11px;line-height:1.6}
.cmd table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
.cmd th{text-align:left;padding:7px 8px;color:var(--tx3);font-size:10px;
letter-spacing:.07em;text-transform:uppercase;border-bottom:1px solid var(--line)}
.cmd td{padding:7px 8px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
.cmd tr:last-child td{border-bottom:none}
.cmd .nm{font-weight:600;font-variant-numeric:normal}
.cmd .sym{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--tx2)}
.cmd .r{text-align:right}
.up{color:var(--ok)} .dn{color:var(--bad)}
.kind{font-size:9.5px;padding:1px 6px;border-radius:20px;border:1px solid var(--line);
color:var(--tx3);text-transform:uppercase;letter-spacing:.05em}
.kind.equity{background:rgba(226,88,94,.13);color:var(--bad);border-color:rgba(226,88,94,.4)}
.kind.etn{background:rgba(201,162,39,.14);color:var(--warn);border-color:rgba(201,162,39,.4)}
.kind.physical{background:rgba(63,191,127,.12);color:var(--ok);border-color:rgba(63,191,127,.35)}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:16px 0 10px;
padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:10px}
.fl{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em;margin-right:2px}
.chip{font-size:12.5px;padding:4px 11px;border-radius:20px;border:1px solid var(--line);
background:transparent;color:var(--tx2);cursor:pointer;font-weight:500}
.chip:hover{border-color:var(--acc);color:var(--tx)}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
.sep{width:1px;height:20px;background:var(--line);margin:0 4px}
.searchwrap{display:flex;align-items:center;gap:7px;flex:1;min-width:230px}
input[type=search]{background:var(--panel2);border:1px solid var(--line);border-radius:7px;
padding:7px 11px;color:var(--tx);font-size:13.5px;flex:1;min-width:170px}
input[type=search]:focus{outline:none;border-color:var(--acc);background:var(--panel)}
.linkbtn{font-size:12px;padding:5px 11px;border-radius:7px;border:1px solid var(--line);
background:transparent;color:var(--tx2);cursor:pointer;white-space:nowrap}
.linkbtn:hover{border-color:var(--acc);color:var(--tx)}
.searchnote{font-size:11.5px;color:var(--tx3);margin:-2px 0 0 2px;width:100%}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 8px;color:var(--tx3);font-size:10.5px;letter-spacing:.06em;
text-transform:uppercase;border-bottom:1px solid var(--line);white-space:nowrap;font-weight:600}
th.c,td.c{text-align:center}
td{padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
tr.row{cursor:pointer}
tr.row:hover{background:var(--panel2)}
.tk{font-weight:650;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.nm{color:var(--tx2);font-size:12px;max-width:210px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk{font-size:10.5px;padding:2px 7px;border-radius:4px;background:var(--panel2);color:var(--tx2);font-weight:600}
.num{font-variant-numeric:tabular-nums;text-align:right}
.dot{display:inline-block;width:19px;height:19px;line-height:19px;border-radius:5px;
font-size:11px;font-weight:700;text-align:center}
.dot.pass{background:rgba(63,191,127,.20);color:var(--ok)}
.dot.fail{background:var(--panel2);color:var(--tx3)}
.dot.unknown{background:rgba(201,162,39,.18);color:var(--warn)}
.dot.na{background:transparent;color:var(--tx3);opacity:.4}
.cnt{font-weight:700;font-variant-numeric:tabular-nums}
tr.detail>td{background:var(--panel);padding:0;border-bottom:2px solid var(--line)}
.dwrap{padding:16px 18px}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.dcard{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.dcard h4{margin:0 0 7px;font-size:12.5px;display:flex;justify-content:space-between;align-items:center}
.dcard .badge{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.badge.pass{background:rgba(63,191,127,.2);color:var(--ok)}
.badge.fail{background:rgba(226,88,94,.13);color:var(--bad)}
.trow{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;padding:2.5px 0;border-top:1px solid var(--line)}
.trow:first-of-type{border-top:none}
.tn{color:var(--tx2);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tv{font-variant-numeric:tabular-nums;white-space:nowrap}
.tv.pass{color:var(--ok)}.tv.fail{color:var(--bad)}.tv.unknown{color:var(--warn)}
.tv.na{color:var(--tx3);opacity:.75;font-style:italic}
.dcard .lim{font-size:10.5px;color:var(--tx3);font-style:italic;margin-bottom:5px}
.kmet{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:7px;margin-bottom:14px}
.kmet div{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:7px 9px}
.kmet .kl{font-size:9.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em}
.kmet .kv{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.cand{display:flex;flex-direction:column;gap:7px}
.cand .sig{display:grid;grid-template-columns:82px 1fr auto;gap:9px;align-items:start;
font-size:11.5px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.cand .sig:last-child{border-bottom:none;padding-bottom:0}
.cand .sig .d{color:var(--tx3);font-variant-numeric:tabular-nums}
.cand .sig .nm{font-weight:600}
.cand .sig .rl{color:var(--tx3);font-size:10.5px;line-height:1.45}
.cand .sig .rep{font-size:9.5px;color:var(--tx3);border:1px solid var(--line);
border-radius:4px;padding:0 4px;margin-left:4px;white-space:nowrap}
.cand .st{font-size:9.5px;font-weight:700;padding:1px 6px;border-radius:4px;
white-space:nowrap;letter-spacing:.03em;text-transform:uppercase}
.st.confirmed{background:rgba(63,191,127,.16);color:var(--ok);border:1px solid rgba(63,191,127,.4)}
.st.unconfirmed{background:rgba(201,162,39,.16);color:var(--warn);border:1px solid rgba(201,162,39,.4)}
.st.failed{background:var(--panel);color:var(--tx3);border:1px solid var(--line)}
/* The neutral state. Escaped because the class is literally "n/a" — an
   indecision candle has no direction, so "confirmed" is the wrong question
   rather than an unanswered one, and the chip should not look like a pending
   one. */
.st.n\/a{background:var(--panel);color:var(--tx3);border:1px dashed var(--line)}
.verdict{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:9px}
.verdict .big{font-size:15px;font-weight:700}
.verdict .big.buy{color:var(--ok)} .verdict .big.sell{color:var(--bad)}
.verdict .big.wait,.verdict .big.watch{color:var(--warn)}
/* The trend as of the LATEST bar — the answer to "what is it doing now",
   which the action alone does not give: a sell signal in an uptrend and one
   in a downtrend are different trades. */
.verdict .tchip{font-size:9.5px;font-weight:700;letter-spacing:.03em;
text-transform:uppercase;padding:1px 6px;border-radius:4px;
border:1px solid var(--line);color:var(--tx3);background:var(--panel)}
.verdict .tchip.t-up{color:var(--ok);border-color:rgba(63,191,127,.4)}
.verdict .tchip.t-down{color:var(--bad);border-color:rgba(214,88,72,.4)}
.pchart,.cchart{position:relative}
.pchart svg,.cchart svg{width:100%;height:auto;display:block}
.pchart .hit,.cchart .hit{cursor:crosshair}
.ptip{position:absolute;top:6px;pointer-events:none;background:var(--panel);
border:1px solid var(--line);border-radius:7px;padding:6px 9px;font-size:11.5px;
line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.22);white-space:nowrap;z-index:3}
.ptip b{display:block;font-variant-numeric:tabular-nums;margin-bottom:2px}
.ptip span{display:flex;align-items:center;gap:5px;color:var(--tx2);
font-variant-numeric:tabular-nums}
.ptip i{width:8px;height:8px;border-radius:2px;display:inline-block;flex:none}
.chartgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
gap:12px;margin-bottom:14px;align-items:start}
.chart{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:11px 12px}
.chart h5{margin:0 0 8px;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
color:var(--tx3);font-weight:600;display:flex;justify-content:space-between;gap:8px}
.chart h5 span{text-transform:none;letter-spacing:0;color:var(--tx2);font-weight:500;
font-variant-numeric:tabular-nums}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart .cap{font-size:10.5px;color:var(--tx3);margin-top:7px;line-height:1.5}
.chart.wide{grid-column:1 / -1}
/* The price strip uses preserveAspectRatio="none" on a 260x68 viewBox. Without
   an explicit height it inflates to a third of the viewport width, and a
   two-year chart does not need 340 pixels of vertical space to be read. */
.chart.wide>svg{height:150px}
.tstrip{display:flex;flex-direction:column;gap:5px}
.tbar{display:grid;grid-template-columns:150px 1fr 96px;gap:10px;align-items:center;font-size:11.5px}
.tbar .tl{color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tbar .tt{text-align:right;font-variant-numeric:tabular-nums;color:var(--tx3)}
.tbar .track{position:relative;height:9px;background:var(--panel);border:1px solid var(--line);
border-radius:5px;overflow:hidden}
.tbar .fill{position:absolute;top:0;bottom:0;border-radius:4px}
.tbar .fill.ok{background:rgba(63,191,127,.55)}
.tbar .fill.no{background:rgba(226,88,94,.45)}
.tbar .fill.na{background:var(--line)}
.tbar .mark{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--tx3)}
.chart .lgd{font-size:10px;color:var(--tx3);display:flex;gap:10px;margin-top:6px;flex-wrap:wrap}
.chart .lgd i{font-style:normal;display:inline-flex;align-items:center;gap:4px}
.chart .lgd b{display:inline-block;width:14px;height:2px;border-radius:2px}
/* The legend swatches for the candle chart carry the SHAPE difference, not
   just the colour — a reader who cannot separate green from red still learns
   the rule here, and the chart then obeys it. */
.chart .lgd b.ck{width:9px;height:13px;border-radius:1px;border:1.4px solid}
.chart .lgd b.ck.up{border-color:var(--ok);background:var(--panel2)}
.chart .lgd b.ck.dn{border-color:var(--bad);background:var(--bad)}
.chart .lgd b.ck.tri{width:0;height:0;border:none;border-left:6px solid transparent;
border-right:6px solid transparent;border-bottom:9px solid var(--ok);border-radius:0}
.syn{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:0 9px 9px 0;padding:12px 15px;margin-bottom:13px;font-size:13px;line-height:1.62}
.syn .what{color:var(--tx2);margin-bottom:8px}
.syn .what b{color:var(--tx)}
.syn p{margin:0 0 6px}
.syn p:last-child{margin-bottom:0}
.syn .src{font-size:10.5px;color:var(--tx3);margin-top:9px;font-style:italic}
.warn{background:rgba(201,162,39,.12);border:1px solid rgba(201,162,39,.3);border-radius:7px;
padding:8px 11px;margin-bottom:12px;font-size:12px;color:var(--warn)}
.note{background:var(--panel);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:11px 15px;margin:14px 0;font-size:13px;color:var(--tx2)}
.tagrow{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tag.lst{border-color:var(--acc);color:var(--acc)}
.tag{font-size:10.5px;padding:2px 8px;border-radius:20px;background:var(--panel2);
 color:var(--tx2);border:1px solid var(--line)}
.ddbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.ddbtn{font-size:12.5px;font-weight:600;padding:6px 13px;border-radius:7px;
 border:1px solid var(--line);color:var(--tx2);text-decoration:none;white-space:nowrap}
.ddbtn:hover{border-color:var(--acc);color:var(--tx)}
.ddbtn.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.ddhint{font-size:11.5px;color:var(--tx3)}
.dd{color:var(--acc);font-size:11px;margin-left:5px}
.empty{text-align:center;padding:50px;color:var(--tx3)}
.foot{color:var(--tx3);font-size:12px;margin-top:32px;padding-top:14px;border-top:1px solid var(--line)}
</style></head><body><div class="wrap">

<div class="eyebrow">__REGION__ refresh · run __RUNID__</div>
<h1>Value + Technical Screener</h1>
<div class="meta">S&amp;P 500 + Nasdaq 100 · Nikkei 225 · SGX · HKEX · SET · IDX &nbsp;·&nbsp; generated __TS__ &nbsp;·&nbsp; <a href="deepdive/">Deep dives &rarr;</a></div>

<div class="statbar" id="statbar"></div>
__GATE__

<div class="filters">
  <span class="fl">Market</span>
  <button class="chip on" data-mkt="ALL">All</button>
  __MKTCHIPS__
  <span class="sep"></span>
  <span class="fl">Must pass</span>
  __FWCHIPS__
  __THEMEROW__
  __LISTINGROW__
  <span class="sep"></span>
  <button class="chip" id="techOnly">Technical pass</button>
  <button class="chip" id="disOnly" title="Every name down more than 30% over six months, whatever the reason">Fell &gt;30% (6m)</button>
  <button class="chip" id="disQual" title="Of those, the ones whose last published accounts do NOT explain the fall">&hellip; accounts intact</button>
  <button class="chip" id="valOnly" title="Priced on assets in place and current cash — cheap against earnings, book or cash flow relative to the rest of this universe">Value</button>
  <button class="chip" id="grwOnly" title="Priced on what it will earn rather than on what it owns — technology and related industries carry a growth tilt by sector">Growth</button>
  <button class="chip" id="bOnly" title="Buffett's three business tenets: inside the circle of competence you declared, showing the footprint of a durable moat, and with a consistent operating history">B &mdash; business tenets</button>
  <button class="chip" id="rfxOnly" title="Soros stage DE or EF — price rising through an earnings setback, or expectations run far ahead of reality">Reflexive risk</button>
  <button class="chip on" id="surfOnly">Surfaced only</button>
  <span class="sep"></span>
  <span class="searchwrap">
    <input type="search" id="q" placeholder="Search ticker or name — e.g. MSFT, DBS, 0700">
    <button class="linkbtn" id="copylink">Copy link to this view</button>
  </span>
  <div class="searchnote">Search looks across the whole universe, including names that
    failed every screen — so you can always check a stock you already hold.</div>
</div>

<table id="tbl"><thead><tr>
  <th>Ticker</th><th>Name</th><th>Market</th><th class="num">Price</th><th class="num">Mkt cap</th>
  __FWHEAD__
  <th class="c">Tech</th><th class="c">Passed</th>
</tr></thead><tbody id="tbody"></tbody></table>
<div class="empty" id="empty" style="display:none">No names match these filters.</div>

<div class="note"><b>Reading this screen.</b> A green cell means the name cleared that
framework's tests at the thresholds in <code>config/thresholds.yml</code>. Amber means the
test could not be evaluated — data was missing — which under the default strict setting
counts as a fail. Faded means the framework doesn't apply (Greenblatt and Klarman skip
financials, REITs and utilities, where EV/EBIT and return-on-capital are meaningless).
Click any row to see exactly which test failed and on what value.</div>

<div class="foot">Fundamentals: SEC EDGAR XBRL (US, audited filings) · Yahoo Finance (SGX, HKEX, SET, IDX).
Prices: Yahoo Finance, split and dividend adjusted, computed on each market's own trading calendar.
Macro: FRED. Not investment advice — a screen is a starting point for research, not a conclusion.<br>
Price series written this run: __SERIES__. Each row's chart is fetched from
<code>series/&lt;market&gt;.json</code> when you open it — a market shows charts only after its own
refresh has run.</div>
</div>

<script>
const DATA = __DATA__;
const FWS = __FWS__;
const CATS = __CATS__;
const STY = __STY__;
const WF_URL = __WFURL__;
const ISSUE_URL = __ISSUEURL__;
let fMkt="ALL", fFw=new Set(), fTech=false, fSurf=true, fQ="", fTheme="ALL", fList="ALL";
let fRfx=false;   // Soros stage DE/EF only
let fDis=false;   // fell >30% in 6m
let fDisQ=false;  // ...and the accounts do not explain it
let fB=false;     // Buffett's three business tenets
let fVal=false;   // value stocks
let fGrw=false;   // growth stocks

function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

function visible(){
  // A search must reach the WHOLE universe, not just what survived the filters.
  // Otherwise looking up a stock you hold returns nothing and you can't tell
  // "it failed the screens" from "it isn't covered" — two very different facts.
  const searching = fQ.trim().length > 0;
  return DATA.filter(r=>{
    if(searching){
      const q=fQ.trim().toLowerCase();
      return r.ticker.toLowerCase().includes(q) || r.name.toLowerCase().includes(q);
    }
    if(fB && !r.b) return false;
    if(fVal && !(r.sty==='value'||r.sty==='deep value')) return false;
    if(fGrw && !(r.sty==='growth'||r.sty==='high growth')) return false;
    if(fRfx && !r.rfx_late) return false;
    if(fDis && !r.dis_fell) return false;
    if(fDisQ && !r.dis) return false;
    if(fSurf && !r.surfaced) return false;
    if(fMkt!=="ALL" && r.market!==fMkt) return false;
    if(fTheme!=="ALL" && !(r.themes||[]).includes(fTheme)) return false;
    if(fList!=="ALL" && (r.listing||"")!==fList) return false;
    if(fTech && !r.tech_pass) return false;
    for(const f of fFw){ if(r.fw[f]!=="pass") return false; }
    return true;
  });
}

// ---- shareable deep links -------------------------------------------------
function syncUrl(){
  const p=new URLSearchParams();
  if(fMkt!=="ALL") p.set('market',fMkt);
  if(fTheme!=="ALL") p.set('theme',fTheme);
  if(fFw.size) p.set('pass',[...fFw].join(','));
  if(fTech) p.set('tech','1');
  if(fB) p.set('b','1');
  if(fVal) p.set('style','value');
  if(fGrw) p.set('style','growth');
  if(!fSurf) p.set('all','1');
  if(fQ.trim()) p.set('q',fQ.trim());
  const s=p.toString();
  history.replaceState(null,'',s?('?'+s):location.pathname);
}

function loadUrl(){
  const p=new URLSearchParams(location.search);
  const th=p.get('theme');
  if(th){ fTheme=th;
    document.querySelectorAll('[data-theme]').forEach(b=>
      b.classList.toggle('on', b.dataset.theme===th)); }
  const m=p.get('market');
  if(m){ fMkt=m;
    document.querySelectorAll('[data-mkt]').forEach(b=>
      b.classList.toggle('on', b.dataset.mkt===m)); }
  const pass=p.get('pass');
  if(pass) pass.split(',').filter(Boolean).forEach(k=>{ fFw.add(k);
    const b=document.querySelector(`[data-fw="${k}"]`); if(b) b.classList.add('on'); });
  if(p.get('tech')==='1'){ fTech=true; document.getElementById('techOnly').classList.add('on'); }
  if(p.get('b')==='1'){ fB=true; document.getElementById('bOnly').classList.add('on'); }
  const sty=p.get('style');
  if(sty==='value'){ fVal=true; document.getElementById('valOnly').classList.add('on'); }
  if(sty==='growth'){ fGrw=true; document.getElementById('grwOnly').classList.add('on'); }
  if(p.get('all')==='1'){ fSurf=false; document.getElementById('surfOnly').classList.remove('on'); }
  const q=p.get('q');
  if(q){ fQ=q; document.getElementById('q').value=q; }
}

function stats(rows){
  const byMkt={};
  rows.forEach(r=>byMkt[r.market_label]=(byMkt[r.market_label]||0)+1);
  const tech=rows.filter(r=>r.tech_pass).length;
  let h=`<div class="stat"><div class="v">${rows.length}</div><div class="l">Names shown</div></div>`;
  h+=`<div class="stat"><div class="v">${tech}</div><div class="l">Technical pass</div></div>`;
  // Universe size makes an empty result readable: 0 of 700 is a strict filter,
  // 0 of 25 means the last run only covered a fraction of the universe.
  h+=`<div class="stat"><div class="v">${DATA.length}</div><div class="l">In universe</div></div>`;
  Object.keys(byMkt).sort().forEach(k=>{
    h+=`<div class="stat"><div class="v">${byMkt[k]}</div><div class="l">${esc(k)}</div></div>`;});
  document.getElementById('statbar').innerHTML=h;
}

// ---- technical timing, drawn rather than listed -------------------------
// Everything below renders from values the row already carries. The price
// series is present only for rows worth opening (see MAX_SPARKLINES in
// render.py); every other chart here needs nothing but scalars, so a row
// without a series still gets a readable panel rather than an empty box.
// ---- the price chart -----------------------------------------------------
// Close with its 50- and 200-day averages: the same three series, the same
// three hues and the same direct labelling as the deep-dive report, so moving
// between the two never means re-learning a chart.
//
// Colour is assigned by IDENTITY — three named series, in fixed order, from a
// palette validated for colour-vision deficiency in both light and dark mode.
// It is never assigned by rank, so filtering the table cannot repaint a line.
// Aqua sits under 3:1 contrast on the light surface, so all three lines carry
// visible end labels: identity is never colour alone here.
function isNum(v){ return typeof v==='number' && isFinite(v); }
function fmtPx(v){
  const a=Math.abs(v);
  return a>=1000?v.toFixed(0):(a>=10?v.toFixed(1):v.toFixed(2));
}


// ---- series loader -------------------------------------------------------
// One fetch per market, cached for the session. The series live in
// series/<MARKET>.json beside this page rather than inside it: embedding ~900
// of them would put well over a megabyte of numbers into the HTML for charts
// most visitors never open, and capping which rows got one — the previous
// design — meant the reader could not tell a missing chart from a missing
// budget.
const SERIES_CACHE = {};
function loadSeries(market){
  if(!market) return Promise.resolve({});
  if(SERIES_CACHE[market]) return SERIES_CACHE[market];
  SERIES_CACHE[market] = fetch('series/' + encodeURIComponent(market) + '.json')
    .then(r => r.ok ? r.json() : {})
    .catch(() => ({}));
  return SERIES_CACHE[market];
}

function fillPriceCharts(root){
  (root || document).querySelectorAll('[data-series-for]').forEach(card => {
    if(card.dataset.filled) return;
    card.dataset.filled = '1';
    const t = card.dataset.seriesFor, mkt = card.dataset.market;
    const slot = card.querySelector('.pslot');
    const head = card.querySelector('.phead');
    loadSeries(mkt).then(all => {
      const sp = all[t];
      if(!sp){
        // Name the exact URL. "Could not load" sends someone hunting; a link
        // that either shows JSON or a 404 answers the question in one click.
        const u = new URL('series/' + mkt + '.json', location.href).href;
        slot.innerHTML = '<div class="cap"><b>No price series for this row yet.</b><br>'
          + 'The chart reads <a href="' + esc(u) + '" target="_blank" rel="noopener">'
          + esc(u) + '</a>, written by the refresh that covers ' + esc(mkt) + '.<br>'
          + 'If that link 404s, this market has not run since the feature was '
          + 'added — run its refresh workflow. If it returns JSON but this row '
          + 'is missing from it, the company has under sixty trading days of '
          + 'price history.</div>';
        return;
      }
      const row = DATA.find(x => x.ticker === t) || {};
      const chg = ((sp.last / sp.first - 1) * 100);
      head.textContent = `${sp.years}y · ${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%`
        + ` · ${row.currency || ''} ${sp.lo} – ${sp.hi}`;
      slot.innerHTML = priceChart(sp, row.currency);
      // The candlestick view lives in its own slot in the same card fetch, so
      // opening a drawer costs one request rather than two.
      const cslot = card.parentElement
        && card.parentElement.querySelector('[data-candles-for="' + t + '"] .cslot');
      if(cslot){
        const ch = sp.ohlc ? candleChart(sp.ohlc, row.currency) : '';
        cslot.innerHTML = ch || '<div class="cap">This row has no daily bars '
          + 'stored yet &mdash; the candlestick view needs at least twenty '
          + 'sessions, and it is written by the same refresh that writes the '
          + 'price series above.</div>';
      }
    });
  });
}

function priceChart(sp, ccy){
  if(!sp || !sp.px || sp.px.length < 8) return '';
  const px=sp.px, m50=sp.ma50||[], m200=sp.ma||[];
  const all=px.concat(m50.filter(isNum)).concat(m200.filter(isNum));
  let lo=Math.min.apply(null,all), hi=Math.max.apply(null,all);
  const span=(hi-lo)||1; lo-=span*0.06; hi+=span*0.06;
  const W=880,H=300,padL=8,padR=78,padT=14,padB=26;
  const pw=W-padL-padR, ph=H-padT-padB;
  const X=i=>padL+i*pw/Math.max(px.length-1,1);
  const Y=v=>padT+ph-(v-lo)/(hi-lo)*ph;
  // The series are normalised to 100 at the left edge; the axis has to speak
  // in money, so every label converts back through the first real close.
  const real=v=>v/100*sp.first;

  let grid='';
  for(let k=0;k<5;k++){
    const v=lo+(hi-lo)*k/4, y=Y(v);
    grid+=`<line x1="${padL}" y1="${y.toFixed(1)}" x2="${(padL+pw).toFixed(1)}"
      y2="${y.toFixed(1)}" stroke="var(--line)" stroke-width="1"/>
      <text x="${(padL+pw+6).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="10.5"
        fill="var(--tx3)">${fmtPx(real(v))}</text>`;
  }

  function path(vals){
    let d='', pen=false;
    vals.forEach((v,i)=>{ if(!isNum(v)){pen=false;return;}
      d+=(pen?'L':'M')+X(i).toFixed(1)+','+Y(v).toFixed(1)+' '; pen=true; });
    return d.trim();
  }
  function endOf(vals){
    for(let i=vals.length-1;i>=0;i--)
      if(isNum(vals[i])) return {x:X(i), y:Y(vals[i])};
    return null;
  }

  const SER=[[px,1,'Close'],[m50,2,'50-day'],[m200,3,'200-day']];
  let lines='', labels='';
  // Drawn slowest-first so the close, which is the thing being read, is never
  // buried under a moving average.
  [[m200,3,'200-day'],[m50,2,'50-day'],[px,1,'Close']].forEach(s=>{
    lines+=`<path d="${path(s[0])}" fill="none" stroke="var(--series-${s[1]})"
      stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  });
  // The three end labels sit at their own line's last value, which means they
  // land on top of each other exactly when the averages converge — the moment
  // the chart is most worth reading. Dodge them apart, dot still on the line.
  const placed=[];
  SER.forEach(s=>{
    const e=endOf(s[0]); if(!e) return;
    let ty=e.y-9;
    while(placed.some(v=>Math.abs(v-ty)<12)) ty+=12;
    ty=Math.max(padT+10,Math.min(padT+ph-2,ty));
    placed.push(ty);
    labels+=`<circle cx="${e.x.toFixed(1)}" cy="${e.y.toFixed(1)}" r="4"
        fill="var(--series-${s[1]})" stroke="var(--panel2)" stroke-width="2"/>
      <text x="${(e.x-7).toFixed(1)}" y="${ty.toFixed(1)}" font-size="11"
        font-weight="600" text-anchor="end" stroke="var(--panel2)"
        stroke-width="3" paint-order="stroke"
        fill="var(--series-${s[1]})">${s[2]}</text>`;
  });

  let xlab='';
  [[0,sp.d0,'start'],[Math.floor((px.length-1)/2),sp.dmid,'middle'],
   [px.length-1,sp.d1,'end']].forEach(t=>{
    if(!t[1]) return;
    xlab+=`<text x="${X(t[0]).toFixed(1)}" y="${H-7}" font-size="10.5"
      fill="var(--tx3)" text-anchor="${t[2]}">${esc(t[1].slice(0,7))}</text>`;
  });

  // The hover layer. A line chart without one makes the reader guess at values
  // between the three axis labels, which is exactly the guess the chart exists
  // to remove.
  const data=JSON.stringify({px:px,m50:m50,m200:m200,first:sp.first,
    d0:sp.d0,d1:sp.d1,ccy:ccy||''});
  return `<div class="pchart"><svg viewBox="0 0 ${W} ${H}" role="img"
    data-series='${data.replace(/'/g,"&#39;")}'
    aria-label="Close with 50-day and 200-day moving averages over ${sp.years} years">
    ${grid}${lines}${labels}${xlab}
    <g class="cross" style="display:none">
      <line y1="${padT}" y2="${padT+ph}" stroke="var(--tx3)" stroke-width="1"
        stroke-dasharray="3 3"/>
      <circle r="3.5" fill="var(--series-1)" stroke="var(--panel2)" stroke-width="2"/>
    </g>
    <rect class="hit" x="${padL}" y="${padT}" width="${pw}" height="${ph}"
      fill="transparent"/>
  </svg><div class="ptip" style="display:none"></div></div>`;
}

// The candlestick chart. Sixty daily bars — the same window the pattern
// scanner reads, so the marks below can never point at a candle the chart does
// not draw.
//
// Rising and falling candles are told apart TWICE: by colour, and by whether
// the body is hollow or filled. That is not decoration. Green against red is
// the one pair that can never pass a colourblind check — measured, the app's
// own green and red separate by ΔE 7 under deuteranopia, which is inside the
// band that is only legal WITH a second, non-colour encoding. Hollow-up and
// filled-down is that encoding, and it happens to be the original Japanese
// convention, so it costs a reader nothing to learn.
function candleChart(o, ccy){
  if(!o || !o.c || o.c.length < 10) return '';
  const n=o.c.length;
  let lo=Math.min.apply(null,o.l), hi=Math.max.apply(null,o.h);
  const span=(hi-lo)||1; lo-=span*0.08; hi+=span*0.12;   // headroom for labels
  const W=880,H=300,padL=8,padR=78,padT=16,padB=26;
  const pw=W-padL-padR, ph=H-padT-padB;
  const step=pw/n, bw=Math.max(2.5,Math.min(11,step*0.62));
  const X=i=>padL+step*(i+0.5);
  const Y=v=>padT+ph-(v-lo)/(hi-lo)*ph;

  let grid='';
  for(let k=0;k<5;k++){
    const v=lo+(hi-lo)*k/4, y=Y(v);
    grid+=`<line x1="${padL}" y1="${y.toFixed(1)}" x2="${(padL+pw).toFixed(1)}"
      y2="${y.toFixed(1)}" stroke="var(--line)" stroke-width="1"/>
      <text x="${(padL+pw+6).toFixed(1)}" y="${(y+3.5).toFixed(1)}" font-size="10.5"
        fill="var(--tx3)">${fmtPx(v)}</text>`;
  }

  let sticks='';
  for(let i=0;i<n;i++){
    const up=o.c[i]>=o.o[i], col=up?'var(--ok)':'var(--bad)';
    const yh=Y(o.h[i]), yl=Y(o.l[i]);
    const yo=Y(o.o[i]), yc=Y(o.c[i]);
    const top=Math.min(yo,yc);
    // A doji has no body at all. Floored to one pixel so the bar does not
    // vanish — a session that opened and closed at the same price is
    // information, and an invisible candle reads as missing data.
    const hgt=Math.max(1,Math.abs(yc-yo));
    sticks+=`<line x1="${X(i).toFixed(1)}" y1="${yh.toFixed(1)}"
        x2="${X(i).toFixed(1)}" y2="${yl.toFixed(1)}" stroke="${col}"
        stroke-width="1"/>
      <rect x="${(X(i)-bw/2).toFixed(1)}" y="${top.toFixed(1)}"
        width="${bw.toFixed(1)}" height="${hgt.toFixed(1)}"
        fill="${up?'var(--panel2)':col}" stroke="${col}" stroke-width="1.2"/>`;
  }

  // Where the stop line will sit, so a pattern label can dodge it. The two
  // collide by construction rather than by accident: the stop IS the low of
  // the bullish pattern, so its marker and its level are at the same height.
  const liveStop=((o.marks||[])[0]||{}).stop;
  const stopY=(liveStop!==null&&liveStop!==undefined&&liveStop>lo&&liveStop<hi)
    ? Y(liveStop) : null;

  // Pattern markers. Directional only, and each is direct-labelled rather than
  // relying on a colour key — the label is what makes the chart readable to
  // someone who cannot separate the two hues at all.
  // Every marker is drawn, but each pattern NAME is spelled out only once.
  // A stock that gapped up five sessions running prints five rising windows,
  // and five copies of the same words stacked on each other is a smear, not
  // five facts. The triangles still show where each one occurred.
  let marks='', taken=[], named={};
  (o.marks||[]).slice(0,6).forEach(mk=>{
    const i=mk.i; if(i<0||i>=n) return;
    const label=named[mk.n]?'':mk.n;
    named[mk.n]=1;
    const bull=mk.dir==='bullish';
    const col=bull?'var(--ok)':'var(--bad)';
    const y=Math.max(padT+8,Math.min(padT+ph-8,
      bull?Y(o.l[i])+13:Y(o.h[i])-13));
    const x=X(i);
    // Nudge a label that would sit on top of one already placed.
    let ly=bull?y+12:y-12, tries=0;
    while(taken.some(t=>Math.abs(t.x-x)<58 && Math.abs(t.y-ly)<11) && tries<4){
      ly+=bull?11:-11; tries++;
    }
    // Keep the label inside the plot, and off the stop line. A bullish mark
    // sits at the low of its candle and the stop IS that low, so the two
    // collide by construction rather than by accident. Where there is no room
    // on the marker's own side, the label moves to the other side of the
    // candle — clamping instead would park it ON the axis or ON the stop line,
    // and a label squeezed into another element is worse than one that moved.
    const ceilY=padT+9, floorY=padT+ph-3;
    if(stopY!==null && Math.abs(ly-stopY)<11) ly=bull?stopY+14:stopY-12;
    // Flipping to the other side of the candle solves the vertical collision
    // and creates a horizontal one: the label is centred on the candle it
    // names, so above the high it lands ON the candle. Flipped labels step
    // aside instead, to whichever side has more room.
    let lx=x, anchor='middle';
    if((bull && ly>floorY) || (!bull && ly<ceilY)){
      ly=bull?Y(o.h[i])-7:Y(o.l[i])+13;
      const room=x-padL>pw*0.5;
      lx=room?x-bw/2-4:x+bw/2+4;
      anchor=room?'end':'start';
    }
    ly=Math.max(ceilY,Math.min(floorY,ly));
    if(label) taken.push({x:lx,y:ly});
    const tri=bull
      ? `${x},${(y-7).toFixed(1)} ${(x-5).toFixed(1)},${(y+1).toFixed(1)} ${(x+5).toFixed(1)},${(y+1).toFixed(1)}`
      : `${x},${(y+7).toFixed(1)} ${(x-5).toFixed(1)},${(y-1).toFixed(1)} ${(x+5).toFixed(1)},${(y-1).toFixed(1)}`;
    const solid=mk.st==='confirmed';
    marks+=`<polygon points="${tri}" fill="${solid?col:'var(--panel2)'}"
        stroke="${col}" stroke-width="1.5"
        ${mk.st==='failed'?'opacity=".45"':''}/>
      <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="9.5"
        font-weight="600" text-anchor="${anchor}" fill="${col}"
        stroke="var(--panel2)" stroke-width="3" paint-order="stroke"
        ${mk.st==='failed'?'opacity=".55"':''}>${esc(label)}</text>`;
  });

  // The invalidation level, drawn where it belongs — on the price axis, so it
  // can be read against the candles instead of looked up as a number.
  let stopline='';
  if(stopY!==null){
    const y=stopY, live={stop:liveStop};
    stopline=`<line x1="${padL}" y1="${y.toFixed(1)}" x2="${(padL+pw).toFixed(1)}"
        y2="${y.toFixed(1)}" stroke="var(--warn)" stroke-width="1.5"
        stroke-dasharray="5 4"/>
      <text x="${(padL+4)}" y="${(y-4).toFixed(1)}" font-size="9.5"
        font-weight="600" fill="var(--warn)" stroke="var(--panel2)"
        stroke-width="3" paint-order="stroke">stop ${fmtPx(live.stop)}</text>`;
  }

  let xlab='';
  const dt=k=>{ const b=Date.parse(o.d0);
    return isNaN(b)?'':new Date(b+(o.dx[k]||0)*864e5).toISOString().slice(0,10); };
  [[0,'start'],[Math.floor((n-1)/2),'middle'],[n-1,'end']].forEach(t=>{
    const s=dt(t[0]); if(!s) return;
    xlab+=`<text x="${X(t[0]).toFixed(1)}" y="${H-7}" font-size="10.5"
      fill="var(--tx3)" text-anchor="${t[1]}">${esc(s)}</text>`;
  });

  const data=JSON.stringify({o:o.o,h:o.h,l:o.l,c:o.c,d0:o.d0,dx:o.dx,
    marks:o.marks||[],ccy:ccy||''});
  return `<div class="cchart"><svg viewBox="0 0 ${W} ${H}" role="img"
    data-candles='${data.replace(/'/g,"&#39;")}'
    aria-label="Candlestick chart of the last ${n} trading sessions">
    ${grid}${stopline}${sticks}${marks}${xlab}
    <g class="cross" style="display:none">
      <line y1="${padT}" y2="${padT+ph}" stroke="var(--tx3)" stroke-width="1"
        stroke-dasharray="3 3"/>
    </g>
    <rect class="hit" x="${padL}" y="${padT}" width="${pw}" height="${ph}"
      fill="transparent"/>
  </svg><div class="ptip" style="display:none"></div></div>`;
}

// The candle hover layer, kept separate from the line-chart one because the
// payload and the readout are different: OHLC for four numbers, not a
// crosshair against three interpolated series.
document.addEventListener('mousemove', function(ev){
  const svg=ev.target.closest ? ev.target.closest('.cchart svg') : null;
  // Clear every OTHER candle chart first, or a tooltip left behind on one
  // drawer reads as a live value on the next.
  document.querySelectorAll('.cchart').forEach(w=>{
    if(svg && w.contains(svg)) return;
    const c=w.querySelector('.cross'), t=w.querySelector('.ptip');
    if(c) c.style.display='none';
    if(t) t.style.display='none';
  });
  if(!svg) return;
  const wrap=svg.parentElement;
  let d; try{ d=JSON.parse(svg.getAttribute('data-candles')); }catch(e){ return; }
  if(!d || !d.c) return;
  const box=svg.getBoundingClientRect();
  const W=880,H=300,padL=8,padR=78,padT=16,padB=26;
  const pw=W-padL-padR, ph=H-padT-padB, n=d.c.length, step=pw/n;
  const sx=(ev.clientX-box.left)/box.width*W;
  let i=Math.floor((sx-padL)/step);
  i=Math.max(0,Math.min(n-1,i));
  const X=k=>padL+step*(k+0.5);
  const g=svg.querySelector('.cross');
  g.style.display='';
  g.querySelector('line').setAttribute('x1',X(i).toFixed(1));
  g.querySelector('line').setAttribute('x2',X(i).toFixed(1));
  const b=Date.parse(d.d0);
  const when=isNaN(b)?'':new Date(b+(d.dx[i]||0)*864e5).toISOString().slice(0,10);
  const up=d.c[i]>=d.o[i];
  const hit=(d.marks||[]).filter(m=>i>=m.i-(m.b||1)+1 && i<=m.i);
  const tip=wrap.querySelector('.ptip');
  tip.innerHTML=`<b>${esc(when)}</b>`
    +`<span>O ${fmtPx(d.o[i])} &nbsp; H ${fmtPx(d.h[i])}</span>`
    +`<span>L ${fmtPx(d.l[i])} &nbsp; C ${fmtPx(d.c[i])}</span>`
    +`<span style="color:${up?'var(--ok)':'var(--bad)'}">${up?'rising (hollow)':'falling (filled)'}</span>`
    +hit.map(m=>`<span><b>${esc(m.n)}</b> · ${esc(m.st)}</span>`).join('');
  tip.style.display='';
  tip.style.left=Math.max(0,Math.min(72,X(i)/W*100))+'%';
});

// One delegated listener rather than one per chart: drawers are built and
// discarded as rows are opened, and per-chart handlers would leak with them.
document.addEventListener('mousemove', function(ev){
  const svg=ev.target.closest ? ev.target.closest('.pchart svg') : null;
  document.querySelectorAll('.pchart').forEach(w=>{
    if(svg && w.contains(svg)) return;
    const c=w.querySelector('.cross'), t=w.querySelector('.ptip');
    if(c) c.style.display='none';
    if(t) t.style.display='none';
  });
  if(!svg) return;
  const wrap=svg.parentElement;
  let d; try{ d=JSON.parse(svg.getAttribute('data-series')); }catch(e){ return; }
  const box=svg.getBoundingClientRect();
  const padL=8,padR=78,padT=14,padB=26,W=880,H=300;
  const pw=W-padL-padR, ph=H-padT-padB;
  const sx=(ev.clientX-box.left)/box.width*W;
  let i=Math.round((sx-padL)/pw*(d.px.length-1));
  i=Math.max(0,Math.min(d.px.length-1,i));
  const all=d.px.concat(d.m50.filter(isNum)).concat(d.m200.filter(isNum));
  let lo=Math.min.apply(null,all), hi=Math.max.apply(null,all);
  const span=(hi-lo)||1; lo-=span*0.06; hi+=span*0.06;
  const X=k=>padL+k*pw/Math.max(d.px.length-1,1);
  const Y=v=>padT+ph-(v-lo)/(hi-lo)*ph;
  const g=svg.querySelector('.cross');
  g.style.display='';
  g.querySelector('line').setAttribute('x1',X(i).toFixed(1));
  g.querySelector('line').setAttribute('x2',X(i).toFixed(1));
  g.querySelector('circle').setAttribute('cx',X(i).toFixed(1));
  g.querySelector('circle').setAttribute('cy',Y(d.px[i]).toFixed(1));
  const real=v=>isNum(v)?fmtPx(v/100*d.first):'—';
  // The date is interpolated between the two endpoints rather than stored per
  // point: 100 date strings would cost more than the price series itself, and
  // the sampling is even, so the interpolation is exact to the sampling step.
  let when='';
  if(d.d0 && d.d1){
    const a=Date.parse(d.d0), b=Date.parse(d.d1);
    if(!isNaN(a)&&!isNaN(b))
      when=new Date(a+(b-a)*i/Math.max(d.px.length-1,1))
        .toISOString().slice(0,10)+' · ';
  }
  const tip=wrap.querySelector('.ptip');
  tip.innerHTML=`<b>${when}${esc(d.ccy)} ${real(d.px[i])}</b>`
    +`<span><i style="background:var(--series-2)"></i>50-day ${real(d.m50[i])}</span>`
    +`<span><i style="background:var(--series-3)"></i>200-day ${real(d.m200[i])}</span>`;
  tip.style.display='';
  const leftPct=X(i)/W*100;
  tip.style.left=Math.max(0,Math.min(72,leftPct))+'%';
});

function svgRsi(rsi){
  if(rsi===null||rsi===undefined) return '';
  const W=260,H=40,x=v=>4+v*(W-8)/100;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="RSI ${rsi.toFixed(0)}">
    <rect x="${x(0)}" y="10" width="${x(30)-x(0)}" height="12" rx="3"
      fill="var(--ok)" opacity=".22"/>
    <rect x="${x(30)}" y="10" width="${x(70)-x(30)}" height="12" rx="3"
      fill="var(--line)"/>
    <rect x="${x(70)}" y="10" width="${x(100)-x(70)}" height="12" rx="3"
      fill="var(--bad)" opacity=".22"/>
    <line x1="${x(rsi)}" y1="5" x2="${x(rsi)}" y2="27" stroke="var(--tx)"
      stroke-width="2"/>
    <text x="${x(0)}" y="36" font-size="8" fill="var(--tx3)">0 oversold</text>
    <text x="${x(100)}" y="36" font-size="8" fill="var(--tx3)"
      text-anchor="end">overbought 100</text>
    <text x="${x(rsi)}" y="7" font-size="9" fill="var(--tx)" text-anchor="middle"
      >${rsi.toFixed(0)}</text></svg>`;
}

function svgRange(t){
  // Where the price sits inside two windows. Drawn as one bar each rather than
  // a number, because "38% below the high" and "62% above the low" are the same
  // fact stated twice and a bar makes that obvious.
  const rows=[];
  if(t.below_52w_high!==null&&t.below_52w_high!==undefined)
    rows.push(['52-week', 1-Math.max(0,Math.min(1,t.below_52w_high)),
               (t.below_52w_high*100).toFixed(0)+'% below high']);
  if(t.in_10y_range!==null&&t.in_10y_range!==undefined)
    rows.push(['10-year', Math.max(0,Math.min(1,t.in_10y_range)),
               (t.in_10y_range*100).toFixed(0)+'% up its decade range']);
  if(!rows.length) return '';
  const W=260,H=rows.length*26+6;
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="price position">`;
  rows.forEach((r,i)=>{
    const y=i*26+4;
    out+=`<text x="0" y="${y+8}" font-size="9" fill="var(--tx3)">${esc(r[0])}</text>
      <rect x="52" y="${y+1}" width="${W-52}" height="9" rx="4" fill="var(--panel)"
        stroke="var(--line)"/>
      <rect x="52" y="${y+1}" width="${((W-52)*r[1]).toFixed(1)}" height="9" rx="4"
        fill="var(--acc)" opacity=".55"/>
      <text x="52" y="${y+21}" font-size="9" fill="var(--tx3)">${esc(r[2])}</text>`;
  });
  return out+'</svg>';
}

function svgReturns(t){
  const bars=[['1m',t.r1],['3m',t.r3],['6m',t.r6],['12m',t.r12]]
    .filter(b=>b[1]!==null&&b[1]!==undefined);
  if(!bars.length) return '';
  const W=260,H=76,mid=H/2-6;
  const max=Math.max(0.05,...bars.map(b=>Math.abs(b[1])));
  const bw=(W-10)/bars.length;
  let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="returns">
    <line x1="0" y1="${mid}" x2="${W}" y2="${mid}" stroke="var(--line)"/>`;
  bars.forEach((b,i)=>{
    const h=Math.abs(b[1])/max*(mid-8), x=5+i*bw+bw*0.18, w=bw*0.64;
    const up=b[1]>=0;
    out+=`<rect x="${x.toFixed(1)}" y="${(up?mid-h:mid).toFixed(1)}"
        width="${w.toFixed(1)}" height="${Math.max(1,h).toFixed(1)}" rx="2"
        fill="${up?'var(--ok)':'var(--bad)'}" opacity=".75"/>
      <text x="${(x+w/2).toFixed(1)}" y="${(up?mid-h-3:mid+h+9).toFixed(1)}"
        font-size="8.5" text-anchor="middle" fill="var(--tx2)"
        >${(b[1]*100).toFixed(0)}%</text>
      <text x="${(x+w/2).toFixed(1)}" y="${H-2}" font-size="8.5"
        text-anchor="middle" fill="var(--tx3)">${b[0]}</text>`;
  });
  return out+'</svg>';
}

function testBars(tests){
  // Each test as a bar against its own threshold, so "how far past the line"
  // is visible at a glance instead of having to be read out of two columns of
  // numbers. Binary tests (above the 200-day average, golden cross) have no
  // meaningful distance, so they render as a full or empty bar.
  if(!tests || !tests.length) return '';
  return '<div class="tstrip">'+tests.map(t=>{
    const v=parseFloat(String(t.value).replace(/,/g,''));
    const th=parseFloat(String(t.threshold).replace(/[^0-9.\\-]/g,''));
    const cls=t.state==='pass'?'ok':(t.state==='fail'?'no':'na');
    // One geometry for every test: the marker is ALWAYS the threshold, sitting
    // at the midpoint, and the bar shows how far past it the value is. Twice
    // the threshold fills the track. A glance down the column then reads as
    // "how far over or under the line", which is the only question a bar chart
    // of mixed units can honestly answer.
    let pctFill, markAt=null;
    if(isFinite(v)&&isFinite(th)&&th!==0){
      pctFill=Math.max(0.03,Math.min(1,(Math.abs(v)/Math.abs(th))/2));
      markAt=0.5;
    } else if(isFinite(v)&&th===0){
      // A zero threshold has no ratio. The sign IS the test, so the bar leans
      // one way or the other from the middle and says so.
      pctFill=v>=0?0.75:0.25; markAt=0.5;
    } else { pctFill=t.state==='pass'?0.75:0.25; }
    if(t.state==='na') pctFill=0.06;
    return `<div class="tbar"><span class="tl" title="${esc(t.name)}">${esc(t.name)}</span>
      <span class="track"><span class="fill ${cls}" style="left:0;width:${(pctFill*100).toFixed(1)}%"></span>
      ${markAt!==null?`<span class="mark" style="left:${(markAt*100).toFixed(1)}%"></span>`:''}</span>
      <span class="tt">${esc(t.value)} / ${esc(t.threshold)}</span></div>`;
  }).join('')+'</div>';
}


// "after a ${trend}trend" reads correctly for "down" and wrongly for
// everything else — "a uptrend", "a flattrend". The phrasing is written out.
function trendPhrase(t){
  if(t==='down') return 'after a downtrend';
  if(t==='up') return 'after an uptrend';
  if(t==='flat') return 'after a flat stretch';
  return 'with no clear prior trend';
}

function candleCard(r){
  const c=r.candles||{};
  if(!c.action && !(c.signals||[]).length)
    return `<div class="chart wide"><h5>Candlestick patterns</h5>
      <div class="cap">${esc(c.why||'not computed for this row')}</div></div>`;
  const cls=(c.action||'').indexOf('buy')>=0?'buy'
    :((c.action||'').indexOf('sell')>=0?'sell':'wait');
  const sigs=(c.signals||[]).map(s=>`<div class="sig">
      <span class="d">${esc(s.date||'')}</span>
      <span><span class="nm">${esc(s.name)}</span>${
        s.repeats>1?`<span class="rep">&times;${s.repeats} sessions</span>`:''}
        <span class="muted-ink">&middot; ${esc(s.direction)},
        ${trendPhrase(s.trend_before)}</span>
        <div class="rl">${esc(s.rule||'')}. ${esc(s.state_note||'')}${
          s.stop!==null&&s.stop!==undefined
            ? ' &middot; stop-loss at '+Number(s.stop).toFixed(2) : ''}${
          s.gap_sensitive
            ? ' &middot; <b>gap-based</b>: prices here are split and dividend '
              +'adjusted, which moves gaps, so treat this one with extra care' : ''}</div>
      </span>
      <span class="st ${esc(s.state)}">${esc(s.state)}</span></div>`).join('');
  return `<div class="chart wide"><h5>Candlestick patterns
      <span>${c.bullish||0} bullish &middot; ${c.bearish||0} bearish confirmed</span></h5>
    <div class="verdict"><span class="big ${cls}">${esc((c.action||'').toUpperCase())}</span>
      ${c.trend?`<span class="tchip t-${esc(c.trend)}">trend: ${esc(c.trend)}</span>`:''}
      <span class="muted-ink">${esc(c.why||'')}</span></div>
    ${c.stop!==null&&c.stop!==undefined
      ? `<div class="cap" style="margin:0 0 8px">Invalidation level from the most
         recent pattern: <b>${Number(c.stop).toFixed(2)}</b> &mdash; the book puts
         the stop at the low of a bullish pattern and the high of a bearish one,
         so a close through it is the signal saying it was wrong.</div>` : ''}
    <div class="cand">${sigs||'<div class="cap">no patterns in the recent window</div>'}</div>
    <div class="cap">${esc(c.caveat||'')}</div></div>`;
}

function technicalPanel(r){
  const t=r.tech_raw||{}, d=r.tech_detail||{};
  let cards='';
  // The price chart is filled in asynchronously: its series lives in a sidecar
  // file so the page itself stays small. The slot is drawn immediately so the
  // layout does not jump when the data arrives.
  if(r.has_series){
    cards+=`<div class="chart wide" data-series-for="${esc(r.ticker)}"
      data-market="${esc(r.market||'')}"><h5>Close, 50-day and 200-day averages
      <span class="phead"></span></h5>
      <div class="pslot"><div class="cap">loading price history&hellip;</div></div>
      <div class="lgd"><i><b style="background:var(--series-1)"></b>Close</i>
      <i><b style="background:var(--series-2)"></b>50-day</i>
      <i><b style="background:var(--series-3)"></b>200-day</i></div></div>`;
  } else {
    cards+=`<div class="chart wide"><h5>Price</h5><div class="cap">This company
      has too little price history to chart &mdash; the series needs at least
      sixty trading days. Every other chart below is drawn from the numbers and
      is unaffected.</div></div>`;
  }
  const rsi=svgRsi(t.rsi);
  if(rsi) cards+=`<div class="chart"><h5>RSI(14)<span>${esc(t.rsi_label||'')}</span></h5>
    ${rsi}<div class="cap">Bands are the app&rsquo;s own thresholds, not the
    textbook 30/70 — this is a value screen, so it wants weakness that is
    recovering rather than strength that is stretched.</div></div>`;
  const rg=svgRange(t);
  if(rg) cards+=`<div class="chart"><h5>Where the price sits</h5>${rg}</div>`;
  const rets=svgReturns(t);
  if(rets) cards+=`<div class="chart"><h5>Returns
    <span>${t.rs!==null&&t.rs!==undefined?((t.rs>=0?'+':'')+(t.rs*100).toFixed(0)+'% vs index, 6m'):''}</span></h5>
    ${rets}</div>`;
  const trend=[];
  if(t.above_sma200!==null&&t.above_sma200!==undefined)
    trend.push(t.above_sma200?'above the 200-day average':'below the 200-day average');
  if(t.golden_cross!==null&&t.golden_cross!==undefined)
    trend.push(t.golden_cross?'50-day above the 200-day':'50-day below the 200-day');
  if(t.macd!==null&&t.macd!==undefined)
    trend.push('MACD histogram '+(t.macd>=0?'positive':'negative'));
  if(t.vol!==null&&t.vol!==undefined)
    trend.push('volume at '+t.vol.toFixed(2)+'× its own baseline');
  // The candlestick chart, then the reading of it. Picture first: the patterns
  // named in the card below are marked on these candles, so the reader can see
  // the thing being described before reading the description.
  if(r.has_series){
    cards+=`<div class="chart wide" data-candles-for="${esc(r.ticker)}">
      <h5>Candlestick chart &mdash; last 60 sessions
      <span>hollow = rising &middot; filled = falling</span></h5>
      <div class="cslot"><div class="cap">loading daily bars&hellip;</div></div>
      <div class="lgd"><i><b class="ck up"></b>rising session (hollow body)</i>
      <i><b class="ck dn"></b>falling session (filled body)</i>
      <i><b class="ck tri"></b>pattern found &mdash; solid marker means the next
      session confirmed it, outlined means it did not</i></div></div>`;
  }
  cards+=candleCard(r);
  cards+=`<div class="chart wide"><h5>Tests against their thresholds
    <span>${esc(d.summary||'')}</span></h5>${testBars(d.tests||[])}
    ${trend.length?`<div class="cap">${esc(trend.join(' · '))}.</div>`:''}</div>`;
  return `<div class="chartgrid">${cards}</div>`;
}

function testList(tests){
  if(!tests.length) return '<div class="trow"><span class="tn">no tests evaluated</span></div>';
  return tests.map(t=>`<div class="trow"><span class="tn">${esc(t.name)}${t.alt?' *':''}</span>
    <span class="tv ${t.state}">${esc(t.value)} <span style="color:var(--tx3)">/ ${esc(t.threshold)}</span></span></div>`
    +(t.note?`<div style="font-size:10.5px;color:var(--tx3);font-style:italic;
       margin:-1px 0 3px;line-height:1.45">${esc(t.note)}</div>`:'')).join('');
}

function detail(r){
  let h='<div class="dwrap">';
  // The synopsis leads, because the panels below it are a reference and this
  // is the read. Everything in it is derived from those same panels, so if the
  // two ever disagree the panels are right and this is a bug.
  if(r.syn && (r.syn.what || (r.syn.numbers && r.syn.numbers.length))){
    h+='<div class="syn">';
    if(r.syn.what) h+=`<div class="what"><b>What it is.</b> ${esc(r.syn.what)}</div>`;
    h+=(r.syn.numbers||[]).map(x=>`<p>${esc(x)}</p>`).join('');
    h+='<div class="src">Written from this row&rsquo;s own figures'
      +(r.syn.what_source==='feed'
        ? ' plus the company description carried by the data feed'
        : (r.syn.what_source==='classification'
           ? ' — the feed carries no business description for this name, only its sector'
           : ''))
      +'. No forecast, no outside commentary.</div>';
    h+='</div>';
  }
  // A static page on a public repo cannot hold a GitHub token, so it cannot
  // trigger a workflow. It can link to an existing report, and it can send you
  // to the workflow with the ticker ready to copy.
  h+='<div class="ddbar">';
  if(r.has_report){
    h+=`<a class="ddbtn primary" href="deepdive/${esc(r.ticker)}.html">Open deep dive &rarr;</a>`;
  } else if(ISSUE_URL){
    // Pre-filled new-issue form: opening the issue is what starts the workflow.
    // Two clicks, no token on the page.
    const u = ISSUE_URL + '?title=' + encodeURIComponent('deepdive: ' + r.ticker)
            + '&body=' + encodeURIComponent('Requested from the screener. This issue closes itself when the report is ready.');
    h+=`<a class="ddbtn primary" href="${u}" target="_blank" rel="noopener">Run a deep dive &rarr;</a>
        <span class="ddhint">opens a pre-filled request &mdash; press <b>Create</b> and the report builds itself (~2 min)</span>`;
  }
  h+='</div>';
  // The listing sits with the theme tags: it is the same kind of fact about
  // the row (how it got here), and it is the only place the page says whether
  // a name arrived on the S&P 500 list or on the wider Nasdaq one.
  const tagbits=[];
  if(r.listing) tagbits.push(`<span class="tag lst">${esc(r.listing)}</span>`);
  (r.themes||[]).forEach(t=>tagbits.push(`<span class="tag">${esc(t)}</span>`));
  if(tagbits.length) h+='<div class="tagrow">'+tagbits.join('')+'</div>';
  // Lynch's category decides which bar this row was judged against, so it is
  // stated before the panels rather than buried inside one of them.
  // Value or growth, with the percentiles that produced it. The badge is one
  // letter; this is where the letter has to justify itself.
  // The stage's practical reading, on the row rather than only in the legend.
  // A badge you have to go and look up is a badge nobody reads twice.
  if(r.rfx_action)
    h+='<div class="note" style="border-left-color:'
      +(r.rfx_group==='caution'?'var(--bad)':(r.rfx_group==='opportunity'?'var(--ok)':'var(--tx3)'))
      +'"><b>Reflexive stage '+esc(r.rfx||'')+' &mdash; '+esc(r.rfx_label||'')+'.</b> '
      +esc(r.rfx_action)
      +'<div style="font-size:11.5px;color:var(--tx3);margin-top:6px">Fires when '
      +esc(r.rfx_rule)+(r.rfx_evidence?' &nbsp;&middot;&nbsp; '+esc(r.rfx_evidence):'')
      +'</div></div>';
  if(r.sty_why)
    h+='<div class="note" style="border-left-color:var(--tx3)"><b>Style: '
      +esc((r.sty||'').charAt(0).toUpperCase()+(r.sty||'').slice(1))+'.</b> '
      +esc(r.sty_why)
      +(r.sty_ev && r.sty_ev.length
        ? '<div style="font-size:11.5px;color:var(--tx3);margin-top:6px">'
          +r.sty_ev.map(esc).join(' &nbsp;&middot;&nbsp; ')+'</div>' : '')
      +'</div>';
  if(r.cat_label)
    h+='<div class="note" style="border-left-color:var(--tx3)"><b>Lynch category: '
      +esc(r.cat_label)+'.</b> '+esc(r.cat_why)+'. The Lynch panel below is scored '
      +'on this category&rsquo;s benchmarks; tests that do not apply to it are '
      +'excluded rather than failed.</div>';
  if(r.cat_warn)
    h+='<div class="warn"><b>Cyclical warning.</b> '+esc(r.cat_warn)+'</div>';
  // Munger's third basket is the one worth showing explicitly: "too tough" is
  // not a rejection, it is a refusal to have an opinion, and a screen that
  // collapsed it into "fail" would be losing his actual point.
  if(r.mun_bucket)
    h+='<div class="note" style="border-left-color:var(--tx3)"><b>Munger basket: '
      +esc(r.mun_bucket_label)+'.</b> '+esc(r.mun_bucket_why)+'.'
      +(r.mun_readings && r.mun_readings.length
        ? '<div style="font-size:11.5px;color:var(--tx3);margin-top:6px">'
          +r.mun_readings.map(esc).join(' &nbsp;&middot;&nbsp; ')+'</div>' : '')
      +'</div>';
  // The B label, with the evidence that earned it — or the reason it did not.
  // A badge whose reasoning is hidden is an instruction, and this app does not
  // give instructions.
  const bd=r.b_detail||{};
  if(bd.circle||bd.moat||bd.history){
    h+='<div class="note" style="border-left-color:'+(r.b?'var(--ok)':'var(--tx3)')+'">'
      +'<b>Buffett business tenets'+(r.b?' — labelled B':'')+'.</b><br>'
      +'<span style="color:var(--tx3)">Circle of competence:</span> '+esc((bd.circle||{}).why||'—')+'<br>'
      +'<span style="color:var(--tx3)">Durable moat:</span> '+esc((bd.moat||{}).why||'—')
      +((bd.moat&&bd.moat.evidence&&bd.moat.evidence.length)
         ? '<br><span style="font-size:11.5px;color:var(--tx3)">'
           +bd.moat.evidence.map(esc).join(' &nbsp;·&nbsp; ')+'</span>' : '')
      +'<br><span style="color:var(--tx3)">Consistent history:</span> '+esc((bd.history||{}).why||'—')
      +((bd.moat&&bd.moat.caveat)
         ? '<div style="font-size:11.5px;font-style:italic;color:var(--tx3);margin-top:7px">'
           +esc(bd.moat.caveat)+'</div>' : '')
      +'</div>';
  }
  if(r.is_fund)
    h+='<div class="warn"><b>This is a fund, not an operating company.</b> Revenue, '
      +'equity, ROE and EV/EBIT are undefined for an ETF, so the six value '
      +'frameworks show n/a rather than fail. Prices, technicals and the Soros '
      +'regime read still apply.</div>';
  if(r.warnings && r.warnings.length)
    h+=`<div class="warn"><b>Data quality:</b> ${r.warnings.map(esc).join(' · ')}</div>`;
  if(r.gates && r.gates.length)
    h+=`<div class="warn"><b>Excluded by size/liquidity gate:</b> ${r.gates.map(esc).join(' · ')}</div>`;
  h+='<div class="kmet">';
  for(const [k,v] of Object.entries(r.key_metrics))
    h+=`<div><div class="kl">${esc(k)}</div><div class="kv">${esc(v)}</div></div>`;
  h+='</div><div class="dgrid">';
  for(const [key,label] of FWS){
    const d=r.fw_detail[key]; if(!d) continue;
    const st=r.fw[key]==="pass"?"pass":"fail";
    h+=`<div class="dcard"><h4>${esc(label)}
        <span class="badge ${st}">${r.fw[key]==="na"?"n/a":(st==="pass"?"pass":"fail")}</span></h4>
        <div style="font-size:11px;color:var(--tx3);margin-bottom:5px">${esc(d.summary)}${
          d.rank?` · rank ${d.rank}`:''}</div>`;
    if(d.note) h+=`<div style="font-size:11px;color:var(--warn);margin-bottom:5px">${esc(d.note)}</div>`;
    h+=testList(d.tests)+'</div>';
  }
  h+='</div>';
  // Technical timing, as charts rather than a column of numbers. The numeric
  // list is kept underneath in a collapsed block: the charts are for reviewing
  // at a glance, the list is for checking a specific figure, and neither
  // replaces the other.
  const t=r.tech_detail;
  h+=`<div class="dcard" style="margin-top:14px"><h4>Technical timing
      <span class="badge ${r.tech_pass?'pass':'fail'}">${r.tech_pass?'pass':'fail'}</span></h4>
      ${technicalPanel(r)}
      <details><summary style="cursor:pointer;font-size:11.5px;color:var(--tx3)">
      the same tests as numbers</summary>
      <div style="margin-top:6px">${testList(t.tests)}</div></details></div>`;
  return h+'</div>';
}

function render(){
  const rows=visible(); stats(rows); syncUrl();
  const tb=document.getElementById('tbody'); tb.innerHTML='';
  const emptyEl=document.getElementById('empty');
  emptyEl.style.display=rows.length?'none':'block';
  if(!rows.length){
    emptyEl.innerHTML = fQ.trim()
      ? `Nothing matching <b>${esc(fQ.trim())}</b> among the ${DATA.length} names in
         the last run.<br><span style="font-size:12.5px">Either it isn't an index
         constituent, or the most recent refresh didn't reach it — check
         <code>config/universe.yml</code>.</span>`
      : (fList!=="ALL"
        // Naming the number is the point. "No names match" reads as "the
        // Nasdaq coverage never arrived"; "62 arrived, none cleared a screen"
        // is a different fact and the one that is actually true.
        ? `<b>${DATA.filter(x=>(x.listing||"")===fList).length}</b> name${
             DATA.filter(x=>(x.listing||"")===fList).length===1?'':'s'} came in on
           the <b>${esc(fList)}</b> list this run, and none of them cleared a
           screen under the filters you have set.<br>
           <span style="font-size:12.5px">Turn off <b>Surfaced only</b> to see
           them all with their per-test results — arriving in the universe and
           passing a screen are different things.</span>`
        : `No names match these filters.<br><span style="font-size:12.5px">Try turning off
           <b>Surfaced only</b> to see every name in the universe with its per-test
           results.</span>`);
  }
  rows.forEach((r,i)=>{
    const tr=document.createElement('tr'); tr.className='row';
    // Soros's stage, inline. A late-stage name is worth seeing whichever
    // framework surfaced it, so this sits on the ticker rather than inside
    // one framework's column.
    const rx = r.rfx && r.rfx!=='EQ'
      ? `<span class="rfx${r.rfx_late?' late':''}" title="${esc(r.rfx_label||'')} — ${esc(r.rfx_note||'')}${r.rfx_evidence?' ['+esc(r.rfx_evidence)+']':''}">${esc(r.rfx)}</span>` : '';
    // Every name that fell gets the badge; the colour says whether the
    // accounts explain the fall. Hiding the unexplained ones made the
    // qualifying list impossible to put in context.
    const dl = r.dis_fell
      ? `<span class="dis${r.dis?'':' expl'}" title="${esc(r.dis_note||'')}">&#8595;${esc(r.dis_6m||'30')}</span>`
      : '';
    // Lynch's label, inline. It is not decoration: it decides which bar every
    // value test on this row was measured against.
    const ct = r.cat
      ? `<span class="cat ${r.cat}${r.cat_warn?' warnflag':''}" title="${esc(r.cat_label||'')} — ${esc(r.cat_why||'')}${r.cat_warn?' ⚠ '+esc(r.cat_warn):''}">${esc(CATS[r.cat]||'?')}</span>`
      : '';
    // Buffett's three business tenets. A single letter, because that is what
    // it is: a business you could understand, protected, and boringly steady.
    const bl = r.b
      ? `<span class="blab" title="${esc(r.b_summary)}">B</span>` : '';
    // Value or growth, in one letter. V, G, and nothing at all for a blend —
    // the middle is where most of the market lives and badging it would be
    // noise on every second row.
    const st = r.sty && r.sty !== 'blend' && r.sty !== 'unscored'
      ? `<span class="sty ${r.sty.replace(' ','')}" title="${esc(r.sty_why||'')}${r.sty_ev&&r.sty_ev.length?' ['+esc(r.sty_ev.join(', '))+']':''}">${STY[r.sty]||'?'}</span>` : '';
    let cells=`<td class="tk">${esc(r.ticker)}${r.has_report?'<span class="dd" title="deep dive available">&#9670;</span>':''}${bl}${st}${ct}${rx}${dl}</td><td class="nm" title="${esc(r.name)}${r.syn&&r.syn.one_liner?' — '+esc(r.syn.one_liner):''}">${esc(r.name)}</td>
      <td><span class="mk">${esc(r.market_label)}</span></td>
      <td class="num">${esc(r.price)}</td><td class="num">${esc(r.mcap_usd)}</td>`;
    for(const [key] of FWS){
      const s=r.fw[key]; const ch=s==="pass"?"✓":(s==="unknown"?"?":(s==="na"?"–":"·"));
      cells+=`<td class="c"><span class="dot ${s}">${ch}</span></td>`;}
    cells+=`<td class="c"><span class="dot ${r.tech_pass?'pass':'fail'}">${r.tech_pass?'✓':'·'}</span></td>
      <td class="c cnt">${r.n_passed}</td>`;
    tr.innerHTML=cells;
    const dr=document.createElement('tr'); dr.className='detail'; dr.style.display='none';
    dr.innerHTML=`<td colspan="${5+FWS.length+2}">${detail(r)}</td>`;
    tr.onclick=()=>{
      const opening = dr.style.display==='none';
      dr.style.display = opening ? 'table-row' : 'none';
      if(opening) fillPriceCharts(dr);
    };
    tb.appendChild(tr); tb.appendChild(dr);
  });
}

document.querySelectorAll('[data-mkt]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-mkt]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); fMkt=b.dataset.mkt; render();});
document.querySelectorAll('[data-theme]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-theme]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); fTheme=b.dataset.theme; render();});
document.querySelectorAll('[data-listing]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-listing]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); fList=b.dataset.listing; render();});
document.querySelectorAll('[data-fw]').forEach(b=>b.onclick=()=>{
  const k=b.dataset.fw;
  if(fFw.has(k)){fFw.delete(k);b.classList.remove('on');}else{fFw.add(k);b.classList.add('on');}
  render();});
document.getElementById('techOnly').onclick=function(){
  fTech=!fTech; this.classList.toggle('on',fTech); render();};
document.getElementById('rfxOnly').onclick=function(){
  fRfx=!fRfx; this.classList.toggle('on',fRfx); render();};
document.getElementById('valOnly').onclick=function(){
  fVal=!fVal; if(fVal){fGrw=false;document.getElementById('grwOnly').classList.remove('on');}
  this.classList.toggle('on',fVal); render();};
document.getElementById('grwOnly').onclick=function(){
  fGrw=!fGrw; if(fGrw){fVal=false;document.getElementById('valOnly').classList.remove('on');}
  this.classList.toggle('on',fGrw); render();};
document.getElementById('bOnly').onclick=function(){
  fB=!fB; this.classList.toggle('on',fB); render();};
document.getElementById('disOnly').onclick=function(){
  fDis=!fDis; this.classList.toggle('on',fDis); render();};
document.getElementById('disQual').onclick=function(){
  fDisQ=!fDisQ; this.classList.toggle('on',fDisQ); render();};
document.getElementById('surfOnly').onclick=function(){
  fSurf=!fSurf; this.classList.toggle('on',fSurf); render();};
document.getElementById('q').oninput=e=>{fQ=e.target.value; render();};
document.getElementById('copylink').onclick=function(){
  const btn=this;
  navigator.clipboard.writeText(location.href).then(()=>{
    btn.textContent='Copied'; setTimeout(()=>btn.textContent='Copy link to this view',1600);
  }).catch(()=>{ btn.textContent=location.href; });
};
loadUrl();
render();
</script></body></html>
"""


def e_attr(s: str) -> str:
    """Escape for an HTML attribute AND for the JS string comparison it feeds."""
    import html as _h
    return _h.escape(str(s), quote=True)


DALIO_STAGES = ["Early", "Bubble", "Top", "Depression", "Beautiful\ndeleverage",
                "Pushing on\na string", "Normalise"]



def _pct_cell(v, dp=1):
    if v is None or (isinstance(v, float) and v != v):
        return '<td class="r">—</td>'
    cls = "up" if v > 0 else ("dn" if v < 0 else "")
    return f'<td class="r {cls}">{v * 100:+.{dp}f}%</td>'




def _dis_tip(d: Optional[Dict[str, Any]]) -> str:
    """One-line summary of a dislocation, for the badge tooltip."""
    if not d:
        return ""
    bits = [f"down {abs(d['return_6m']):.0%} in 6 months"]
    for k in ("scope_reading", "shape_reading", "volume_reading"):
        if d.get(k):
            bits.append(d[k])
    if d.get("evidence_grade") == "observed":
        for e in (d.get("events") or [])[:3]:
            lab = (", ".join(e.get("labels") or [])
                   or e.get("title")
                   or (f"M{e['magnitude']} quake, {e.get('place')}"
                       if e.get("magnitude") else ""))
            if lab:
                bits.append(f"OBSERVED {e.get('source', '')}: {lab}"
                            + (f" ({e['date']})" if e.get("date") else ""))
    causes = d.get("candidate_causes") or []
    if causes:
        bits.append(("observed cause: " if d.get("evidence_grade") == "observed"
                     else "consistent with: ")
                    + ", ".join(f"#{c['n']} {c['name']}" for c in causes[:4])
                    + ("…" if len(causes) > 4 else ""))
    return " · ".join(bits)



def _dislocation_panel(summary: Dict[str, Any]) -> str:
    """What the Dislocation filter is, and the warning that must travel with it."""
    if not summary or not summary.get("fell_30pct"):
        return ""
    from . import dislocation as _d
    fell, hits = summary["fell_30pct"], summary["fundamentals_intact"]
    thr = abs(summary.get("threshold", -0.30))
    causes = "".join(
        f'<div class="drow"><span class="k">#{c["n"]} {e_attr(c["name"])}</span>'
        f'<span class="v"><span class="pill">'
        f'{"market-wide" if c["scope"] == "market" else "name-specific" if c["scope"] == "name" else "either"}'
        f' &middot; {c["shape"]}</span></span></div>'
        for c in _d.CAUSES)
    return (
        '<details class="dalio"><summary>'
        f'<span class="stg">Fell &gt;{thr:.0%} in 6m &middot; {fell} names, '
        f'{hits} the accounts do not explain</span>'
        '<span class="muted-ink">filter with the two chips above &mdash; '
        'every faller carries a &#8595; badge showing its six-month return'
        '</span>'
        '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        '</summary><div class="body">'
        f'<div class="dnote"><b>Where to find them.</b> Every name down more '
        f'than {thr:.0%} over six months carries a <b>&#8595;</b> badge next to '
        'its ticker showing the actual figure. A <b>blue</b> badge means the '
        'last published accounts do NOT explain the fall; a <b>grey</b> one '
        'means they do &mdash; revenue, cash flow or the balance sheet '
        'deteriorated too, so the market had a reason. Use <b>Fell &gt;30% '
        '(6m)</b> to see them all and <b>&hellip; accounts intact</b> to '
        'narrow to the unexplained. The six-month return is also in every '
        'name\'s metrics drawer.</div>'
        '<div class="dnote"><b>Read this first.</b> The commonest reason a '
        'stock falls 30% while its filings look healthy is not panic &mdash; '
        'it is that the filings are stale and the market is right. Annual '
        'statements can be a year old; the market prices tomorrow. Every name '
        'on this list is one where the market disagrees with the last filing, '
        'and the market usually wins that argument. This is a list of '
        '<i>questions</i>: read the last three announcements before treating '
        'any of it as a mispricing.</div>'
        '<div class="dnote"><b>What the app can and cannot tell you.</b> It can '
        'establish the <i>divergence</i> &mdash; price down hard, accounts '
        'intact &mdash; and it can separate a fall the whole market shared from '
        'one the name took alone, a cascade concentrated in days from a steady '
        're-rating, and heavy volume from an empty order book. It <b>cannot</b> '
        'identify a cause. Nothing in a price series can. So each hit narrows '
        'the fifteen candidates below to the ones the evidence is consistent '
        'with, and leaves the diagnosis to you.</div>'
        f'<div class="dgrid"><div class="dcard">'
        f'<h4>The fifteen candidate causes</h4>{causes}</div></div>'
        '<div class="dnote"><b>Where the causes come from.</b> Three feeds run '
        'on the names that fell. <b>SEC 8-K item codes</b> &mdash; free, '
        'structured, US issuers only; item 5.02 <i>is</i> an officer '
        'departure, it is not a model guessing from a headline. <b>USGS '
        'quakes</b>, matched to a market by location. <b>GDELT</b> headlines, '
        'which narrow but never conclude. A name with an observed event shows '
        '<b>OBSERVED</b> in its tooltip and its shortlist collapses to what was '
        'actually seen; a name with none keeps the inferred shortlist. '
        '<b>Silence is not evidence that nothing happened</b> &mdash; 8-K '
        'covers US filers only, and news coverage of SGX, SET and IDX names is '
        'thin, so an empty result says more about the feeds than the company. '
        'Two codes REMOVE a name outright rather than annotate it: '
        '<b>4.02</b> (previously issued accounts can no longer be relied on) '
        'and <b>1.03</b> (bankruptcy) &mdash; because the intact fundamentals '
        'this screen just measured are, in those cases, fiction.</div>'
        '</div></details>')

def _reflexive_legend(census: Dict[str, int]) -> str:
    """The stage key, what each one means for a buyer, and the census.

    Three jobs, in order of how easy they are to get wrong:

      1. A two-letter code with no key is a puzzle, not a label.
      2. A stage that fires on most of the universe has stopped
         discriminating, so the distribution is printed rather than described.
      3. A taxonomy is not advice. The sequence AB->HI is a story about a
         market; what an investor needs is which END of it they are standing
         at, so the stages are grouped by that and each carries the practical
         reading rather than only the quotation.
    """
    from . import reflexivity as rfx
    if not census:
        return ""
    total = sum(census.values()) or 1
    unclassified = census.get("n/a", 0)
    classified = total - unclassified

    def row(code):
        st = rfx.STAGES[code]
        n = census.get(code, 0)
        share = n / total
        cls = "rfx late" if code in rfx.LATE_STAGES else "rfx"
        return (f'<tr><td><span class="{cls}">{code}</span></td>'
                f'<td class="nm">{e_attr(st["label"])}</td>'
                f'<td class="r"><b>{n}</b> <span class="muted-ink">'
                f'{share:.0%}</span></td>'
                f'<td class="muted-ink">{e_attr(st["rule"])}</td>'
                f'<td>{e_attr(st["action"])}</td></tr>')

    blocks = ""
    for group in ("opportunity", "caution", "neutral", "off"):
        codes = [c for c in rfx.GROUPS[group] if census.get(c)]
        if not codes:
            continue
        blocks += (
            f'<h4 style="margin:15px 0 2px;font-size:12px;letter-spacing:.06em;'
            f'text-transform:uppercase;color:var(--tx3)">'
            f'{e_attr(rfx.GROUP_LABELS[group])}</h4>'
            f'<div class="dnote" style="margin:0 0 6px">'
            f'{e_attr(rfx.GROUP_NOTES[group])}</div>'
            '<table><thead><tr><th></th><th>Stage</th><th class="r">Names</th>'
            '<th>Fires when</th><th>What it means for a buyer</th>'
            '</tr></thead><tbody>'
            + "".join(row(c) for c in codes) + '</tbody></table>')
    if not blocks:
        return ""

    notes = []
    # The coverage line comes FIRST, because every count above it is a share of
    # this number and not of the classified subset.
    if unclassified:
        notes.append(
            f'<b>{classified} of {total} names carry a badge.</b> The other '
            f'{unclassified} have no stage at all: the read needs a full year '
            'of earnings AND a full year of price, and much of the Asian '
            'universe arrives with a four-year statement feed that cannot '
            'always supply the earnings half. Every percentage above is a share '
            'of the whole table, so a stage showing 3% is 3% of everything, not '
            '3% of what was classified.')
    late = census.get("DE", 0) + census.get("EF", 0)
    if classified and late / max(classified, 1) > 0.5:
        notes.append(
            '<b>Read this before trusting the badges.</b> '
            f'{late / classified:.0%} of the classified names are reading DE or '
            'EF. A stage that fires on most names is not identifying anything — '
            'it is describing a market where prices rose faster than earnings '
            'across the board, which is a fact about the index rather than '
            'about these companies.')
    notes.append(
        'The path runs AB to HI, and it is a LOOP rather than a ladder: a name '
        'can sit in one stage for years or skip several in a quarter. Two are '
        'worth knowing by name. <b>DE</b> is the diagnostic stage — price '
        'rising through an earnings setback near the highs, where the market '
        'has stopped listening to what it was originally responding to. '
        '<b>GH</b> is the only stage where reflexivity is confirmed rather '
        'than asserted, because the earnings deteriorated AFTER the price did.')
    notes.append(
        '<span class="muted-ink">What this measures is POSITIONING AND '
        'NARRATIVE, not value. A badge tells you what the market has already '
        'believed and paid for; it says nothing about what the business is '
        'worth. That is what the framework columns are for, and the two '
        'questions are best answered separately.</span>')

    return ('<details class="dalio cmd"><summary>'
            '<span class="stg">Reflexive stage &middot; what the badges mean</span>'
            f'<span class="muted-ink">Soros&rsquo;s boom/bust path &mdash; '
            f'{classified} of {total} names classified, grouped by what to do '
            'about it</span><span class="muted-ink" style="margin-left:auto">'
            'details &#9662;</span></summary><div class="body">'
            + blocks
            + "".join(f'<div class="dnote">{n}</div>' for n in notes)
            + '</div></details>')


def _b_badge(row: Dict[str, Any]) -> str:
    """The B label as a chip, or nothing. Kept out of the f-string because an
    f-string expression cannot contain a backslash, and escaping quotes inside
    one is how that rule gets discovered the hard way."""
    return '<span class="blab">B</span> ' if row.get("b_label") else ''


def _ranked_panel(title: str, blurb: str, data: Dict[str, Any],
                  extra_notes: Optional[List[str]] = None) -> str:
    """A two-factor ranked list, one table per exchange.

    Grouped by market rather than merged, because a Hong Kong list and a US
    list are two different opportunity sets: merged, whichever market is
    structurally cheaper this decade takes most of the slots and the other
    exchanges effectively vanish.
    """
    markets = (data or {}).get("markets") or {}
    if not markets:
        return ""
    fa, fb = data["factor_a"], data["factor_b"]

    def fmt(v, spec):
        if not isinstance(v, (int, float)):
            return "—"
        return f"{v * 100:.1f}%" if spec.get("format") == "pct" else f"{v:,.2f}"

    blocks = ""
    for mkt in sorted(markets, key=lambda k: MARKET_LABELS.get(k, k)):
        blk = markets[mkt]
        rows = blk.get("rows") or []
        if not rows:
            continue
        body = "".join(
            f'<tr><td class="r">{r["rank"]}</td>'
            f'<td class="sym">{e_attr(r["ticker"])}</td>'
            f'<td>{_b_badge(r)}{e_attr(r.get("category") or "")}</td>'
            f'<td class="r">{fmt(r["a"], fa)}</td>'
            f'<td class="r">{fmt(r["b"], fb)}</td>'
            f'<td class="r">{r["a_rank"]}</td>'
            f'<td class="r">{r["b_rank"]}</td>'
            f'<td class="r">{r["score"]}</td></tr>'
            for r in rows)
        blocks += (
            f'<h4 style="margin:16px 0 2px;font-size:12px;letter-spacing:.06em;'
            f'text-transform:uppercase;color:var(--tx3)">'
            f'{e_attr(MARKET_LABELS.get(mkt, mkt))} '
            f'<span style="text-transform:none;letter-spacing:0;color:var(--tx2)">'
            f'&mdash; top {len(rows)} of {blk["eligible"]} eligible'
            + (' · <b>gate relaxed</b>' if blk.get("fallback") else '')
            + '</span></h4>'
            '<table><thead><tr><th class="r">#</th><th>Ticker</th><th>Kind</th>'
            f'<th class="r">{e_attr(fa["label"])}</th>'
            f'<th class="r">{e_attr(fb["label"])}</th>'
            '<th class="r">rank A</th><th class="r">rank B</th>'
            '<th class="r">Score</th></tr></thead><tbody>'
            + body + '</tbody></table>')

    notes = [blurb]
    notes.append(
        f'Ranked separately on <b>{e_attr(fa["label"])}</b> and '
        f'<b>{e_attr(fb["label"])}</b>; the two ranks are added and the lowest '
        'total wins — Greenblatt\'s arithmetic applied to a different pair of '
        'questions. Ranks are computed WITHIN each exchange, so a rank of 3 '
        'means third on that exchange, not third in the world.')
    if data.get("excluded"):
        top = list((data.get("exclusion_reasons") or {}).items())[:3]
        notes.append(
            f'{data["excluded"]} names were excluded before ranking: '
            + "; ".join(f'{e_attr(k)} ({v})' for k, v in top)
            + '. A top thirty drawn from forty survivors is a different object '
              'from one drawn from three hundred, so the count is published.')
    if data.get("fallback_markets"):
        notes.append(
            '<b>Gate relaxed on '
            + ", ".join(e_attr(MARKET_LABELS.get(m, m))
                        for m in data["fallback_markets"])
            + '.</b> Fewer than '
            + str(data.get("fallback_floor", 8))
            + ' names on those exchanges clear the three business tenets, so '
              'the list there ranks everything eligible instead. It is still a '
              'quality-and-price ranking, but it is not the tenet-gated list '
              'the other exchanges show.')
    for n in (extra_notes or []):
        notes.append(n)
    return (
        '<details class="dalio cmd"><summary>'
        f'<span class="stg">{e_attr(title)}</span>'
        f'<span class="muted-ink">top {data.get("top_n", 30)} per exchange &mdash; '
        f'{data.get("total_eligible", 0)} names ranked</span>'
        '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        '</summary><div class="body">'
        + blocks
        + "".join(f'<div class="dnote">{n}</div>' for n in notes)
        + '</div></details>')


def _magic_formula_panel(mf: Dict[str, Any]) -> str:
    """The list the Magic Formula actually prescribes, and what to do with it.

    Separated from the pass/fail column deliberately. The green tick in the
    Greenblatt column means "inside the top N of ITS OWN MARKET", which is this
    app's deviation from the book; the list below is the book's own — one
    ranking across the whole universe. Publishing both is the only way the
    deviation stays visible.
    """
    rows = (mf or {}).get("rows") or []
    if not rows:
        return ""
    body = "".join(
        f'<tr><td class="r">{r["rank"]}</td>'
        f'<td class="sym">{e_attr(r["ticker"])}</td>'
        f'<td>{e_attr(MARKET_LABELS.get(r["market"], r["market"] or ""))}</td>'
        f'<td class="r">{r["earnings_yield"] * 100:.1f}%</td>'
        f'<td class="r">{r["return_on_capital"] * 100:.0f}%</td>'
        f'<td class="r">{r["ey_rank"]}</td>'
        f'<td class="r">{r["roc_rank"]}</td>'
        f'<td class="r">{r["combined_score"]}</td></tr>'
        for r in rows)
    notes = [
        "<b>How to use this.</b> Greenblatt's instruction is mechanical: buy "
        f"the top {mf.get('top_n', 30)} by combined rank, hold for a year, then "
        "rebalance. The holding period is not incidental — it is what makes the "
        "formula survive the stretches where it underperforms, which the book "
        "says will be roughly one year in three. A list re-picked every month "
        "is a different strategy with the same inputs.",
        f"Ranked across all {mf.get('universe_size', 0)} eligible names, "
        f"{mf.get('excluded', 0)} excluded — financials, REITs and utilities "
        "(where EV/EBIT and return on capital are meaningless because the "
        "balance sheet IS the business) and anything under the market-cap floor.",
    ]
    if mf.get("scope_in_use") == "market":
        notes.append(
            "<b>Note the two rankings.</b> The list above is the book's: one "
            "ranking across the whole universe. The green tick in the "
            "Greenblatt column of the table below is this app's deviation — "
            "ranked within each market, so one structurally cheap market "
            "cannot take every slot. A name can be top-30 in its own market "
            "and nowhere near this list.")
    if mf.get("basis_mixed"):
        notes.append(
            f"{mf.get('on_greenblatt_basis', 0)} of {mf.get('universe_size', 0)} "
            "names are ranked on Greenblatt's stricter definitions — excess "
            "cash rather than all cash, and a capital base net of "
            "interest-bearing current liabilities. The rest fall back to the "
            "general definitions because their feed lacks the split, and a "
            "ranking that mixes the two is not quite comparing like with like.")
    return (
        '<details class="dalio cmd"><summary>'
        '<span class="stg">Magic Formula portfolio</span>'
        f'<span class="muted-ink">top {len(rows)} by combined rank across '
        f'{mf.get("universe_size", 0)} eligible names — buy the list, hold a '
        'year, rebalance</span>'
        '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        '</summary><div class="body">'
        '<table><thead><tr><th class="r">#</th><th>Ticker</th><th>Market</th>'
        '<th class="r">Earnings yield</th><th class="r">Return on capital</th>'
        '<th class="r">EY rank</th><th class="r">ROC rank</th>'
        '<th class="r">Score</th></tr></thead><tbody>'
        + body + '</tbody></table>'
        + "".join(f'<div class="dnote">{n}</div>' for n in notes)
        + '</div></details>')


def _sentiment_panel(s: Dict[str, Any]) -> str:
    """The six psychology gauges, each with its contrarian reading.

    Built so a missing feed is visible rather than absorbed: an unavailable
    gauge prints its reason in the same place a reading would have gone. The
    alternative — quietly showing five gauges where there should be six — makes
    a degraded panel look like a calm market.
    """
    if not s or s.get("error"):
        if s and s.get("error"):
            return ('<div class="gate"><b>Market psychology:</b> not computed '
                    '— ' + e_attr(str(s["error"])) + '.</div>')
        return ""
    g = s.get("gauges", {})
    fg = g.get("fear_greed", {})
    vix = g.get("vix", {})
    rsi = g.get("rsi", {})
    pcr = g.get("put_call", {})
    ad = g.get("advance_decline", {})
    part = g.get("participation", {})
    cot = g.get("cot", []) or []

    crowd = s.get("crowd", "mixed")
    cls = {"greedy": "hot", "fearful": "cool"}.get(crowd, "warm")
    head = ""
    if fg.get("available"):
        head = (f'<span class="pill {cls}">Fear &amp; Greed '
                f'{fg["score"]:.0f} &mdash; {e_attr(fg["label"])}</span>')

    def _row(k, v):
        return (f'<div class="drow"><span class="k">{k}</span>'
                f'<span class="v">{v}</span></div>')

    def _gauge(title, gg, rows):
        if not gg.get("available"):
            return (f'<div class="dcard"><h4>{title}</h4>'
                    f'<div class="drow"><span class="k muted-ink">not available '
                    f'&mdash; {e_attr(str(gg.get("reason", "no data")))}</span>'
                    f'</div></div>')
        return f'<div class="dcard"><h4>{title}</h4>{rows}</div>'

    vix_rows = ""
    if vix.get("available"):
        vix_rows = _row("Level", f'{vix["level"]:.1f}')
        if vix.get("percentile_1y") is not None:
            vix_rows += _row("In its own year",
                             f'{vix["percentile_1y"] * 100:.0f}th percentile')
        if vix.get("term_structure") is not None:
            vix_rows += _row("Spot ÷ 3-month", f'{vix["term_structure"]:.2f}')
        vix_rows += (f'<div class="dnote">{e_attr(vix["reading"])}.'
                     + (f' {e_attr(vix.get("term_reading", ""))}.'
                        if vix.get("term_reading") else '') + '</div>')

    rsi_rows = ""
    if rsi.get("available"):
        if rsi.get("index_rsi") is not None:
            rsi_rows += _row("Index RSI(14)", f'{rsi["index_rsi"]:.0f}')
        if rsi.get("universe_median_rsi") is not None:
            rsi_rows += _row("Median stock RSI",
                             f'{rsi["universe_median_rsi"]:.0f}')
        rsi_rows += f'<div class="dnote">{e_attr(rsi["reading"])}.</div>'
        if rsi.get("divergence"):
            rsi_rows += f'<div class="dnote">{e_attr(rsi["divergence"])}.</div>'

    pcr_rows = ""
    if pcr.get("available"):
        pcr_rows = _row("Put/call", f'{pcr["ratio"]:.2f}')
        if pcr.get("percentile_1y") is not None:
            pcr_rows += _row("In its own year",
                             f'{pcr["percentile_1y"] * 100:.0f}th percentile')
        pcr_rows += f'<div class="dnote">{e_attr(pcr["reading"])}.</div>'

    ad_rows = ""
    if ad.get("available"):
        ad_rows = _row("Advancers / decliners",
                       f'{ad["advancers"]} / {ad["decliners"]}')
        ad_rows += _row("McClellan Oscillator",
                        f'{ad["mcclellan_oscillator"]:+.0f}')
        if ad.get("ad_change_20d") is not None:
            ad_rows += _row("A/D line, 20 days", f'{ad["ad_change_20d"]:+.0f}')
        ad_rows += f'<div class="dnote">{e_attr(ad["oscillator_state"])}.</div>'
        if ad.get("divergence"):
            ad_rows += f'<div class="dnote">{e_attr(ad["divergence"])}.</div>'
        ad_rows += (f'<div class="dnote muted-ink">{e_attr(ad["caveat"])}</div>')

    part_rows = ""
    if part.get("available"):
        if part.get("index_6m_return") is not None:
            part_rows += _row("Index, 6 months",
                              f'{part["index_6m_return"] * 100:+.1f}%')
        part_rows += _row("Median stock, 6 months",
                          f'{part["median_6m_return"] * 100:+.1f}%')
        part_rows += _row("New highs / new lows",
                          f'{part["new_highs"]} / {part["new_lows"]}')
        part_rows += _row("Names up over 6 months",
                          f'{part["pct_positive_6m"] * 100:.0f}%')
        if part.get("reading"):
            part_rows += f'<div class="dnote">{e_attr(part["reading"])}.</div>'

    cot_rows = ""
    for c in cot:
        if c.get("available"):
            cot_rows += _row(
                e_attr(str(c.get("contract", ""))),
                f'{c["spec_net_pct_oi"] * 100:+.1f}% of OI · '
                f'percentile {c["percentile"] * 100:.0f}')
        else:
            cot_rows += (f'<div class="drow"><span class="k muted-ink">'
                         f'{e_attr(str(c.get("contract", "")))} — '
                         f'{e_attr(str(c.get("reason", "no data")))}</span></div>')
    extremes = [c for c in cot if c.get("available")
                and c.get("state") in ("crowded long", "crowded short")]
    for c in extremes:
        cot_rows += (f'<div class="dnote"><b>{e_attr(str(c["contract"]))}:</b> '
                     f'{e_attr(c["reading"])}.</div>')
    if cot and any(c.get("available") for c in cot):
        dates = {c.get("report_date") for c in cot if c.get("available")}
        cot_rows += ('<div class="dnote muted-ink">The COT report covers the '
                     'preceding Tuesday and is published on Friday, so it is '
                     'always several days old. Week of '
                     + e_attr(", ".join(sorted(d for d in dates if d)))
                     + '.</div>')

    fg_rows = ""
    if fg.get("available"):
        for k, v in fg["subscores"].items():
            fg_rows += _row(e_attr(fg["subscore_labels"].get(k, k)), f'{v:.0f}')
        if fg.get("missing"):
            fg_rows += ('<div class="dnote muted-ink">Not included: '
                        + e_attr("; ".join(fg["missing"])) + '.</div>')
        fg_rows += f'<div class="dnote muted-ink">{e_attr(fg["caveat"])}</div>'

    body = (
        '<div class="dgrid">'
        + _gauge("Fear &amp; Greed (our replication)", fg, fg_rows)
        + _gauge("Volatility &amp; fear (VIX)", vix, vix_rows)
        + _gauge("Momentum (RSI)", rsi, rsi_rows)
        + _gauge("Options hedging (put/call)", pcr, pcr_rows)
        + _gauge("Participation breadth (A/D, McClellan)", ad, ad_rows)
        + _gauge("Whose rally is it?", part, part_rows)
        + ('<div class="dcard"><h4>Institutional positioning (COT)</h4>'
           + (cot_rows or '<div class="drow"><span class="k muted-ink">no COT '
                          'contracts configured</span></div>') + '</div>')
        + '</div>'
        + f'<div class="dnote"><b>The crowd looks {e_attr(crowd)}.</b> '
        + e_attr(s.get("action", "")) + '.</div>'
        + (f'<div class="dnote">{e_attr(s["cycle_note"])}</div>'
           if s.get("cycle_note") else '')
        + f'<div class="dnote muted-ink">{e_attr(s.get("caveat", ""))}</div>')

    return ('<details class="dalio"><summary>'
            '<span class="stg">Market psychology</span>'
            + head
            + f'<span class="muted-ink">{s.get("gauges_available", 0)} of 6 '
              'gauges reporting — every one of them contrarian, none of them a '
              'timing signal</span>'
            '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
            '</summary><div class="body">' + body + '</div></details>')


def _buffett_indicator_line(bi: Dict[str, Any]) -> str:
    """Buffett's market-level gauge, or an honest account of why there isn't one."""
    if not bi:
        return ""
    if not bi.get("available"):
        return ('<div class="dnote muted-ink"><b>Buffett Indicator:</b> not '
                'computed — ' + e_attr(str(bi.get("reason", "unknown"))) + '.</div>')
    cls = {"significantly undervalued": "cool", "fairly valued": "",
           "on the expensive side": "warm",
           "substantially overvalued": "hot"}.get(bi["verdict"], "")
    return (f'<div class="dnote"><b>Buffett Indicator: {bi["pct"]:.0f}% '
            f'<span class="pill {cls}">{e_attr(bi["verdict"])}</span></b> — '
            f'{e_attr(bi["reading"])} '
            f'<span class="muted-ink">{e_attr(bi["caveat"])}</span></div>')


STYLES_ORDER = ("deep value", "value", "blend", "growth", "high growth")

STYLE_SHORT = {"deep value": "VV", "value": "V", "blend": "—",
               "growth": "G", "high growth": "GG"}

CAT_SHORT = {"fast_grower": "FG", "stalwart": "SW", "slow_grower": "SG",
             "cyclical": "CY", "turnaround": "TA", "asset_play": "AP",
             "unclassified": "?"}


def _lynch_panel(census: Dict[str, int], regime: Dict[str, Any],
                 rule20: Optional[Dict[str, Any]] = None,
                 buffett_ind: Optional[Dict[str, Any]] = None,
                 b_count: Optional[Dict[str, Any]] = None,
                 style: Optional[Dict[str, Any]] = None) -> str:
    """Lynch's six categories across the universe, plus Schloss's value regime.

    Both are published for the same reason: they change how every row below is
    scored, and a classifier or a regime switch that nobody can see is one
    nobody can argue with.
    """
    from . import lynch as ly
    rule20 = rule20 or {}
    total = sum(census.values())
    if not total and not regime:
        return ""
    r20 = ""
    if rule20.get("available"):
        cls = {"cheap": "cool", "fair": "warm", "expensive": "hot"}.get(
            rule20.get("verdict"), "")
        r20 = (f'<span class="pill {cls}">Rule of 20: '
               f'{rule20["total"]:.1f}</span>')
    elif rule20.get("reason"):
        r20 = '<span class="pill">Rule of 20 unavailable</span>'
    order = [c for c in ly.CATEGORIES if census.get(c)]
    sty = style or {}
    sty_cells = ""
    if sty.get("census"):
        sty_order = [k for k in STYLES_ORDER if sty["census"].get(k)]
        tot = sum(sty["census"].values()) or 1
        sty_cells = "".join(
            f'<div class="drow"><span class="k">{e_attr(k.title())} '
            f'<span class="pill">{STYLE_SHORT.get(k, "?")}</span></span>'
            f'<span class="v">{sty["census"][k]} · '
            f'{sty["census"][k] / tot:.0%}</span></div>'
            for k in sty_order)
        if sty["census"].get("unscored"):
            sty_cells += (
                f'<div class="drow"><span class="k muted-ink">Unscored</span>'
                f'<span class="v muted-ink">{sty["census"]["unscored"]}</span>'
                '</div>')
    cells = "".join(
        f'<div class="drow"><span class="k">{e_attr(ly.LABELS[c])} '
        f'<span class="pill">{CAT_SHORT[c]}</span></span>'
        f'<span class="v">{census[c]} · {census[c] / total:.0%}</span></div>'
        for c in order)
    notes = []
    # The self-check. One category swallowing the universe means the classifier
    # is wrong, not that the market is uniform — and the page should say so
    # before anyone trades on a label.
    if total:
        top = max(census.items(), key=lambda kv: kv[1])
        if top[1] / total > 0.60:
            notes.append(
                f'<b>Check this.</b> {top[1] / total:.0%} of the universe is '
                f'landing in one category ({ly.LABELS[top[0]]}). Lynch\'s split '
                'is meant to be uneven but not this uneven — the industry word '
                'list in <code>src/lynch.py</code> is probably too greedy.')
    if rule20.get("available"):
        notes.append(
            f'<b>Rule of 20: {rule20["total"]:.1f}</b> — a median trailing P/E '
            f'of {rule20["median_pe"]:.1f} across {rule20["names"]} US names '
            f'plus inflation of {rule20["inflation_pct"]:.1f}%. That is '
            f'{rule20["reading"]}. The P/E is this screener\'s own median, not '
            'a vendor\'s cap-weighted S&amp;P figure — a median runs cooler, '
            'and the gap between the two is the mega-caps.')
    elif rule20.get("reason"):
        notes.append('<span class="muted-ink">Rule of 20 not computed: '
                     + e_attr(str(rule20["reason"])) + '.</span>')
    if sty.get("caveat"):
        notes.append(
            f'<b>{sty.get("value_names", 0)} value, '
            f'{sty.get("growth_names", 0)} growth, '
            f'{sty.get("blend_names", 0)} blend.</b> ' + sty["caveat"])
    notes.append(
        'Lynch looked for names institutions had <em>not</em> found — under '
        '15–20% institutional ownership. This universe is built from index '
        'constituents, which are by construction the most heavily '
        'institution-owned equities in the world, so that bar is set here at '
        'what counts as under-owned <em>within</em> an index: 70%. It is a '
        'relative test, not Lynch\'s original one.')
    if regime.get("regime"):
        share = regime.get("below_book_share")
        share_txt = (f'{share:.1%} of {regime.get("names_scored", 0)} names'
                     if isinstance(share, (int, float)) else "not measurable")
        if regime["regime"] == "relative_value":
            notes.append(
                f'<b>Schloss is in relative-value mode.</b> Only {share_txt} '
                'trade below tangible book, under the '
                f'{regime.get("floor", 0.05):.0%} floor — so his valuation test '
                'switches to price-to-sales, which is what Edwin Schloss did in '
                'buoyant markets rather than stand aside for a decade.')
        else:
            notes.append(
                f'<b>Schloss is in deep-value mode.</b> {share_txt} trade below '
                'tangible book, so the book discount is still available and the '
                'valuation test stays on it.')
    # The two market-level gauges sit here rather than in six separate boxes:
    # both answer "what is the whole market priced at", and both belong beside
    # the categories they should temper.
    notes.append(_buffett_indicator_line(buffett_ind or {})
                 .replace('<div class="dnote">', '<span>')
                 .replace('<div class="dnote muted-ink">', '<span class="muted-ink">')
                 .replace('</div>', '</span>'))
    if b_count and b_count.get("names"):
        notes.append(
            f'<b>{b_count.get("b_labelled", 0)} of {b_count["names"]} names '
            'carry the <span class="blab">B</span> label</b> — inside the '
            'circle of competence declared in <code>config/thresholds.yml</code>, '
            'showing the footprint of a durable moat, and with a consistent '
            'operating history. Edit that list and the label moves: the circle '
            'of competence is a fact about you, not about the company, and it '
            'is the one thing here the app refuses to guess.')
    return (
        '<details class="dalio"><summary>'
        '<span class="stg">Categories &amp; market gauges</span>'
        + r20 +
        f'<span class="muted-ink">{total} names classified — the category sets '
        'the bar each name is judged against</span>'
        '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        '</summary><div class="body"><div class="dgrid">'
        f'<div class="dcard"><h4>Lynch categories</h4>{cells}</div>'
        + (f'<div class="dcard"><h4>Value or growth</h4>{sty_cells}</div>'
           if sty_cells else '')
        + '<div class="dcard"><h4>What each is judged on</h4>'
        + "".join(f'<div class="drow"><span class="k">{e_attr(ly.LABELS[c])}</span>'
                  f'<span class="v" style="text-align:left;max-width:60%">'
                  f'{e_attr(ly.RATIONALE[c])}</span></div>'
                  for c in (order or list(ly.CATEGORIES)[:4]))
        + '</div></div>'
        + "".join(f'<div class="dnote">{n}</div>' for n in notes)
        + '</div></details>')


def _commodity_panel(cb: Dict[str, Any]) -> str:
    """The commodity board — the underlying beside the instrument.

    Collapsed by default. The column that earns its place is the last one:
    what the fund returned minus what the commodity returned, over a year.
    """
    rows = (cb or {}).get("rows") or []
    if not rows:
        return ""
    body = ""
    for r in rows:
        fut = (f'<span class="sym">{e_attr(r["future"])}</span> '
               f'{_fmt_num(r.get("future_price"))}' if r.get("future_price")
               else '<span class="muted-ink">no futures line</span>')
        etf = (f'<span class="sym">{e_attr(r["etf"])}</span> '
               f'{_fmt_num(r.get("etf_price"))}' if r.get("etf_price")
               else '<span class="muted-ink">—</span>')
        gap = r.get("tracking_gap_12m")
        gap_cell = ('<td class="r">—</td>' if gap is None
                    else f'<td class="r {"up" if gap > -0.02 else "dn"}" '
                         f'title="{e_attr(str(r.get("tracking_reading", "")))}">'
                         f'{gap * 100:+.1f}%</td>')
        a = r.get("assessment") or {}
        call = a.get("commodity_call", "")
        grade = a.get("instrument_grade", "")
        gcls = {"clean": "cool", "mild drag": "", "heavy drag": "warm",
                "poor": "hot", "not the commodity": "hot"}.get(grade, "")
        ccls = ("up" if str(call).startswith("uptrend")
                else "dn" if call == "downtrend" else "")
        verdict = (f'<div style="margin-top:3px"><span class="{ccls}">'
                   f'{e_attr(call)}</span> &middot; '
                   f'<span class="pill {gcls}">{e_attr(grade)}</span></div>'
                   + (f'<div class="muted-ink" style="font-size:11px">'
                      f'{e_attr(a["flag"])}</div>' if a.get("flag") else ""))
        body += (f'<tr><td class="nm">{e_attr(r["name"])}{verdict}'
                 + (f'<div class="muted-ink" style="font-size:11px">'
                    f'{e_attr(r["note"])}</div>' if r.get("note") else "")
                 + f'</td><td>{fut}</td>'
                 + _pct_cell(r.get("future_12m"))
                 + f'<td>{etf} <span class="kind {e_attr(r["kind"])}">'
                   f'{e_attr(r["kind"])}</span></td>'
                 + _pct_cell(r.get("etf_12m"))
                 + gap_cell + '</tr>')

    miss = cb.get("missing") or []
    foot = (f'<div class="dnote"><span class="muted-ink">{len(miss)} symbol(s) '
            f'unavailable this run: {e_attr(", ".join(miss))}. Their rows show '
            f'blanks rather than stale prices.</span></div>' if miss else "")

    return (
        '<details class="dalio cmd"><summary>'
        '<span class="stg">Commodities &middot; the thing vs the instrument</span>'
        '<span class="muted-ink">Rogers: buy the commodity, not the miner &mdash; '
        'so here is what each costs and what the fund actually returned</span>'
        '<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        '</summary><div class="body">'
        '<table><thead><tr><th>Commodity</th><th>Futures</th>'
        '<th class="r">12m</th><th>Buyable ETF</th><th class="r">12m</th>'
        '<th class="r">Fund &minus; commodity</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        f'<div class="dnote">{e_attr(cb.get("caveat", ""))}</div>'
        '<div class="dnote"><b>The trend reading is not a supply call.</b> '
        'Rogers analyses the commodity first &mdash; inventories, days of '
        'consumption, rig and mine counts, project pipelines &mdash; and none '
        'of that is in a free feed. His sell rule fires on <i>rising '
        'stockpiles</i>, and this app cannot see stockpiles. So the left-hand '
        'verdict is price trend only, and a commodity can be in a clean '
        'uptrend while its warehouses are filling.</div>'
        '<div class="dnote"><b>Reading the last column.</b> It is the ETF\'s '
        'twelve-month return minus the commodity\'s. It already contains roll '
        'cost, fees, tracking error and storage, so it is what you got versus '
        'what the headline price did. A <b>negative</b> number on a futures '
        'fund is contango &mdash; the roll sells cheap and buys dear &mdash; '
        'and it is the best argument against holding one for the long term. A '
        '<b>positive</b> number is the mirror case: the curve is backwardated, '
        'each roll buys a cheaper deferred contract, and the fund legitimately '
        'beats the front-month price change. Neither state is permanent; the '
        'curve flips, and the fund\'s result flips with it.</div>'
        + foot + '</div></details>')


def _dalio_panel(d: Dict[str, Any]) -> str:
    """Ray Dalio's Big Debt Cycle stage, recomputed on every run.

    Collapsed by default: the stage and the alert level are the two things
    worth seeing at a glance, and the evidence is one click away for when the
    reading is surprising enough to want checking.
    """
    if not d:
        return ""
    if not d.get("enabled"):
        return ('<div class="dalio"><summary style="display:block">'
                '<b>Debt cycle: unavailable</b> <span class="muted-ink">— '
                f'{e_attr(str(d.get("reason", "not computed")))}</span>'
                '</summary></div>')

    stage = d.get("stage")
    chk = d.get("checklist") or {}
    lvl = chk.get("level", "GREY")
    cls_ = d.get("classification") or {}

    bar = "".join(
        f'<i class="{"on" if (i + 1) == stage else ""}">{i + 1}. '
        f'{DALIO_STAGES[i].replace(chr(10), " ")}</i>' for i in range(7))

    def rows(pairs):
        return "".join(
            f'<div class="drow"><span class="k">{k}</span>'
            f'<span class="v">{v}</span></div>' for k, v in pairs if v)

    # --- bubble checklist -------------------------------------------------
    words = {0: ("cool", "COOL"), 1: ("warm", "WARM"), 2: ("hot", "HOT")}
    chk_rows = ""
    for t in chk.get("tests", []):
        s = t.get("score")
        cl, wd = words.get(s, ("", "N/A"))
        chk_rows += (f'<div class="drow"><span class="k">{e_attr(t["label"])}'
                     f'<br><span class="muted-ink" style="font-size:11.5px">'
                     f'{e_attr(str(t.get("detail", "")))}</span></span>'
                     f'<span class="v"><span class="pill {cl}">{wd}</span></span>'
                     f'</div>')

    # --- tug of war -------------------------------------------------------
    tug = d.get("tug_of_war") or {}
    arrow = {1: "&#9650; inflationary", -1: "&#9660; deflationary",
             0: "&#9679; neutral"}
    tug_rows = ""
    for l in tug.get("levers", []):
        p = l.get("pull")
        cl = "hot" if p == 1 else "cool" if p == -1 else ""
        tug_rows += (f'<div class="drow"><span class="k"><b>'
                     f'{e_attr(l["lever"])}</b><br>'
                     f'<span class="muted-ink" style="font-size:11.5px">'
                     f'{e_attr(str(l.get("detail", "")))} — '
                     f'{e_attr(str(l.get("reading", "")))}</span></span>'
                     f'<span class="v"><span class="pill {cl}">'
                     f'{arrow.get(p, "n/a")}</span></span></div>')

    # --- the numbered checks ---------------------------------------------
    v = d.get("velocity") or {}
    sus = d.get("sustainability") or {}
    tp = d.get("tipping_point") or {}
    ew = d.get("early_warnings") or {}

    def pc(x, dp=1, sign=False):
        if x is None or (isinstance(x, float) and x != x):
            return None
        return f"{x:+.{dp}f}" if sign else f"{x:.{dp}f}"

    debt_rows = rows([
        ("Federal debt / GDP", f'{pc(v.get("fed_level"))}%'
         if v.get("fed_level") else None),
        ("&nbsp;&nbsp;3-year change", f'{pc(v.get("fed_3y_pp"), 1, True)}pp'
         if v.get("fed_3y_pp") is not None else None),
        ("Household debt / GDP", f'{pc(v.get("hh_level"))}%'
         if v.get("hh_level") else None),
        ("&nbsp;&nbsp;3-year change", f'{pc(v.get("hh_3y_pp"), 1, True)}pp'
         if v.get("hh_3y_pp") is not None else None),
        ("Federal interest, annualised",
         f'${sus["interest_saar_bn"] / 1000:.2f}tn'
         if sus.get("interest_saar_bn") else None),
        ("Interest as % of the deficit",
         f'<b>{sus["interest_to_deficit"] * 100:.0f}%</b>'
         if sus.get("interest_to_deficit") else None),
    ])
    mkt_rows = rows([
        ("10y &minus; 2y", f'{pc(tp.get("curve_10y2y"), 2, True)}%'
         if tp.get("curve_10y2y") is not None else None),
        ("10y &minus; 3m", f'{pc(tp.get("curve_10y3m"), 2, True)}%'
         if tp.get("curve_10y3m") is not None else None),
        ("Curve shape", e_attr(str(tp.get("shape", "")))),
        ("Real policy rate", f'{pc(tp.get("real_policy_rate"), 2, True)}%'
         if tp.get("real_policy_rate") is not None else None),
        ("High-yield OAS", f'{ew["hy_oas"]:.0f}bp'
         if ew.get("hy_oas") else None),
        ("&nbsp;&nbsp;3-month change", f'{ew["hy_3m_bp"]:+.0f}bp'
         if ew.get("hy_3m_bp") is not None else None),
        ("Card delinquency", f'{pc(ew.get("cc_delinq"), 2)}%'
         if ew.get("cc_delinq") else None),
        ("&nbsp;&nbsp;year-on-year", f'{pc(ew.get("cc_delinq_4q_pp"), 2, True)}pp'
         if ew.get("cc_delinq_4q_pp") is not None else None),
    ])

    assets = d.get("assets") or {}
    notes = []
    if cls_.get("sector_note"):
        notes.append(e_attr(cls_["sector_note"]))
    if cls_.get("contested"):
        notes.append(e_attr(cls_["contested"]))
    if assets.get("reading"):
        notes.append("<b>Real vs financial assets:</b> favours "
                     f'<b>{e_attr(str(assets.get("favours", "—")))}</b> — '
                     f'{e_attr(assets["reading"])}')
    if tug.get("balance"):
        notes.append(f'<b>Tug of war:</b> {e_attr(tug["balance"])}')
    miss = d.get("missing_series") or []
    if miss:
        notes.append('<span class="muted-ink">Ran with '
                     f'{len(miss)} series unavailable: '
                     f'{e_attr(", ".join(miss))}. Their tests were dropped from '
                     'the denominator rather than scored as passes.</span>')
    if d.get("unavailable"):
        notes.append('<span class="muted-ink">Not machine-readable for free, '
                     'and therefore not in the score: '
                     + e_attr("; ".join(d["unavailable"])) + '.</span>')
    cd = d.get("cape_detail") or {}
    if cd.get("cape"):
        notes.append('<span class="muted-ink">CAPE '
                     f'{cd["cape"]:.1f} is our own aggregate across '
                     f'{cd.get("names_used", 0)} US constituents with 10 years '
                     'of EDGAR earnings — not the official Shiller series.'
                     '</span>')
    elif cd.get("error"):
        notes.append('<span class="muted-ink">CAPE not computed: '
                     + e_attr(str(cd["error"])) + '.</span>')

    pct = chk.get("pct")
    score_txt = (f'{chk.get("score", 0)}/{chk.get("max", 0)}'
                 + (f' ({pct * 100:.0f}%)' if pct is not None else ''))

    return (
        f'<details class="dalio"><summary>'
        f'<span class="stg">Debt cycle &middot; stage {stage or "?"} of 7 &mdash; '
        f'{e_attr(str(d.get("stage_name", "unknown")))}</span>'
        f'<span class="alert {lvl}">{lvl}</span>'
        f'<span class="muted-ink">Bubble checklist {score_txt} &mdash; '
        f'{e_attr(str(chk.get("reason", "")))}</span>'
        f'<span class="muted-ink" style="margin-left:auto">details &#9662;</span>'
        f'</summary><div class="body">'
        f'<div class="stages">{bar}</div>'
        f'<div class="dgrid">'
        f'<div class="dcard"><h4>Bubble checklist</h4>{chk_rows}</div>'
        f'<div class="dcard"><h4>The tug of war</h4>{tug_rows}</div>'
        f'<div class="dcard"><h4>Debt &amp; sustainability</h4>{debt_rows}</div>'
        f'<div class="dcard"><h4>Tipping point &amp; early warnings</h4>'
        f'{mkt_rows}</div>'
        f'</div>'
        + "".join(f'<div class="dnote">{n}</div>' for n in notes)
        + '</div></details>')


def render(results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]],
           screened: Dict[str, Any], thresholds: Dict[str, Any],
           universe_cfg: Dict[str, Any], out_dir: str = "out",
           region: str = "all", run_id: str = "",
           report_tickers: Optional[set] = None) -> str:
    rows = build_payload(results, metrics, screened, report_tickers)

    # GITHUB_REPOSITORY is set inside Actions; omit the link when absent.
    repo = os.environ.get("GITHUB_REPOSITORY")
    wf_url = (f"https://github.com/{repo}/actions/workflows/deep-dive.yml"
              if repo else None)
    issue_url = f"https://github.com/{repo}/issues/new" if repo else None

    markets_present = sorted({r["market"] for r in rows if r["market"]})
    mkt_chips = "".join(
        f'<button class="chip" data-mkt="{m}">{MARKET_LABELS.get(m, m)}</button>'
        for m in markets_present)
    themes = list((universe_cfg.get("themes") or {}).keys())
    theme_row = ""
    if themes:
        theme_row = ('<span class="sep"></span><span class="fl">Theme</span>'
                     '<button class="chip on" data-theme="ALL">All</button>'
                     + "".join(f'<button class="chip" data-theme="{e_attr(t)}">'
                               f'{e_attr(t)}</button>' for t in themes))

    # Listing chips are built from what is ACTUALLY in the data, not from the
    # config. A config-driven row would offer a "Nasdaq listed" button on a run
    # where the directory fetch failed and nothing was added — a filter that
    # returns zero rows and looks like a bug in the page rather than what it
    # is: coverage that did not arrive. Counting the rows also answers the
    # question directly, which is the whole reason this row exists.
    listings: Dict[str, int] = {}
    for r in rows:
        lb = (r.get("listing") or "").strip()
        if lb:
            listings[lb] = listings.get(lb, 0) + 1
    listing_row = ""
    if len(listings) > 1:
        listing_row = (
            '<span class="sep"></span><span class="fl">Listing</span>'
            '<button class="chip on" data-listing="ALL">All</button>'
            + "".join(
                f'<button class="chip" data-listing="{e_attr(k)}" '
                f'title="{v} names came into this run on the {e_attr(k)} list">'
                f'{e_attr(k)} <b>{v}</b></button>'
                for k, v in sorted(listings.items(), key=lambda kv: -kv[1])))

    fw_chips = "".join(
        f'<button class="chip" data-fw="{k}">{lbl}</button>' for k, lbl in FRAMEWORKS)
    fw_head = "".join(f'<th class="c">{lbl[:4]}</th>' for _k, lbl in FRAMEWORKS)

    open_ = screened.get("macro_gate_open", True)
    reason = screened.get("macro_gate_reason", "")
    cyc = screened.get("cycle") or {}
    cycle_html = ""
    if cyc.get("mode"):
        mode = cyc["mode"]
        shift = cyc.get("threshold_shift", 0)
        shift_txt = ("Marks needs one MORE passing test in this posture."
                     if shift > 0 else
                     "Marks accepts one FEWER passing test in this posture."
                     if shift < 0 else "Marks runs at its normal bar.")
        cycle_html = (
            f'<div class="gate cyc-{mode}"><b>Market cycle: {mode.upper()}</b> — '
            f'{cyc.get("evidence", "")}.<br>{cyc.get("reason", "")} '
            f'<span class="muted-ink">{shift_txt}</span></div>')

    debt_html = _dalio_panel(screened.get("debt_cycle") or {})
    cmd_html = _commodity_panel(screened.get("commodity_board") or {})
    rfx_html = _reflexive_legend(screened.get("reflexive_census") or {})
    dis_html = _dislocation_panel(screened.get("dislocation_summary") or {})
    sen_html = _sentiment_panel(screened.get("sentiment") or {})
    mf_html = _magic_formula_panel(screened.get("magic_formula") or {})
    buf_html = _ranked_panel(
        "Buffett list",
        "A wonderful business at a fair price, as two rankings: how much the "
        "business earns on the tangible capital it actually needs, and how much "
        "cash it throws off against what it costs. Eligibility is the three "
        "business tenets — the B label — because a list carrying his name that "
        "includes businesses outside your circle of competence, without a moat, "
        "or mid-turnaround would not be his list.",
        screened.get("buffett_ranking") or {})
    mun_html = _ranked_panel(
        "Munger list",
        "Munger's contribution is not a valuation method, it is inversion: "
        "&ldquo;all I want to know is where I&rsquo;m going to die, so I&rsquo;ll "
        "never go there.&rdquo; So this ranks return on capital against how many "
        "of the obvious ways to lose money are ABSENT — earnings that turn into "
        "cash, book value that is not mostly goodwill, no dangerous leverage, a "
        "margin that is not eroding, earnings predictable enough to estimate, no "
        "dilution, no loss years. That second axis is what makes this a "
        "different list from Buffett&rsquo;s rather than a reordering of it.",
        screened.get("munger_ranking") or {},
        ["The inversion score is the share of those checks that could be "
         "evaluated and came back clean, so a four-year feed is judged on the "
         "same standard as a ten-year one rather than marked down for its "
         "provider. The number evaluated travels with each row in the drawer."])
    lyn_html = _lynch_panel(screened.get("lynch_census") or {},
                            screened.get("value_regime") or {},
                            screened.get("rule_of_20") or {},
                            screened.get("buffett_indicator") or {},
                            screened.get("buffett_valuation") or {},
                            screened.get("style_census") or {})

    gate = (f'<div class="gate {"open" if open_ else "closed"}">'
            f'<b>Soros macro gate: {"OPEN" if open_ else "CLOSED"}</b> — {reason}.'
            + ("" if open_ else " Every Soros signal is suppressed while credit "
               "conditions are stressed — single-name momentum stops meaning what "
               "it normally means in this regime.")
            + '</div>')

    # Written BEFORE the page is assembled, because the footer reports what was
    # written and a count that is computed after the string it appears in is a
    # count that never appears.
    os.makedirs(out_dir, exist_ok=True)
    series_counts = write_series(results, metrics, out_dir)

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, default=str))
            .replace("__FWS__", json.dumps(FRAMEWORKS))
            .replace("__CATS__", json.dumps(CAT_SHORT))
            .replace("__STY__", json.dumps(STYLE_SHORT))
            .replace("__WFURL__", json.dumps(wf_url))
            .replace("__ISSUEURL__", json.dumps(issue_url))
            .replace("__MKTCHIPS__", mkt_chips)
            .replace("__FWCHIPS__", fw_chips)
            .replace("__FWHEAD__", fw_head)
            .replace("__THEMEROW__", theme_row)
            .replace("__LISTINGROW__", listing_row)
            .replace("__GATE__", debt_html + sen_html + mf_html + buf_html
                     + mun_html + cmd_html
                     + lyn_html + rfx_html + dis_html + cycle_html + gate)
            .replace("__REGION__", {"us": "US", "asia": "Asia", "all": "Full"}[region])
            .replace("__RUNID__", run_id or "—")
            .replace("__SERIES__", ", ".join(
                f"{MARKET_LABELS.get(k, k)} {v}"
                for k, v in sorted(series_counts.items())) or "none written")
            .replace("__TS__", datetime.now(timezone.utc)
                     .strftime("%Y-%m-%d %H:%M UTC")))

    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "region": region,
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "macro_gate_open": open_, "macro_gate_reason": reason,
                   "rows": rows}, f, indent=2, default=str)
    return path
