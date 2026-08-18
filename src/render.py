"""Static HTML screener output — self-contained, no build step, no server.

Pass/fail presentation as specified: a name either clears a framework's tests
or it doesn't. The detail drawer exists so a fail is never a black box — you can
always see which test failed, on what value, against what threshold. That
matters more than it sounds: most of the time a surprising result is a data
problem, not an investment insight, and this is how you tell the difference.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import synopsis as syn

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
)

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
            # Buffett's three business tenets, as a label plus its evidence.
            "b": bool(m.get("buffett_b_label")),
            "b_summary": m.get("buffett_tenets_summary") or "",
            "b_detail": m.get("business_tenets") or {},
            "surfaced": bool(r.get("surfaced")),
            "has_report": ticker in report_tickers,
            "themes": r.get("themes") or [],
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
                "Years CFO positive (5y)": _fmt_num(m.get("cfo_positive_share_5y"),
                                                    0, pct=True),
                "Loss years in 5": _fmt_num(m.get("loss_years_in_5"), 0),
            },
        })
    rows.sort(key=lambda x: (-x["n_passed"], -x["mcap_sort"]))
    return rows


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Value + Technical Screener</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2b303b;--tx:#e6e8ec;
--tx2:#a3aab8;--tx3:#6f7789;--acc:#5b9dff;--ok:#3fbf7f;--bad:#e2585e;--warn:#c9a227;}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f2f6;
--line:#dfe3ea;--tx:#12151b;--tx2:#4d5567;--tx3:#798193;--acc:#1f6feb;--warn:#8a6d10;}}
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
  <span class="sep"></span>
  <button class="chip" id="techOnly">Technical pass</button>
  <button class="chip" id="disOnly" title="Every name down more than 30% over six months, whatever the reason">Fell &gt;30% (6m)</button>
  <button class="chip" id="disQual" title="Of those, the ones whose last published accounts do NOT explain the fall">&hellip; accounts intact</button>
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
Macro: FRED. Not investment advice — a screen is a starting point for research, not a conclusion.</div>
</div>

<script>
const DATA = __DATA__;
const FWS = __FWS__;
const CATS = __CATS__;
const WF_URL = __WFURL__;
const ISSUE_URL = __ISSUEURL__;
let fMkt="ALL", fFw=new Set(), fTech=false, fSurf=true, fQ="", fTheme="ALL";
let fRfx=false;   // Soros stage DE/EF only
let fDis=false;   // fell >30% in 6m
let fDisQ=false;  // ...and the accounts do not explain it
let fB=false;     // Buffett's three business tenets

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
    if(fRfx && !r.rfx_late) return false;
    if(fDis && !r.dis_fell) return false;
    if(fDisQ && !r.dis) return false;
    if(fSurf && !r.surfaced) return false;
    if(fMkt!=="ALL" && r.market!==fMkt) return false;
    if(fTheme!=="ALL" && !(r.themes||[]).includes(fTheme)) return false;
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
  if(r.themes && r.themes.length)
    h+='<div class="tagrow">'+r.themes.map(t=>`<span class="tag">${esc(t)}</span>`).join('')+'</div>';
  // Lynch's category decides which bar this row was judged against, so it is
  // stated before the panels rather than buried inside one of them.
  if(r.cat_label)
    h+='<div class="note" style="border-left-color:var(--tx3)"><b>Lynch category: '
      +esc(r.cat_label)+'.</b> '+esc(r.cat_why)+'. The Lynch panel below is scored '
      +'on this category&rsquo;s benchmarks; tests that do not apply to it are '
      +'excluded rather than failed.</div>';
  if(r.cat_warn)
    h+='<div class="warn"><b>Cyclical warning.</b> '+esc(r.cat_warn)+'</div>';
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
  const t=r.tech_detail;
  h+=`<div class="dcard"><h4>Technical timing
      <span class="badge ${r.tech_pass?'pass':'fail'}">${r.tech_pass?'pass':'fail'}</span></h4>
      <div style="font-size:11px;color:var(--tx3);margin-bottom:5px">${esc(t.summary)}</div>
      ${testList(t.tests)}</div>`;
  return h+'</div></div>';
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
      : `No names match these filters.<br><span style="font-size:12.5px">Try turning off
         <b>Surfaced only</b> to see every name in the universe with its per-test
         results.</span>`;
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
    let cells=`<td class="tk">${esc(r.ticker)}${r.has_report?'<span class="dd" title="deep dive available">&#9670;</span>':''}${bl}${ct}${rx}${dl}</td><td class="nm" title="${esc(r.name)}${r.syn&&r.syn.one_liner?' — '+esc(r.syn.one_liner):''}">${esc(r.name)}</td>
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
    tr.onclick=()=>{dr.style.display=dr.style.display==='none'?'table-row':'none';};
    tb.appendChild(tr); tb.appendChild(dr);
  });
}

document.querySelectorAll('[data-mkt]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-mkt]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); fMkt=b.dataset.mkt; render();});
document.querySelectorAll('[data-theme]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-theme]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); fTheme=b.dataset.theme; render();});
document.querySelectorAll('[data-fw]').forEach(b=>b.onclick=()=>{
  const k=b.dataset.fw;
  if(fFw.has(k)){fFw.delete(k);b.classList.remove('on');}else{fFw.add(k);b.classList.add('on');}
  render();});
document.getElementById('techOnly').onclick=function(){
  fTech=!fTech; this.classList.toggle('on',fTech); render();};
document.getElementById('rfxOnly').onclick=function(){
  fRfx=!fRfx; this.classList.toggle('on',fRfx); render();};
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
    """A legend for the stage badges, plus the census that keeps them honest.

    Two jobs. First, a two-letter code with no key is a puzzle, not a label.
    Second — and this is why the counts are shown rather than just the key —
    a stage that fires on most of the universe has stopped discriminating.
    Printing the distribution means over-firing is visible on the page rather
    than something you have to go and audit.
    """
    if not census:
        return ""
    order = [("AB", "trend unrecognised"), ("BC", "recognition"),
             ("CD", "tested and held"), ("DE", "conviction through a setback"),
             ("EF", "expectations excessive"), ("FG", "de-rating"),
             ("GH", "break, fundamentals following"), ("HI", "pessimism overdone"),
             ("EQ", "near-equilibrium — framework off")]
    total = sum(census.values()) or 1
    bits = []
    for code, label in order:
        n = census.get(code, 0)
        if not n:
            continue
        share = n / total
        cls = "rfx late" if code in ("DE", "EF") else "rfx"
        bits.append(f'<span class="{cls}">{code}</span>&nbsp;{label} '
                    f'<b>{n}</b> <span class="muted-ink">({share:.0%})</span>')
    if not bits:
        return ""

    # The honesty check, stated on the page.
    late = census.get("DE", 0) + census.get("EF", 0)
    warn = ""
    if late / total > 0.5:
        warn = ('<div class="dnote"><b>Read this before trusting the badges.</b> '
                f'{late / total:.0%} of the universe is reading DE or EF. A '
                'stage that fires on most names is not identifying anything — '
                'it is describing a market where prices rose faster than '
                'earnings across the board, which is a fact about the index '
                'rather than about these companies. Treat the badge as '
                'contextual until that share falls.</div>')
    elif census.get("EQ", 0) / total > 0.8:
        warn = ('<div class="dnote"><span class="muted-ink">Most names show '
                'no reflexive channel at all, which is the expected result — '
                'Soros\'s framework describes prices that CHANGE fundamentals, '
                'and most companies do not transact in their own equity.</span>'
                '</div>')

    return ('<details class="dalio"><summary>'
            '<span class="stg">Reflexive stage &middot; what the badges mean</span>'
            '<span class="muted-ink">Soros\'s boom/bust path, one label per '
            'name</span><span class="muted-ink" style="margin-left:auto">'
            'details &#9662;</span></summary><div class="body">'
            + '<div class="dnote">' + ' &nbsp;&middot;&nbsp; '.join(bits) + '</div>'
            + warn
            + '<div class="dnote"><span class="muted-ink">The path runs AB to '
              'HI. DE is the diagnostic one: price rising through an earnings '
              'setback near the highs. GH is where reflexivity is confirmed '
              'rather than asserted, because the earnings deteriorated AFTER '
              'the price did.</span></div>'
            '</div></details>')


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


CAT_SHORT = {"fast_grower": "FG", "stalwart": "SW", "slow_grower": "SG",
             "cyclical": "CY", "turnaround": "TA", "asset_play": "AP",
             "unclassified": "?"}


def _lynch_panel(census: Dict[str, int], regime: Dict[str, Any],
                 rule20: Optional[Dict[str, Any]] = None,
                 buffett_ind: Optional[Dict[str, Any]] = None,
                 b_count: Optional[Dict[str, Any]] = None) -> str:
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
        f'<div class="dcard"><h4>Universe split</h4>{cells}</div>'
        '<div class="dcard"><h4>What each is judged on</h4>'
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
    lyn_html = _lynch_panel(screened.get("lynch_census") or {},
                            screened.get("value_regime") or {},
                            screened.get("rule_of_20") or {},
                            screened.get("buffett_indicator") or {},
                            screened.get("buffett_valuation") or {})

    gate = (f'<div class="gate {"open" if open_ else "closed"}">'
            f'<b>Soros macro gate: {"OPEN" if open_ else "CLOSED"}</b> — {reason}.'
            + ("" if open_ else " Every Soros signal is suppressed while credit "
               "conditions are stressed — single-name momentum stops meaning what "
               "it normally means in this regime.")
            + '</div>')

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, default=str))
            .replace("__FWS__", json.dumps(FRAMEWORKS))
            .replace("__CATS__", json.dumps(CAT_SHORT))
            .replace("__WFURL__", json.dumps(wf_url))
            .replace("__ISSUEURL__", json.dumps(issue_url))
            .replace("__MKTCHIPS__", mkt_chips)
            .replace("__FWCHIPS__", fw_chips)
            .replace("__FWHEAD__", fw_head)
            .replace("__THEMEROW__", theme_row)
            .replace("__GATE__", debt_html + sen_html + mf_html + cmd_html
                     + lyn_html + rfx_html + dis_html + cycle_html + gate)
            .replace("__REGION__", {"us": "US", "asia": "Asia", "all": "Full"}[region])
            .replace("__RUNID__", run_id or "—")
            .replace("__TS__", datetime.now(timezone.utc)
                     .strftime("%Y-%m-%d %H:%M UTC")))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "region": region,
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "macro_gate_open": open_, "macro_gate_reason": reason,
                   "rows": rows}, f, indent=2, default=str)
    return path
