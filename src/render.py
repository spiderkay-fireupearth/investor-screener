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

FRAMEWORKS = [
    ("buffett", "Buffett"), ("munger", "Munger"), ("schloss", "Schloss"),
    ("klarman", "Klarman"), ("lynch", "Lynch"), ("templeton", "Templeton"),
    ("marks", "Marks"), ("greenblatt", "Greenblatt"), ("soros", "Soros"),
    ("rogers", "Rogers"), ("graham", "Graham"),
]

MARKET_LABELS = {"US": "US large cap", "JP": "Nikkei 225", "SG": "SGX",
                 "HK": "HKEX", "TH": "SET", "ID": "IDX"}


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
            # Not enough YEARS to judge — a property of the data feed, not the
            # company. Shown greyed and excluded from the score, so a shallow
            # feed never masquerades as a failing business.
            state = "na"
            thresh = t.get("note") or "insufficient history"
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
            if f.get("limited_history") and not note:
                note = (f"judged on {m.get('history_years', '?')} years of statements — "
                        f"{f.get('n_insufficient')} test(s) need a longer window and "
                        f"were excluded, with the bar scaled down to match")
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
            "surfaced": bool(r.get("surfaced")),
            "has_report": ticker in report_tickers,
            "themes": r.get("themes") or [],
            "is_fund": bool(r.get("is_fund")),
            "gates": r.get("gates_failed", []),
            "warnings": r.get("warnings", []),
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
                "RSI(14)": _fmt_num(m.get("rsi_14"), 1),
                "vs 200d MA": "above" if m.get("price_above_sma200") else "below",
                "RS 6m vs index": _fmt_num(m.get("rs_vs_market_index_6m"), 1, pct=True),
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
.rfx.late{background:rgba(226,88,94,.16);color:var(--bad);border-color:rgba(226,88,94,.45)}
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
const WF_URL = __WFURL__;
const ISSUE_URL = __ISSUEURL__;
let fMkt="ALL", fFw=new Set(), fTech=false, fSurf=true, fQ="", fTheme="ALL";
let fRfx=false;   // Soros stage DE/EF only

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
    if(fRfx && !r.rfx_late) return false;
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
    <span class="tv ${t.state}">${esc(t.value)} <span style="color:var(--tx3)">/ ${esc(t.threshold)}</span></span></div>`).join('');
}

function detail(r){
  let h='<div class="dwrap">';
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
    let cells=`<td class="tk">${esc(r.ticker)}${r.has_report?'<span class="dd" title="deep dive available">&#9670;</span>':''}${rx}</td><td class="nm" title="${esc(r.name)}">${esc(r.name)}</td>
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

    gate = (f'<div class="gate {"open" if open_ else "closed"}">'
            f'<b>Soros macro gate: {"OPEN" if open_ else "CLOSED"}</b> — {reason}.'
            + ("" if open_ else " Every Soros signal is suppressed while credit "
               "conditions are stressed — single-name momentum stops meaning what "
               "it normally means in this regime.")
            + '</div>')

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, default=str))
            .replace("__FWS__", json.dumps(FRAMEWORKS))
            .replace("__WFURL__", json.dumps(wf_url))
            .replace("__ISSUEURL__", json.dumps(issue_url))
            .replace("__MKTCHIPS__", mkt_chips)
            .replace("__FWCHIPS__", fw_chips)
            .replace("__FWHEAD__", fw_head)
            .replace("__THEMEROW__", theme_row)
            .replace("__GATE__", debt_html + cmd_html + cycle_html + gate)
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
