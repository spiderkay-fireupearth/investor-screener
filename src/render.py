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
    ("klarman", "Klarman"), ("lynch", "Lynch"), ("greenblatt", "Greenblatt"),
    ("soros", "Soros"),
]

MARKET_LABELS = {"US": "S&P 500", "SG": "SGX", "HK": "HKEX", "TH": "SET", "ID": "IDX"}


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
        rows.append({
            "name": t["name"].replace("_", " "),
            "metric": t.get("metric") or "",
            "value": _fmt_num(t.get("value"), 3),
            "threshold": ("rank " + str(t.get("rank"))) if t.get("operator") == "rank"
                         else f"{t.get('operator', '')} {_fmt_num(t.get('threshold'), 3)}",
            "state": "pass" if res is True else ("fail" if res is False else "unknown"),
            "alt": bool(t.get("via_alt")),
        })
    return rows


def build_payload(results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]],
                  screened: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            fw_detail[key] = {
                "label": f.get("label", key),
                "passed": bool(f.get("passed")),
                "summary": f"{f.get('n_passed', 0)}/{f.get('n_total', 0)} tests"
                           + (f", need {f.get('required')}" if f.get("required") else ""),
                "note": f.get("ineligible_reason") or f.get("macro_gate_blocked") or "",
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
            "surfaced": bool(r.get("surfaced")),
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
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:16px 0 10px;
padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:10px}
.fl{font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.06em;margin-right:2px}
.chip{font-size:12.5px;padding:4px 11px;border-radius:20px;border:1px solid var(--line);
background:transparent;color:var(--tx2);cursor:pointer;font-weight:500}
.chip:hover{border-color:var(--acc);color:var(--tx)}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
.sep{width:1px;height:20px;background:var(--line);margin:0 4px}
input[type=search]{background:var(--panel2);border:1px solid var(--line);border-radius:7px;
padding:5px 10px;color:var(--tx);font-size:13px;min-width:150px}
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
.kmet{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:7px;margin-bottom:14px}
.kmet div{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:7px 9px}
.kmet .kl{font-size:9.5px;color:var(--tx3);text-transform:uppercase;letter-spacing:.05em}
.kmet .kv{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.warn{background:rgba(201,162,39,.12);border:1px solid rgba(201,162,39,.3);border-radius:7px;
padding:8px 11px;margin-bottom:12px;font-size:12px;color:var(--warn)}
.note{background:var(--panel);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:11px 15px;margin:14px 0;font-size:13px;color:var(--tx2)}
.empty{text-align:center;padding:50px;color:var(--tx3)}
.foot{color:var(--tx3);font-size:12px;margin-top:32px;padding-top:14px;border-top:1px solid var(--line)}
</style></head><body><div class="wrap">

<div class="eyebrow">__REGION__ refresh · run __RUNID__</div>
<h1>Value + Technical Screener</h1>
<div class="meta">S&amp;P 500 · SGX · HKEX · SET · IDX &nbsp;·&nbsp; generated __TS__</div>

<div class="statbar" id="statbar"></div>
__GATE__

<div class="filters">
  <span class="fl">Market</span>
  <button class="chip on" data-mkt="ALL">All</button>
  __MKTCHIPS__
  <span class="sep"></span>
  <span class="fl">Must pass</span>
  __FWCHIPS__
  <span class="sep"></span>
  <button class="chip" id="techOnly">Technical pass</button>
  <button class="chip on" id="surfOnly">Surfaced only</button>
  <span class="sep"></span>
  <input type="search" id="q" placeholder="ticker or name">
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
let fMkt="ALL", fFw=new Set(), fTech=false, fSurf=true, fQ="";

function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

function visible(){
  return DATA.filter(r=>{
    if(fSurf && !r.surfaced) return false;
    if(fMkt!=="ALL" && r.market!==fMkt) return false;
    if(fTech && !r.tech_pass) return false;
    for(const f of fFw){ if(r.fw[f]!=="pass") return false; }
    if(fQ){ const q=fQ.toLowerCase();
      if(!r.ticker.toLowerCase().includes(q) && !r.name.toLowerCase().includes(q)) return false; }
    return true;
  });
}

function stats(rows){
  const byMkt={};
  rows.forEach(r=>byMkt[r.market_label]=(byMkt[r.market_label]||0)+1);
  const tech=rows.filter(r=>r.tech_pass).length;
  let h=`<div class="stat"><div class="v">${rows.length}</div><div class="l">Names shown</div></div>`;
  h+=`<div class="stat"><div class="v">${tech}</div><div class="l">Technical pass</div></div>`;
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
  const rows=visible(); stats(rows);
  const tb=document.getElementById('tbody'); tb.innerHTML='';
  document.getElementById('empty').style.display=rows.length?'none':'block';
  rows.forEach((r,i)=>{
    const tr=document.createElement('tr'); tr.className='row';
    let cells=`<td class="tk">${esc(r.ticker)}</td><td class="nm" title="${esc(r.name)}">${esc(r.name)}</td>
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
document.querySelectorAll('[data-fw]').forEach(b=>b.onclick=()=>{
  const k=b.dataset.fw;
  if(fFw.has(k)){fFw.delete(k);b.classList.remove('on');}else{fFw.add(k);b.classList.add('on');}
  render();});
document.getElementById('techOnly').onclick=function(){
  fTech=!fTech; this.classList.toggle('on',fTech); render();};
document.getElementById('surfOnly').onclick=function(){
  fSurf=!fSurf; this.classList.toggle('on',fSurf); render();};
document.getElementById('q').oninput=e=>{fQ=e.target.value; render();};
render();
</script></body></html>
"""


def render(results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]],
           screened: Dict[str, Any], thresholds: Dict[str, Any],
           universe_cfg: Dict[str, Any], out_dir: str = "out",
           region: str = "all", run_id: str = "") -> str:
    rows = build_payload(results, metrics, screened)

    markets_present = sorted({r["market"] for r in rows if r["market"]})
    mkt_chips = "".join(
        f'<button class="chip" data-mkt="{m}">{MARKET_LABELS.get(m, m)}</button>'
        for m in markets_present)
    fw_chips = "".join(
        f'<button class="chip" data-fw="{k}">{lbl}</button>' for k, lbl in FRAMEWORKS)
    fw_head = "".join(f'<th class="c">{lbl[:4]}</th>' for _k, lbl in FRAMEWORKS)

    open_ = screened.get("macro_gate_open", True)
    reason = screened.get("macro_gate_reason", "")
    gate = (f'<div class="gate {"open" if open_ else "closed"}">'
            f'<b>Soros macro gate: {"OPEN" if open_ else "CLOSED"}</b> — {reason}.'
            + ("" if open_ else " Every Soros signal is suppressed while credit "
               "conditions are stressed — single-name momentum stops meaning what "
               "it normally means in this regime.")
            + '</div>')

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, default=str))
            .replace("__FWS__", json.dumps(FRAMEWORKS))
            .replace("__MKTCHIPS__", mkt_chips)
            .replace("__FWCHIPS__", fw_chips)
            .replace("__FWHEAD__", fw_head)
            .replace("__GATE__", gate)
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
