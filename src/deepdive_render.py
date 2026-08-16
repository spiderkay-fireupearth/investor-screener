"""HTML report for the single-ticker deep dive.

Charts are inline SVG built here rather than pulled from a library: the report
has to be a single self-contained file that opens from a static host with no
build step. Colours come from the validated categorical palette — blue, orange,
aqua in fixed slot order, never cycled — and every series is direct-labelled, so
identity never rests on colour alone (the light-mode aqua sits below 3:1 against
the surface, which makes labels mandatory rather than optional).
"""
from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

FRAMEWORKS = [("buffett", "Buffett"), ("munger", "Munger"), ("schloss", "Schloss"),
              ("klarman", "Klarman"), ("lynch", "Lynch"),
              ("greenblatt", "Greenblatt"), ("soros", "Soros")]


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and math.isnan(x))


def f(v, dp=2, pct=False, money=False, dash="—"):
    if not _n(v):
        return dash
    if pct:
        return f"{v * 100:.{dp}f}%"
    if money:
        for u, d in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if abs(v) >= d:
                return f"{v / d:.2f}{u}"
        return f"{v:,.0f}"
    return f"{v:,.{dp}f}"


def pct_diff(target, spot):
    if not (_n(target) and _n(spot)) or spot == 0:
        return "—"
    d = (target - spot) / spot
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}%"


def e(s):
    return html.escape(str(s if s is not None else ""))


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------
def _scale(vals, lo, hi, out_lo, out_hi):
    if hi == lo:
        return [(out_lo + out_hi) / 2 for _ in vals]
    return [out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo) for v in vals]


def price_chart(series: Dict[str, Any], w=880, h=300) -> str:
    """Close with 50- and 200-day averages. Three series, direct-labelled."""
    close = series.get("close") or []
    if len(close) < 30:
        return '<p class="muted">Not enough history to chart.</p>'
    sma50 = series.get("sma50") or []
    sma200 = series.get("sma200") or []
    dates = series.get("dates") or []

    pad_l, pad_r, pad_t, pad_b = 8, 74, 14, 24
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    allv = [v for v in close + sma50 + sma200 if _n(v)]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1
    lo -= span * 0.06
    hi += span * 0.06

    n = len(close)
    xs = [pad_l + i / max(n - 1, 1) * plot_w for i in range(n)]

    def path(vals):
        pts, started = [], False
        for i, v in enumerate(vals):
            if not _n(v):
                started = False
                continue
            y = pad_t + plot_h - (v - lo) / (hi - lo) * plot_h
            pts.append(f"{'M' if not started else 'L'}{xs[i]:.1f},{y:.1f}")
            started = True
        return " ".join(pts)

    def endpoint(vals):
        for i in range(len(vals) - 1, -1, -1):
            if _n(vals[i]):
                return xs[i], pad_t + plot_h - (vals[i] - lo) / (hi - lo) * plot_h, vals[i]
        return None

    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = pad_t + plot_h - (v - lo) / (hi - lo) * plot_h
        grid += (f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                 f'class="grid"/><text x="{pad_l + plot_w + 6}" y="{y + 3.5:.1f}" '
                 f'class="axis">{f(v)}</text>')

    labels = ""
    for vals, cls, name in ((close, "s1", "Close"), (sma50, "s2", "50-day"),
                            (sma200, "s3", "200-day")):
        ep = endpoint(vals)
        if ep:
            x, y, _v = ep
            labels += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" class="dot {cls}"/>'
                       f'<text x="{x - 6:.1f}" y="{y - 8:.1f}" class="dlabel {cls}" '
                       f'text-anchor="end">{name}</text>')

    xlab = ""
    if dates:
        for i in (0, n // 2, n - 1):
            if i < len(dates):
                anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
                xlab += (f'<text x="{xs[i]:.1f}" y="{h - 6}" class="axis" '
                         f'text-anchor="{anchor}">{e(dates[i][:7])}</text>')

    return f'''<svg viewBox="0 0 {w} {h}" class="chart" role="img"
  aria-label="Closing price with 50-day and 200-day moving averages">
  {grid}
  <path d="{path(sma200)}" class="line s3" fill="none"/>
  <path d="{path(sma50)}" class="line s2" fill="none"/>
  <path d="{path(close)}" class="line s1" fill="none"/>
  {labels}{xlab}
</svg>'''


def fan_chart(gbm: Dict[str, Any], w=880, h=300) -> str:
    """Monte Carlo percentile bands — one sequential blue hue, light to dark."""
    band = gbm.get("band") or {}
    if not band.get("p50"):
        return '<p class="muted">Simulation unavailable.</p>'
    p10, p25, p50 = band["p10"], band["p25"], band["p50"]
    p75, p90 = band["p75"], band["p90"]
    s0 = gbm.get("s0")

    pad_l, pad_r, pad_t, pad_b = 8, 74, 14, 24
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    allv = p10 + p90 + ([s0] if _n(s0) else [])
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1
    lo -= span * 0.05
    hi += span * 0.05
    n = len(p50)
    xs = [pad_l + i / max(n - 1, 1) * plot_w for i in range(n)]

    def y(v):
        return pad_t + plot_h - (v - lo) / (hi - lo) * plot_h

    def area(a, b):
        top = " ".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{y(v):.1f}"
                       for i, v in enumerate(a))
        bot = " ".join(f"L{xs[i]:.1f},{y(v):.1f}"
                       for i in range(len(b) - 1, -1, -1) for v in [b[i]])
        return top + " " + bot + " Z"

    def line(vals):
        return " ".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{y(v):.1f}"
                        for i, v in enumerate(vals))

    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{pad_l + plot_w}" '
                 f'y2="{y(v):.1f}" class="grid"/>'
                 f'<text x="{pad_l + plot_w + 6}" y="{y(v) + 3.5:.1f}" '
                 f'class="axis">{f(v)}</text>')

    spot = ""
    if _n(s0):
        spot = (f'<line x1="{pad_l}" y1="{y(s0):.1f}" x2="{pad_l + plot_w}" '
                f'y2="{y(s0):.1f}" class="spot"/>'
                f'<text x="{pad_l + 4}" y="{y(s0) - 6:.1f}" class="dlabel muted-ink">'
                f'spot {f(s0)}</text>')

    return f'''<svg viewBox="0 0 {w} {h}" class="chart" role="img"
  aria-label="Monte Carlo simulated price bands">
  {grid}
  <path d="{area(p90, p10)}" class="band b1"/>
  <path d="{area(p75, p25)}" class="band b2"/>
  <path d="{line(p50)}" class="line med" fill="none"/>
  {spot}
  <text x="{xs[-1] - 6:.1f}" y="{y(p90[-1]) - 6:.1f}" class="dlabel muted-ink"
    text-anchor="end">90th</text>
  <text x="{xs[-1] - 6:.1f}" y="{y(p50[-1]) - 8:.1f}" class="dlabel med-ink"
    text-anchor="end">median</text>
  <text x="{xs[-1] - 6:.1f}" y="{y(p10[-1]) + 14:.1f}" class="dlabel muted-ink"
    text-anchor="end">10th</text>
</svg>'''


# ---------------------------------------------------------------------------
def _fw_table(frameworks: Dict[str, Any]) -> str:
    out = []
    for key, label in FRAMEWORKS:
        fw = frameworks.get(key, {})
        passed = fw.get("passed")
        note = fw.get("ineligible_reason") or fw.get("macro_gate_blocked") or ""
        if fw.get("limited_history") and not note:
            note = f"{fw.get('n_insufficient')} test(s) need a longer history"
        state = ("pass" if passed else
                 ("na" if fw.get("ineligible_reason") else "fail"))
        verdict = {"pass": "Buy-side signal", "fail": "Avoid on this lens",
                   "na": "Not applicable"}[state]
        tests = "".join(
            f'<div class="trow"><span class="tn">{e(t["name"].replace("_", " "))}</span>'
            f'<span class="tv {"pass" if t.get("result") is True else ("na" if t.get("insufficient") else ("fail" if t.get("result") is False else "unknown"))}">'
            f'{f(t.get("value"), 3) if not t.get("insufficient") else "—"}'
            f'<span class="thr"> / {e(t.get("note") or (str(t.get("operator", "")) + " " + f(t.get("threshold"), 3)))}</span>'
            f'</span></div>' for t in fw.get("tests", []))
        eff = fw.get("effective_total", fw.get("n_total", 0))
        out.append(f'''<div class="fwcard {state}">
  <h4>{e(label)}<span class="badge {state}">{"PASS" if state == "pass" else ("N/A" if state == "na" else "FAIL")}</span></h4>
  <div class="sub">{e(fw.get("label", ""))} · {fw.get("n_passed", 0)}/{eff} tests, need {fw.get("required", "—")}</div>
  {f'<div class="lim">{e(note)}</div>' if note else ""}
  <div class="verdict">{verdict}</div>
  {tests}
</div>''')
    return '<div class="fwgrid">' + "".join(out) + "</div>"


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TICKER__ — Deep Dive</title>
<style>
.viz-root, body{
  color-scheme:light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --seq-200:#9ec5f4; --seq-350:#5598e7; --seq-450:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){ .viz-root, body{
  color-scheme:dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --seq-200:#184f95; --seq-350:#256abf; --seq-450:#3987e5;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--text-primary);
 font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:30px 20px 80px}
h1{font-size:26px;margin:0 0 2px;letter-spacing:-.02em}
h2{font-size:17px;margin:38px 0 6px;padding-top:16px;border-top:1px solid var(--border);letter-spacing:-.01em}
h3{font-size:14px;margin:20px 0 6px}
h4{font-size:13.5px;margin:0 0 4px;display:flex;justify-content:space-between;align-items:center;gap:8px}
p{margin:8px 0}
.eyebrow{font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.sub{color:var(--text-secondary);font-size:12.5px}
.muted,.muted-ink{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th{text-align:left;padding:8px;color:var(--muted);font-size:10.5px;letter-spacing:.06em;
 text-transform:uppercase;border-bottom:1px solid var(--border);font-weight:600}
td{padding:8px;border-bottom:1px solid var(--border);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.hero{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;
 padding:20px 22px;margin:16px 0;display:flex;gap:26px;flex-wrap:wrap;align-items:center}
.call{font-size:34px;font-weight:700;letter-spacing:-.03em}
.call.BUY{color:var(--good)} .call.HOLD{color:var(--warning)} .call.AVOID{color:var(--critical)}
.heroitem .l{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.heroitem .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:12px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
.fwgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(238px,1fr));gap:12px;margin:12px 0}
.fwcard{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.fwcard.pass{border-color:rgba(12,163,12,.42)}
.badge{font-size:9.5px;font-weight:700;padding:2px 7px;border-radius:4px;letter-spacing:.05em}
.badge.pass{background:rgba(12,163,12,.16);color:var(--good)}
.badge.fail{background:rgba(208,59,59,.13);color:var(--critical)}
.badge.na{background:var(--plane);color:var(--muted)}
.verdict{font-size:11.5px;color:var(--text-secondary);margin:5px 0 7px}
.lim{font-size:10.5px;color:var(--muted);font-style:italic;margin-bottom:4px}
.trow{display:flex;justify-content:space-between;gap:8px;font-size:11px;padding:2px 0;border-top:1px solid var(--border)}
.tn{color:var(--text-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tv{font-variant-numeric:tabular-nums;white-space:nowrap}
.tv.pass{color:var(--good)}.tv.fail{color:var(--critical)}
.tv.unknown{color:var(--warning)}.tv.na{color:var(--muted);font-style:italic}
.thr{color:var(--muted)}
.chart{width:100%;height:auto;background:var(--surface-1);border:1px solid var(--border);
 border-radius:10px;padding:6px}
.line{stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
.s1{stroke:var(--series-1)} .s2{stroke:var(--series-2)} .s3{stroke:var(--series-3)}
.dot.s1{fill:var(--series-1)}.dot.s2{fill:var(--series-2)}.dot.s3{fill:var(--series-3)}
.line.med{stroke:var(--seq-450)}
.band{stroke:none}.b1{fill:var(--seq-200);opacity:.55}.b2{fill:var(--seq-350);opacity:.55}
.grid{stroke:var(--grid);stroke-width:1}
.spot{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:4 3}
.axis{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.dlabel{font-size:10.5px;font-weight:600}
.dlabel.s1{fill:var(--series-1)}.dlabel.s2{fill:var(--series-2)}.dlabel.s3{fill:var(--series-3)}
.dlabel.med-ink{fill:var(--seq-450)}
.note{background:var(--surface-1);border-left:3px solid var(--series-1);border-radius:0 8px 8px 0;
 padding:11px 15px;margin:12px 0;font-size:13px;color:var(--text-secondary)}
.note.warn{border-left-color:var(--warning)}
.note.crit{border-left-color:var(--critical)}
ul{margin:6px 0;padding-left:20px}li{margin:3px 0}
.foot{color:var(--muted);font-size:12px;margin-top:36px;padding-top:14px;border-top:1px solid var(--border)}
code{background:var(--plane);padding:1px 5px;border-radius:4px;font-size:12px}
</style></head><body class="viz-root"><div class="wrap">
__BODY__
</div></body></html>"""


def render_deepdive(d: Dict[str, Any], out_dir: str) -> str:
    spot = d.get("price")
    rx = d.get("recommendation", {})
    gbm = d.get("gbm", {})
    val = d.get("valuation", {})
    m = d.get("metrics", {})
    tech = d.get("metrics", {})
    ctx = d.get("context", {})
    hur = d.get("hurst", {})
    sw = d.get("swings", {})
    b: List[str] = []

    # ---------- header + hero
    b.append(f'<div class="eyebrow">Deep dive · {e(d.get("market_name"))} · '
             f'generated {e(d.get("generated_utc"))}</div>')
    b.append(f'<h1>{e(d.get("name"))} <span class="muted">{e(d["ticker"])}</span></h1>')
    b.append(f'<p class="sub">{e(d.get("sector") or "—")} · {e(d.get("industry") or "")} · '
             f'reporting in {e(d.get("financial_currency") or d.get("currency"))}</p>')

    b.append('<div class="hero">'
             f'<div><div class="call {e(rx.get("call"))}">{e(rx.get("call"))}</div>'
             f'<div class="sub">{e(rx.get("conviction"))} conviction</div></div>'
             f'<div class="heroitem"><div class="l">Price</div><div class="v">{f(spot)} {e(d.get("currency"))}</div></div>'
             f'<div class="heroitem"><div class="l">Market cap</div><div class="v">{f(d.get("market_cap_usd"), money=True)} USD</div></div>'
             f'<div class="heroitem"><div class="l">Frameworks</div><div class="v">{sum(1 for v in d.get("frameworks", {}).values() if v.get("passed"))}/7</div></div>'
             f'<div class="heroitem"><div class="l">Position size</div><div class="v">{f(rx.get("position_size", 0) * 100, 1)}%</div></div>'
             f'<div class="heroitem"><div class="l">Ann. volatility</div><div class="v">{f(gbm.get("sigma_annual"), 1, pct=True)}</div></div>'
             '</div>')

    if d.get("warnings"):
        b.append('<div class="note warn"><b>Data quality:</b> '
                 + " · ".join(e(w) for w in d["warnings"]) + '</div>')

    # ---------- 1. Investor philosophy
    b.append('<h2>1 · Investor-philosophy analysis</h2>')
    b.append('<p class="sub">The same seven engines the nightly screen runs, at the '
             'same thresholds. A pass is a buy-side signal on that lens alone, not a '
             'recommendation.</p>')
    b.append(_fw_table(d.get("frameworks", {})))

    # ---------- 2. Fundamentals & valuation frameworks
    b.append('<h2>2 · Fundamentals and valuation frameworks</h2>')
    rows = [
        ("P/E (trailing)", f(m.get("pe_ttm"))), ("P/B", f(m.get("price_to_book"))),
        ("P/tangible book", f(m.get("price_to_tangible_book"))),
        ("EV/EBIT", f(m.get("ev_to_ebit"))),
        ("Earnings yield (EBIT/EV)", f(m.get("ebit_to_ev"), 1, pct=True)),
        ("FCF yield", f(m.get("fcf_yield"), 1, pct=True)),
        ("ROE (trailing)", f(m.get("roe_ttm"), 1, pct=True)),
        ("ROIC 5-year average", f(m.get("roic_5y_avg"), 1, pct=True)),
        ("Gross margin", f(m.get("gross_margin_ttm"), 1, pct=True)),
        ("Operating margin", f(m.get("operating_margin_ttm"), 1, pct=True)),
        ("Debt / equity", f(m.get("debt_to_equity"))),
        ("Net debt / EBITDA", f(m.get("net_debt_to_ebitda"))),
        ("Net cash / market cap", f(m.get("net_cash_to_market_cap"), 1, pct=True)),
        ("EPS CAGR 5y", f(m.get("eps_cagr_5y"), 1, pct=True)),
        ("PEG", f(m.get("peg_ratio"))),
        ("Accruals ratio", f(m.get("accruals_ratio"), 3)),
        ("Years of statements", f(d.get("history_years"), 0)),
    ]
    b.append('<div class="grid2"><table><tbody>'
             + "".join(f'<tr><td>{e(k)}</td><td class="num">{e(v)}</td></tr>'
                       for k, v in rows[:9]) + '</tbody></table>'
             '<table><tbody>'
             + "".join(f'<tr><td>{e(k)}</td><td class="num">{e(v)}</td></tr>'
                       for k, v in rows[9:]) + '</tbody></table></div>')

    fedm = val.get("fed_model", {})
    tq = val.get("tobins_q", {})
    cape = val.get("cape", {})
    b.append('<h3>2.1 Cross-asset and long-horizon valuation</h3>')
    b.append('<table><thead><tr><th>Model</th><th class="num">Value</th>'
             '<th>Reading</th></tr></thead><tbody>')
    b.append(f'<tr><td>Fed Model — earnings yield</td><td class="num">'
             f'{f(fedm.get("earnings_yield"), 2, pct=True)}</td>'
             f'<td>{e(fedm.get("verdict") or fedm.get("error") or "")}'
             + (f' (10-year at {f(fedm.get("treasury_10y"), 2, pct=True)}, spread '
                f'{f(fedm.get("spread"), 2, pct=True)})' if _n(fedm.get("spread")) else "")
             + '</td></tr>')
    b.append(f'<tr><td>Tobin\'s Q</td><td class="num">{f(tq.get("q"))}</td>'
             f'<td>{e(tq.get("verdict") or tq.get("error") or "")} '
             f'<span class="muted">book value stands in for replacement cost</span></td></tr>')
    if cape.get("cape"):
        b.append(f'<tr><td>Shiller CAPE</td><td class="num">{f(cape["cape"])}</td>'
                 f'<td>{cape.get("years_used")} years of EPS'
                 f'{" (inflation-adjusted)" if cape.get("inflation_adjusted") else ""}</td></tr>')
    else:
        b.append(f'<tr><td>Shiller CAPE</td><td class="num">—</td>'
                 f'<td class="muted">{e(cape.get("error", "unavailable"))}</td></tr>')
    bbb_f, bbb_a = val.get("bbb_formula"), val.get("bbb_actual")
    b.append(f'<tr><td>BBB rate — your formula</td><td class="num">{f(bbb_f, 2, pct=True)}</td>'
             f'<td>earnings yield − 2.8% risk premium + 6.7% growth'
             + (f' · <b>actual BBB effective yield {f(bbb_a / 100 if _n(bbb_a) else None, 2, pct=True)}</b> (FRED)'
                if _n(bbb_a) else "") + '</td></tr>')
    ggm = val.get("ggm", {})
    ggm_note = ggm.get("error") or (
        "r = " + f(ggm.get("required_return"), 1, pct=True)
        + ", g = " + f(ggm.get("growth"), 1, pct=True))
    b.append(f'<tr><td>Gordon Growth value</td><td class="num">{f(ggm.get("value"))}</td>'
             f'<td>{e(ggm_note)}</td></tr>')
    dv = val.get("dividend", {})
    b.append(f'<tr><td>Dividend yield / R²</td>'
             f'<td class="num">{f(val.get("dividend_yield"), 2, pct=True)} / {f(dv.get("r_squared"), 3)}</td>'
             f'<td class="muted">{e(dv.get("caveat") or dv.get("error") or "")}</td></tr>')
    b.append('</tbody></table>')

    # ---------- 3. GBM
    b.append('<h2>3 · Geometric Brownian Motion and Monte Carlo</h2>')
    if gbm.get("error"):
        b.append(f'<p class="muted">{e(gbm["error"])}</p>')
    else:
        b.append(f'<p class="sub">dS = μS·dt + σS·dW, solved in log space. '
                 f'{gbm["n_paths"]:,} paths over {gbm["horizon_days"]} trading days, '
                 f'estimated on three years of daily returns. '
                 f'Drift {f(gbm.get("mu_annual"), 1, pct=True)}, volatility '
                 f'{f(gbm.get("sigma_annual"), 1, pct=True)} annualised.</p>')
        if gbm.get("drift_capped"):
            b.append('<div class="note warn"><b>Drift was capped.</b> The historical '
                     f'estimate was {f(gbm.get("mu_annual_raw"), 1, pct=True)} a year, '
                     'which extrapolates a past run straight into the forecast. It is '
                     'held to ±25%. Treat the upper percentiles as a mechanical '
                     'consequence of past volatility, not a forecast.</div>')
        b.append(fan_chart(gbm))
        q = gbm.get("percentiles", {})
        b.append('<table><thead><tr><th>Percentile</th><th class="num">Price</th>'
                 '<th class="num">vs spot</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{p}th</td><td class="num">{f(q.get(f"p{p}"))}</td>'
                           f'<td class="num">{pct_diff(q.get(f"p{p}"), spot)}</td></tr>'
                           for p in (5, 10, 25, 50, 75, 90, 95))
                 + '</tbody></table>')
        b.append(f'<div class="note"><b>GBM entry {f(gbm.get("entry"))} '
                 f'({pct_diff(gbm.get("entry"), spot)}) · exit {f(gbm.get("exit"))} '
                 f'({pct_diff(gbm.get("exit"), spot)}).</b> Entry at the 25th '
                 f'percentile, exit at the 75th — accumulate into the cheap tail of '
                 f'the modelled distribution, trim into the rich one. '
                 f'{f(gbm.get("prob_above_spot"), 0, pct=True)} of simulated outcomes '
                 f'finish above today\'s price.</div>')

    # ---------- 4. Technicals
    b.append('<h2>4 · Technical analysis</h2>')
    b.append(price_chart(d.get("price_series", {})))
    trows = [
        ("Price vs 200-day", "above" if m.get("price_above_sma200") else "below"),
        ("50-day vs 200-day", "golden-cross regime" if m.get("sma50_above_sma200") else "death-cross regime"),
        ("RSI(14)", f(m.get("rsi_14"), 1)),
        ("MACD histogram", f(m.get("macd_histogram"), 3)),
        ("ATR percentile", f(m.get("atr_pct_percentile"), 2)),
        ("% below 52-week high", f(m.get("pct_below_52w_high"), 1, pct=True)),
        ("% above 5-year low", f(m.get("pct_above_5y_low"), 1, pct=True)),
        ("RS vs own index, 6m", f(m.get("rs_vs_market_index_6m"), 1, pct=True)),
    ]
    b.append('<table><tbody>' + "".join(
        f'<tr><td>{e(k)}</td><td class="num">{e(v)}</td></tr>' for k, v in trows)
        + '</tbody></table>')

    if sw.get("support") or sw.get("resistance"):
        b.append('<h3>4.1 Support, resistance and Fibonacci</h3>')
        b.append('<table><thead><tr><th>Level</th><th class="num">Price</th>'
                 '<th class="num">vs spot</th><th>Type</th></tr></thead><tbody>')
        for lv, k in (sw.get("resistance") or []):
            b.append(f'<tr><td>Resistance</td><td class="num">{f(lv)}</td>'
                     f'<td class="num">{pct_diff(lv, spot)}</td>'
                     f'<td class="muted">{k} touch(es)</td></tr>')
        for lv, k in (sw.get("support") or []):
            b.append(f'<tr><td>Support</td><td class="num">{f(lv)}</td>'
                     f'<td class="num">{pct_diff(lv, spot)}</td>'
                     f'<td class="muted">{k} touch(es)</td></tr>')
        for lbl, lv in (d.get("fibonacci") or {}).items():
            b.append(f'<tr><td>Fibonacci {e(lbl)}</td><td class="num">{f(lv)}</td>'
                     f'<td class="num">{pct_diff(lv, spot)}</td>'
                     f'<td class="muted">retracement from swing high</td></tr>')
        b.append('</tbody></table>')
        sup = (sw.get("support") or [[None]])[0][0] if sw.get("support") else None
        res = (sw.get("resistance") or [[None]])[0][0] if sw.get("resistance") else None
        b.append(f'<div class="note"><b>Technical entry {f(sup)} '
                 f'({pct_diff(sup, spot)}) · exit {f(res)} ({pct_diff(res, spot)}).</b> '
                 f'Nearest tested support and resistance.</div>')

    # ---------- 8. Fractal / Hurst
    b.append('<h2>5 · Persistence structure (in place of the fractal model)</h2>')
    b.append('<div class="note crit"><b>On the requested formula.</b> '
             '<code>Zₙ₊₁ = f(Zₙ, C)</code> is the Mandelbrot set iteration. It has no '
             'established mapping onto price levels, and any entry and exit prices I '
             'derived from it would look precise while meaning nothing. The Hurst '
             'exponent below answers the question it was reaching for — does this '
             'series trend or revert? — and is a checkable statistic.</div>')
    if hur.get("error"):
        b.append(f'<p class="muted">{e(hur["error"])}</p>')
    else:
        b.append(f'<div class="card"><h4>Hurst exponent '
                 f'<span class="badge {"pass" if hur.get("regime") == "persistent" else "na"}">'
                 f'{e(hur.get("regime"))}</span></h4>'
                 f'<p class="v" style="font-size:26px;font-weight:700">'
                 f'{f(hur.get("hurst"), 3)}</p>'
                 f'<p class="sub">{e(hur.get("interpretation"))} '
                 f'0.5 is a random walk; above 0.55 trends persist; below 0.45 they '
                 f'reverse. Estimated by rescaled-range analysis across '
                 f'{len(hur.get("scales", []))} time scales.</p></div>')

    # ---------- 6/7. Recommendation and scenarios
    b.append('<h2>6 · Recommendation and scenarios</h2>')
    q = gbm.get("percentiles", {})
    entries = [("GBM (25th pct)", gbm.get("entry")),
               ("Technical support", (sw.get("support") or [[None]])[0][0] if sw.get("support") else None),
               ("Gordon Growth value", ggm.get("value"))]
    exits = [("GBM (75th pct)", gbm.get("exit")),
             ("Technical resistance", (sw.get("resistance") or [[None]])[0][0] if sw.get("resistance") else None),
             ("90th percentile", q.get("p90"))]
    b.append('<div class="grid2"><table><thead><tr><th>Entry basis</th>'
             '<th class="num">Price</th><th class="num">vs spot</th></tr></thead><tbody>'
             + "".join(f'<tr><td>{e(k)}</td><td class="num">{f(v)}</td>'
                       f'<td class="num">{pct_diff(v, spot)}</td></tr>' for k, v in entries)
             + '</tbody></table><table><thead><tr><th>Exit basis</th>'
             '<th class="num">Price</th><th class="num">vs spot</th></tr></thead><tbody>'
             + "".join(f'<tr><td>{e(k)}</td><td class="num">{f(v)}</td>'
                       f'<td class="num">{pct_diff(v, spot)}</td></tr>' for k, v in exits)
             + '</tbody></table></div>')

    p3 = d.get("probs_3y", {})
    b.append('<h3>6.1 Probability-weighted scenarios (1 year, from the simulation)</h3>')
    b.append('<table><thead><tr><th>Scenario</th><th class="num">Probability</th>'
             '<th class="num">Target</th><th class="num">vs spot</th><th>Basis</th>'
             '</tr></thead><tbody>'
             f'<tr><td>Best case</td><td class="num">10%</td>'
             f'<td class="num">{f(q.get("p90"))}</td><td class="num">{pct_diff(q.get("p90"), spot)}</td>'
             f'<td class="muted">90th percentile of simulated outcomes</td></tr>'
             f'<tr><td>Base case</td><td class="num">50%</td>'
             f'<td class="num">{f(q.get("p50"))}</td><td class="num">{pct_diff(q.get("p50"), spot)}</td>'
             f'<td class="muted">median path</td></tr>'
             f'<tr><td>Worst case</td><td class="num">10%</td>'
             f'<td class="num">{f(q.get("p10"))}</td><td class="num">{pct_diff(q.get("p10"), spot)}</td>'
             f'<td class="muted">10th percentile</td></tr>'
             '</tbody></table>')
    if p3:
        b.append(f'<p class="sub"><b>Three-year outlook (closed-form GBM):</b> '
                 f'{f(p3.get("p_up"), 0, pct=True)} chance of finishing above today\'s '
                 f'price, {f(p3.get("p_down"), 0, pct=True)} below. '
                 f'{f(p3.get("p_up_50"), 0, pct=True)} chance of a 50%+ gain, '
                 f'{f(p3.get("p_down_30"), 0, pct=True)} chance of a 30%+ loss.</p>')

    b.append('<div class="card"><h4>Why this call</h4>')
    if rx.get("reasons_for"):
        b.append('<p class="sub"><b>For:</b></p><ul>'
                 + "".join(f'<li>{e(r)}</li>' for r in rx["reasons_for"]) + '</ul>')
    if rx.get("reasons_against"):
        b.append('<p class="sub"><b>Against:</b></p><ul>'
                 + "".join(f'<li>{e(r)}</li>' for r in rx["reasons_against"]) + '</ul>')
    b.append(f'<p class="sub"><b>Time frame:</b> {e(rx.get("horizon"))}<br>'
             f'<b>Position size:</b> {f(rx.get("position_size", 0) * 100, 1)}% of portfolio — '
             f'scaled inversely to {f(gbm.get("sigma_annual"), 0, pct=True)} annual '
             f'volatility, so a jumpier name gets a smaller slice at the same conviction.</p></div>')

    # ---------- 15. Peers
    b.append('<h2>7 · Peers in your universe</h2>')
    peers = d.get("peers") or []
    if peers:
        b.append('<table><thead><tr><th>Ticker</th><th>Name</th><th>Market</th>'
                 '<th class="num">Passed</th><th class="num">P/E</th>'
                 '<th class="num">EV/EBIT</th><th class="num">ROE</th>'
                 '<th class="num">FCF yld</th></tr></thead><tbody>'
                 + "".join(
                     f'<tr><td><b>{e(p["ticker"])}</b></td><td>{e(p.get("name"))}</td>'
                     f'<td>{e(p.get("market"))}</td><td class="num">{p.get("n_passed")}</td>'
                     f'<td class="num">{f(p.get("pe"))}</td>'
                     f'<td class="num">{f(p.get("ev_ebit"))}</td>'
                     f'<td class="num">{f(p.get("roe"), 1, pct=True)}</td>'
                     f'<td class="num">{f(p.get("fcf_yield"), 1, pct=True)}</td></tr>'
                     for p in peers) + '</tbody></table>')
        b.append('<div class="note"><b>Bounded on purpose.</b> These are same-sector '
                 'names from your own six indices. A genuinely global peer search — '
                 'the two better-upside candidates anywhere in the world — needs '
                 'judgment and a wider source than this app has. Ask for that '
                 'separately rather than treating this table as exhaustive.</div>')
    else:
        b.append('<p class="muted">No same-sector peers in the stored universe yet — '
                 'run a full refresh first.</p>')

    # ---------- 17. Market context
    b.append('<h2>8 · Market context</h2>')
    br = ctx.get("breadth", {})
    crows = [
        ("VIX", f(ctx.get("vix"), 1), ctx.get("vix_reading", "")),
        ("VIX / 3-month VIX", f(ctx.get("vix_vix3m"), 3), ctx.get("vix_reading", "")),
        ("XLY/XLP 4-week ROC", f(ctx.get("xly_xlp_roc_4w"), 2, pct=True),
         ctx.get("xly_xlp_reading", "")),
        ("Buffett Indicator (mkt cap ÷ GDP)", f(ctx.get("buffett_indicator")),
         ctx.get("buffett_reading", "")),
        ("US 10-year", f(ctx.get("macro", {}).get("us_10y"), 2) + "%", ""),
        ("10y − 2y curve", f(ctx.get("macro", {}).get("yield_curve_10y2y"), 2), ""),
        ("High-yield spread", f(ctx.get("macro", {}).get("hy_spread"), 2) + "%", ""),
    ]
    if br and not br.get("error"):
        crows += [
            ("% above 50-day MA", f(br.get("pct_above_50dma"), 1, pct=True),
             f'{br.get("universe_size")} US names'),
            ("% above 200-day MA", f(br.get("pct_above_200dma"), 1, pct=True),
             br.get("internals", "")),
            ("TRIN (proxy)", f(br.get("trin_proxy"), 2),
             f'{e(br.get("trin_reading", ""))} — computed over this universe, '
             f'not the NYSE index'),
        ]
    b.append('<table><thead><tr><th>Indicator</th><th class="num">Value</th>'
             '<th>Reading</th></tr></thead><tbody>'
             + "".join(f'<tr><td>{e(k)}</td><td class="num">{e(v)}</td>'
                       f'<td class="muted">{e(n)}</td></tr>' for k, v, n in crows)
             + '</tbody></table>')

    if ctx.get("unavailable"):
        b.append('<div class="note warn"><b>Deliberately absent.</b> These inputs from '
                 'the brief are not obtainable free, and approximating them would be '
                 'worse than omitting them:<ul>'
                 + "".join(f'<li>{e(u)}</li>' for u in ctx["unavailable"])
                 + '</ul>The free substitutes above — VIX term structure, XLY/XLP, the '
                 'Buffett Indicator and breadth computed from your own universe — carry '
                 'much of the same signal about fear, risk appetite, valuation and '
                 'internals.</div>')

    b.append('<div class="foot">Fundamentals: SEC EDGAR (US) · Yahoo Finance (Asia). '
             'Prices: Yahoo, split and dividend adjusted. Macro: FRED. '
             'Simulations are seeded from the ticker, so this report reproduces exactly. '
             'A model is an argument, not a forecast — and none of this is investment '
             'advice.</div>')

    html_doc = TEMPLATE.replace("__TICKER__", e(d["ticker"])).replace("__BODY__", "".join(b))
    path = os.path.join(out_dir, f"{d['ticker']}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return path
