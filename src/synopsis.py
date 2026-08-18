"""A short, plain-English synopsis of each name.

Two things are stitched together here, and the difference between them matters
enough to keep visible in the output:

  * WHAT IT IS — one or two sentences of business description, taken verbatim
    from the data feed's company profile. This is reported, not computed. If
    the feed has no description the synopsis says so rather than inventing one.

  * WHAT THE NUMBERS SAY — generated from the screener's own fields: the
    frameworks passed and failed, the valuation and quality ratios, the price
    behaviour, and the reflexive / dislocation flags. Every clause traces to a
    number already displayed elsewhere in the row, so nothing here can assert
    something the detail panels contradict.

Nothing in this module fetches, models, or forecasts. It is a rewrite of data
the row already carries into a sentence a person can read in five seconds.

THE PERSISTENCE TRAP. Rows from the other region are read back from the store,
which holds only the fields in ``render.DISPLAY_METRICS``. A synopsis that
reached for a metric outside that list would render fully for the region that
just ran and silently degrade to "not available" for every other market — the
exact failure mode that has bitten this app twice before. So metric reads go
through ``_g``, which refuses any key not declared in ``SYNOPSIS_FIELDS``, and
a test asserts ``SYNOPSIS_FIELDS`` is a subset of ``DISPLAY_METRICS``. Adding a
new metric to the synopsis without also publishing it now fails loudly at the
first call instead of quietly on half the table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Every metrics key this module is allowed to read. Must be a subset of
# render.DISPLAY_METRICS — enforced by _g() at runtime and by a test.
SYNOPSIS_FIELDS = (
    "pe_ttm", "price_to_tangible_book", "price_to_book", "ev_to_ebit",
    "fcf_yield", "peg_ratio",
    "roe_ttm", "roic_5y_avg", "eps_cagr_5y", "debt_to_equity",
    "net_cash_to_market_cap", "ncav_to_market_cap",
    "return_6m", "return_12m", "pct_below_52w_high", "pct_above_5y_low",
    "rs_vs_market_index_6m", "price_above_sma200", "sma50_above_sma200",
    "rsi_14", "rsi_label", "rsi_regime", "rsi_divergence",
    "history_years", "statement_currency",
)

MAX_DESCRIPTION_CHARS = 260


def _g(m: Dict[str, Any], key: str):
    """Read one metric, refusing anything not declared as publishable."""
    if key not in SYNOPSIS_FIELDS:
        raise KeyError(
            f"{key!r} is not in SYNOPSIS_FIELDS. Declare it there AND in "
            "render.DISPLAY_METRICS, or merged rows will show it blank.")
    v = m.get(key)
    if isinstance(v, float) and v != v:      # NaN
        return None
    return v


def _pct(v, dp=0) -> str:
    return f"{v * 100:.{dp}f}%"


def _x(v, dp=1) -> str:
    return f"{v:.{dp}f}×"


def _join(bits: List[str]) -> str:
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + " and " + bits[-1]


def trim_description(text: Optional[str], limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """First sentence or two of a feed description, cut at a sentence end.

    Profile blurbs run to a full page. Truncating mid-word looks broken and
    truncating mid-sentence reads as if a fact were withheld, so the cut lands
    on a full stop wherever one is available.
    """
    if not text:
        return ""
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    stop = max(cut.rfind(". "), cut.rfind(".’"), cut.rfind('."'))
    # Take the sentence break only if it leaves a useful amount of text; a cut
    # at the first full stop of a long blurb can throw away everything that
    # said what the company does.
    floor = max(40, int(limit * 0.4))
    if stop >= floor:
        return cut[:stop + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space >= floor else cut).rstrip(",;: ") + "…"


def _failed_tests(frameworks: Dict[str, Any]) -> List[str]:
    """Test names that failed, most frequently failed first.

    A name usually fails six frameworks for one or two underlying reasons. The
    frequency count finds that reason: if 'debt to equity' fails in five
    frameworks, the objection is the balance sheet, not five separate things.
    """
    counts: Dict[str, int] = {}
    for f in (frameworks or {}).values():
        if f.get("ineligible_reason") or f.get("macro_gate_blocked"):
            continue
        for t in f.get("tests", []) or []:
            if t.get("insufficient") or t.get("result") is not False:
                continue
            nm = str(t.get("name", "")).replace("_", " ").strip()
            if nm:
                counts[nm] = counts.get(nm, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _verdict(res: Dict[str, Any], frameworks: Dict[str, Any],
             labels: Dict[str, str], total: int) -> str:
    passed = [labels.get(k, k) for k, f in (frameworks or {}).items()
              if f.get("passed")]
    n = len(passed)
    tech = res.get("technical_passed")
    if n == 0:
        head = f"Clears none of the {total} value frameworks,"
        head += (" though the technical timing test does agree" if tech
                 else " and the technical timing test does not agree either")
    else:
        if n <= 3:
            head = (f"Clears {n} of {total} value frameworks — "
                    + _join(sorted(passed)) + " —")
        else:
            head = (f"Clears {n} of {total} value frameworks, among them "
                    + _join(sorted(passed)[:3]) + " —")
        head += (" and the technical timing test agrees" if tech
                 else " though the technical timing test does not agree")
    objections = _failed_tests(frameworks)
    if objections:
        head += ("; the objection that recurs across frameworks is "
                 + " and ".join(objections[:2]))
    return head + "."


def _valuation(m: Dict[str, Any]) -> str:
    bits = []
    pe = _g(m, "pe_ttm")
    if pe is not None and pe > 0:
        bits.append(f"{_x(pe)} trailing earnings")
    elif pe is not None:
        bits.append("no P/E — it is loss-making on a trailing basis")
    ptb = _g(m, "price_to_tangible_book")
    if ptb is not None and ptb > 0:
        bits.append(f"{_x(ptb, 2)} tangible book")
    ev = _g(m, "ev_to_ebit")
    if ev is not None and ev > 0:
        bits.append(f"{_x(ev)} EV/EBIT")
    fcf = _g(m, "fcf_yield")
    if fcf is not None:
        bits.append(f"a {_pct(fcf, 1)} free-cash-flow yield")
    if not bits:
        return "The feed carries none of the headline valuation ratios for this name."
    s = "It is priced at " + _join(bits)
    nc = _g(m, "net_cash_to_market_cap")
    if nc is not None and nc > 0.10:
        s += (f", and {_pct(nc)} of the market capitalisation is cash net of "
              "debt — the operating business costs less than the headline price")
    ncav = _g(m, "ncav_to_market_cap")
    if ncav is not None and ncav > 1.0:
        s += (", and it trades below net current asset value, Graham's "
              "deep-value marker")
    return s + "."


def _quality(m: Dict[str, Any]) -> str:
    roe, roic = _g(m, "roe_ttm"), _g(m, "roic_5y_avg")
    g, de = _g(m, "eps_cagr_5y"), _g(m, "debt_to_equity")
    parts = []
    if roe is not None:
        parts.append(f"return on equity of {_pct(roe)}")
    if roic is not None:
        parts.append(f"a five-year return on capital of {_pct(roic)}")
    if g is not None:
        parts.append("earnings flat over five years" if abs(g) < 0.005 else
                     "earnings "
                     + ("compounding at " if g > 0 else "shrinking at ")
                     + _pct(abs(g)) + " a year over five years")
    if de is not None:
        parts.append(("debt at " + _x(de, 2) + " equity") if de > 0
                     else "no net borrowing against equity")
    if not parts:
        return ""
    s = ("Behind the price sit " if len(parts) > 1 else "Behind the price sits ") \
        + _join(parts)
    hy = _g(m, "history_years")
    if hy is not None and hy < 5:
        s += (f", though only {int(hy)} years of statements are on file, so the "
              "longer-window tests were set aside rather than failed")
    return s + "."


def _price_action(m: Dict[str, Any]) -> str:
    r12, r6 = _g(m, "return_12m"), _g(m, "return_6m")
    parts = []
    if r12 is not None:
        parts.append(("up " if r12 >= 0 else "down ") + _pct(abs(r12)) + " over a year")
    if r6 is not None:
        parts.append(("up " if r6 >= 0 else "down ") + _pct(abs(r6)) + " over six months")
    if not parts:
        return ""
    s = "The shares are " + " and ".join(parts)
    below = _g(m, "pct_below_52w_high")
    if below is not None and below > 0.02:
        s += f", {_pct(below)} below the 52-week high"
    rs = _g(m, "rs_vs_market_index_6m")
    if rs is not None:
        s += (f", {_pct(abs(rs))} "
              + ("ahead of" if rs >= 0 else "behind")
              + " their own index over six months")
    s += ("; they trade above the 200-day average"
          if _g(m, "price_above_sma200") else
          "; they trade below the 200-day average")
    rsi, lab = _g(m, "rsi_14"), _g(m, "rsi_label")
    if rsi is not None:
        s += f", with RSI at {rsi:.0f}"
        if lab:
            s += f" — {lab}"
        reg = _g(m, "rsi_regime")
        if reg:
            s += f" in a {reg} tape"
    div = _g(m, "rsi_divergence")
    if div:
        s += f", and momentum shows {div} divergence"
    return s + "."


def _flags(res: Dict[str, Any]) -> List[str]:
    out = []
    r = res.get("reflexive") or {}
    if r.get("stage") and r.get("stage") != "EQ":
        line = f"Soros stage {r['stage']}"
        if r.get("label"):
            line += f" — {r['label']}"
        if r.get("late"):
            line += (". This is the part of the loop where price and fundamentals "
                     "have come apart, and the app flags it as risk rather than "
                     "as an opportunity")
        out.append(line + ".")
    d = res.get("dislocation") or {}
    if d:
        r6 = d.get("return_6m")
        fell = f"It fell {_pct(abs(r6))}" if r6 is not None else "It fell sharply"
        if d.get("disqualified_by_filing"):
            out.append(f"{fell} in six months, and a filing takes it off the "
                       "dislocation list rather than qualifying it: "
                       + str(d["disqualified_by_filing"]) + ".")
        elif d.get("qualifies"):
            observed = d.get("evidence_grade") == "observed"
            causes = (d.get("observed_causes") if observed
                      else d.get("candidate_causes")) or []
            names = [c.get("name") for c in causes[:2] if c.get("name")]
            line = (f"{fell} in six months while the last published accounts "
                    "stayed intact")
            if names:
                line += ((" — a feed actually reported " if observed
                          else " — the shape of the fall is consistent with ")
                         + _join(names).lower())
            out.append(line + ". The accounts may simply be stale; treat this as "
                       "a question, not an answer.")
        else:
            out.append(f"{fell} in six months, and the accounts do appear to "
                       "explain it — a business problem rather than a "
                       "price accident.")
    if res.get("is_fund"):
        out.append("This is a fund rather than an operating company, so the value "
                   "frameworks are skipped rather than failed.")
    if res.get("gates_failed"):
        out.append("It is held back from the surfaced list by a size or liquidity "
                   "gate: " + "; ".join(str(x) for x in res["gates_failed"]) + ".")
    return out


def _one_liner(res: Dict[str, Any], m: Dict[str, Any], total: int) -> str:
    """A single sentence, for the table tooltip where there is no room."""
    n = res.get("n_frameworks_passed", 0)
    pe, roe = _g(m, "pe_ttm"), _g(m, "roe_ttm")
    r6 = _g(m, "return_6m")
    bits = [f"{n}/{total} frameworks"]
    if pe is not None and pe > 0:
        bits.append(f"P/E {pe:.1f}")
    if roe is not None:
        bits.append(f"ROE {_pct(roe)}")
    if r6 is not None:
        bits.append(("6m " + ("+" if r6 >= 0 else "−") + _pct(abs(r6))))
    bits.append("technicals agree" if res.get("technical_passed")
                else "technicals disagree")
    return " · ".join(bits)


def build(res: Dict[str, Any], metrics: Dict[str, Any],
          framework_labels: Optional[Dict[str, str]] = None,
          frameworks_total: Optional[int] = None) -> Dict[str, Any]:
    """Return {'what', 'what_source', 'numbers': [...], 'one_liner'} for one row."""
    m = metrics or {}
    fw = res.get("frameworks") or {}
    labels = framework_labels or {}
    total = frameworks_total or len(fw) or len(labels)

    desc = trim_description(res.get("business_summary"))
    what_source = "feed" if desc else ""
    if not desc:
        sector, industry = res.get("sector"), res.get("industry")
        if industry and sector and industry != sector:
            desc = f"Classified by the feed as {industry}, within {sector}."
            what_source = "classification"
        elif sector or industry:
            desc = f"Classified by the feed as {industry or sector}."
            what_source = "classification"

    numbers = [_verdict(res, fw, labels, total), _valuation(m)]
    for fn in (_quality, _price_action):
        s = fn(m)
        if s:
            numbers.append(s)
    numbers.extend(_flags(res))

    cur, stmt = res.get("currency"), _g(m, "statement_currency")
    if cur and stmt and cur != stmt:
        numbers.append(f"The shares trade in {cur} while the accounts are "
                       f"reported in {stmt}; the ratios above are computed on "
                       "matched currencies, not on the mixed pair.")

    return {
        "what": desc,
        "what_source": what_source,
        "numbers": numbers,
        "one_liner": _one_liner(res, m, total),
    }
