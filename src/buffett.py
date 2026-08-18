"""Buffett: owner earnings, intrinsic value, and the three business tenets.

Three things live here that do not belong in `metrics.py`, because each is a
judgement wearing a number's clothes and each needs its assumptions stated on
the page rather than buried in a ratio:

  * OWNER EARNINGS. From the 1986 letter's Appendix: "(a) reported earnings
    plus (b) depreciation, depletion, amortization and certain other non-cash
    charges less (c) the average annual amount of capitalized expenditures for
    plant and equipment, etc. that the business requires to fully maintain its
    long-term competitive position and its unit volume." He then adds the part
    every screen ignores: "(c) must be a guess — and one sometimes very
    difficult to make." So the maintenance-capex estimate here is labelled as
    an estimate everywhere it appears, and the method is stated.

  * INTRINSIC VALUE. A two-stage discounted cash flow on owner earnings. The
    output of a DCF is dominated by its inputs, and Buffett himself never
    published one; this reports the assumptions beside the answer and refuses
    to run at all where the inputs are not defensible (negative owner
    earnings, no growth history, a discount rate below the growth rate).

  * THE BUSINESS TENETS — the "B" label. Circle of competence, durable moat,
    consistent operating history. Two of these have measurable proxies. The
    first does not: the circle of competence is a fact about the INVESTOR, not
    the company, so it is read from a list the user keeps in
    `config/thresholds.yml` rather than inferred. A screener that claimed to
    know what you understand would be making the most Buffett-hostile mistake
    available to it.
"""
from __future__ import annotations

import re as _re
from typing import Any, Dict, List, Optional


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x


def _safe_div(a, b) -> Optional[float]:
    if not (_n(a) and _n(b)) or b == 0:
        return None
    return a / b


# ---------------------------------------------------------------- owner earnings
def owner_earnings(years: List[Any], n: int = 1) -> Dict[str, Any]:
    """Reported earnings + non-cash charges − maintenance capital expenditure.

    Maintenance capex is not disclosed by anyone, so it must be estimated. Two
    estimates are computed and the more conservative (higher capex, lower owner
    earnings) is used:

      * DEPRECIATION AS PROXY. The textbook approach: over a full cycle a
        business that is merely standing still spends roughly what it charges
        in depreciation. Fails for a genuinely growing business, where it
        understates the spend, and for an inflating one, where historical-cost
        depreciation understates replacement cost.

      * BRUCE GREENWALD'S SPLIT. Capex above the historical capex-to-revenue
        ratio applied to this year's revenue growth is treated as GROWTH capex
        and excluded; the rest is maintenance. Fails when revenue is falling.

    Returning both, and flagging which was used, is the point: a single number
    here would hide a guess that can move intrinsic value by half.
    """
    if not years:
        return {"available": False, "reason": "no statements"}
    y = years[n - 1] if len(years) >= n else None
    if y is None:
        return {"available": False, "reason": "not enough years"}
    ni, da, capex = y.net_income, y.depreciation_amortization, y.capex
    if not _n(ni):
        return {"available": False, "reason": "no net income"}
    if not _n(da):
        return {"available": False, "reason": "no depreciation line"}
    capex = abs(capex) if _n(capex) else None
    if capex is None:
        return {"available": False, "reason": "no capital expenditure line"}

    maint_dep = min(capex, da)          # cannot spend more on upkeep than total
    maint_gw = None
    if len(years) > n and _n(y.revenue):
        prev = years[n]
        if _n(prev.revenue) and prev.revenue > 0:
            rev_growth = y.revenue - prev.revenue
            hist = [(_safe_div(abs(x.capex), x.revenue))
                    for x in years[:7] if _n(x.capex) and _n(x.revenue) and x.revenue]
            hist = [h for h in hist if _n(h)]
            if hist and rev_growth > 0:
                cap_to_rev = sum(hist) / len(hist)
                growth_capex = min(cap_to_rev * rev_growth, capex)
                maint_gw = capex - growth_capex
            elif hist:
                maint_gw = capex          # no growth: all of it is maintenance

    candidates = [c for c in (maint_dep, maint_gw) if _n(c)]
    maintenance = max(candidates)         # the conservative one
    method = ("depreciation proxy" if maintenance == maint_dep
              else "growth-capex split")
    if maint_gw is not None and maint_dep is not None and maint_gw == maint_dep:
        method = "both methods agree"

    # Working capital movement, where two consecutive years are available.
    dwc = None
    if len(years) > n:
        prev = years[n]
        cur_wc = ((y.current_assets or 0) - (y.cash_and_equivalents or 0)
                  - ((y.current_liabilities or 0)))
        prv_wc = ((prev.current_assets or 0) - (prev.cash_and_equivalents or 0)
                  - ((prev.current_liabilities or 0)))
        if _n(y.current_assets) and _n(prev.current_assets):
            dwc = cur_wc - prv_wc

    oe = ni + da - maintenance - (dwc or 0.0)
    return {
        "available": True,
        "owner_earnings": oe,
        "net_income": ni,
        "depreciation": da,
        "capex_total": capex,
        "maintenance_capex": maintenance,
        "maintenance_method": method,
        "maintenance_depreciation_proxy": maint_dep,
        "maintenance_growth_split": maint_gw,
        "working_capital_change": dwc,
        "caveat": (
            "Maintenance capital expenditure is not disclosed by any filer. "
            "Buffett's own words on this line: it \"must be a guess — and one "
            f"sometimes very difficult to make\". Here it is the {method}, "
            "taking the more conservative of two estimates."),
    }


# -------------------------------------------------------------- intrinsic value
def intrinsic_value(owner_earnings_ps: Optional[float],
                    growth_rate: Optional[float],
                    discount_rate: Optional[float],
                    cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Two-stage DCF on owner earnings per share.

    Ten years at the estimated growth rate, then a perpetuity at a terminal
    rate no higher than long-run nominal GDP growth. The refusals matter more
    than the arithmetic:

      * negative owner earnings → no discounting, because a DCF on a loss is a
        number generator, not a valuation;
      * no growth history → refuse rather than assume;
      * growth capped at the configured ceiling, because a five-year rate
        extrapolated for a decade is how DCFs produce absurdities;
      * terminal growth forced below the discount rate, or the formula returns
        an infinite value and the page reports it with a straight face.
    """
    c = cfg or {}
    cap = c.get("growth_cap", 0.15)
    floor_disc = c.get("min_discount_rate", 0.09)
    term = c.get("terminal_growth", 0.025)
    years = int(c.get("stage_one_years", 10))

    if not _n(owner_earnings_ps) or owner_earnings_ps <= 0:
        return {"available": False,
                "reason": "owner earnings are not positive — a discounted cash "
                          "flow on a loss produces a number, not a valuation"}
    if not _n(growth_rate):
        return {"available": False,
                "reason": "no usable growth history to project from"}
    g = max(min(float(growth_rate), cap), -0.05)
    d = max(float(discount_rate), floor_disc) if _n(discount_rate) else floor_disc
    t = min(term, d - 0.02)               # terminal growth must stay below d
    if t >= d:
        return {"available": False,
                "reason": "terminal growth is not below the discount rate"}

    pv, cash = 0.0, float(owner_earnings_ps)
    for i in range(1, years + 1):
        cash *= (1 + g)
        pv += cash / ((1 + d) ** i)
    terminal = cash * (1 + t) / (d - t)
    pv_terminal = terminal / ((1 + d) ** years)
    value = pv + pv_terminal
    return {
        "available": True,
        "value_per_share": value,
        "growth_used": g,
        "growth_capped": _n(growth_rate) and growth_rate > cap,
        "discount_rate": d,
        "terminal_growth": t,
        "stage_one_years": years,
        "terminal_share": pv_terminal / value if value else None,
        "caveat": (
            f"Projected at {g:.1%} for {years} years, then {t:.1%} for ever, "
            f"discounted at {d:.1%}. "
            f"{pv_terminal / value:.0%} of this value sits in the terminal "
            "figure — that is normal for a DCF and it is also why the answer "
            "is an opinion about the far future dressed as arithmetic. Treat "
            "the margin of safety as the whole point of the exercise."),
    }


def margin_of_safety(price: Optional[float],
                     value: Optional[float]) -> Optional[float]:
    """How far below intrinsic value the price sits. Positive is the discount."""
    if not (_n(price) and _n(value)) or value <= 0:
        return None
    return (value - price) / value


# ------------------------------------------------------- the Buffett Indicator
# "Probably the best single measure of where valuations stand at any given
# moment" — Fortune, December 2001. Total market value of listed equities against
# GDP. Two FRED series, and the trap is that they are NOT published in the same
# units: the Z.1 corporate-equities series is in millions, GDP in billions.
# Assuming otherwise gives a ratio wrong by a factor of a thousand that still
# reads like a number, so the units are FETCHED and applied rather than assumed.
EQUITIES_SERIES = "NCBEILQ027S"   # Nonfinancial corporate business, equities
GDP_SERIES = "GDP"                # Nominal GDP, seasonally adjusted annual rate

# Buffett's own bands, from the 2001 article and its later restatements.
INDICATOR_BANDS = (
    (0.80, "significantly undervalued",
     "under 80% — the territory Buffett described as where 'buying stocks is "
     "likely to work very well for you'"),
    (1.15, "fairly valued",
     "between 80% and 115% — the range he treated as ordinary"),
    (1.40, "on the expensive side",
     "between 115% and 140% — expensive, though not yet at the level he called "
     "playing with fire"),
    (float("inf"), "substantially overvalued",
     "above 140% — the zone Buffett said meant 'you are playing with fire'"),
)
# A sanity window. Anything outside it means the units are wrong, a series was
# revised into a different shape, or FRED returned something unexpected — and
# the honest answer is to refuse rather than to publish an absurd figure.
PLAUSIBLE_RANGE = (0.20, 5.00)


def buffett_indicator(fred) -> Dict[str, Any]:
    """Total corporate equity value ÷ GDP, with the units read from FRED.

    Returns `available: False` and a reason for every failure mode rather than
    a number: no API key, a missing series, mismatched observation dates, or a
    result outside the plausible range. A macro gauge that is silently off by
    a thousand is worse than one that says it could not be computed.
    """
    if not getattr(fred, "enabled", False):
        return {"available": False,
                "reason": "no FRED API key — set FRED_API_KEY in the repo "
                          "secrets and this fills in on the next run"}
    eq_meta = fred.series_meta(EQUITIES_SERIES)
    gdp_meta = fred.series_meta(GDP_SERIES)
    eq = fred.latest_observation(EQUITIES_SERIES)
    gdp = fred.latest_observation(GDP_SERIES)
    if not eq or not gdp:
        missing = [s for s, v in ((EQUITIES_SERIES, eq), (GDP_SERIES, gdp))
                   if not v]
        return {"available": False,
                "reason": f"FRED returned no observations for {', '.join(missing)}"}
    eq_date, eq_val = eq
    gdp_date, gdp_val = gdp
    eq_scale = eq_meta.get("scale")
    gdp_scale = gdp_meta.get("scale")
    assumed = False
    if not eq_scale or not gdp_scale:
        # Documented defaults, used only when the metadata call failed — and
        # flagged, so a wrong assumption is visible instead of invisible.
        eq_scale, gdp_scale, assumed = eq_scale or 1e6, gdp_scale or 1e9, True
    if not gdp_val:
        return {"available": False, "reason": "GDP came back as zero"}

    ratio = (eq_val * eq_scale) / (gdp_val * gdp_scale)
    if not (PLAUSIBLE_RANGE[0] <= ratio <= PLAUSIBLE_RANGE[1]):
        return {"available": False,
                "reason": (f"computed {ratio:.2%}, outside the plausible "
                           f"{PLAUSIBLE_RANGE[0]:.0%}–{PLAUSIBLE_RANGE[1]:.0%} "
                           "range — the series units or definitions have "
                           "changed, so the figure is being withheld rather "
                           "than published"),
                "raw": {"equities": eq_val, "equities_units": eq_meta.get("units"),
                        "gdp": gdp_val, "gdp_units": gdp_meta.get("units")}}

    # Both halves are quarterly. A numerator from a materially older quarter
    # than the denominator is a stale reading wearing a current date.
    lag_note = ""
    if eq_date[:4] != gdp_date[:4] or eq_date[5:7] != gdp_date[5:7]:
        lag_note = (f" The two series are dated differently — equities to "
                    f"{eq_date}, GDP to {gdp_date} — so this is as current as "
                    "the older of the two.")

    verdict, reading = next((v, r) for lim, v, r in INDICATOR_BANDS if ratio < lim)
    return {
        "available": True,
        "ratio": ratio,
        "pct": ratio * 100.0,
        "verdict": verdict,
        "reading": reading + lag_note,
        "equities_date": eq_date, "gdp_date": gdp_date,
        "equities_units": eq_meta.get("units"),
        "gdp_units": gdp_meta.get("units"),
        "units_assumed": assumed,
        "caveat": (
            "This is corporate equities from the Federal Reserve's Z.1 tables "
            "over nominal GDP — the version Buffett's own 2001 article "
            "described. It is not the Wilshire-over-GDP chart quoted in the "
            "press, and the two do not agree in level. It also says nothing "
            "about any single company: it is a statement about the price of "
            "the whole market."
            + (" NOTE: FRED's units metadata was unavailable, so the standard "
               "millions/billions scaling was assumed." if assumed else "")),
    }


# ------------------------------------------------------------ the three tenets
DISRUPTION_WORDS = (
    "biotech", "crypto", "blockchain", "cannabis", "airline", "shipping",
    "solar", "electric vehicle", "space", "quantum",
)


def business_tenets(m: Dict[str, Any], sector: Optional[str],
                    industry: Optional[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The "B" label: circle of competence, durable moat, consistent history.

    All three must hold. Each returns its own evidence, because "B" on a row is
    worth nothing if you cannot see what earned it — and because two of the
    three are proxies for a judgement rather than the judgement itself.
    """
    moat = _moat(m, cfg)
    circle = _circle_of_competence(sector, industry, cfg, moat)
    history = _consistent_history(m, sector, industry, cfg)
    passed = bool(circle["ok"] and moat["ok"] and history["ok"])
    return {
        "label": "B" if passed else None,
        "passed": passed,
        "circle": circle,
        "moat": moat,
        "history": history,
        "summary": _summary(passed, circle, moat, history),
    }


def _summary(passed, circle, moat, history) -> str:
    if passed:
        return ("Clears all three of Buffett's business tenets: "
                + circle["why"] + "; " + moat["why"] + "; " + history["why"] + ".")
    failed = [x for x in (circle, moat, history) if not x["ok"]]
    return ("Does not clear Buffett's business tenets — "
            + "; ".join(f["why"] for f in failed) + ".")


def _circle_of_competence(sector: Optional[str], industry: Optional[str],
                          cfg: Dict[str, Any],
                          moat: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read from a list YOU keep, never inferred.

    The circle of competence is a fact about the investor. No screen can
    measure it, and one that pretended to would be making exactly the error the
    tenet exists to prevent. So this checks membership of the list in
    `config/thresholds.yml` — edit it and the label changes.

    One widening route exists, and only because it was asked for: "products or
    services with a moat and a high barrier to entry" is a qualifier rather
    than an industry, so a name outside the list is admitted if it clears a
    STRICTER moat test than the tenet itself requires — every marker, not a
    majority. Admission by that route is labelled as such on the row, never
    silently folded into the list.
    """
    allowed = [str(x).strip().lower()
               for x in (cfg.get("circle_of_competence") or [])]
    if not allowed:
        return {"ok": True, "declared": False,
                "why": ("no circle of competence declared, so this tenet is "
                        "not being tested — set `circle_of_competence` in "
                        "config/thresholds.yml to make it bite")}
    s, i = (sector or "").strip().lower(), (industry or "").strip().lower()
    hit = next((a for a in allowed if a and (a == s or a in i or a in s)), None)
    if hit:
        return {"ok": True, "declared": True, "matched": hit, "via_moat": False,
                "why": (f"{industry or sector} is inside the circle of "
                        "competence you declared")}

    # The high-barrier route.
    if cfg.get("admit_on_strong_moat") and moat:
        strong = (moat.get("durable_roe")
                  and moat.get("scored", 0) >= 4
                  and moat.get("hits", 0) >= (moat.get("scored", 0)
                                              if cfg.get("strong_moat_requires_all",
                                                         True) else 3))
        if strong:
            return {"ok": True, "declared": True, "matched": None,
                    "via_moat": True,
                    "why": (f"{industry or sector} is not on your list, but it "
                            "is admitted on the high-barrier route — every "
                            "moat marker holds and returns on equity stayed "
                            "above 15% through the decade, which is the "
                            "'products or services with a moat' qualifier "
                            "rather than an industry you named")}
    return {"ok": False, "declared": True, "matched": None, "via_moat": False,
            "why": (f"{industry or sector or 'this sector'} is outside the "
                    "circle of competence declared in config, and does not "
                    "clear the high-barrier route either")}


def _moat(m: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A moat is not observable. Its FOOTPRINT is.

    High returns on capital sustained over a decade are not a normal outcome:
    capitalism removes them. When they persist, something is doing the
    protecting — pricing power, a brand, switching costs, scale. The screen
    cannot see which, and says so; what it can see is that competition has not
    yet done its usual work.
    """
    need = cfg.get("moat", {})
    gm = m.get("gross_margin_ttm")
    gm_cv = m.get("gross_margin_cv")
    roic = m.get("roic_5y_avg")
    rota = m.get("return_on_net_tangible_assets")
    roe_years = m.get("roe_years_above_15")
    roe_eval = m.get("roe_years_evaluated") or 0
    evidence, missing = [], []

    def _test(label, val, thr, op="gte"):
        if not _n(val):
            missing.append(label)
            return None
        ok = val >= thr if op == "gte" else val <= thr
        evidence.append(("✓" if ok else "✗") + f" {label}")
        return ok

    checks = [
        _test(f"gross margin {gm:.0%} vs {need.get('gross_margin', 0.40):.0%}"
              if _n(gm) else "gross margin", gm, need.get("gross_margin", 0.40)),
        _test(f"margin stability {gm_cv:.2f} vs {need.get('gross_margin_cv', 0.15):.2f}"
              if _n(gm_cv) else "margin stability", gm_cv,
              need.get("gross_margin_cv", 0.15), "lte"),
        _test(f"5-year return on capital {roic:.0%} vs "
              f"{need.get('roic', 0.12):.0%}" if _n(roic) else "return on capital",
              roic, need.get("roic", 0.12)),
        _test(f"return on net tangible assets {rota:.0%} vs "
              f"{need.get('return_on_tangible', 0.20):.0%}" if _n(rota)
              else "return on net tangible assets", rota,
              need.get("return_on_tangible", 0.20)),
    ]
    scored = [c for c in checks if c is not None]
    hits = sum(1 for c in scored if c)
    # A decade of high ROE is the strongest single piece of evidence, so it is
    # counted separately rather than averaged away.
    durable_roe = (_n(roe_years) and roe_eval >= 5
                   and roe_years / roe_eval >= need.get("roe_years_share", 0.80))
    if durable_roe:
        evidence.append(f"✓ return on equity above 15% in {roe_years} of "
                        f"{roe_eval} years")
    elif roe_eval:
        evidence.append(f"✗ return on equity above 15% in only {roe_years} of "
                        f"{roe_eval} years")
    need_hits = need.get("min_checks_passed", 3)
    ok = bool(len(scored) >= 3 and hits >= need_hits and durable_roe)
    why = (f"high returns have persisted — {hits} of {len(scored)} moat "
           "markers hold and returns on equity stayed above 15% through the "
           "decade, which competition does not normally allow"
           if ok else
           f"the moat markers do not hold ({hits} of {len(scored)}"
           + (", and returns on equity did not stay high" if not durable_roe else "")
           + ")")
    return {"ok": ok, "hits": hits, "scored": len(scored), "missing": missing,
            "durable_roe": durable_roe, "evidence": evidence, "why": why,
            "caveat": ("These are the FOOTPRINTS of a moat, not the moat. The "
                       "screen cannot see a brand, a switching cost or a "
                       "network effect — only that returns competition should "
                       "have eroded have not been eroded yet.")}


def _consistent_history(m: Dict[str, Any], sector: Optional[str],
                        industry: Optional[str],
                        cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stable products, enduring demand, no turnaround, no disruption zone."""
    need = cfg.get("history", {})
    losses = m.get("loss_years_in_10")
    recent = m.get("loss_years_in_3")
    cv = m.get("eps_cv_5y")
    age = m.get("listing_age_years")
    hist = m.get("history_years") or 0
    cat = m.get("lynch_category")
    text = f"{(sector or '').lower()} {(industry or '').lower()}"
    # Word boundaries, not substrings: "space" must not match "aerospace",
    # which is a mature manufacturing industry and precisely the sort of thing
    # this list should not be quietly excluding.
    disrupted = next((w for w in DISRUPTION_WORDS
                      if _re.search(r"\b" + _re.escape(w), text)), None)

    reasons = []
    ok = True
    if _n(losses) and losses > need.get("max_loss_years_in_10", 0):
        ok = False
        reasons.append(f"{losses} loss-making year(s) in the last ten")
    if _n(recent) and recent > 0:
        ok = False
        reasons.append("a loss inside the last three years")
    if _n(cv) and cv > need.get("max_eps_cv", 0.35):
        ok = False
        reasons.append(f"earnings vary too much year to year (variation {cv:.2f})")
    if _n(age) and age < need.get("min_listing_age_years", 10):
        ok = False
        reasons.append(f"listed only {age:.0f} years")
    if hist < need.get("min_history_years", 5):
        ok = False
        reasons.append(f"only {hist} years of statements on file")
    if cat == "turnaround":
        ok = False
        reasons.append("it is currently a turnaround, which Buffett explicitly "
                       "avoided in favour of businesses that never needed one")
    if disrupted and need.get("exclude_disruption_zones", True):
        ok = False
        reasons.append(f"'{disrupted}' is an industry where the product itself "
                       "keeps changing")
    why = ("the record is steady — no loss years, earnings that do not swing, "
           "and a long enough listing to have been tested"
           if ok else "; ".join(reasons))
    return {"ok": ok, "reasons": reasons, "why": why}
