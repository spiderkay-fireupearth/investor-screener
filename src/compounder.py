"""The compounder test: is the engine strong, and is the price asking too much?

Four questions, in the order they have to be asked. The first three are about
the BUSINESS and the fourth is about the PRICE, and that ordering is the point —
a wonderful company at an impossible price is not an investment, and a cheap
price on a business that destroys capital is not a bargain.

    1. Operating margin above 20%, and expanding rather than bought.
    2. ROIC above 15%, AND above the cost of capital by more than 5 points.
    3. Maintenance capex under half of operating cash flow.
    4. The revenue growth the market is already paying for, under 10%.

WHAT IS ESTIMATED HERE, AND WHAT THAT MEANS
-------------------------------------------
Two of these four cannot be read off a filing. They are estimates, and the
module is built so that is never in doubt on the page.

  WACC is not disclosed by anyone. Cost of equity comes from CAPM — the
  ten-year Treasury this run already fetches, plus beta times an equity risk
  premium — and cost of debt from interest expense over total debt, tax
  effected. Every input is reported alongside the answer, because a spread of
  "ROIC 18% against WACC 9%" is worth nothing to a reader who cannot see that
  the 9% assumed a beta of 1.0.

  MAINTENANCE CAPEX is not disclosed either. This reuses the estimate the
  Buffett framework already makes — the more conservative of Buffett's
  depreciation proxy and Greenwald's growth-capex split — rather than inventing
  a second one, so two panels on the same page can never disagree about how
  much of this company's spending is upkeep.

THE REVERSE DCF IS NOT A VALUATION
----------------------------------
It does not say what the company is worth. It says what growth rate the current
enterprise value is already paying for, holding today's operating margin and
capital intensity constant. That is a much narrower and much more checkable
claim, and it is the only honest way to use a DCF on a business whose future
nobody knows: instead of asserting a growth rate and deriving a price, take the
price the market has set and derive the growth rate it implies. Then the reader
does the one thing a model cannot — judge whether that rate is plausible.

WHAT THIS CANNOT DO
-------------------
It cannot check the implied revenue against a total addressable market. There
is no free, machine-readable source for TAM, it is contested even among people
who follow one industry closely, and a fabricated denominator would make the
most important sanity check in the whole exercise look rigorous. The implied
ten-year revenue figure IS computed and shown, so the reader can weigh it
against a market they know; the weighing is theirs.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULTS = {
    # CAPM inputs.
    "equity_risk_premium": 0.05,     # the long-run US equity premium
    "default_beta": 1.0,             # used, and LABELLED, when the feed has none
    "fallback_risk_free": 0.042,     # only if FRED is unreachable this run
    "max_tax_rate": 0.35,
    "min_tax_rate": 0.0,
    # Reverse DCF.
    "horizon_years": 10,
    "terminal_growth": 0.025,        # capped at the risk-free rate below
    "min_spread_over_terminal": 0.01,  # WACC must clear terminal growth
    "search_low": -0.25,
    "search_high": 0.60,
    "search_iterations": 80,
    # Floors that keep a nonsense input from producing a confident answer.
    "min_revenue": 1.0,
    "min_wacc": 0.04,
    "max_wacc": 0.25,
}


def _n(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _div(a, b) -> Optional[float]:
    if not _n(a) or not _n(b) or float(b) == 0:
        return None
    return float(a) / float(b)


# ---------------------------------------------------------------------------
# Cost of capital
# ---------------------------------------------------------------------------

def _f(y, attr):
    """One field off the latest fiscal year, or None."""
    v = getattr(y, attr, None) if y is not None else None
    return float(v) if _n(v) else None


def effective_tax_rate(y0, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The rate the company actually paid, clamped to something usable.

    Clamped because the reported rate is unusable at the edges: a loss-making
    year gives a negative rate, a one-off settlement can give 80%, and either
    would swing the whole discount rate. The clamp is stated rather than
    silently applied.
    """
    c = {**DEFAULTS, **(cfg or {})}
    pre, net = _f(y0, "pretax_income"), _f(y0, "net_income")
    raw = None
    if _n(pre) and float(pre) > 0 and _n(net):
        raw = (float(pre) - float(net)) / float(pre)
    if raw is None:
        return {"rate": 0.21, "source": "assumed 21% — the company's own rate "
                                        "could not be computed from its last "
                                        "statements", "raw": None}
    clamped = max(c["min_tax_rate"], min(c["max_tax_rate"], raw))
    src = "from the last income statement"
    if abs(clamped - raw) > 1e-9:
        src = (f"its reported rate was {raw:.0%}, clamped to "
               f"{clamped:.0%} — outside that band the figure is a one-off, "
               f"not a rate")
    return {"rate": clamped, "source": src, "raw": raw}


def wacc(m: Dict[str, Any], y0, beta: Optional[float],
         risk_free: Optional[float],
         cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Weighted average cost of capital, with every input on the record.

    Returned as a dict rather than a number because the number alone is not
    reviewable. A reader who cannot see which beta and which premium produced
    9.1% has no way to disagree with it, and a figure nobody can disagree with
    is not analysis.
    """
    c = {**DEFAULTS, **(cfg or {})}
    rf = float(risk_free) if _n(risk_free) else c["fallback_risk_free"]
    rf_source = ("the 10-year Treasury from FRED" if _n(risk_free)
                 else f"assumed {c['fallback_risk_free']:.1%} — FRED was "
                      f"unreachable on this run")

    if _n(beta) and 0 < float(beta) < 5:
        b, b_source = float(beta), "from the data feed"
    else:
        b, b_source = c["default_beta"], (
            f"assumed {c['default_beta']:.1f} — the feed carried no beta for "
            f"this company, so this cost of equity is the market's, not its own")

    ke = rf + b * c["equity_risk_premium"]

    tax = effective_tax_rate(y0, c)
    debt = _f(y0, "total_debt")
    # Market cap in the STATEMENT currency, because debt and interest are in
    # that currency too. Mixing a USD market cap with HKD debt would set the
    # equity and debt weights by an exchange rate rather than by the balance
    # sheet — the same trap the app already guards in its valuation ratios.
    equity = m.get("market_cap_stmt_ccy") or m.get("market_cap")
    interest = _f(y0, "interest_expense")

    kd_pre = _div(abs(float(interest)) if _n(interest) else None, debt)
    kd_source = "interest expense over total debt"
    if kd_pre is None or not (0.001 < kd_pre < 0.30):
        # No debt, or a ratio that cannot be a borrowing rate — a company with
        # trivial debt often books interest income here, giving a negative or
        # absurd figure. Fall back to the risk-free rate plus a spread.
        kd_pre = rf + 0.015
        kd_source = (f"assumed {kd_pre:.1%} — risk-free plus 150bp, because "
                     f"interest expense over total debt did not produce a "
                     f"usable borrowing rate")
    kd = kd_pre * (1.0 - tax["rate"])

    if not _n(equity) or float(equity) <= 0:
        return {"available": False,
                "reason": "no market capitalisation, so the equity and debt "
                          "weights cannot be set"}
    e = float(equity)
    d = float(debt) if (_n(debt) and float(debt) > 0) else 0.0
    total = e + d
    w_e, w_d = e / total, d / total
    value = w_e * ke + w_d * kd

    if not (c["min_wacc"] <= value <= c["max_wacc"]):
        return {"available": False,
                "reason": (f"the computed cost of capital, {value:.1%}, falls "
                           f"outside the {c['min_wacc']:.0%}–{c['max_wacc']:.0%} "
                           f"band this page will report; one of its inputs is "
                           f"wrong rather than the company being unusual")}

    return {
        "available": True,
        "wacc": value,
        "cost_of_equity": ke,
        "cost_of_debt_pretax": kd_pre,
        "cost_of_debt_after_tax": kd,
        "beta": b, "beta_source": b_source,
        "risk_free": rf, "risk_free_source": rf_source,
        "equity_risk_premium": c["equity_risk_premium"],
        "tax_rate": tax["rate"], "tax_source": tax["source"],
        "weight_equity": w_e, "weight_debt": w_d,
        "basis": (f"{w_e:.0%} equity at {ke:.1%} (risk-free {rf:.2%} + beta "
                  f"{b:.2f} × {c['equity_risk_premium']:.1%} premium) and "
                  f"{w_d:.0%} debt at {kd:.1%} after tax"),
    }


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------

def _ev_at_growth(g: float, revenue0: float, margin: float, tax: float,
                  sales_to_capital: float, disc: float, years: int,
                  terminal_g: float) -> float:
    """Enterprise value implied by a constant revenue growth rate.

    Reinvestment is tied to REVENUE GROWTH rather than to earnings, through the
    sales-to-capital ratio the company actually runs at. That is what stops the
    model producing free growth: a business cannot double revenue without
    funding the capital to carry it, and a DCF that forgets this will justify
    any price at all.
    """
    pv = 0.0
    rev_prev = revenue0
    fcff_last = 0.0
    for t in range(1, years + 1):
        rev = rev_prev * (1.0 + g)
        nopat = rev * margin * (1.0 - tax)
        reinvest = (rev - rev_prev) / sales_to_capital if sales_to_capital else 0.0
        fcff = nopat - reinvest
        pv += fcff / ((1.0 + disc) ** t)
        rev_prev = rev
        fcff_last = fcff
    # Terminal value on a steady-state year: growth drops to the terminal rate
    # and reinvestment falls with it.
    nopat_term = rev_prev * (1.0 + terminal_g) * margin * (1.0 - tax)
    reinvest_term = (rev_prev * terminal_g) / sales_to_capital if sales_to_capital else 0.0
    fcff_term = nopat_term - reinvest_term
    tv = fcff_term / (disc - terminal_g)
    pv += tv / ((1.0 + disc) ** years)
    return pv


def reverse_dcf(m: Dict[str, Any], y0, w: Dict[str, Any],
                cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What revenue growth is today's enterprise value already paying for?

    Solved by bisection rather than a formula because the relationship between
    growth and value is not invertible in closed form once reinvestment is tied
    to growth — faster growth raises earnings and consumes capital at the same
    time, and which wins depends on the company.
    """
    c = {**DEFAULTS, **(cfg or {})}
    if not w.get("available"):
        return {"available": False,
                "reason": "no cost of capital, so there is no rate to discount at"}

    ev = m.get("enterprise_value")
    revenue = _f(y0, "revenue")
    margin = m.get("operating_margin_ttm")
    invested = _f(y0, "invested_capital")
    tax = w["tax_rate"]
    disc = w["wacc"]

    missing = [k for k, v in (("enterprise value", ev), ("revenue", revenue),
                              ("operating margin", margin),
                              ("invested capital", invested)) if not _n(v)]
    if missing:
        return {"available": False,
                "reason": f"missing {', '.join(missing)}"}
    ev, revenue, margin, invested = (float(ev), float(revenue), float(margin),
                                     float(invested))
    if ev <= 0:
        return {"available": False,
                "reason": "enterprise value is zero or negative — net cash "
                          "exceeds the market value of the equity, and a "
                          "growth rate cannot be solved against it"}
    if revenue < c["min_revenue"] or margin <= 0:
        return {"available": False,
                "reason": ("the company is not currently profitable at the "
                           "operating line, so there is no margin to hold "
                           "constant" if margin <= 0 else "revenue too small")}
    if invested <= 0:
        return {"available": False,
                "reason": "invested capital is zero or negative, so the "
                          "reinvestment the growth would require cannot be sized"}

    terminal_g = min(c["terminal_growth"], w["risk_free"])
    if disc - terminal_g < c["min_spread_over_terminal"]:
        return {"available": False,
                "reason": (f"the cost of capital ({disc:.1%}) is too close to "
                           f"the terminal growth rate ({terminal_g:.1%}) for a "
                           f"terminal value to mean anything")}

    s2c = revenue / invested
    lo, hi = c["search_low"], c["search_high"]

    def f(g):
        return _ev_at_growth(g, revenue, margin, tax, s2c, disc,
                             c["horizon_years"], terminal_g) - ev

    f_lo, f_hi = f(lo), f(hi)
    # Both ends on the same side means no rate in the searched range reproduces
    # today's price. That is a real answer and it is reported as one — "more
    # than 60%" is the finding, not a failure to compute.
    if f_lo > 0 and f_hi > 0:
        return {"available": True, "implied_growth": lo, "bounded": "below",
                "note": (f"Even at {lo:.0%} revenue growth the discounted cash "
                         f"flows exceed today's enterprise value — the market "
                         f"is pricing in decline, or the margin is unusual."),
                "sales_to_capital": s2c, "terminal_growth": terminal_g,
                "discount_rate": disc, "margin_held": margin,
                "horizon_years": c["horizon_years"]}
    if f_lo < 0 and f_hi < 0:
        return {"available": True, "implied_growth": hi, "bounded": "above",
                "note": (f"Even {hi:.0%} revenue growth every year for "
                         f"{c['horizon_years']} years does not justify today's "
                         f"enterprise value on current margins — this is priced "
                         f"beyond what the model can express."),
                "sales_to_capital": s2c, "terminal_growth": terminal_g,
                "discount_rate": disc, "margin_held": margin,
                "horizon_years": c["horizon_years"]}

    for _ in range(c["search_iterations"]):
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    g = (lo + hi) / 2.0
    rev10 = revenue * ((1.0 + g) ** c["horizon_years"])
    return {
        "available": True, "implied_growth": g, "bounded": None,
        "revenue_in_10y": rev10,
        "revenue_now": revenue,
        "revenue_multiple": rev10 / revenue if revenue else None,
        "sales_to_capital": s2c,
        "terminal_growth": terminal_g,
        "discount_rate": disc,
        "margin_held": margin,
        "horizon_years": c["horizon_years"],
        "note": (f"Holding the operating margin at {margin:.1%} and the "
                 f"sales-to-capital ratio at {s2c:.2f}, revenue must compound "
                 f"at {g:.1%} for {c['horizon_years']} years to justify "
                 f"today's enterprise value."),
    }


# ---------------------------------------------------------------------------
# The four tests, as metrics
# ---------------------------------------------------------------------------

def assess(m: Dict[str, Any], y0=None, beta: Optional[float] = None,
           risk_free: Optional[float] = None,
           cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every metric the compounder framework tests, plus the evidence behind it.

    Written back into the metrics dict by the caller, so the config-driven
    threshold tests can read them like any other metric.
    """
    c = {**DEFAULTS, **(cfg or {})}
    out: Dict[str, Any] = {}

    # --- 1. Operating margin, level and direction --------------------------
    om = m.get("operating_margin_ttm")
    out["compounder_operating_margin"] = om
    # The slope over five years, already computed for the quality screens. A
    # margin of 22% that was 30% three years ago is a business buying its
    # growth, and the level alone cannot see that.
    slope = m.get("operating_margin_slope_5y")
    out["compounder_margin_slope"] = slope
    # Expanding is the test; "not contracting" is the bar, because a flat
    # high margin is a moat holding and a rising one is a moat widening.
    out["compounder_margin_expanding"] = (
        1 if _n(slope) and float(slope) >= 0 else
        (0 if _n(slope) else None))

    # --- 2. ROIC against the cost of capital -------------------------------
    # EBIT over invested capital on the latest year — the app already computes
    # it for the Greenblatt and Buffett screens. Falls back to the five-year
    # average when the latest year is unusable, which is the more forgiving
    # reading and is labelled as such in the evidence line.
    roic = m.get("ebit_to_invested_capital")
    if not _n(roic):
        roic = m.get("roic_5y_avg")
    out["compounder_roic"] = roic
    w = wacc(m, y0, beta, risk_free, c)
    out["compounder_wacc_detail"] = w
    out["compounder_wacc"] = w.get("wacc") if w.get("available") else None
    if _n(roic) and w.get("available"):
        out["compounder_spread"] = float(roic) - w["wacc"]
    else:
        out["compounder_spread"] = None

    # --- 3. Maintenance capex against operating cash flow ------------------
    # Reuses the Buffett framework's estimate rather than making a second one,
    # so the two panels can never disagree about the same company.
    maint, cfo = m.get("maintenance_capex"), _f(y0, "cfo")
    out["compounder_maint_capex_to_cfo"] = _div(maint, cfo)
    out["compounder_maint_capex"] = maint

    # --- 4. What the price is already paying for ---------------------------
    rd = reverse_dcf(m, y0, w, c)
    out["compounder_reverse_dcf"] = rd
    out["compounder_implied_growth"] = (rd.get("implied_growth")
                                        if rd.get("available") else None)
    out["compounder_implied_revenue_10y"] = rd.get("revenue_in_10y")
    # Stated plainly because it is the number a reader can actually check
    # against a market they know — which is the TAM test this module cannot
    # perform for them.
    out["compounder_revenue_multiple_10y"] = rd.get("revenue_multiple")
    return out


def explain(m: Dict[str, Any]) -> List[str]:
    """Plain-language evidence lines for the row drawer."""
    lines: List[str] = []
    om = m.get("compounder_operating_margin")
    slope = m.get("compounder_margin_slope")
    if _n(om):
        # "Flat" and "we could not strike the slope" are different statements
        # and must not share a word. A company with three years of statements
        # has an unknown trend, not a flat one, and calling it flat would have
        # the page report our feed's depth as a fact about the business.
        if not _n(slope):
            lines.append(f"Operating margin {om:.1%}; the five-year trend "
                         f"could not be struck from the statements available.")
        else:
            d = ("expanding" if slope > 0.001 else
                 ("contracting" if slope < -0.001 else "flat"))
            lines.append(f"Operating margin {om:.1%}, {d} over five years "
                         f"({slope * 100:+.2f} points a year).")
    roic, w, sp = (m.get("compounder_roic"), m.get("compounder_wacc"),
                   m.get("compounder_spread"))
    if _n(roic) and _n(w) and _n(sp):
        verb = "above" if sp > 0 else "below"
        # Percentage POINTS, not the raw fraction: a spread of 0.404 is 40
        # points, and printing "0.4 points above" for a business earning five
        # times its cost of capital is the kind of slip a reader corrects for
        # once and then stops trusting the page.
        det = m.get("compounder_wacc_detail") or {}
        lines.append(f"ROIC {roic:.1%} against a cost of capital of {w:.1%} — "
                     f"{abs(sp) * 100:.1f} points {verb} it. "
                     + (det.get("basis") or ""))
        # Which inputs were assumed rather than measured. The basis line prints
        # the beta and the risk-free rate but not where they came from, and a
        # reader who cannot tell a fed beta from a placeholder 1.0 has no way
        # to know how much weight this spread will bear.
        caveats = [s for s in (det.get("beta_source"), det.get("risk_free_source"))
                   if s and s.startswith("assumed")]
        if caveats:
            s = "; ".join(caveats)
            lines.append(s[0].upper() + s[1:] + ".")
    elif _n(roic):
        lines.append(f"ROIC {roic:.1%}; the cost of capital could not be "
                     f"estimated, so the spread is unknown rather than zero.")
    mc = m.get("compounder_maint_capex_to_cfo")
    if _n(mc):
        lines.append(f"Estimated maintenance capital expenditure takes "
                     f"{mc:.0%} of operating cash flow.")
    rd = m.get("compounder_reverse_dcf") or {}
    if rd.get("available"):
        lines.append(rd.get("note") or "")
        if rd.get("revenue_multiple"):
            lines.append(f"That is {rd['revenue_multiple']:.1f}× today's "
                         f"revenue within {rd.get('horizon_years', 10)} years. "
                         f"Whether that is a plausible share of this "
                         f"company's addressable market is the one judgement "
                         f"this page cannot make for you.")
    elif rd.get("reason"):
        lines.append(f"The implied growth rate could not be solved: "
                     f"{rd['reason']}.")
    return [x for x in lines if x]
