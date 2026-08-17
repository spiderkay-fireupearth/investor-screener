"""Derive every metric the screens reference, from a CompanyRecord.

Ratios are computed entirely within the filing's reporting currency, so no FX
conversion is needed for them — a Thai company's EV/EBIT is the same number in
THB or USD. FX is applied only to the absolute-size gates (market cap floor,
turnover floor) and to cross-market ranking displays. That keeps exactly one
conversion point in the system.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import numpy as np

from .schema import CompanyRecord, FundamentalYear


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and math.isnan(x))


def _safe_div(a, b) -> Optional[float]:
    if not (_n(a) and _n(b)) or b == 0:
        return None
    return a / b


def _cv(vals: List[float]) -> Optional[float]:
    """Coefficient of variation — our stability proxy for margins and ROIC."""
    v = [x for x in vals if _n(x)]
    if len(v) < 3:
        return None
    mean = float(np.mean(v))
    if mean == 0:
        return None
    return float(np.std(v) / abs(mean))


def _cagr(newest: Optional[float], oldest: Optional[float], years: int) -> Optional[float]:
    if not (_n(newest) and _n(oldest)) or years <= 0:
        return None
    # Growth from a negative or zero base is not a meaningful CAGR.
    if oldest <= 0 or newest <= 0:
        return None
    return (newest / oldest) ** (1.0 / years) - 1.0


def _slope(vals: List[float]) -> Optional[float]:
    """Normalised least-squares slope of a newest-first series."""
    v = [x for x in vals if _n(x)]
    if len(v) < 3:
        return None
    y = np.array(list(reversed(v)), dtype=float)      # oldest -> newest
    x = np.arange(len(y), dtype=float)
    m = float(np.polyfit(x, y, 1)[0])
    denom = abs(float(np.mean(y)))
    return m / denom if denom else None


def enterprise_value(market_cap: Optional[float],
                     fy: FundamentalYear) -> Optional[float]:
    """EV in the STATEMENT currency — `market_cap` must already be restated."""
    if not _n(market_cap):
        return None
    debt = fy.total_debt or 0.0
    cash = (fy.cash_and_equivalents or 0.0) + (fy.short_term_investments or 0.0)
    mi = fy.minority_interest or 0.0
    ev = market_cap + debt + mi - cash
    return ev if ev > 0 else None


def reconcile_currency(rec: CompanyRecord,
                       fx_rates: Optional[Dict[str, float]]) -> tuple:
    """Return (market_cap, price) restated into the STATEMENT currency.

    Shares can trade in one currency while the financials are reported in
    another — routine on HKEX, where mainland issuers report in CNY and trade
    in HKD, and on SGX, where several names report in USD. Every valuation
    ratio in this module divides a market figure by a statement figure, so
    without this step P/E, P/B, EV/EBIT and NCAV are all quietly wrong by the
    exchange rate. Wrong by 8% is worse than wrong by 800%: nobody notices.
    """
    mcap, price = rec.market_cap, rec.price
    trade_ccy = (rec.currency or "USD").upper()
    stmt_ccy = (rec.financial_currency or trade_ccy).upper()

    if stmt_ccy == trade_ccy:
        return mcap, price, None

    if not fx_rates:
        return mcap, price, (f"statements in {stmt_ccy} but shares trade in "
                             f"{trade_ccy}; no FX available — ratios unreliable")

    r_trade = fx_rates.get(trade_ccy)
    r_stmt = fx_rates.get(stmt_ccy)
    if not (_n(r_trade) and _n(r_stmt)) or r_stmt == 0:
        return mcap, price, (f"statements in {stmt_ccy} but shares trade in "
                             f"{trade_ccy}; FX missing — ratios unreliable")

    factor = r_trade / r_stmt          # trading ccy -> statement ccy
    return ((mcap * factor if _n(mcap) else None),
            (price * factor if _n(price) else None),
            f"restated {trade_ccy}->{stmt_ccy} at {factor:.4f} for ratio maths")


def sanity_check(m: Dict[str, Any]) -> List[str]:
    """Catch ratios that are arithmetically valid but physically impossible.

    These almost always mean a unit or currency mismatch upstream, not a
    remarkable company. Surfacing them as warnings beats letting a P/B of
    7,000,000 sit in the table looking like a number.
    """
    flags = []
    checks = [
        ("pe_ttm", 5000, "P/E"),
        ("price_to_tangible_book", 1000, "P/TB"),
        ("price_to_book", 1000, "P/B"),
        ("ev_to_ebit", 5000, "EV/EBIT"),
    ]
    for key, limit, label in checks:
        v = m.get(key)
        if _n(v) and abs(v) > limit:
            flags.append(f"{label} of {v:,.0f} is implausible — likely a "
                         f"currency or units mismatch; treat this row as suspect")
    return flags


def compute_metrics(rec: CompanyRecord,
                    fx_rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    ys = rec.years
    latest = rec.latest
    m["years_of_data"] = len(ys)

    if not latest:
        m["_no_fundamentals"] = True
        return m

    # ---------------------------------------------------------- profitability
    roe_series: List[Optional[float]] = []
    roic_series: List[Optional[float]] = []
    gm_series: List[Optional[float]] = []
    om_series: List[Optional[float]] = []
    fcf_series: List[Optional[float]] = []
    ni_series: List[Optional[float]] = []

    for y in ys:
        roe_series.append(_safe_div(y.net_income, y.total_equity))
        roic_series.append(_safe_div(y.ebit, y.invested_capital))
        gm_series.append(_safe_div(y.gross_profit, y.revenue))
        om_series.append(_safe_div(y.ebit, y.revenue))
        fcf_series.append(y.free_cash_flow)
        ni_series.append(y.net_income)

    m["roe_series"] = roe_series[:10]
    m["roe_ttm"] = roe_series[0] if roe_series else None
    m["roe_years_above_15"] = sum(1 for v in roe_series[:10] if _n(v) and v >= 0.15)
    # How many years could actually be evaluated, as distinct from how many
    # passed. EDGAR gives 10+; Yahoo gives ~4. Without this denominator an
    # "8 of the last 10 years" test is unreachable for every Asian name, and
    # they fail on data depth rather than on merit.
    m["roe_years_evaluated"] = sum(1 for v in roe_series[:10] if _n(v))
    m["history_years"] = len(ys)

    m["roic_series"] = roic_series[:10]
    valid_roic = [v for v in roic_series[:5] if _n(v)]
    m["roic_5y_avg"] = float(np.mean(valid_roic)) if len(valid_roic) >= 3 else None
    m["roic_cv_5y"] = _cv(roic_series[:5])

    m["gross_margin_ttm"] = gm_series[0] if gm_series else None
    m["gross_margin_cv"] = _cv(gm_series[:10])
    m["operating_margin_ttm"] = om_series[0] if om_series else None
    m["operating_margin_slope_5y"] = _slope(om_series[:5])

    m["free_cash_flow_ttm"] = fcf_series[0] if fcf_series else None
    m["fcf_years_positive"] = sum(1 for v in fcf_series[:10] if _n(v) and v > 0)
    m["fcf_years_evaluated"] = sum(1 for v in fcf_series[:10] if _n(v))
    m["loss_years_in_10"] = sum(1 for v in ni_series[:10] if _n(v) and v < 0)

    # ------------------------------------------------------------- leverage
    m["debt_to_equity"] = _safe_div(latest.total_debt, latest.total_equity)
    m["net_debt_to_ebitda"] = _safe_div(latest.net_debt, latest.ebitda)
    m["goodwill_to_assets"] = _safe_div(latest.goodwill, latest.total_assets)

    # Accrual quality — Munger's inversion test. High accruals mean reported
    # earnings are not turning into cash.
    m["accruals_ratio"] = _safe_div(
        (latest.net_income - latest.cfo)
        if (_n(latest.net_income) and _n(latest.cfo)) else None,
        latest.total_assets)

    # -------------------------------------------------------- share dilution
    sc_now = latest.shares_diluted or latest.shares_outstanding or rec.shares_outstanding
    sc_5y = None
    if len(ys) > 5:
        y5 = ys[5]
        sc_5y = y5.shares_diluted or y5.shares_outstanding
    m["share_count_change_5y"] = _safe_div(
        (sc_now - sc_5y) if (_n(sc_now) and _n(sc_5y)) else None, sc_5y)

    # -------------------------------------------------------------- valuation
    # Restate market figures into the statement currency BEFORE any ratio.
    mcap, price, fx_note = reconcile_currency(rec, fx_rates)
    m["market_cap"] = rec.market_cap          # display value, trading currency
    m["market_cap_stmt_ccy"] = mcap
    m["statement_currency"] = rec.financial_currency or rec.currency
    if fx_note:
        m["currency_note"] = fx_note
        if "unreliable" in fx_note:
            rec.warnings.append(fx_note)

    eps = latest.eps_diluted
    if not _n(eps) and _n(latest.net_income) and _n(sc_now) and sc_now:
        eps = latest.net_income / sc_now
    m["eps_ttm"] = eps
    m["pe_ttm"] = _safe_div(price, eps) if (_n(price) and _n(eps) and eps > 0) else None

    tbv = latest.tangible_book_value
    m["price_to_tangible_book"] = _safe_div(mcap, tbv) if (_n(tbv) and tbv > 0) else None
    m["price_to_book"] = _safe_div(mcap, latest.total_equity) if (
        _n(latest.total_equity) and latest.total_equity > 0) else None

    ev = enterprise_value(mcap, latest)
    m["enterprise_value"] = ev
    m["ev_to_ebit"] = _safe_div(ev, latest.ebit) if (
        _n(latest.ebit) and latest.ebit > 0) else None
    m["ebit_to_ev"] = _safe_div(latest.ebit, ev) if (
        _n(latest.ebit) and latest.ebit > 0) else None
    m["ebit_to_invested_capital"] = _safe_div(latest.ebit, latest.invested_capital)

    net_cash = None
    if _n(latest.total_debt) or _n(latest.cash_and_equivalents):
        cash = (latest.cash_and_equivalents or 0.0) + (latest.short_term_investments or 0.0)
        net_cash = cash - (latest.total_debt or 0.0)
    m["net_cash_to_market_cap"] = _safe_div(net_cash, mcap)
    m["ncav_to_market_cap"] = _safe_div(latest.ncav, mcap)
    m["fcf_yield"] = _safe_div(latest.free_cash_flow, mcap)
    # Templeton screens on price-to-cash-flow as well as P/E and P/B — cash
    # flow is harder to dress up than earnings, which is the point.
    m["price_to_cash_flow"] = _safe_div(mcap, latest.cfo) if (
        _n(latest.cfo) and latest.cfo > 0) else None
    m["earnings_yield"] = _safe_div(eps, price) if (
        _n(eps) and _n(price) and price > 0) else None
    # "Maximum pessimism" needs both distance-from-high and nearness-to-low.
    pbh = rec.technicals.get("pct_below_52w_high")
    m["pct_below_52w_high"] = pbh
    m["drawdown_score"] = pbh if _n(pbh) else None

    # ------------------------------------------------------------------ growth
    if len(ys) > 5 and _n(ys[0].eps_diluted) and _n(ys[5].eps_diluted):
        m["eps_cagr_5y"] = _cagr(ys[0].eps_diluted, ys[5].eps_diluted, 5)
    elif len(ys) > 5:
        m["eps_cagr_5y"] = _cagr(
            _safe_div(ys[0].net_income, sc_now),
            _safe_div(ys[5].net_income, ys[5].shares_diluted or sc_5y), 5)
    else:
        m["eps_cagr_5y"] = None

    if len(ys) > 10 and _n(ys[0].eps_diluted) and _n(ys[10].eps_diluted):
        m["eps_cagr_10y"] = _cagr(ys[0].eps_diluted, ys[10].eps_diluted, 10)
    else:
        m["eps_cagr_10y"] = None

    m["revenue_cagr_5y"] = (_cagr(ys[0].revenue, ys[5].revenue, 5)
                            if len(ys) > 5 else None)

    growth_pct = m["eps_cagr_5y"] * 100 if _n(m["eps_cagr_5y"]) else None
    m["peg_ratio"] = _safe_div(m["pe_ttm"], growth_pct) if (
        _n(m["pe_ttm"]) and _n(growth_pct) and growth_pct > 0) else None

    # Lynch's inventory tell — inventory outgrowing sales means product isn't moving.
    if len(ys) > 1:
        inv_g = _safe_div(
            (ys[0].inventory - ys[1].inventory)
            if (_n(ys[0].inventory) and _n(ys[1].inventory)) else None,
            ys[1].inventory)
        rev_g = _safe_div(
            (ys[0].revenue - ys[1].revenue)
            if (_n(ys[0].revenue) and _n(ys[1].revenue)) else None,
            ys[1].revenue)
        m["inventory_growth"] = inv_g
        m["revenue_growth_1y"] = rev_g
        m["inventory_growth_less_revenue_growth"] = (
            inv_g - rev_g if (_n(inv_g) and _n(rev_g)) else None)
    else:
        m["inventory_growth_less_revenue_growth"] = None

    # ------------------------------------------------------------------------
    # Soros, "The Alchemy of Finance" — the mechanics of a reflexive loop.
    #
    # His unit of analysis is a company whose SHARE PRICE is an input to its
    # own fundamentals, not merely a reflection of them. The channel is equity
    # issuance: "The true attraction of mortgage trusts lies in their ability
    # to generate capital gains for their shareholders by selling additional
    # shares at a premium over book value... The higher the premium, the easier
    # it is for the trust to fulfil this expectation."
    #
    # So the diagnostic is not momentum. It is: is this company *converting its
    # own multiple into reported earnings*? These metrics measure that.
    # ------------------------------------------------------------------------
    if len(ys) >= 2:
        m["eps_growth_1y"] = _cagr(ys[0].eps_diluted, ys[1].eps_diluted, 1)
        # Revenue per share strips out the effect of issuing paper. If EPS is
        # growing much faster than revenue per share, the growth is arriving
        # through the share count and the accounting, not through the business.
        rps0 = _safe_div(ys[0].revenue, ys[0].shares_diluted)
        rps1 = _safe_div(ys[1].revenue, ys[1].shares_diluted)
        m["revenue_per_share_growth_1y"] = _cagr(rps0, rps1, 1)
        m["share_count_change_1y"] = _safe_div(
            (ys[0].shares_diluted - ys[1].shares_diluted)
            if (_n(ys[0].shares_diluted) and _n(ys[1].shares_diluted))
            else None, ys[1].shares_diluted)
        # The conglomerate fingerprint, in one number. Soros: "Investors had
        # come to value growth in per-share earnings and failed to discriminate
        # about the way the earnings growth was accomplished."
        m["eps_over_revenue_per_share_gap"] = (
            m["eps_growth_1y"] - m["revenue_per_share_growth_1y"]
            if (_n(m.get("eps_growth_1y"))
                and _n(m.get("revenue_per_share_growth_1y"))) else None)
    else:
        for k in ("eps_growth_1y", "revenue_per_share_growth_1y",
                  "share_count_change_1y", "eps_over_revenue_per_share_gap"):
            m[k] = None

    # Act Two of the mortgage-trust scenario: "With higher leverage, the rate
    # of return on equity can be maintained despite a lower effective yield."
    # A flat ROE held up by rising leverage and a falling margin is the exact
    # configuration he describes — and it is invisible in the ROE alone.
    m["leverage_change_3y"] = None
    m["roe_held_up_by_leverage"] = None
    if len(ys) >= 4:
        lev0 = _safe_div(ys[0].total_assets, ys[0].total_equity)
        lev3 = _safe_div(ys[3].total_assets, ys[3].total_equity)
        mar0 = _safe_div(ys[0].net_income, ys[0].revenue)
        mar3 = _safe_div(ys[3].net_income, ys[3].revenue)
        roe0 = _safe_div(ys[0].net_income, ys[0].total_equity)
        roe3 = _safe_div(ys[3].net_income, ys[3].total_equity)
        if all(_n(x) for x in (lev0, lev3, mar0, mar3, roe0, roe3)) and lev3:
            m["leverage_change_3y"] = lev0 / lev3 - 1.0
            # 1 = the Act Two signature is present. Reported as a flag rather
            # than a score so the reason survives into the UI.
            m["roe_held_up_by_leverage"] = int(
                roe0 >= roe3 * 0.95 and lev0 > lev3 * 1.10 and mar0 < mar3)

    # Stage DE, the diagnostic one: "conviction develops and it is no longer
    # shaken by a setback in the earning trend... Expectations become
    # excessive, and fail to be sustained by reality." Measured as the gap
    # between what the price did and what the earnings did over the same year.
    _r12 = rec.technicals.get("return_12m")
    m["reflexive_divergence"] = (
        _r12 - m["eps_growth_1y"]
        if (_n(_r12) and _n(m.get("eps_growth_1y"))) else None)

    # ------------------------------------------------------------------------
    # Rogers, "Hot Commodities" — the supply side, at company level.
    #
    # His thesis is under-investment: "Virtually no new mine shafts have been
    # opened in 20 years worldwide"; "There were 4,530 rigs in the U.S. at the
    # end of 1981... In 2004... the total was 1,201." Capex against depreciation
    # is the company-level version of that: below 1.0 and the asset base is
    # shrinking, which is bullish for the commodity and eventually for whoever
    # still owns production.
    # ------------------------------------------------------------------------
    if ys:
        m["capex_to_depreciation"] = _safe_div(
            ys[0].capex, ys[0].depreciation_amortization)
        m["capex_to_revenue"] = _safe_div(ys[0].capex, ys[0].revenue)
    else:
        m["capex_to_depreciation"] = m["capex_to_revenue"] = None

    # ------------------------------------------- Soros: reflexivity divergence
    # Compare the direction of earnings against the direction of price. When
    # price is running well ahead of (or well behind) fundamentals, that gap is
    # the reflexive signal — the market's belief is doing the work, not the
    # business. +1 aligned, 0 neutral, -1 divergent.
    eps_trend = _slope([y.eps_diluted for y in ys[:4]])
    price_trend = rec.technicals.get("return_12m")
    if _n(eps_trend) and _n(price_trend):
        if eps_trend > 0.02 and price_trend > 0:
            m["eps_trend_vs_price_trend"] = 1
        elif eps_trend < -0.02 and price_trend > 0.15:
            m["eps_trend_vs_price_trend"] = -1      # price up, earnings down
        elif eps_trend > 0.02 and price_trend < -0.15:
            m["eps_trend_vs_price_trend"] = -1      # earnings up, price down
        else:
            m["eps_trend_vs_price_trend"] = 0
    else:
        m["eps_trend_vs_price_trend"] = None
    m["eps_trend_slope"] = eps_trend

    # -------------------------------------------------- merge in technicals
    for k, v in rec.technicals.items():
        m.setdefault(k, v)

    # ------------------------------------------------------- sanity guardrail
    flags = sanity_check(m)
    if flags:
        m["sanity_flags"] = flags
        rec.warnings.extend(flags)

    return m
