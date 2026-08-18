"""Derive every metric the screens reference, from a CompanyRecord.

Ratios are computed entirely within the filing's reporting currency, so no FX
conversion is needed for them — a Thai company's EV/EBIT is the same number in
THB or USD. FX is applied only to the absolute-size gates (market cap floor,
turnover floor) and to cross-market ranking displays. That keeps exactly one
conversion point in the system.
"""
from __future__ import annotations

import math
from datetime import date as _date
from typing import Dict, List, Optional, Any

import numpy as np

from . import buffett as _buffett
from . import lynch as _lynch
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
    # Buffett, from the 2010-2021 letters, and the classic Graham ratios the
    # value guide names. Read the attribution carefully — it is not uniform:
    #
    #   * RONTA is Buffett's own frame. On See's: "annually earning about
    #     $4 million pre-tax while utilizing only $8 million of net tangible
    #     assets", and the 2017 acquisition criteria ask for "good returns on
    #     the net tangible assets required to operate the business". He gives
    #     NO threshold. The 50% See's figure is a fact about one company.
    #   * The current, quick and payout ratios are named and defined in the
    #     value guide with NO thresholds attached — it says only "the lower the
    #     PE, the better" and "you should not be happy to see D/A and D/E
    #     rising". Any cutoff is mine.
    #   * The Graham number and the P/E x P/B rule are NOT in either document.
    #     They are Graham's, from The Intelligent Investor, and are computed
    #     here because they are the standard value yardsticks — but nothing in
    #     the uploaded books supports them.
    # ------------------------------------------------------------------------
    if ys:
        y0 = ys[0]
        m["current_ratio"] = _safe_div(y0.current_assets, y0.current_liabilities)
        m["quick_ratio"] = _safe_div(
            (y0.current_assets - y0.inventory)
            if (_n(y0.current_assets) and _n(y0.inventory)) else None,
            y0.current_liabilities)
        # Dividends are stored as paid (negative in most feeds); take the
        # magnitude so the payout ratio is positive whichever sign arrives.
        m["payout_ratio"] = _safe_div(
            abs(y0.dividends_paid) if _n(y0.dividends_paid) else None,
            y0.net_income if (_n(y0.net_income) and y0.net_income > 0) else None)
        # Return on NET TANGIBLE assets. The point of excluding goodwill is
        # that a serial acquirer's ROE flatters it: the capital really employed
        # includes what it paid over book, and RONTA puts that back.
        tbv = y0.tangible_book_value
        m["return_on_net_tangible_assets"] = _safe_div(
            y0.pretax_income if _n(y0.pretax_income) else y0.net_income,
            tbv if (_n(tbv) and tbv > 0) else None)
    else:
        for k in ("current_ratio", "quick_ratio", "payout_ratio",
                  "return_on_net_tangible_assets"):
            m[k] = None

    # "We first have to decide whether we can sensibly estimate an earnings
    # range for five years out... If, however, we lack the ability to estimate
    # future earnings — which is usually the case — we simply move on."
    # Past earnings variability is the only observable proxy for that.
    eps_hist = [y.eps_diluted for y in ys[:5] if _n(y.eps_diluted)]
    if len(eps_hist) >= 4 and all(e > 0 for e in eps_hist):
        mu = float(np.mean(eps_hist))
        m["eps_cv_5y"] = float(np.std(eps_hist, ddof=1) / mu) if mu else None
    else:
        # A loss year makes a coefficient of variation meaningless rather than
        # merely large, so it is refused instead of reported.
        m["eps_cv_5y"] = None

    # "To date, See's has earned $1.9 billion pre-tax, with its growth having
    # required added investment of only $40 million." Incremental return on
    # capital separates a 15% ROE bought with heavy reinvestment from one that
    # needed almost none — nothing else in the Buffett screen sees that.
    m["incremental_roic_5y"] = None
    if len(ys) >= 6:
        e0, e5 = ys[0].ebit, ys[5].ebit
        ic0, ic5 = ys[0].invested_capital, ys[5].invested_capital
        if all(_n(x) for x in (e0, e5, ic0, ic5)) and (ic0 - ic5) > 0:
            m["incremental_roic_5y"] = (e0 - e5) / (ic0 - ic5)

    # ------------------------------------------------------------------------
    # CNAV and the POF score — the value guide's own proprietary method, which
    # I skipped on the first pass and should not have.
    #
    #   "we only count the full value of cash and properties, and half the
    #    value for equipment, receivables, investments, inventories and
    #    intangibles (income generating intangibles). Goodwill and other
    #    non-income generating intangibles are excluded."
    #
    # The idea is Graham's NCAV taken one step further: instead of counting
    # current assets at face and ignoring everything else, haircut each class
    # by how confident you are of realising it.
    #
    # APPROXIMATION, stated because it changes the number: the guide splits
    # PP&E into "properties" at 100% and "equipment" at 50%. No free feed
    # separates them, so net PP&E is taken at 50% throughout. That is the
    # conservative side of his split — a property-heavy company will score
    # lower here than under his method, never higher.
    if ys:
        y0 = ys[0]
        full = sum(v for v in (y0.cash_and_equivalents,
                               y0.short_term_investments) if _n(v))
        half_items = [y0.net_ppe, y0.receivables, y0.inventory]
        # Intangibles EXCLUDING goodwill: goodwill is scored at zero.
        if _n(y0.intangibles):
            gw = y0.goodwill if _n(y0.goodwill) else 0.0
            half_items.append(max(0.0, y0.intangibles - gw))
        half = sum(v for v in half_items if _n(v))
        if _n(y0.total_liabilities) and (full or half):
            cnav_total = full + 0.5 * half - y0.total_liabilities
            m["cnav"] = cnav_total
            m["cnav_per_share"] = _safe_div(cnav_total, y0.shares_diluted)
            # Below 1.0 means the price is below conservative asset value.
            m["price_to_cnav"] = _safe_div(rec.price, m["cnav_per_share"]) \
                if (_n(m.get("cnav_per_share")) and m["cnav_per_share"] > 0) else None
            m["cnav_discount"] = (1.0 - m["price_to_cnav"]) \
                if _n(m.get("price_to_cnav")) else None
        else:
            m["cnav"] = m["cnav_per_share"] = None
            m["price_to_cnav"] = m["cnav_discount"] = None

        # POF: "A 3-point system based on Dr Joseph Piotroski's F-score to find
        # fundamentally strong low price-to-book stocks." P = profitable,
        # O = positive operating cash flow, F = "the lower debt the better".
        # The guide gives NO numeric cutoff for any of the three; all three
        # below are mine, and deliberately the mildest reading of his words.
        pof = 0
        pof_detail = []
        if _n(y0.net_income):
            ok = y0.net_income > 0
            pof += int(ok)
            pof_detail.append(("profitable", ok))
        if _n(y0.cfo):
            ok = y0.cfo > 0
            pof += int(ok)
            pof_detail.append(("positive operating cash flow", ok))
        de = m.get("debt_to_equity")
        if _n(de):
            ok = de <= 1.0
            pof += int(ok)
            pof_detail.append(("debt not above equity", ok))
        m["pof_score"] = pof if len(pof_detail) == 3 else None
        m["pof_detail"] = pof_detail
    else:
        for k in ("cnav", "cnav_per_share", "price_to_cnav", "cnav_discount",
                  "pof_score"):
            m[k] = None
        m["pof_detail"] = []

    # Graham's own yardsticks. NOT from the uploaded books — see the note above.
    bvps = _safe_div(y0.total_equity, y0.shares_diluted) if ys else None
    eps0 = ys[0].eps_diluted if ys else None
    if _n(bvps) and _n(eps0) and bvps > 0 and eps0 > 0:
        m["graham_number"] = float(np.sqrt(22.5 * eps0 * bvps))
        m["price_to_graham_number"] = _safe_div(rec.price, m["graham_number"])
    else:
        m["graham_number"] = m["price_to_graham_number"] = None
    m["pe_times_pb"] = (m["pe_ttm"] * m["price_to_book"]
                        if (_n(m.get("pe_ttm")) and _n(m.get("price_to_book"))
                            and m["pe_ttm"] > 0) else None)

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

    # "Ideally, these assets should have the ability in inflationary times to
    # deliver output that will retain its purchasing-power value while
    # requiring a minimum of new capital investment." Three observables:
    # pricing power holding the margin, revenue per share growing, and capital
    # intensity staying low. Computed HERE rather than with the other Buffett
    # metrics because it depends on capex/revenue, which is set just above.
    _om_slope = m.get("operating_margin_slope_5y")
    _rps = m.get("revenue_per_share_growth_1y")
    _cxr = m.get("capex_to_revenue")
    if all(_n(x) for x in (_om_slope, _rps, _cxr)):
        m["inflation_resilient"] = int(
            _om_slope >= -0.005 and _rps >= 0.02 and _cxr <= 0.08)
    else:
        m["inflation_resilient"] = None

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

    # ========================================================================
    # Peter Lynch — GARP, and the ratios that make a category legible
    # ========================================================================
    # Ownership and listing facts come from the profile feed rather than the
    # statements, and are simply carried through so the threshold engine can
    # see them. Where the feed has nothing, the test is marked not-applicable
    # rather than failed: an absent field is our provider's gap, not a
    # judgement on the company.
    m["insider_ownership"] = getattr(rec, "insider_ownership", None)
    m["institutional_ownership"] = getattr(rec, "institutional_ownership", None)

    # Dividend yield: prefer the feed's own figure, fall back to cash actually
    # paid over the market value. The fallback matters for Asian names where
    # the profile is thin but the cash-flow statement is not.
    dy = getattr(rec, "dividend_yield", None)
    if not _n(dy) and ys and _n(ys[0].dividends_paid) and _n(mcap) and mcap:
        dy = abs(ys[0].dividends_paid) / mcap
    m["dividend_yield"] = dy

    # Listing age — the only 20-year-history signal available in every market.
    m["listing_age_years"] = None
    ftd = getattr(rec, "first_trade_date", None)
    if ftd:
        try:
            y, mo, d = (int(x) for x in str(ftd)[:10].split("-"))
            today = _date.today()
            m["listing_age_years"] = round(
                (today - _date(y, mo, d)).days / 365.25, 1)
        except (ValueError, TypeError):
            pass

    # --- the balance sheet Lynch actually read ------------------------------
    # "75% equity, 25% debt" is his healthy structure; and of that debt, what
    # matters is whether a bank can call it in a bad quarter.
    if ys:
        y0 = ys[0]
        std, ltd = y0.short_term_debt, y0.long_term_debt
        total_debt = y0.total_debt
        if not _n(total_debt) and (_n(std) or _n(ltd)):
            total_debt = (std or 0.0) + (ltd or 0.0)
        m["short_term_debt_share"] = (
            _safe_div(std, total_debt)
            if (_n(std) and _n(total_debt) and total_debt > 0) else
            (0.0 if (_n(total_debt) and total_debt == 0) else None))
        m["long_term_debt_to_equity"] = _safe_div(ltd, y0.total_equity)
        # Net cash per share, and the price of the BUSINESS once the cash on
        # the balance sheet is taken out of the price. Lynch's Ford example is
        # exactly this arithmetic: a $38 stock with $16.60 of net cash is a
        # $21.40 business, and every multiple should be struck on the $21.40.
        cash_tot = (y0.cash_and_equivalents or 0.0) + (y0.short_term_investments or 0.0)
        nc = cash_tot - (y0.total_debt or 0.0) if (
            _n(y0.cash_and_equivalents) or _n(y0.total_debt)) else None
        m["net_cash_per_share"] = _safe_div(nc, sc_now)
        if _n(m["net_cash_per_share"]) and _n(price) and price > 0:
            ex = price - m["net_cash_per_share"]
            m["price_ex_cash"] = ex
            m["net_cash_share_of_price"] = m["net_cash_per_share"] / price
            m["pe_ex_cash"] = (ex / eps) if (_n(eps) and eps > 0 and ex > 0) else None
        else:
            m["price_ex_cash"] = m["pe_ex_cash"] = m["net_cash_share_of_price"] = None
        m["price_to_sales"] = _safe_div(mcap, y0.revenue) if (
            _n(y0.revenue) and y0.revenue > 0) else None
        # A turnaround lives or dies on the debt maturity schedule. The nearest
        # readable proxy: liquid assets against the debt that comes due first.
        # No short-term debt at all is reported as a wide margin rather than a
        # division by zero — that company is not the one Lynch worried about.
        if _n(std):
            m["cash_to_short_term_debt"] = (
                min(_safe_div(cash_tot, std), 99.0) if std > 0 else 99.0)
        else:
            m["cash_to_short_term_debt"] = None
    else:
        for k in ("short_term_debt_share", "long_term_debt_to_equity",
                  "net_cash_per_share", "price_ex_cash", "pe_ex_cash",
                  "net_cash_share_of_price", "price_to_sales",
                  "cash_to_short_term_debt"):
            m[k] = None

    # --- Lynch on FIVE years of statements, not six -------------------------
    # `eps_cagr_5y` needs a year -5 to compare against, i.e. six statements —
    # the fencepost that made Lynch unevaluable for a company with exactly five
    # years on file. This is the same growth idea measured over the longest
    # window the feed actually carries, up to five years, with the span it used
    # reported beside it. A four-year rate labelled as a four-year rate is
    # honest; a four-year rate called a five-year rate is not.
    m["eps_cagr_lynch"] = None
    m["eps_cagr_lynch_years"] = None
    _eps_pts = [(k, y.eps_diluted) for k, y in enumerate(ys[:6])
                if _n(y.eps_diluted)]
    if len(_eps_pts) >= 2:
        newest_i, newest_v = _eps_pts[0]
        oldest_i, oldest_v = _eps_pts[-1]
        span = oldest_i - newest_i
        if span >= 4 and _n(newest_v) and _n(oldest_v) and oldest_v > 0:
            m["eps_cagr_lynch"] = _cagr(newest_v, oldest_v, span)
            m["eps_cagr_lynch_years"] = span
    if m["eps_cagr_lynch"] is None and _n(m.get("eps_cagr_5y")):
        m["eps_cagr_lynch"] = m["eps_cagr_5y"]
        m["eps_cagr_lynch_years"] = 5
    _lynch_growth_pct = (m["eps_cagr_lynch"] * 100
                         if _n(m["eps_cagr_lynch"]) else None)
    m["peg_ratio_lynch"] = _safe_div(m["pe_ttm"], _lynch_growth_pct) if (
        _n(m["pe_ttm"]) and _n(_lynch_growth_pct) and _lynch_growth_pct > 0) else None

    # --- PEGY: the stalwart's ratio -----------------------------------------
    # Lynch: "the ratio of the long-term growth rate PLUS the dividend yield,
    # divided by the P/E." A stalwart's return arrives partly as a cheque, and
    # a plain PEG cannot see the cheque. Both forms are kept: the divisor form
    # (under 1 is good) and Lynch's own multiple form (over 1.5 is good, over
    # 2 is excellent), because his books quote both and they invert each other.
    yld_pct = dy * 100 if _n(dy) else 0.0
    # Struck on the Lynch growth rate so the whole framework runs on one window.
    _g_pct = _lynch_growth_pct if _n(_lynch_growth_pct) else growth_pct
    if _n(m.get("pe_ttm")) and _n(_g_pct) and (_g_pct + yld_pct) > 0:
        m["pegy_ratio"] = m["pe_ttm"] / (_g_pct + yld_pct)
        m["growth_plus_yield_to_pe"] = (_g_pct + yld_pct) / m["pe_ttm"]
    else:
        m["pegy_ratio"] = m["growth_plus_yield_to_pe"] = None

    # --- where the cycle sits ------------------------------------------------
    # Current earnings against their own five-year average. For a cyclical this
    # is the only honest read on the P/E: the multiple tells you nothing until
    # you know whether the E underneath it is at a peak or a trough.
    _eps5 = [y.eps_diluted for y in ys[:5] if _n(y.eps_diluted)]
    m["eps_vs_5y_avg"] = None
    if len(_eps5) >= 3:
        _mu = float(np.mean(_eps5))
        if _mu > 0:
            m["eps_vs_5y_avg"] = _safe_div(_eps5[0], _mu)

    # Recent trouble, as distinct from trouble a decade ago. A turnaround is a
    # company that has JUST been in difficulty; ten-year loss counts cannot
    # tell the two apart.
    m["loss_years_in_3"] = sum(1 for y in ys[:3]
                               if _n(y.net_income) and y.net_income < 0)

    # Cash-flow record, Schloss's "baseline cash generation capability". Kept
    # as a share of the years actually evaluated so a four-year feed and a
    # ten-year one are judged on the same standard.
    _cfo_years = [y.cfo for y in ys[:10] if _n(y.cfo)]
    m["positive_cfo_years_in_10"] = sum(1 for v in _cfo_years if v > 0)
    m["cfo_years_evaluated"] = len(_cfo_years)
    m["cfo_positive_share_10y"] = (
        m["positive_cfo_years_in_10"] / len(_cfo_years) if _cfo_years else None)

    # The same two records over FIVE years. Schloss is run on a five-year
    # statement window so that the Asian markets, where the feed carries four
    # or five years, are judged on the window they actually have rather than
    # on a decade the provider cannot supply. The ten-year versions above stay
    # for the frameworks that genuinely want a decade — Buffett's consistency
    # tenets read the whole run.
    _cfo5 = [y.cfo for y in ys[:5] if _n(y.cfo)]
    m["positive_cfo_years_in_5"] = sum(1 for v in _cfo5 if v > 0)
    m["cfo_years_evaluated_5"] = len(_cfo5)
    m["cfo_positive_share_5y"] = (
        m["positive_cfo_years_in_5"] / len(_cfo5) if _cfo5 else None)
    m["loss_years_in_5"] = sum(1 for y in ys[:5]
                               if _n(y.net_income) and y.net_income < 0)
    m["statement_years_used_schloss"] = min(len(ys), 5)

    # Buybacks: `share_count_change_1y` is computed above from diluted shares.
    # When that line is missing from the feed, fall back to shares outstanding
    # rather than leaving Lynch's buyback test unevaluable.
    if not _n(m.get("share_count_change_1y")) and len(ys) > 1:
        sc_1y = ys[1].shares_diluted or ys[1].shares_outstanding
        m["share_count_change_1y"] = _safe_div(
            (sc_now - sc_1y) if (_n(sc_now) and _n(sc_1y)) else None, sc_1y)

    # ========================================================================
    # Buffett — owner earnings, overheads, and the one-dollar test
    # ========================================================================
    if ys:
        y0 = ys[0]
        m["net_margin_ttm"] = _safe_div(y0.net_income, y0.revenue)
        m["sga_to_gross_profit"] = _safe_div(y0.sga_expense, y0.gross_profit) if (
            _n(y0.gross_profit) and y0.gross_profit > 0) else None
        m["capex_to_net_income"] = _safe_div(
            abs(y0.capex) if _n(y0.capex) else None,
            y0.net_income if (_n(y0.net_income) and y0.net_income > 0) else None)
        # "Long-term debt should be payable in under three or four years of
        # earnings." Stated as the number of years, so the threshold reads the
        # way the rule is spoken.
        m["debt_payoff_years"] = _safe_div(
            y0.long_term_debt if _n(y0.long_term_debt) else y0.total_debt,
            y0.net_income if (_n(y0.net_income) and y0.net_income > 0) else None)
    else:
        for k in ("net_margin_ttm", "sga_to_gross_profit",
                  "capex_to_net_income", "debt_payoff_years"):
            m[k] = None

    # Owner earnings, from the 1986 letter. The maintenance-capex component is
    # an estimate and is labelled as one wherever it surfaces.
    _oe = _buffett.owner_earnings(ys)
    m["owner_earnings_detail"] = _oe
    if _oe.get("available"):
        m["owner_earnings"] = _oe["owner_earnings"]
        m["maintenance_capex"] = _oe["maintenance_capex"]
        m["owner_earnings_per_share"] = _safe_div(_oe["owner_earnings"], sc_now)
        m["owner_earnings_yield"] = _safe_div(_oe["owner_earnings"], mcap)
        m["owner_earnings_to_net_income"] = _safe_div(
            _oe["owner_earnings"], _oe["net_income"]
            if (_n(_oe["net_income"]) and _oe["net_income"] > 0) else None)
    else:
        for k in ("owner_earnings", "maintenance_capex",
                  "owner_earnings_per_share", "owner_earnings_yield",
                  "owner_earnings_to_net_income"):
            m[k] = None
    # Five-year owner-earnings growth, the input the DCF projects from. Taken
    # from owner earnings rather than reported EPS on purpose: the whole point
    # of the measure is that reported earnings are not the spendable cash.
    _oe5 = _buffett.owner_earnings(ys, n=6) if len(ys) >= 6 else {"available": False}
    m["owner_earnings_cagr_5y"] = None
    if _oe.get("available") and _oe5.get("available"):
        a, b = _oe["owner_earnings"], _oe5["owner_earnings"]
        if _n(a) and _n(b) and b > 0 and a > 0:
            m["owner_earnings_cagr_5y"] = _cagr(a, b, 5)

    # The one-dollar premise. Every dollar retained must become at least a
    # dollar of market value over five to ten years. Needs a market cap from
    # five years ago, which is why the price series carries `price_5y_ago`.
    m["retained_earnings_5y"] = None
    m["market_cap_change_5y"] = None
    m["one_dollar_premise"] = None
    if len(ys) >= 6:
        retained = 0.0
        seen = 0
        for y in ys[:5]:
            if _n(y.net_income):
                retained += y.net_income - abs(y.dividends_paid or 0.0)
                seen += 1
        px5 = rec.technicals.get("price_5y_ago")
        sh5 = ys[5].shares_diluted or ys[5].shares_outstanding
        if seen >= 4 and _n(px5) and _n(sh5) and _n(rec.price) and _n(sc_now):
            mcap_then = px5 * sh5
            mcap_now = rec.price * sc_now
            m["retained_earnings_5y"] = retained
            m["market_cap_change_5y"] = mcap_now - mcap_then
            if retained > 0:
                m["one_dollar_premise"] = (mcap_now - mcap_then) / retained

    # --- the category, and the benchmarks that follow from it ---------------
    cat = _lynch.classify(m, getattr(rec, "sector", None),
                          getattr(rec, "industry", None))
    m["lynch_category"] = cat["category"]
    m["lynch_category_label"] = cat["label"]
    m["lynch_category_why"] = cat["why"]
    m["lynch_category_rationale"] = cat["rationale"]
    m["lynch_peak_earnings_warning"] = _lynch.peak_earnings_warning(
        m, cat["category"])

    # ------------------------------------------------------- sanity guardrail
    flags = sanity_check(m)
    if flags:
        m["sanity_flags"] = flags
        rec.warnings.extend(flags)

    return m
