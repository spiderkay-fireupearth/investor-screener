"""Logic verification with synthetic fixtures.

Network egress to Yahoo/SEC is blocked in the build sandbox, so these tests
pin the parts that can be wrong silently: derived accounting identities,
metric maths, threshold evaluation, unknown-handling, Greenblatt's ranking,
the macro gate, and the renderer. Each fixture has a hand-computed expected
outcome so a regression in the maths fails loudly.

Run: python -m tests.test_pipeline
"""
from __future__ import annotations

import sys
import os
import math
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import CompanyRecord, FundamentalYear      # noqa: E402
from src import metrics as mx, screens as sc, technicals as ta, render as rn  # noqa: E402
import yaml                                                 # noqa: E402

PASS = FAIL = 0
FAILURES = []


def check(label, got, want, tol=1e-6):
    global PASS, FAIL
    if isinstance(want, float) and isinstance(got, (int, float)) and got is not None:
        ok = abs(got - want) < tol
    else:
        ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{label}: got {got!r}, expected {want!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got!r}")


def mkyear(fy, **kw):
    base = dict(ticker="TEST", fiscal_year=fy, period_end=f"{fy}-12-31",
                standard="us-gaap", currency="USD", source="test")
    base.update(kw)
    return FundamentalYear(**base)


# ---------------------------------------------------------------------------
def test_schema_identities():
    print("\n[1] Schema derived identities")
    y = mkyear(2025,
               revenue=1000.0, gross_profit=400.0, operating_income=200.0,
               net_income=140.0, total_assets=2000.0, total_liabilities=800.0,
               current_assets=600.0, current_liabilities=300.0,
               cash_and_equivalents=150.0, short_term_investments=50.0,
               total_debt=500.0, total_equity=1200.0, goodwill=100.0,
               intangibles=60.0, net_ppe=900.0, cfo=250.0, capex=80.0,
               depreciation_amortization=70.0)

    check("EBIT", y.ebit, 200.0)
    check("EBITDA = EBIT + D&A", y.ebitda, 270.0)
    check("FCF = CFO - capex", y.free_cash_flow, 170.0)
    check("net debt = debt - cash - STI", y.net_debt, 300.0)
    check("tangible BV = equity - GW - intangibles", y.tangible_book_value, 1040.0)
    # NWC strips cash: (600 - 150) - 300 = 150
    check("net working capital (ex-cash)", y.net_working_capital, 150.0)
    check("invested capital = NWC + net PPE", y.invested_capital, 1050.0)
    # NCAV = current assets - TOTAL liabilities = 600 - 800
    check("NCAV = CA - total liabilities", y.ncav, -200.0)

    # EBIT reconstruction for IFRS filers with no operating income line
    y2 = mkyear(2025, pretax_income=180.0, interest_expense=-25.0)
    check("EBIT reconstructed from pretax + interest", y2.ebit, 205.0)

    # Missing inputs must yield None, never 0 — a zero would pass value screens.
    y3 = mkyear(2025, total_debt=None, cash_and_equivalents=100.0)
    check("net debt is None when debt missing", y3.net_debt, None)
    check("invested capital None without PPE", mkyear(2025).invested_capital, None)


def test_metrics_math():
    print("\n[2] Metric computation")
    years = []
    # 10 years of a Buffett-shaped compounder: ROE ~20%, stable 40% gross margin,
    # low leverage, growing EPS, shrinking share count.
    for i in range(11):
        fy = 2025 - i
        scale = 1.0 / (1.08 ** i)
        years.append(mkyear(
            fy,
            revenue=1000.0 * scale, gross_profit=400.0 * scale,
            operating_income=200.0 * scale, net_income=140.0 * scale,
            eps_diluted=1.40 * scale * (1 + 0.01 * i),
            total_assets=2000.0 * scale, total_liabilities=800.0 * scale,
            current_assets=600.0 * scale, current_liabilities=300.0 * scale,
            cash_and_equivalents=150.0 * scale, short_term_investments=50.0 * scale,
            total_debt=300.0 * scale, total_equity=700.0 * scale,
            goodwill=100.0 * scale, intangibles=60.0 * scale,
            inventory=120.0 * scale, net_ppe=900.0 * scale,
            cfo=250.0 * scale, capex=80.0 * scale,
            depreciation_amortization=70.0 * scale,
            shares_diluted=100.0 * (1 + 0.01 * i)))

    rec = CompanyRecord(ticker="GOOD", market="US", sector="Technology",
                        currency="USD", years=years)
    rec.price = 28.0
    rec.market_cap = 2800.0
    rec.technicals = {"return_12m": 0.12}
    m = mx.compute_metrics(rec)

    check("ROE ttm = 140/700", m["roe_ttm"], 0.2)
    check("ROE years >=15% (of 10)", m["roe_years_above_15"], 10)
    check("gross margin ttm", m["gross_margin_ttm"], 0.4)
    check("gross margin CV ~0 (stable)", round(m["gross_margin_cv"], 6), 0.0)
    check("debt/equity = 300/700", round(m["debt_to_equity"], 6), round(300 / 700, 6))
    check("FCF positive years", m["fcf_years_positive"], 10)
    check("loss years in 10", m["loss_years_in_10"], 0)
    # shares today 100, 5y ago 105 -> change = -5/105
    check("share count change 5y (buyback)", round(m["share_count_change_5y"], 6),
          round((100.0 - 105.0) / 105.0, 6))
    check("ROIC 5y avg = EBIT/IC", round(m["roic_5y_avg"], 6),
          round(200.0 / (150.0 + 900.0), 6))
    # EV = 2800 + 300 - 200 = 2900
    check("enterprise value", m["enterprise_value"], 2900.0)
    check("EV/EBIT = 2900/200", m["ev_to_ebit"], 14.5)
    check("earnings yield = 200/2900", round(m["ebit_to_ev"], 6), round(200 / 2900, 6))
    # accruals = (NI - CFO)/assets = (140-250)/2000
    check("accruals ratio", round(m["accruals_ratio"], 6), round(-110 / 2000, 6))
    check("goodwill/assets", m["goodwill_to_assets"], 0.05)
    # P/TB: 2800 / (700 - 100 - 60) = 2800/540
    check("price/tangible book", round(m["price_to_tangible_book"], 6),
          round(2800 / 540, 6))
    check("net cash/mcap = (200-300)/2800", round(m["net_cash_to_market_cap"], 6),
          round(-100 / 2800, 6))
    check("NCAV/mcap = (600-800)/2800", round(m["ncav_to_market_cap"], 6),
          round(-200 / 2800, 6))

    # A CAGR from a negative base is meaningless and must be None, not a number.
    neg = [mkyear(2025, eps_diluted=1.0), mkyear(2024), mkyear(2023),
           mkyear(2022), mkyear(2021), mkyear(2020, eps_diluted=-0.5)]
    rec2 = CompanyRecord(ticker="NEG", market="US", years=neg)
    m2 = mx.compute_metrics(rec2)
    check("EPS CAGR from negative base is None", m2["eps_cagr_5y"], None)


def test_threshold_engine():
    print("\n[3] Threshold evaluation and unknown handling")
    spec = {"metric": "roe_ttm", "threshold": 0.15, "operator": "gte"}
    check("gte pass", sc.evaluate_test("t", spec, {"roe_ttm": 0.20})["result"], True)
    check("gte fail", sc.evaluate_test("t", spec, {"roe_ttm": 0.10})["result"], False)
    check("missing metric -> unknown",
          sc.evaluate_test("t", spec, {})["result"], None)
    check("NaN metric -> unknown",
          sc.evaluate_test("t", spec, {"roe_ttm": float("nan")})["result"], None)

    lte = {"metric": "pe_ttm", "threshold": 25, "operator": "lte"}
    check("lte pass", sc.evaluate_test("t", lte, {"pe_ttm": 18})["result"], True)
    check("lte boundary is inclusive",
          sc.evaluate_test("t", lte, {"pe_ttm": 25})["result"], True)

    # Klarman's alternative route: fails EV/EBIT but passes as a net-net.
    alt = {"metric": "ev_to_ebit", "threshold": 8.0, "operator": "lte",
           "alt_metric": "ncav_to_market_cap", "alt_threshold": 0.66,
           "alt_operator": "gte"}
    r = sc.evaluate_test("t", alt, {"ev_to_ebit": 20.0, "ncav_to_market_cap": 0.80})
    check("alt route rescues a fail", r["result"], True)
    check("alt route is flagged", r["via_alt"], True)

    # min_tests_passed semantics
    cfg = {"label": "X", "min_tests_passed": 2, "tests": {
        "a": {"metric": "m1", "threshold": 1, "operator": "gte"},
        "b": {"metric": "m2", "threshold": 1, "operator": "gte"},
        "c": {"metric": "m3", "threshold": 1, "operator": "gte"}}}
    r = sc.run_framework("x", cfg, {"m1": 5, "m2": 5, "m3": 0}, "fail")
    check("2 of 3 passed meets requirement", r["passed"], True)
    r = sc.run_framework("x", cfg, {"m1": 5, "m2": 0, "m3": 0}, "fail")
    check("1 of 3 fails requirement", r["passed"], False)

    # Strict mode: missing data must NOT pass a value screen.
    r = sc.run_framework("x", cfg, {"m1": 5}, "fail")
    check("unknown=fail: 1 pass 2 unknown -> fail", r["passed"], False)
    check("unknown counted", r["n_unknown"], 2)
    # Lenient mode scales the requirement to the tests that could be evaluated.
    r = sc.run_framework("x", cfg, {"m1": 5}, "skip")
    check("unknown=skip: 1/1 evaluated -> pass", r["passed"], True)


def test_greenblatt_ranking():
    print("\n[4] Greenblatt ranking and sector exclusion")
    recs, mets = [], {}
    # Six industrials with deliberately crossed EY and ROC ranks.
    spec = [("A", 0.20, 0.50), ("B", 0.18, 0.45), ("C", 0.15, 0.60),
            ("D", 0.10, 0.20), ("E", 0.05, 0.10), ("F", 0.02, 0.05)]
    for t, ey, roc in spec:
        recs.append(CompanyRecord(ticker=t, market="US", sector="Industrials"))
        mets[t] = {"ebit_to_ev": ey, "ebit_to_invested_capital": roc}
    # A bank must be excluded — EV/EBIT is meaningless when the balance sheet
    # IS the business.
    recs.append(CompanyRecord(ticker="BANK", market="US", sector="Financial Services"))
    mets["BANK"] = {"ebit_to_ev": 0.90, "ebit_to_invested_capital": 0.90}
    # Negative earnings yield must be ineligible, not top-ranked.
    recs.append(CompanyRecord(ticker="LOSS", market="US", sector="Industrials"))
    mets["LOSS"] = {"ebit_to_ev": -0.10, "ebit_to_invested_capital": 0.30}

    cfg = {"rank_scope": "market", "top_n": 3, "label": "Magic Formula"}
    out = sc.run_greenblatt(recs, mets, cfg,
                            ["Financial Services", "Real Estate", "Utilities"])

    # A: EY rank 1 + ROC rank 2 = 3. C: EY 3 + ROC 1 = 4. B: EY 2 + ROC 3 = 5.
    check("A combined score", out["A"]["combined_rank_score"], 3)
    check("C combined score", out["C"]["combined_rank_score"], 4)
    check("B combined score", out["B"]["combined_rank_score"], 5)
    check("A ranks first", out["A"]["combined_rank"], 1)
    check("C ranks second (ROC beats EY here)", out["C"]["combined_rank"], 2)
    check("top_n=3 -> A passes", out["A"]["passed"], True)
    check("top_n=3 -> D fails", out["D"]["passed"], False)
    check("bank excluded", out["BANK"]["passed"], False)
    check("bank reason recorded",
          "sector" in out["BANK"].get("ineligible_reason", ""), True)
    check("negative EY ineligible", out["LOSS"]["passed"], False)
    check("eligible universe size", out["A"]["universe_size"], 6)

    # Market scoping: a cheap market must not crowd out the rest.
    recs2 = [CompanyRecord(ticker=f"HK{i}", market="HK", sector="Industrials")
             for i in range(5)]
    # Vary both factors so HK4 is unambiguously best on each — an all-ties
    # fixture would make the assertion depend on sort stability, not on logic.
    mets2 = {f"HK{i}": {"ebit_to_ev": 0.30 + i * 0.01,
                        "ebit_to_invested_capital": 0.30 + i * 0.05}
             for i in range(5)}
    out2 = sc.run_greenblatt(recs + recs2, {**mets, **mets2}, cfg,
                             ["Financial Services"])
    check("US names still rank within US scope", out2["A"]["scope"], "US")
    check("HK ranked separately", out2["HK4"]["scope"], "HK")
    check("HK top name passes despite US universe", out2["HK4"]["passed"], True)


def test_macro_gate():
    print("\n[5] Soros macro gate")
    cfg = {"enabled": True, "hy_spread_max": 6.0, "yield_curve_min": -0.50}
    ok, why = sc.macro_gate_open(
        {"_enabled": True, "hy_credit_spread": 3.2, "yield_curve_10y2y": 0.4}, cfg)
    check("normal conditions -> open", ok, True)
    ok, why = sc.macro_gate_open(
        {"_enabled": True, "hy_credit_spread": 7.5, "yield_curve_10y2y": 0.4}, cfg)
    check("wide HY spread -> closed", ok, False)
    check("reason names the spread", "high-yield spread" in why, True)
    ok, _ = sc.macro_gate_open(
        {"_enabled": True, "hy_credit_spread": 3.0, "yield_curve_10y2y": -0.9}, cfg)
    check("deep inversion -> closed", ok, False)
    ok, why = sc.macro_gate_open({"_enabled": False}, cfg)
    check("no FRED key -> gate skipped, not failed", ok, True)
    ok, _ = sc.macro_gate_open({"_enabled": True, "hy_credit_spread": 99},
                               {"enabled": False})
    check("gate disabled in config -> open", ok, True)


def test_technicals():
    print("\n[6] Technical indicators")
    # A clean uptrend: price above both MAs, 50 above 200.
    n = 600
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                       "Close": close, "Volume": np.full(n, 1_000_000.0)}, index=idx)
    t = ta.compute(df, fx_to_usd=1.0)
    check("price above SMA200 in uptrend", t["price_above_sma200"], 1)
    check("SMA50 above SMA200 in uptrend", t["sma50_above_sma200"], 1)
    check("RSI high in pure uptrend", t["rsi_14"] > 70, True)
    check("MACD histogram present", t["macd_histogram"] is not None, True)
    check("12m return computed", t["return_12m"] is not None, True)
    check("5y low = series min", round(t["low_5y"], 4), 100.0)

    # Downtrend must invert the trend flags.
    close_d = pd.Series(np.linspace(200, 100, n), index=idx)
    df_d = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        df_d[c] = close_d * (1.01 if c == "High" else 0.99 if c == "Low" else 1.0)
    t2 = ta.compute(df_d, fx_to_usd=1.0)
    check("price below SMA200 in downtrend", t2["price_above_sma200"], 0)
    check("death cross regime", t2["sma50_above_sma200"], 0)
    check("RSI low in downtrend", t2["rsi_14"] < 30, True)

    # Relative strength must be measured against the stock's OWN index and
    # must align on shared dates only (different holiday calendars).
    idx_close = pd.Series(np.linspace(100, 150, n), index=idx)
    idx_df = pd.DataFrame({"Close": idx_close}, index=idx)
    t3 = ta.compute(df, index_df=idx_df, fx_to_usd=1.0)
    check("RS positive when stock outruns index",
          t3["rs_vs_market_index_6m"] > 0, True)

    # A market with a different holiday calendar (drop 30 random sessions)
    rng = np.random.default_rng(7)
    keep = sorted(rng.choice(n, n - 30, replace=False))
    idx_holiday = idx_df.iloc[keep]
    t4 = ta.compute(df, index_df=idx_holiday, fx_to_usd=1.0)
    check("RS still computes across mismatched calendars",
          t4["rs_vs_market_index_6m"] is not None, True)

    # Too little history must be reported, not silently produce garbage.
    t5 = ta.compute(df.iloc[:10])
    check("short series flagged", t5["insufficient_history"], True)


def test_currency_reconciliation():
    print("\n[7] Reporting currency != trading currency")
    # An HKEX-listed mainland issuer: reports in CNY, trades in HKD.
    # Book value 1,000 CNY, market cap 1,400 HKD. At HKD 0.1282/USD and
    # CNY 0.1400/USD, 1,400 HKD = 179.48 USD = 1,281.99 CNY, so the correct
    # P/B is 1.28 — NOT the 1.40 you get by dividing HKD by CNY directly.
    fx = {"USD": 1.0, "HKD": 0.1282, "CNY": 0.1400}
    years = [mkyear(2025 - i, total_equity=1000.0, total_assets=2000.0,
                    total_liabilities=1000.0, net_income=100.0,
                    eps_diluted=1.0, operating_income=150.0,
                    current_assets=500.0, current_liabilities=300.0,
                    cash_and_equivalents=100.0, total_debt=200.0,
                    net_ppe=800.0, cfo=180.0, capex=50.0) for i in range(6)]
    rec = CompanyRecord(ticker="9999.HK", market="HK", sector="Industrials",
                        currency="HKD", financial_currency="CNY", years=years)
    rec.price, rec.market_cap = 14.0, 1400.0
    rec.technicals = {"return_12m": 0.05}

    mcap, price, note = mx.reconcile_currency(rec, fx)
    check("market cap restated into CNY", round(mcap, 2),
          round(1400.0 * (0.1282 / 0.1400), 2))
    check("price restated too", round(price, 4),
          round(14.0 * (0.1282 / 0.1400), 4))
    check("restatement is noted", "restated HKD->CNY" in (note or ""), True)

    m = mx.compute_metrics(rec, fx_rates=fx)
    check("P/B uses restated cap (1.28, not 1.40)",
          round(m["price_to_book"], 3), round(1400.0 * (0.1282 / 0.1400) / 1000.0, 3))
    check("display market cap stays in trading currency", m["market_cap"], 1400.0)
    check("statement currency recorded", m["statement_currency"], "CNY")

    # Same currency on both sides must be a no-op, not a rounding drift.
    rec2 = CompanyRecord(ticker="D05.SI", market="SG", currency="SGD",
                         financial_currency="SGD", years=years)
    rec2.price, rec2.market_cap = 14.0, 1400.0
    rec2.technicals = {}
    mcap2, price2, note2 = mx.reconcile_currency(rec2, fx)
    check("no-op when currencies match", (mcap2, price2, note2), (1400.0, 14.0, None))

    # Missing FX for the reporting currency must warn, not silently mis-rank.
    rec3 = CompanyRecord(ticker="X.HK", market="HK", currency="HKD",
                         financial_currency="KRW", years=years)
    rec3.price, rec3.market_cap = 14.0, 1400.0
    rec3.technicals = {}
    _, _, note3 = mx.reconcile_currency(rec3, fx)
    check("missing FX flags unreliability", "unreliable" in (note3 or ""), True)

    # The sanity guardrail must catch a units mismatch that slipped through.
    flags = mx.sanity_check({"price_to_tangible_book": 7_304_347.0, "pe_ttm": 12.0})
    check("implausible P/TB is flagged", len(flags), 1)
    check("flag names the cause", "currency or units" in flags[0], True)
    check("sane ratios raise nothing",
          len(mx.sanity_check({"pe_ttm": 18.0, "price_to_book": 2.1})), 0)


def test_history_depth_parity():
    print("\n[8] Data depth must not decide the verdict")
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config")
    with open(os.path.join(cfg_dir, "thresholds.yml")) as f:
        thresholds = yaml.safe_load(f)

    def compounder(ticker, n_years, source):
        """Identical business, differing only in how many years the feed returns."""
        ys = []
        for i in range(n_years):
            s = 1.0 / (1.08 ** i)
            ys.append(mkyear(
                2025 - i, revenue=1000.0 * s, gross_profit=400.0 * s,
                operating_income=200.0 * s, net_income=140.0 * s,
                eps_diluted=1.40 * s, total_assets=2000.0 * s,
                total_liabilities=800.0 * s, current_assets=600.0 * s,
                current_liabilities=300.0 * s, cash_and_equivalents=150.0 * s,
                short_term_investments=50.0 * s, total_debt=300.0 * s,
                total_equity=700.0 * s, goodwill=100.0 * s, intangibles=60.0 * s,
                inventory=120.0 * s, net_ppe=900.0 * s, cfo=250.0 * s,
                capex=80.0 * s, depreciation_amortization=70.0 * s,
                shares_diluted=100.0 * (1 + 0.01 * i)))
            ys[-1].source = source
        r = CompanyRecord(ticker=ticker, market="US" if source == "edgar" else "HK",
                          sector="Technology", name=ticker, currency="USD", years=ys)
        r.price, r.market_cap = 28.0, 2800.0
        r.technicals = {"return_12m": 0.10, "price_above_sma200": 1,
                        "sma50_above_sma200": 1, "rsi_14": 55.0,
                        "macd_histogram": 0.5, "vol20_over_vol50": 1.1,
                        "atr_pct_percentile": 0.5, "rs_vs_market_index_6m": 0.05,
                        "pct_above_5y_low": 0.20, "median_turnover_usd": 5e7}
        return r

    deep = compounder("USDEEP", 11, "edgar")     # EDGAR: 10+ years
    shallow = compounder("HKSHAL", 4, "yahoo")   # Yahoo: ~4 years
    recs = [deep, shallow]
    mets = {}
    for r in recs:
        m = mx.compute_metrics(r)
        m["market_cap_usd"] = 2_000_000_000.0
        mets[r.ticker] = m

    check("deep feed reports 11 years", mets["USDEEP"]["history_years"], 11)
    check("shallow feed reports 4 years", mets["HKSHAL"]["history_years"], 4)

    out = sc.screen_universe(recs, mets, thresholds,
                             {"_enabled": True, "hy_credit_spread": 3.0,
                              "yield_curve_10y2y": 0.5})["results"]

    d = out["USDEEP"]["frameworks"]
    s = out["HKSHAL"]["frameworks"]

    # The whole point: same business, same verdict, regardless of feed depth.
    check("Buffett passes on the deep feed", d["buffett"]["passed"], True)
    check("Buffett REACHABLE on the shallow feed", s["buffett"]["passed"], True)
    check("Lynch reachable on the shallow feed",
          s["lynch"]["passed"] is not None, True)

    # The shallow name must be marked as history-limited, not silently equal.
    check("shallow Buffett flagged limited_history",
          s["buffett"]["limited_history"], True)
    check("deep Buffett not flagged", d["buffett"].get("limited_history"), False)
    check("shallow share_count test is insufficient, not failed",
          next(t["insufficient"] for t in s["buffett"]["tests"]
               if t["name"] == "share_count_discipline"), True)
    check("insufficient tests leave the denominator",
          s["buffett"]["effective_total"] < s["buffett"]["n_total"], True)
    check("bar scales down with the denominator",
          s["buffett"]["required"] <= d["buffett"]["required"], True)

    # An 8-of-10 rule must become 4-of-4, not stay unreachable at 8.
    roe = next(t for t in s["buffett"]["tests"] if t["name"] == "roe_consistency")
    check("8-of-10 rescaled for a 4-year window", roe["threshold"], 3)
    check("rescaled ROE test actually passes", roe["result"], True)

    # Missing DATA must still count against a company — only missing YEARS is forgiven.
    broken = compounder("BROKEN", 11, "edgar")
    for y in broken.years:
        y.total_debt = None
        y.gross_profit = None
    mb = mx.compute_metrics(broken)
    mb["market_cap_usd"] = 2_000_000_000.0
    ob = sc.screen_universe([broken], {"BROKEN": mb}, thresholds,
                            {"_enabled": True, "hy_credit_spread": 3.0,
                             "yield_curve_10y2y": 0.5})["results"]
    check("missing data still fails, unlike missing years",
          ob["BROKEN"]["frameworks"]["buffett"]["passed"], False)


def test_end_to_end_with_config():
    print("\n[9] End-to-end against the real config")
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config")
    with open(os.path.join(cfg_dir, "thresholds.yml")) as f:
        thresholds = yaml.safe_load(f)

    def make(ticker, sector, quality="high"):
        years = []
        for i in range(11):
            fy = 2025 - i
            s = 1.0 / (1.08 ** i)
            if quality == "high":
                ni, eq, debt, gp = 140.0 * s, 700.0 * s, 200.0 * s, 400.0 * s
            else:
                ni, eq, debt, gp = 20.0 * s, 900.0 * s, 900.0 * s, 150.0 * s
            years.append(mkyear(
                fy, revenue=1000.0 * s, gross_profit=gp, operating_income=200.0 * s,
                net_income=ni, eps_diluted=(ni / 100.0) * (1 + 0.01 * i),
                total_assets=2000.0 * s, total_liabilities=800.0 * s,
                current_assets=600.0 * s, current_liabilities=300.0 * s,
                cash_and_equivalents=150.0 * s, short_term_investments=50.0 * s,
                total_debt=debt, total_equity=eq, goodwill=100.0 * s,
                intangibles=60.0 * s, inventory=120.0 * s, net_ppe=900.0 * s,
                cfo=250.0 * s, capex=80.0 * s, depreciation_amortization=70.0 * s,
                shares_diluted=100.0 * (1 + 0.01 * i)))
        r = CompanyRecord(ticker=ticker, market="US", sector=sector,
                          name=ticker, currency="USD", years=years)
        r.price, r.market_cap = 20.0, 2000.0
        r.technicals = {"return_12m": 0.10, "price_above_sma200": 1,
                        "sma50_above_sma200": 1, "rsi_14": 55.0,
                        "macd_histogram": 0.5, "vol20_over_vol50": 1.1,
                        "atr_pct_percentile": 0.5, "rs_vs_market_index_6m": 0.05,
                        "pct_above_5y_low": 0.20, "median_turnover_usd": 5e7}
        return r

    good = make("GOOD", "Technology", "high")
    weak = make("WEAK", "Technology", "low")
    bank = make("BANK", "Financial Services", "high")
    recs = [good, weak, bank]

    mets = {}
    for r in recs:
        m = mx.compute_metrics(r)
        m["market_cap_usd"] = 2_000_000_000.0
        mets[r.ticker] = m

    out = sc.screen_universe(recs, mets, thresholds,
                             {"_enabled": True, "hy_credit_spread": 3.0,
                              "yield_curve_10y2y": 0.5})
    res = out["results"]

    check("macro gate open", out["macro_gate_open"], True)
    check("GOOD passes Buffett", res["GOOD"]["frameworks"]["buffett"]["passed"], True)
    check("WEAK fails Buffett", res["WEAK"]["frameworks"]["buffett"]["passed"], False)
    check("bank excluded from Klarman",
          "sector" in res["BANK"]["frameworks"]["klarman"].get("ineligible_reason", ""),
          True)
    check("bank excluded from Greenblatt",
          res["BANK"]["frameworks"]["greenblatt"]["passed"], False)
    check("GOOD technical overlay passes", res["GOOD"]["technical_passed"], True)
    check("GOOD is surfaced", res["GOOD"]["surfaced"], True)
    check("every framework evaluated",
          sorted(res["GOOD"]["frameworks"].keys()),
          ["buffett", "graham", "greenblatt", "klarman", "lynch", "marks",
           "munger", "rogers", "schloss", "soros", "templeton"])

    # Size/liquidity gate must suppress a name entirely.
    mets["GOOD"]["market_cap_usd"] = 1_000_000.0
    out2 = sc.screen_universe(recs, mets, thresholds,
                              {"_enabled": True, "hy_credit_spread": 3.0,
                               "yield_curve_10y2y": 0.5})
    check("sub-scale name is gated out", out2["results"]["GOOD"]["surfaced"], False)
    check("gate reason recorded",
          len(out2["results"]["GOOD"]["gates_failed"]) > 0, True)

    # Closed macro gate must suppress Soros for everyone.
    out3 = sc.screen_universe(recs, mets, thresholds,
                              {"_enabled": True, "hy_credit_spread": 9.0,
                               "yield_curve_10y2y": 0.5})
    check("closed gate blocks Soros",
          out3["results"]["GOOD"]["frameworks"]["soros"]["passed"], False)

    # Renderer must produce a real page with the data embedded.
    outdir = "/tmp/screener_test_out"
    with open(os.path.join(cfg_dir, "universe.yml")) as f:
        universe = yaml.safe_load(f)
    path = rn.render(res, mets, out, thresholds, universe,
                     out_dir=outdir, region="us", run_id="test-run")
    html = open(path, encoding="utf-8").read()
    check("HTML written", os.path.exists(path), True)
    check("HTML non-trivial size", len(html) > 12000, True)
    check("no unreplaced placeholders", "__DATA__" not in html and "__GATE__" not in html, True)
    check("tickers embedded", "GOOD" in html and "BANK" in html, True)
    check("results.json written",
          os.path.exists(os.path.join(outdir, "results.json")), True)


def test_cycle_and_new_frameworks():
    print("\n[10] Market cycle gauge, Templeton and Marks")
    from src import cycle as cy
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config")
    with open(os.path.join(cfg_dir, "thresholds.yml")) as f:
        thresholds = yaml.safe_load(f)
    cyc_cfg = thresholds["market_cycle"]

    idx = pd.bdate_range("2023-01-02", periods=400)
    up = pd.DataFrame({"Close": pd.Series(np.linspace(100, 200, 400), index=idx)})
    down = pd.DataFrame({"Close": pd.Series(np.linspace(200, 100, 400), index=idx)})

    # Euphoria: hot index, broad participation, cheap insurance.
    hot = cy.assess(up, {"pct_above_200dma": 0.85}, 12.0, cyc_cfg)
    check("euphoria -> defensive", hot["mode"], "defensive")
    check("defensive raises the Marks bar", hot["threshold_shift"], 1)

    # Panic: washed-out index, narrow participation, expensive insurance.
    cold = cy.assess(down, {"pct_above_200dma": 0.20}, 38.0, cyc_cfg)
    check("panic -> opportunistic", cold["mode"], "opportunistic")
    check("opportunistic lowers the bar", cold["threshold_shift"], -1)

    # Mixed signals must not produce a confident call in either direction.
    mixed = cy.assess(up, {"pct_above_200dma": 0.50}, 20.0, cyc_cfg)
    check("mixed evidence -> core", mixed["mode"], "core")
    check("core leaves the bar alone", mixed["threshold_shift"], 0)
    check("no signals at all -> core, not a guess",
          cy.assess(None, None, None, cyc_cfg)["mode"], "core")

    # The cycle shift must actually move the requirement.
    marks_cfg = thresholds["marks"]
    # Exactly 5 of the 8 pass: clears the core bar, fails the defensive one.
    # This is the borderline name whose verdict flips with the gauge, which is
    # the entire point of cycle_adjust.
    metrics_pass5 = {"ev_to_ebit": 10.0,            # pass
                     "ev_ebit_vs_market": 0.70,     # pass
                     "max_drawdown_5y": 0.30,       # pass
                     "downside_capture": 0.85,      # pass
                     "loss_years_in_10": 0,         # pass
                     "net_debt_to_ebitda": 9.0,     # fail
                     "accruals_ratio": 0.40,        # fail
                     "pct_below_52w_high": 0.00,    # fail
                     "history_years": 10}
    base = sc.run_framework("marks", marks_cfg, metrics_pass5, "fail", 0)
    defn = sc.run_framework("marks", marks_cfg, metrics_pass5, "fail", +1)
    oppo = sc.run_framework("marks", marks_cfg, metrics_pass5, "fail", -1)
    check("5 tests pass on this fixture", base["n_passed"], 5)
    check("neutral bar is 5 of 8", base["required"], 5)
    check("defensive bar rises to 6", defn["required"], 6)
    check("opportunistic bar falls to 4", oppo["required"], 4)
    check("same evidence passes in core", base["passed"], True)
    check("same evidence FAILS when defensive", defn["passed"], False)
    check("and passes when opportunistic", oppo["passed"], True)

    # Templeton: a cheap, beaten-down, solvent business should clear it.
    bargain = {"pe_ttm": 9.0, "price_to_book": 0.9, "price_to_cash_flow": 6.0,
               "pct_below_52w_high": 0.42, "loss_years_in_10": 1,
               "debt_to_equity": 0.3, "history_years": 10}
    t = sc.run_framework("templeton", thresholds["templeton"], bargain, "fail")
    check("classic bargain clears Templeton", t["passed"], True)
    # A quality name at a full price is a wish-list item, not a buy.
    dear = {"pe_ttm": 34.0, "price_to_book": 12.0, "price_to_cash_flow": 28.0,
            "pct_below_52w_high": 0.01, "loss_years_in_10": 0,
            "debt_to_equity": 0.2, "history_years": 10}
    t2 = sc.run_framework("templeton", thresholds["templeton"], dear, "fail")
    check("quality at a full price fails Templeton", t2["passed"], False)
    check("but it is a near miss, not a rejection", t2["n_passed"], 2)


def test_macro_carry():
    print("\n[11] Carry-trade analytics")
    from src import macro as mc
    idx = pd.bdate_range("2023-01-02", periods=600)
    rng2 = np.random.default_rng(11)

    # Calm yen: low vol, wide differential -> carry well paid.
    calm = pd.DataFrame({"Close": pd.Series(
        150 + np.cumsum(rng2.normal(0, 0.15, 600)), index=idx)})
    r = mc.carry_analysis({"JPY": calm},
                          {"us_policy": 4.5, "jp_policy": 0.5,
                           "us_10y": 4.2, "jp_10y": 1.1})
    check("policy differential", round(r["policy_differential"], 2), 4.0)
    check("long differential", round(r["long_differential"], 2), 3.1)
    check("carry/vol computed", r["carry_to_vol"] is not None, True)
    check("well-paid carry reads as attractive",
          "still attractive" in r["carry_reading"], True)

    # Stressed yen: the shock must be RECENT for 30-day vol to exceed 90-day.
    # A uniformly volatile tail lifts both windows equally and the ratio stays
    # flat — which is exactly the distinction the indicator is built to make.
    stress = list(150 + np.cumsum(rng2.normal(0, 0.10, 570)))
    stress += list(stress[-1] + np.cumsum(rng2.normal(-0.55, 1.6, 30)))
    stressed = pd.DataFrame({"Close": pd.Series(stress, index=idx)})
    r2 = mc.carry_analysis({"JPY": stressed},
                           {"us_policy": 3.0, "jp_policy": 1.5})
    check("vol ratio rises under stress", r2["jpy_vol_ratio"] > 1.15, True)
    check("yen strengthened over 3m", r2["usdjpy_3m"] < 0, True)
    check("unwind pressure detected", r2["unwind_pressure"], True)
    check("thin carry flagged", r2["carry_to_vol"] < r["carry_to_vol"], True)

    # HKD peg: position within the band, not the raw level.
    for spot, expect in ((7.849, "weak end"), (7.751, "strong end"), (7.80, "Mid-band")):
        hk = pd.DataFrame({"Close": pd.Series([spot] * 30,
                          index=pd.bdate_range("2026-01-01", periods=30))})
        p = mc.peg_pressure({"HKD": hk})
        check(f"HKD at {spot} -> {expect}", expect.lower() in p["reading"].lower(), True)
    check("band position 0..1", round(mc.peg_pressure(
        {"HKD": pd.DataFrame({"Close": pd.Series([7.80] * 30,
         index=pd.bdate_range("2026-01-01", periods=30))})})["band_position"], 2), 0.5)


def test_debt_cycle():
    """Dalio staging: severity ordering, missing-data handling, the levers."""
    from src import debtcycle as dc

    def q(latest, step, n=20):
        return [(f"2026-{max(1, (i % 4) * 3 + 1):02d}-01", latest - step * i)
                for i in range(n)]

    def dly(v, n=300):
        return [(f"2026-08-{(i % 28) + 1:02d}", v) for i in range(n)]

    # --- a late-bubble sovereign next to a healthy household sector --------
    h = {
        "fed_debt_gdp": q(122.6, 0.5, 20), "hh_debt_gdp": q(68.6, -0.4, 20),
        "gdp": q(30000.0, 150.0, 20), "fed_interest": q(1247.0, 20.0, 20),
        "fed_deficit": [(f"2026-{i + 1:02d}-01", -112000.0) for i in range(24)],
        "curve_10y2y": dly(0.51), "curve_10y3m": dly(0.82), "policy": dly(3.62),
        "cpi": [(f"2026-{i + 1:02d}-01", 330.0 - 0.96 * i) for i in range(24)],
        "hy_oas": dly(271.0), "cc_delinq": q(2.92, -0.035, 12),
        "fed_assets": [(f"w{i}", 6759955.0 - 200.0 * i) for i in range(60)],
        "m2": [(f"2026-{i + 1:02d}-01", 22000.0 - 60.0 * i) for i in range(24)],
        "top1_wealth": q(31.6, 0.06, 16),
    }
    c = dc.classify(h, cape=42.18, vix=14.6)
    check("public sector reads as a bubble", c["public_stage"], 2)
    check("private sector reads as calm", c["private_stage"] in (1, 7), True)
    # The regression that matters: stage numbers are a sequence, not a
    # severity scale, so max() on the number would headline the SAFE sector.
    check("headline follows the riskier sector, not the higher number",
          c["stage"], 2)
    check("sector disagreement is reported", bool(c["sector_note"]), True)

    # --- the sustainability test -----------------------------------------
    s = dc.sustainability(h)
    check("interest annualised read", round(s["interest_saar_bn"]), 1247)
    check("deficit annualised from 12 monthly obs",
          round(s["deficit_ttm_bn"]), 1344)
    check("interest as share of deficit", round(s["interest_to_deficit"], 2), 0.93)

    # --- velocity: rising, but nowhere near Dalio's +20pp -----------------
    v = dc.debt_velocity(h)
    check("federal debt/GDP 3y change computed", round(v["fed_3y_pp"], 1), 6.0)
    check("velocity test not tripped at +6pp", v["fails_velocity_test"], False)

    # --- checklist: red on valuation/sentiment/credit, cool on velocity ---
    chk = dc.bubble_checklist(h, 42.18, 14.6)
    check("bubble checklist is RED here", chk["level"], "RED")
    by = {t["key"]: t["score"] for t in chk["tests"]}
    check("CAPE 42 scores hot", by["valuation"], 2)
    check("VIX 14.6 scores hot", by["sentiment"], 2)
    check("271bp high-yield scores hot", by["leverage"], 2)
    check("debt velocity scores cool", by["velocity"], 0)
    check("interest at 93% of deficit scores hot", by["sustainability"], 2)

    # --- missing data must shrink the denominator, never score as a pass --
    thin = {k: v2 for k, v2 in h.items() if k not in ("hy_oas", "cpi", "policy")}
    c2 = dc.bubble_checklist(thin, None, None)
    check("unavailable tests are not scored", c2["evaluable"] < c2["of"], True)
    check("max scales with evaluable tests", c2["max"], 2 * c2["evaluable"])
    unscored = [t for t in c2["tests"] if t["score"] is None]
    check("unscored tests say why", all(t["detail"] for t in unscored), True)

    # --- the four levers --------------------------------------------------
    t = dc.tug_of_war(h)
    lev = {l["lever"]: l for l in t["levers"]}
    check("defaults lever idle when spreads tight and delinquency falling",
          lev["Defaults / restructuring"]["pull"], 0)
    check("printing lever neutral on a flat balance sheet",
          lev["Money printing"]["pull"], 0)
    check("wealth gap widening reads inflationary",
          lev["Wealth redistribution"]["pull"], 1)

    # --- a genuine depression fixture must reach stage 4 -------------------
    dep = dict(h)
    dep["policy"] = dly(0.08)
    dep["hy_oas"] = dly(950.0)
    # q() walks BACKWARDS from the latest, so a positive step means older
    # observations were lower — i.e. delinquency rising into the present.
    dep["cc_delinq"] = q(6.5, 0.4, 12)
    dep["fed_assets"] = [(f"w{i}", 9_000_000.0 - 40_000.0 * i) for i in range(60)]
    c3 = dc.classify(dep, cape=12.0, vix=44.0)
    check("zero rates + 950bp spreads + rising defaults reads as depression",
          c3["public_stage"], 4)

    # --- asset call flips with real rates and the printing lever ----------
    a = dc.asset_implications(dep, 4, dc.tug_of_war(dep))
    check("negative real rates plus printing favours real assets",
          a["favours"], "real assets")

    # --- index CAPE is an aggregate, not a mean of ratios ------------------
    # These MUST be real CompanyRecord objects. An earlier version of this test
    # used a hand-rolled stand-in class that stored its statements on a made-up
    # `.fundamentals` attribute; the real dataclass calls it `.years`. The test
    # passed, and the shipped code found zero years on every name and reported
    # "no earnings history" — a wrong answer delivered confidently. A fixture
    # that invents the interface cannot catch an interface mismatch.
    def mkrec(t, cap, earn, market="US", n_years=10):
        return CompanyRecord(
            ticker=t, market=market, currency="USD",
            market_cap=cap,
            years=[mkyear(2025 - i, net_income=earn) for i in range(n_years)])

    recs = [mkrec(f"T{i}", 1000.0, 50.0) for i in range(40)]
    # One name whose earnings are a rounding error: a mean-of-ratios would let
    # its 1,000,000x multiple swamp 40 healthy names. An aggregate must not care.
    recs.append(mkrec("TINY", 1000.0, 0.001))
    mt = {r.ticker: {"market_cap_usd": r.market_cap, "fx_to_usd": 1.0}
          for r in recs}
    res = dc.universe_cape(recs, mt)
    check("aggregate CAPE reads CompanyRecord.years", "error" in res, False)
    check("aggregate CAPE ignores a near-zero-earnings outlier",
          round(res["cape"]), 20)
    check("aggregate CAPE counts every usable name", res["names_used"], 41)

    # --- and it must say WHY it gave up, not just that it did -------------
    thin = recs[:5]
    e1 = dc.universe_cape(thin, {r.ticker: mt[r.ticker] for r in thin})
    check("too few names refuses to print a number", "error" in e1, True)
    check("failure counts names in the market", e1["breakdown"]["in_market"], 5)

    short = [mkrec(f"S{i}", 1000.0, 50.0, n_years=4) for i in range(40)]
    e2 = dc.universe_cape(
        short, {r.ticker: {"market_cap_usd": 1000.0, "fx_to_usd": 1.0}
                for r in short})
    check("short history is reported as short history, not as no names",
          e2["breakdown"]["too_few_years"], 40)
    e3 = dc.universe_cape(recs, {})       # no metrics at all
    check("missing market caps are named as the cause",
          e3["breakdown"]["no_market_cap"], 41)
    check("Asian names are excluded, not counted as failures",
          dc.universe_cape([mkrec("9988.HK", 1000.0, 50.0, market="HK")],
                           mt)["breakdown"]["in_market"], 0)

    # --- the renderer must survive every one of these shapes --------------
    panel = rn._dalio_panel({"enabled": True, "stage": 2,
                             "stage_name": "The bubble", "classification": c,
                             "checklist": chk, "velocity": v,
                             "sustainability": s, "tipping_point":
                             dc.tipping_point(h), "early_warnings":
                             dc.early_warnings(h), "tug_of_war": t,
                             "assets": dc.asset_implications(h, 2, t),
                             "missing_series": ["demo (XYZ)"],
                             "unavailable": dc.UNAVAILABLE,
                             "cape_detail": res})
    check("panel renders the stage", "stage 2 of 7" in panel, True)
    check("panel shows the alert level", 'class="alert RED"' in panel, True)
    check("panel names missing series rather than hiding them",
          "demo (XYZ)" in panel, True)
    check("panel labels CAPE as ours, not Shiller's",
          "not the official Shiller series" in panel, True)
    check("disabled state renders without throwing",
          "unavailable" in rn._dalio_panel({"enabled": False,
                                            "reason": "no key"}), True)
    check("absent debt cycle renders nothing", rn._dalio_panel({}), "")



def test_marks_from_source():
    """Marks rebuilt from 'The Truth about Investing'. Tests the new metrics
    and the cycle-adjusted bar, which is what the page actually reports."""
    import yaml as _y
    th = _y.safe_load(open("config/thresholds.yml"))
    cfg = th["marks"]
    n = len(cfg["tests"])
    check("Marks now has 8 tests", n, 8)
    check("base bar is 5 of 8", cfg["min_tests_passed"], 5)
    check("Marks stays cycle-adjusted", cfg["cycle_adjust"], True)
    for t in ("cheaper_than_its_market", "survives_the_worst_day",
              "avoids_the_losers", "no_history_of_losses"):
        check(f"new test present: {t}", t in cfg["tests"], True)

    # The bar the page reports, at each posture.
    m_pass = {"ev_to_ebit": 10.0, "ev_ebit_vs_market": 0.70,
              "max_drawdown_5y": 0.35, "downside_capture": 0.80,
              "loss_years_in_10": 0, "net_debt_to_ebitda": 1.0,
              "accruals_ratio": 0.02, "pct_below_52w_high": 0.20}
    for shift, needed, posture in ((0, 5, "core"), (1, 6, "defensive"),
                                   (-1, 4, "opportunistic")):
        r = sc.run_framework("marks", cfg, m_pass, "fail", cycle_shift=shift)
        check(f"{posture} bar is {needed} of 8", r["required"], needed)
        check(f"all-pass name passes in {posture}", r["passed"], True)

    # Exactly 5 passing: clears core, fails defensive. This is the case the
    # user actually sees change when the gauge flips.
    m5 = dict(m_pass, net_debt_to_ebitda=9.0, accruals_ratio=0.40,
              pct_below_52w_high=0.00)
    check("5-of-8 name passes in core",
          sc.run_framework("marks", cfg, m5, "fail", cycle_shift=0)["passed"],
          True)
    check("same name fails once the gauge turns defensive",
          sc.run_framework("marks", cfg, m5, "fail", cycle_shift=1)["passed"],
          False)

    # --- downside capture and drawdown, from prices ------------------------
    idx = pd.Series([100, 90, 99, 89, 98, 88] * 45,
                    index=pd.bdate_range("2023-01-02", periods=270))
    # Defensive name: halves every index move, so capture is 0.5 both ways.
    defensive = pd.Series(
        [100.0], index=[pd.Timestamp("2023-01-02")])
    vals = [100.0]
    ir = idx.pct_change().fillna(0.0)
    for x in ir.iloc[1:]:
        vals.append(vals[-1] * (1 + 0.5 * x))
    defensive = pd.Series(vals, index=idx.index)
    t = ta.compute(pd.DataFrame({"Close": defensive, "High": defensive,
                                 "Low": defensive, "Volume": [1e6] * len(idx)},
                                index=idx.index),
                   index_df=pd.DataFrame({"Close": idx}, index=idx.index))
    check("downside capture of a half-beta name is ~0.5",
          round(t["downside_capture"], 2), 0.5)
    check("upside capture is measured separately",
          round(t["upside_capture"], 2), 0.5)
    check("capture ratio is upside over downside",
          round(t["capture_ratio"], 2), 1.0)

    # A name that falls harder than the index must FAIL avoids_the_losers.
    vals = [100.0]
    for x in ir.iloc[1:]:
        vals.append(vals[-1] * (1 + (1.8 if x < 0 else 0.9) * x))
    frag = pd.Series(vals, index=idx.index)
    t2 = ta.compute(pd.DataFrame({"Close": frag, "High": frag, "Low": frag,
                                  "Volume": [1e6] * len(idx)}, index=idx.index),
                    index_df=pd.DataFrame({"Close": idx}, index=idx.index))
    check("a name that falls harder captures more downside",
          t2["downside_capture"] > 1.5, True)
    check("and it fails the avoids_the_losers test",
          sc.evaluate_test("avoids_the_losers",
                           cfg["tests"]["avoids_the_losers"],
                           {"downside_capture": t2["downside_capture"]}
                           )["result"], False)

    # Max drawdown is peak-to-trough, not standard deviation.
    crash = pd.Series(list(range(100, 200)) + list(range(200, 100, -1))
                      + list(range(100, 160)),
                      index=pd.bdate_range("2023-01-02", periods=260),
                      dtype=float)
    t3 = ta.compute(pd.DataFrame({"Close": crash, "High": crash, "Low": crash,
                                  "Volume": [1e6] * 260}, index=crash.index))
    check("max drawdown is peak-to-trough and reported positive",
          round(t3["max_drawdown_5y"], 3), 0.5)

    # Too few down days must yield None, not a two-observation ratio.
    up = pd.Series([100.0 * (1.001 ** i) for i in range(270)], index=idx.index)
    t4 = ta.compute(pd.DataFrame({"Close": up, "High": up, "Low": up,
                                  "Volume": [1e6] * 270}, index=idx.index),
                    index_df=pd.DataFrame({"Close": up}, index=idx.index))
    check("no down sample -> capture is None, not a made-up number",
          t4["downside_capture"], None)

    # --- relative value is scoped per market ------------------------------
    recs, mt = [], {}
    for i in range(25):
        recs.append(CompanyRecord(ticker=f"U{i}", market="US"))
        mt[f"U{i}"] = {"ev_to_ebit": 20.0 + i}          # median 32
    for i in range(25):
        recs.append(CompanyRecord(ticker=f"T{i}", market="TH"))
        mt[f"T{i}"] = {"ev_to_ebit": 5.0 + i}           # median 17
    sc.add_relative_value(recs, mt)
    check("US median is its own, not the world's",
          mt["U0"]["market_median_ev_ebit"], 32.0)
    check("Thai median is separate", mt["T0"]["market_median_ev_ebit"], 17.0)
    # The whole point: a US name at 20 is cheap for the US even though a Thai
    # name at 20 is expensive for Thailand. A global median would invert this.
    check("cheapest US name reads cheap for the US",
          round(mt["U0"]["ev_ebit_vs_market"], 3), round(20.0 / 32.0, 3))
    check("a Thai name at the same multiple reads expensive for Thailand",
          mt["T15"]["ev_ebit_vs_market"] > 1.0, True)

    # A thin market must get no ratio rather than a median off three names.
    thin = [CompanyRecord(ticker="X1", market="XX")]
    mtx = {"X1": {"ev_to_ebit": 9.0}}
    sc.add_relative_value(thin, mtx)
    check("a market too thin to have a median gets no ratio",
          mtx["X1"]["ev_ebit_vs_market"], None)
    # Negative EV/EBIT means negative EBIT — must not enter the median.
    neg = [CompanyRecord(ticker=f"N{i}", market="US") for i in range(25)]
    mtn = {f"N{i}": {"ev_to_ebit": (-50.0 if i < 5 else 10.0)} for i in range(25)}
    sc.add_relative_value(neg, mtn)
    check("loss makers are excluded from the median",
          mtn["N10"]["market_median_ev_ebit"], 10.0)



def test_soros_and_rogers_from_source():
    """Soros rebuilt from Alchemy, Rogers added from Hot Commodities."""
    th = yaml.safe_load(open("config/thresholds.yml"))

    # ---- Soros: the reflexive fingerprints ------------------------------
    sor = th["soros"]
    check("Soros now has 7 tests", len(sor["tests"]), 7)
    check("Soros bar is 5 of 7", sor["min_tests_passed"], 5)
    check("macro gate survives the rebuild", sor["macro_gate"]["enabled"], True)

    # The conglomerate: EPS bought with paper. Share count UP, EPS up fast,
    # revenue per share flat. Soros: "they could offer their own highly priced
    # stock in acquiring other companies."
    congl = []
    for i in range(4):
        fy = 2025 - i
        sh = 100.0 * (1.25 ** i)          # count SHRINKS going back = issuing
        congl.append(mkyear(fy, revenue=1000.0 * (1.25 ** (3 - i)),
                            net_income=100.0 * (1.30 ** (3 - i)),
                            eps_diluted=(100.0 * (1.30 ** (3 - i))) / (100.0 * (1.25 ** (3 - i))),
                            shares_diluted=100.0 * (1.25 ** (3 - i)),
                            total_assets=2000.0, total_equity=1000.0))
    rec = CompanyRecord(ticker="CONGL", market="US", years=congl)
    rec.price, rec.market_cap = 50.0, 5000.0
    rec.technicals = {"return_12m": 0.60}
    m = mx.compute_metrics(rec)
    check("share count growth is measured over 1 year",
          m["share_count_change_1y"] > 0.20, True)
    check("EPS growth and revenue-per-share growth are separated",
          m["revenue_per_share_growth_1y"] is not None, True)
    # EPS grows 30%/yr on a 25%/yr share count: revenue per share is flat, so
    # the gap is the accretion. This is the fingerprint.
    check("EPS outruns revenue per share",
          m["eps_over_revenue_per_share_gap"] > 0.0, True)
    check("a 60% price move against 4% EPS growth is a wide divergence",
          round(m["reflexive_divergence"], 2) > 0.5, True)
    check("and it fails 'expectations not yet excessive'",
          sc.evaluate_test("x", sor["tests"]["expectations_not_yet_excessive"],
                           m)["result"], False)

    # Act Two: ROE held flat by rising leverage while the margin falls.
    act2 = [mkyear(2025, revenue=1000.0, net_income=100.0,
                   total_assets=3000.0, total_equity=500.0,
                   eps_diluted=1.0, shares_diluted=100.0),
            mkyear(2024), mkyear(2023),
            mkyear(2022, revenue=600.0, net_income=100.0,
                   total_assets=1500.0, total_equity=500.0,
                   eps_diluted=1.0, shares_diluted=100.0)]
    r2 = CompanyRecord(ticker="ACT2", market="US", years=act2)
    r2.price, r2.market_cap = 10.0, 1000.0
    m2 = mx.compute_metrics(r2)
    # margin 10% vs 16.7%, leverage 6.0 vs 3.0, ROE 20% vs 20%.
    check("leverage doubled over 3 years", round(m2["leverage_change_3y"], 2), 1.0)
    check("the Act Two signature is detected", m2["roe_held_up_by_leverage"], 1)
    check("and it fails 'ROE not propped by leverage'",
          sc.evaluate_test("x", sor["tests"]["roe_not_propped_by_leverage"],
                           m2)["result"], False)

    # A clean organic grower must NOT trip either fingerprint.
    org = [mkyear(2025 - i, revenue=1000.0 * (1.1 ** (3 - i)),
                  net_income=100.0 * (1.1 ** (3 - i)),
                  eps_diluted=1.0 * (1.1 ** (3 - i)),
                  shares_diluted=100.0, total_assets=2000.0,
                  total_equity=1000.0) for i in range(4)]
    r3 = CompanyRecord(ticker="ORG", market="US", years=org)
    r3.price, r3.market_cap = 20.0, 2000.0
    r3.technicals = {"return_12m": 0.12}
    m3 = mx.compute_metrics(r3)
    check("an organic grower issues no stock", m3["share_count_change_1y"], 0.0)
    check("its EPS and revenue per share move together",
          round(m3["eps_over_revenue_per_share_gap"], 6), 0.0)
    check("no Act Two signature", m3["roe_held_up_by_leverage"], 0)
    check("and its divergence is modest",
          round(m3["reflexive_divergence"], 3), round(0.12 - 0.1, 3))

    # Stage CD: a trend must have been TESTED. An untested melt-up fails.
    smooth = pd.Series([100.0 * (1.002 ** i) for i in range(260)],
                       index=pd.bdate_range("2025-01-01", periods=260))
    t = ta.compute(pd.DataFrame({"Close": smooth, "High": smooth, "Low": smooth,
                                 "Volume": [1e6] * 260}, index=smooth.index))
    check("a never-corrected melt-up has ~zero 1y drawdown",
          t["max_drawdown_1y"] < 0.01, True)
    check("and fails 'tested and held'",
          sc.evaluate_test("x", sor["tests"]["tested_and_held"],
                           {"max_drawdown_1y": t["max_drawdown_1y"]})["result"],
          False)

    # ---- Rogers ----------------------------------------------------------
    rog = th["rogers"]
    check("Rogers has 7 tests", len(rog["tests"]), 7)
    check("Rogers is gated to commodity themes", rog["themes_only"], True)
    check("Rogers is registered in the renderer",
          "rogers" in [k for k, _ in rn.FRAMEWORKS], True)
    check("renderer now shows 11 frameworks", len(rn.FRAMEWORKS), 11)

    # Capex against depreciation is the company-level supply gauge.
    miner = [mkyear(2025, revenue=1000.0, net_income=120.0, capex=60.0,
                    depreciation_amortization=100.0, cfo=200.0,
                    eps_diluted=1.2, shares_diluted=100.0,
                    total_assets=2000.0, total_equity=1200.0,
                    total_debt=200.0, cash_and_equivalents=150.0,
                    goodwill=0.0, intangibles=0.0,
                    operating_income=160.0)] + [mkyear(2024 - i) for i in range(3)]
    rm = CompanyRecord(ticker="MINE", market="US", years=miner)
    rm.price, rm.market_cap = 12.0, 1200.0
    m4 = mx.compute_metrics(rm)
    check("capex/depreciation below 1 means a shrinking asset base",
          round(m4["capex_to_depreciation"], 2), 0.6)
    check("which passes 'not adding to supply'",
          sc.evaluate_test("x", rog["tests"]["not_adding_to_supply"],
                           m4)["result"], True)

    # A name with no commodity theme must be NOT-APPLICABLE, not failed —
    # Rogers analyses the commodity first, and there is no commodity here.
    plain = CompanyRecord(ticker="BANK", market="US", themes=[], years=org)
    plain.price, plain.market_cap = 20.0, 2000.0
    out = sc.screen_universe([plain], {"BANK": m3}, th, {}, cycle=None)
    rres = out["results"]["BANK"]["frameworks"]["rogers"]
    check("a non-commodity name is ineligible for Rogers, not failed",
          bool(rres.get("ineligible_reason")), True)
    check("and the reason names the commodity gap",
          "supply cycle" in rres["ineligible_reason"], True)

    themed = CompanyRecord(ticker="COPR", market="US", themes=["Copper"],
                           years=miner)
    themed.price, themed.market_cap = 12.0, 1200.0
    out2 = sc.screen_universe([themed], {"COPR": m4}, th, {}, cycle=None)
    check("a themed name IS evaluated by Rogers",
          bool(out2["results"]["COPR"]["frameworks"]["rogers"]
               .get("ineligible_reason")), False)



def test_buffett_additions_and_graham():
    """New value metrics, the two Buffett additions, and the Graham screen."""
    th = yaml.safe_load(open("config/thresholds.yml"))
    check("Buffett now has 16 tests", len(th["buffett"]["tests"]), 16)
    check("Buffett bar is 11 of 16", th["buffett"]["min_tests_passed"], 11)
    check("Graham has 9 tests", len(th["graham"]["tests"]), 9)
    check("renderer shows 11 frameworks", len(rn.FRAMEWORKS), 11)

    def mk(**over):
        base = dict(revenue=1000.0, gross_profit=400.0, operating_income=200.0,
                    pretax_income=190.0, net_income=140.0, eps_diluted=1.40,
                    current_assets=600.0, current_liabilities=250.0,
                    inventory=120.0, dividends_paid=-40.0, total_assets=2000.0,
                    total_equity=700.0, total_debt=200.0, goodwill=100.0,
                    intangibles=60.0, cash_and_equivalents=150.0, net_ppe=900.0,
                    cfo=250.0, capex=50.0, depreciation_amortization=70.0,
                    shares_diluted=100.0)
        base.update(over)
        return base

    ys = [mkyear(2025 - i, **mk(revenue=1000.0 * (1.06 ** (6 - i)),
                                gross_profit=400.0 * (1.06 ** (6 - i)),
                                operating_income=200.0 * (1.06 ** (6 - i)),
                                pretax_income=190.0 * (1.06 ** (6 - i)),
                                net_income=140.0 * (1.06 ** (6 - i)),
                                eps_diluted=1.40 * (1.06 ** (6 - i))))
          for i in range(7)]
    rec = CompanyRecord(ticker="Q", market="US", years=ys)
    rec.price, rec.market_cap = 28.0, 2800.0
    rec.technicals = {"return_12m": 0.10}
    m = mx.compute_metrics(rec)

    check("current ratio = 600/250", m["current_ratio"], 2.4)
    check("quick ratio strips inventory", m["quick_ratio"], (600.0 - 120.0) / 250.0)
    check("payout ratio uses the magnitude of dividends paid",
          round(m["payout_ratio"], 4), round(40.0 / ys[0].net_income, 4))
    # Tangible book = 700 - 100 - 60 = 540; pretax at 2025 = 190*1.06^6.
    check("RONTA is pre-tax over TANGIBLE book, not equity",
          round(m["return_on_net_tangible_assets"], 5),
          round(ys[0].pretax_income / 540.0, 5))
    check("and it is higher than ROE would be, because goodwill is excluded",
          m["return_on_net_tangible_assets"] > (ys[0].net_income / 700.0), True)

    # The point of RONTA: a serial acquirer with heavy goodwill.
    acq = [mkyear(2025 - i, **mk(goodwill=900.0, intangibles=100.0,
                                 total_equity=1400.0)) for i in range(7)]
    ra = CompanyRecord(ticker="ACQ", market="US", years=acq)
    ra.price, ra.market_cap = 28.0, 2800.0
    ma = mx.compute_metrics(ra)
    check("a goodwill-heavy acquirer still shows a respectable ROE",
          round(140.0 / 1400.0, 3), 0.1)
    check("but RONTA on 400 of tangible book is a different number",
          round(ma["return_on_net_tangible_assets"], 4), round(190.0 / 400.0, 4))

    # Graham number and the P/E x P/B rule.
    check("Graham number = sqrt(22.5 x EPS x BVPS)",
          round(m["graham_number"], 4),
          round((22.5 * ys[0].eps_diluted * (700.0 / 100.0)) ** 0.5, 4))
    check("P/E x P/B is a product, not a ratio",
          round(m["pe_times_pb"], 4),
          round(m["pe_ttm"] * m["price_to_book"], 4))

    # Earnings predictability refuses to score a loss-maker rather than
    # returning a large number — a CV across a sign change is meaningless.
    check("EPS CV is computed for a steady grower", m["eps_cv_5y"] > 0, True)
    lossy = [mkyear(2025, **mk(eps_diluted=1.0)), mkyear(2024, **mk(eps_diluted=-0.5)),
             mkyear(2023, **mk(eps_diluted=0.8)), mkyear(2022, **mk(eps_diluted=0.9))]
    rl = CompanyRecord(ticker="L", market="US", years=lossy)
    rl.price, rl.market_cap = 10.0, 1000.0
    check("a loss year makes EPS CV unevaluable, not merely large",
          mx.compute_metrics(rl)["eps_cv_5y"], None)

    # Inflation resilience needs all three legs; break one and it goes to 0.
    check("steady margin + growing revenue/share + light capex = resilient",
          m["inflation_resilient"], 1)
    heavy = [mkyear(2025 - i, **mk(capex=300.0,
                                   revenue=1000.0 * (1.06 ** (6 - i)),
                                   operating_income=200.0 * (1.06 ** (6 - i)),
                                   eps_diluted=1.40 * (1.06 ** (6 - i)),
                                   net_income=140.0 * (1.06 ** (6 - i))))
             for i in range(7)]
    rh = CompanyRecord(ticker="H", market="US", years=heavy)
    rh.price, rh.market_cap = 28.0, 2800.0
    mh = mx.compute_metrics(rh)
    check("heavy capital intensity breaks the resilience flag",
          mh["inflation_resilient"], 0)
    check("and that fails the Buffett inflation test",
          sc.evaluate_test("x", th["buffett"]["tests"]["inflation_resilience"],
                           mh)["result"], False)

    # Graham: the two-thirds net-net rule is our NCAV/mcap >= 1.5.
    g = th["graham"]["tests"]
    check("net-net at 2/3 NCAV means NCAV/mcap >= 1.5",
          g["net_net_two_thirds"]["threshold"], 1.5)
    check("a stock at exactly 2/3 of NCAV passes",
          sc.evaluate_test("x", g["net_net_two_thirds"],
                           {"ncav_to_market_cap": 1.5})["result"], True)
    check("one at 90% of NCAV does not",
          sc.evaluate_test("x", g["net_net_two_thirds"],
                           {"ncav_to_market_cap": 1.11})["result"], False)
    check("current ratio 2.4 clears Graham's liquidity bar",
          sc.evaluate_test("x", g["liquidity"], m)["result"], True)
    check("PEG limit is the one the guide states", g["peg_fair_or_better"]["threshold"], 1.0)
    check("P/E x P/B limit is 22.5", g["pe_times_pb_limit"]["threshold"], 22.5)

    # Graham runs on every company, and is registered end to end.
    out = sc.screen_universe([rec], {"Q": m}, th, {}, cycle=None)
    fw = out["results"]["Q"]["frameworks"]
    check("Graham is evaluated in the pipeline", "graham" in fw, True)
    check("Buffett reports 16 tests in the pipeline", fw["buffett"]["n_total"], 16)
    check("a fund is ineligible for Graham, not failed",
          bool(sc.screen_universe(
              [CompanyRecord(ticker="ETF", market="US", quote_type="ETF",
                             years=ys)], {"ETF": m}, th, {}, cycle=None
          )["results"]["ETF"]["frameworks"]["graham"].get("ineligible_reason")),
          True)



def test_commodities_cnav_and_gauge():
    """Rogers's board, the guide's CNAV/POF method, and the 5-signal gauge."""
    from src import commodities as cm, cycle as cy
    th = yaml.safe_load(open("config/thresholds.yml"))

    # ---- the board -------------------------------------------------------
    syms = cm.all_symbols()
    check("board covers both futures and ETFs", "HG=F" in syms and "CPER" in syms, True)
    check("symbols are deduplicated", len(syms), len(set(syms)))
    kinds = {r["kind"] for r in cm.BOARD}
    check("instrument structure is declared", kinds >= {"futures", "physical", "etn", "equity"}, True)
    ura = next(r for r in cm.BOARD if r["etf"] == "URA")
    check("URA is flagged as MINERS, not the commodity", ura["kind"], "equity")
    check("and the note says so", "MINERS" in ura["note"], True)

    class FakeStore:
        def __init__(s_, frames): s_.f = frames
        def save_prices(s_, t, df): pass
        def load_prices(s_, t): return s_.f.get(t)

    idx = pd.bdate_range("2024-01-01", periods=300)
    # Commodity up 30% over the year; the fund up only 5% -> 25pp of drag.
    fut = pd.DataFrame({"Close": pd.Series(
        [100.0 * (1.30 ** (i / 252)) for i in range(300)], index=idx)})
    etf = pd.DataFrame({"Close": pd.Series(
        [50.0 * (1.05 ** (i / 252)) for i in range(300)], index=idx)})
    board = cm.build(None, FakeStore({"NG=F": fut, "UNG": etf}))
    row = next(r for r in board["rows"] if r["etf"] == "UNG")
    check("futures 12m return computed", round(row["future_12m"], 2), 0.30)
    check("ETF 12m return computed", round(row["etf_12m"], 2), 0.05)
    check("tracking gap is ETF minus commodity", round(row["tracking_gap_12m"], 2), -0.25)
    check("severe drag is called severe", "severe" in row["tracking_reading"], True)

    # Backwardation: the case the module originally could not express. A
    # rolling fund in a backwardated market BEATS the front-month price change
    # because every roll buys a cheaper deferred contract. The first version
    # graded that as "kept up with the commodity", which understated the most
    # informative reading the column has.
    back = {"kind": "futures", "future_12m": 0.289, "future_vs_200dma": 0.07,
            "future_off_52w_high": 0.05, "tracking_gap_12m": 0.432}
    a = cm.assess(back)
    check("a large positive gap is graded, not lumped in with 'clean'",
          a["instrument_grade"], "roll paying")
    check("and it is explained as backwardation",
          "backwardated" in a["instrument_note"], True)
    mild = dict(back, tracking_gap_12m=0.05)
    check("a small positive gap reads roll positive",
          cm.assess(mild)["instrument_grade"], "roll positive")
    flat = dict(back, tracking_gap_12m=0.00)
    check("a flat gap is clean", cm.assess(flat)["instrument_grade"], "clean")

    # A commodity up on the year but under its 200-day must say so, or a
    # +33.8% row labelled "no clear trend" reads as a bug.
    gold = {"kind": "physical", "future_12m": 0.338, "future_vs_200dma": -0.02,
            "future_off_52w_high": 0.20, "tracking_gap_12m": -0.022}
    check("a strong year below the 200-day is explained, not just 'no trend'",
          "below its 200-day" in cm.assess(gold)["commodity_call"], True)
    check("the panel explains both directions of the gap",
          "backwardated" in rn._commodity_panel(
              {"rows": [dict(gold, name="Gold", etf="GLD",
                             assessment=cm.assess(gold))],
               "missing": [], "caveat": "x"}), True)
    check("absent symbols are named, not blanked silently",
          len(board["missing"]) > 0, True)
    gold = next(r for r in board["rows"] if r["etf"] == "GLD")
    check("a symbol with no data has no price rather than a stale one",
          gold.get("future_price"), None)
    check("panel renders", "the thing vs the instrument" in rn._commodity_panel(board), True)
    check("empty board renders nothing", rn._commodity_panel({}), "")

    # ---- CNAV, hand-computed --------------------------------------------
    def yy(fy):
        return mkyear(fy, cash_and_equivalents=100.0, short_term_investments=20.0,
                      net_ppe=200.0, receivables=60.0, inventory=40.0,
                      intangibles=30.0, goodwill=10.0, total_liabilities=150.0,
                      total_assets=500.0, total_equity=350.0, total_debt=120.0,
                      net_income=25.0, cfo=40.0, revenue=400.0,
                      eps_diluted=0.25, shares_diluted=100.0,
                      current_assets=220.0, current_liabilities=90.0,
                      operating_income=35.0, pretax_income=30.0)
    rec = CompanyRecord(ticker="SG1", market="SG", years=[yy(2025 - i) for i in range(5)])
    rec.price, rec.market_cap = 1.20, 120.0
    m = mx.compute_metrics(rec)
    # full = 100+20 = 120; half = (200+60+40+(30-10)) x 0.5 = 160; less 150.
    check("CNAV counts cash at full and the rest at half", m["cnav"], 130.0)
    check("goodwill is excluded entirely", m["cnav_per_share"], 1.30)
    check("price/CNAV", round(m["price_to_cnav"], 4), round(1.20 / 1.30, 4))
    check("CNAV discount", round(m["cnav_discount"], 4), round(1 - 1.20 / 1.30, 4))
    check("POF is 3 when profitable, cash-generative and not over-levered",
          m["pof_score"], 3)
    check("POF components are itemised", len(m["pof_detail"]), 3)

    lossy = CompanyRecord(ticker="SG2", market="SG",
                          years=[mkyear(2025 - i, net_income=-5.0, cfo=-2.0,
                                        total_equity=100.0, total_debt=300.0,
                                        total_liabilities=400.0,
                                        cash_and_equivalents=10.0,
                                        shares_diluted=100.0, revenue=100.0)
                                 for i in range(5)])
    lossy.price, lossy.market_cap = 1.0, 100.0
    m2 = mx.compute_metrics(lossy)
    check("a loss-making, over-levered name scores POF 0", m2["pof_score"], 0)
    check("and fails the POF test",
          sc.evaluate_test("x", th["graham"]["tests"]["pof_score"], m2)["result"], False)
    check("negative CNAV yields no ratio rather than a negative one",
          m2["price_to_cnav"], None)

    # ---- the gauge now votes on five signals -----------------------------
    cyc_cfg = th["market_cycle"]
    up = pd.DataFrame({"Close": pd.Series(
        [100.0 * (1.0008 ** i) for i in range(300)], index=idx)})
    hot = cy.assess(up, {"pct_above_200dma": 0.80}, 13.0, cyc_cfg,
                    hy_oas=271.0, cape=42.18)
    check("tight credit and a rich CAPE both vote defensive",
          hot["votes"]["defensive"], 5)
    check("the evidence line names them", "high-yield 271bp" in hot["evidence"], True)
    check("and names CAPE", "CAPE 42.2" in hot["evidence"], True)
    check("gauge reads defensive", hot["mode"], "defensive")
    # Wide spreads and a cheap market must pull the other way.
    calm = cy.assess(up, {"pct_above_200dma": 0.30}, 35.0, cyc_cfg,
                     hy_oas=900.0, cape=12.0)
    check("stressed credit and a cheap CAPE vote opportunistic",
          calm["votes"]["opportunistic"] >= 4, True)
    # Absent inputs must simply not vote.
    none_ = cy.assess(up, {"pct_above_200dma": 0.50}, 20.0, cyc_cfg)
    check("missing credit and CAPE cast no vote at all",
          sum(none_["votes"].values()), 3)
    check("backwards compatible: 3-signal call still works", none_["mode"], "core")

    # The credit key regression. universe.yml calls it `hy_credit_spread`;
    # run.py once asked for `hy_spread`, got None, and the gauge silently
    # dropped to four signals while still describing itself as five. The only
    # symptom was a missing phrase in one banner.
    uni = yaml.safe_load(open("config/universe.yml"))
    check("the macro series key is hy_credit_spread",
          "hy_credit_spread" in uni["macro_series"], True)
    src = open("src/run.py").read()
    check("run.py reads that exact key", 'macro.get("hy_credit_spread")' in src, True)
    check("and warns when no credit spread is found at all",
          "WITHOUT its credit vote" in src, True)

    # A tie must be disclosed rather than resolved silently by dict order.
    tie = cy.assess(up, {"pct_above_200dma": 0.50}, 13.0, cyc_cfg,
                    hy_oas=None, cape=20.0)
    check("the vote split is reported", "vote " in tie["evidence"], True)
    check("a contested call is flagged", tie["signals"]["contested"] in (True, False), True)
    both = cy.assess(up, {"pct_above_200dma": 0.80}, 13.0, cyc_cfg,
                     hy_oas=271.0, cape=42.0)
    check("an uncontested call is not flagged as tied",
          both["signals"]["contested"], False)
    # RSI, breadth, VIX, credit and CAPE all read hot here — five votes, which
    # is the point: the gauge now has five signals to cast, not three.
    check("all five signals can vote",
          both["signals"]["vote_split"]["defensive"], 5)



def test_reflexive_stages():
    """Soros's nine stages as a per-name label, not a pass/fail."""
    from src import reflexivity as rfx

    def m(**kw):
        base = dict(share_count_change_1y=0.06, goodwill_to_assets=0.20,
                    price_above_sma200=1, max_drawdown_1y=0.09,
                    pct_below_52w_high=0.03)
        base.update(kw)
        return base

    cases = [
        ("AB", m(eps_growth_1y=0.18, return_12m=0.01, reflexive_divergence=-0.17,
                 pct_below_52w_high=0.12, max_drawdown_1y=0.11)),
        ("BC", m(eps_growth_1y=0.15, return_12m=0.25, reflexive_divergence=0.10,
                 max_drawdown_1y=0.05)),
        ("CD", m(eps_growth_1y=0.15, return_12m=0.25, reflexive_divergence=0.10,
                 max_drawdown_1y=0.18)),
        ("DE", m(eps_growth_1y=-0.08, return_12m=0.45, reflexive_divergence=0.53)),
        ("FG", m(eps_growth_1y=0.12, return_12m=-0.22, reflexive_divergence=-0.34,
                 price_above_sma200=0, pct_below_52w_high=0.35)),
        ("GH", m(eps_growth_1y=-0.25, return_12m=-0.40, reflexive_divergence=-0.15,
                 price_above_sma200=0, pct_below_52w_high=0.55)),
    ]
    for want, met in cases:
        check(f"stage {want} classified", rfx.stage(met)["stage"], want)

    # The separation that matters most: DE and HI are numerically identical on
    # earnings and price direction, and they are OPPOSITE trades. Only the
    # position in the range tells them apart. Getting this backwards means the
    # app tells you to sell the bottom.
    near_high = m(eps_growth_1y=-0.08, return_12m=0.30, reflexive_divergence=0.38,
                  pct_below_52w_high=0.02)
    far_off = m(eps_growth_1y=-0.08, return_12m=0.30, reflexive_divergence=0.38,
                pct_below_52w_high=0.45)
    check("earnings down + price up NEAR the high is DE",
          rfx.stage(near_high)["stage"], "DE")
    check("the same numbers FAR off the high are HI",
          rfx.stage(far_off)["stage"], "HI")
    check("DE is flagged late", rfx.stage(near_high)["late"], True)
    check("HI is not flagged late", rfx.stage(far_off).get("late"), False)

    # The framework must switch OFF where there is no price->fundamentals path.
    no_ch = m(eps_growth_1y=-0.08, return_12m=0.45, reflexive_divergence=0.53,
              share_count_change_1y=0.001, goodwill_to_assets=0.01)
    st = rfx.stage(no_ch)
    check("no issuance and no goodwill reads near-equilibrium", st["stage"], "EQ")
    check("and the channel is reported closed", st["channel"]["open"], False)
    check("a company transacting in its own equity has an open channel",
          rfx.channel_open({"share_count_change_1y": 0.06,
                            "goodwill_to_assets": 0.02})["open"], True)
    check("so does a serial acquirer",
          rfx.channel_open({"share_count_change_1y": 0.0,
                            "goodwill_to_assets": 0.25})["open"], True)
    check("unknown inputs give an unknown channel, not a closed one",
          rfx.channel_open({})["open"], None)
    check("missing price or earnings refuses to label a stage",
          rfx.stage({"share_count_change_1y": 0.06})["stage"], None)

    # Evidence must travel with the label.
    check("evidence names the numbers behind the call",
          any("EPS growth" in e for e in rfx.stage(near_high)["evidence"]), True)
    check("DE carries an explicit warning",
          "earnings setback" in rfx.stage(near_high)["warning"], True)

    # And the pipeline must mark a near-equilibrium name INELIGIBLE for Soros
    # rather than failing it — the book says the framework does not apply.
    th = yaml.safe_load(open("config/thresholds.yml"))
    rec = CompanyRecord(ticker="EQ1", market="US",
                        years=[mkyear(2025 - i, net_income=100.0, revenue=1000.0,
                                      eps_diluted=1.0, shares_diluted=100.0,
                                      total_equity=800.0, total_assets=1000.0,
                                      goodwill=5.0) for i in range(5)])
    rec.price, rec.market_cap = 20.0, 2000.0
    mm = mx.compute_metrics(rec)
    out = sc.screen_universe([rec], {"EQ1": mm}, th, {}, cycle=None)
    sor = out["results"]["EQ1"]["frameworks"]["soros"]
    check("near-equilibrium name is ineligible for Soros, not failed",
          "near-equilibrium" in (sor.get("ineligible_reason") or ""), True)

    # The census counts every labelled row.
    res = {"A": {"metrics": near_high}, "B": {"metrics": far_off},
           "C": {"metrics": no_ch}}
    cen = rfx.annotate(res, {})
    check("census counts each stage", cen, {"DE": 1, "HI": 1, "EQ": 1})
    check("the label is attached to the row", res["A"]["reflexive"]["stage"], "DE")

    # Renderer: the badge and the filter must both exist.
    t = rn.TEMPLATE
    check("stage badge is rendered on the ticker", 'class="rfx' in t, True)
    check("late stages get their own style", ".rfx.late" in t, True)
    check("a Reflexive risk filter chip exists", "Reflexive risk" in t, True)
    check("and the filter is wired to the late flag", "fRfx && !r.rfx_late" in t, True)

    # A two-letter badge with no key is a puzzle, not a label.
    leg = rn._reflexive_legend({"BC": 120, "CD": 40, "DE": 60, "EQ": 600, "GH": 20})
    check("the legend explains the codes", "what the badges mean" in leg, True)
    check("it shows each stage's share", "(7%)" in leg or "(70%)" in leg, True)
    check("a healthy spread raises no warning",
          "not identifying anything" in leg, False)

    # The self-check that matters: a stage firing on most of the universe has
    # stopped discriminating, and the page must say so rather than leaving the
    # user to notice that every row carries the same badge.
    dom = rn._reflexive_legend({"DE": 500, "EF": 60, "BC": 40, "EQ": 30})
    check("DE dominance is called out on the page",
          "not identifying anything" in dom, True)
    check("and quantified", "89%" in dom, True)
    check("mostly near-equilibrium is explained as expected, not alarming",
          "expected result" in rn._reflexive_legend({"EQ": 900, "BC": 50}), True)
    check("no census renders nothing", rn._reflexive_legend({}), "")



def test_ownership_and_government_theme():
    """13F parsing, Form 4 transaction codes, and the hand-kept stake list."""
    from src import ownership as ow

    # ---- 13F INFOTABLE aggregation --------------------------------------
    tsv = "\t".join(["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT"]) + "\n"
    rows = [
        ("0001-A", "037833100", "1000", "500"),   # Apple, manager A
        ("0001-A", "037833100", "2000", "700"),   # same manager, 2nd account
        ("0002-B", "037833100", "3000", "900"),   # manager B
        ("0003-C", "594918104", "5000", "100"),   # Microsoft, manager C
        ("0004-D", "BADCUSIP", "9999", "999"),    # malformed, must be dropped
    ]
    tsv += "\n".join("\t".join(r) for r in rows)
    agg = ow.parse_infotable(tsv)
    check("malformed CUSIPs are dropped", "BADCUSIP" in agg, False)
    # THE regression that matters: a manager reporting one holding across two
    # accounts is ONE holder, not two. Counting rows would inflate every
    # widely-held name and make the change-on-quarter number meaningless.
    check("holders counts distinct filers, not rows",
          agg["037833100"]["holders"], 2)
    check("values still sum across all rows", agg["037833100"]["value"], 6000.0)
    check("shares sum across all rows", agg["037833100"]["shares"], 2100.0)
    check("a single-filer name reads 1 holder", agg["594918104"]["holders"], 1)

    # ---- Form 4: only an open-market purchase is a purchase ---------------
    def f4(code, shares="1000", price="50"):
        return (f"<transactionCode>{code}</transactionCode>"
                f"<transactionShares><value>{shares}</value></transactionShares>"
                f"<transactionPricePerShare><value>{price}</value>"
                f"</transactionPricePerShare>")

    check("code P is a purchase", ow.parse_form4(f4("P"))["buys"], 1)
    check("and its dollar value is captured",
          ow.parse_form4(f4("P"))["buy_value"], 50000.0)
    check("code S is a sale", ow.parse_form4(f4("S"))["sells"], 1)
    # The distinction the whole module rests on. A stock AWARD is not a
    # director buying with their own money, and counting it as one would turn
    # routine pay into a bullish signal.
    check("a stock award is NOT a purchase", ow.parse_form4(f4("A"))["buys"], 0)
    check("an option exercise is NOT a purchase", ow.parse_form4(f4("M"))["buys"], 0)
    check("a gift is NOT a purchase", ow.parse_form4(f4("G"))["buys"], 0)
    check("awards are counted separately as other",
          ow.parse_form4(f4("A"))["other"], 1)
    mixed = f4("P", "100", "10") + f4("A", "500", "10") + f4("S", "50", "20")
    r = ow.parse_form4(mixed)
    check("a mixed filing splits correctly",
          (r["buys"], r["sells"], r["other"]), (1, 1, 1))
    check("only the purchase leg is valued", r["buy_value"], 1000.0)
    check("an empty document yields zeros, not an error",
          ow.parse_form4("")["buys"], 0)

    # ---- the quarter slugs must respect the 45-day filing lag ------------
    slugs = ow._quarter_slugs(2)
    check("two quarters requested, two returned", len(slugs), 2)
    check("slugs are shaped like 2026q1", bool(re.match(r"^\d{4}q[1-4]$", slugs[0])), True)

    # ---- graceful failure ------------------------------------------------
    class Dead:
        def get(self, *a, **k):
            raise RuntimeError("no network")
    out = ow.institutional(Dead(), ["AAPL"])
    check("an unreachable SEC disables the feed rather than crashing",
          out["enabled"], False)
    check("and it says which quarters it tried", "tried" in out["reason"], True)
    check("no CIK map disables insiders cleanly",
          ow.insider_activity(Dead(), {})["enabled"], False)

    # ---- the government-stake theme -------------------------------------
    uni = yaml.safe_load(open("config/universe.yml"))
    gov = uni["themes"]["US government stake"]
    check("government stake theme exists", len(gov["tickers"]) >= 10, True)
    check("Intel is in it", "INTC" in gov["tickers"], True)
    check("MP Materials is in it", "MP" in gov["tickers"], True)
    # A hand-kept list with no upstream feed MUST carry a date, or it goes
    # stale silently and nothing anywhere will say so.
    check("it is dated", bool(gov.get("as_of")), True)
    check("and it names its source", "no official register" in gov["source"], True)



def test_dislocation():
    """Hard falls the accounts do not explain, and the causes ruled out."""
    from src import dislocation as ds
    cfg = yaml.safe_load(open("config/thresholds.yml"))["dislocation"]

    def m(**kw):
        base = dict(return_6m=-0.38, rs_vs_market_index_6m=-0.31,
                    worst_month_in_6m=0.26, vol20_over_vol50=1.6,
                    revenue_growth_1y=0.08, eps_growth_1y=0.04,
                    free_cash_flow_ttm=120.0, net_debt_to_ebitda=1.2,
                    loss_years_in_10=0, accruals_ratio=0.01)
        base.update(kw)
        return base

    # Nothing that has not fallen far enough may appear at all.
    check("a 10% fall does not qualify", ds.assess(m(return_6m=-0.10), cfg), None)
    check("a rise does not qualify", ds.assess(m(return_6m=0.20), cfg), None)
    check("no price history does not qualify",
          ds.assess(m(return_6m=None), cfg), None)

    a = ds.assess(m(), cfg)
    check("a 38% fall with intact accounts qualifies", a["qualifies"], True)

    # The three discriminators. These are the whole value of the module: the
    # app cannot know WHY a stock fell, but it can rule causes in and out.
    check("fell far more than its index -> name-specific", a["scope"], "name")
    check("68% of the fall in one month -> cliff", a["shape"], "cliff")
    check("volume 1.6x baseline -> heavy", a["volume"], "heavy")
    ns = [c["n"] for c in a["candidate_causes"]]
    check("market-wide causes are ruled out for a solo fall",
          any(n in ns for n in (1, 4, 8, 9, 10)), False)
    check("a scandal stays on the list", 5 in ns, True)
    check("and a short-seller loop stays on the list", 14 in ns, True)

    # Same fall, but the whole market went with it: the opposite shortlist.
    b = ds.assess(m(rs_vs_market_index_6m=-0.03, worst_month_in_6m=0.09,
                    vol20_over_vol50=0.7), cfg)
    check("in-line with a falling market -> macro", b["scope"], "market")
    check("spread over the period -> grind", b["shape"], "grind")
    bn = [c["n"] for c in b["candidate_causes"]]
    check("flight to safety is now a candidate", 8 in bn, True)
    check("a CEO scandal is not", 5 in bn, False)
    check("thin volume reads as a liquidity vacuum, not forced selling",
          b["volume"], "thin")
    check("and forced liquidation is dropped when volume is thin", 10 in bn, False)

    # The gate that stops this being a list of broken businesses.
    broken = ds.assess(m(revenue_growth_1y=-0.34, eps_growth_1y=-0.70,
                         free_cash_flow_ttm=-50.0), cfg)
    check("a collapsing business does NOT qualify", broken["qualifies"], False)
    check("but it is still assessed rather than hidden",
          broken["return_6m"], -0.38)

    # Missing fundamentals must not quietly count as intact.
    thin = ds.assess({"return_6m": -0.40}, cfg)
    check("a name with no accounts at all cannot qualify",
          thin["qualifies"], False)
    check("and the caution says how many tests were unevaluable",
          "could not be evaluated" in thin["caution"], True)

    # The warning must travel with every hit, without exception.
    check("every hit carries the stale-accounts caution",
          "filings are stale and the market is right" in a["caution"], True)

    # The scan labels rows and counts them.
    res = {"A": {"metrics": m()}, "B": {"metrics": m(return_6m=-0.05)},
           "C": {"metrics": m(revenue_growth_1y=-0.5, eps_growth_1y=-0.8,
                              free_cash_flow_ttm=-10.0)}}
    summ = ds.scan(res, {}, cfg)
    check("two names fell more than 30%", summ["fell_30pct"], 2)
    check("only one had intact accounts", summ["fundamentals_intact"], 1)
    check("the qualifying row is labelled", res["A"]["dislocation"]["qualifies"], True)
    check("the un-fallen row is not labelled", "dislocation" in res["B"], False)

    # Renderer.
    pan = rn._dislocation_panel(summ)
    # The header must lead with how many FELL, not only how many survived the
    # fundamentals gate — otherwise the shortlist has no denominator and the
    # question "which stocks dropped 30%?" has no answer anywhere on the page.
    check("panel leads with the number that fell", "2 names" in pan, True)
    check("and names the qualifying subset",
          "1 the accounts do not explain" in pan, True)
    check("panel says where to find them", "Where to find them" in pan, True)
    check("and explains the two badge colours",
          "blue" in pan and "grey" in pan, True)
    check("panel leads with the stale-accounts warning",
          "filings are stale and the market is right" in pan, True)
    check("panel says it cannot identify a cause",
          "cannot</b> identify a cause" in pan, True)
    check("panel lists all fifteen causes",
          all(f"#{i} " in pan for i in range(1, 16)), True)
    check("panel names the three feeds",
          all(k in pan for k in ("SEC 8-K item codes", "USGS", "GDELT")), True)
    check("panel says silence is not evidence",
          "Silence is not evidence" in pan, True)
    check("panel warns that 4.02 removes a name", "4.02" in pan, True)
    check("no fallers renders nothing",
          rn._dislocation_panel({"fell_30pct": 0, "fundamentals_intact": 0}), "")
    t = rn.TEMPLATE
    check("a dislocation badge exists", 'class="dis${' in t, True)
    check("the badge shows the actual 6m return, not a fixed label",
          "esc(r.dis_6m" in t, True)
    check("it renders for EVERY faller, not only the qualifying ones",
          "r.dis_fell\n      ?" in t or "r.dis_fell" in t, True)
    check("explained falls get their own style", ".dis.expl{" in t, True)
    check("a 'fell >30%' chip exists", "Fell &gt;30% (6m)" in t, True)
    check("and a second chip narrows to the unexplained", "disQual" in t, True)
    check("the wide filter is wired to dis_fell", "fDis && !r.dis_fell" in t, True)
    check("the narrow filter is wired to dis", "fDisQ && !r.dis" in t, True)
    check("the 6-month return is in the metrics drawer",
          '"Return 6m"' in open("src/render.py").read(), True)

    # The technical input the shape test depends on.
    idx = pd.bdate_range("2025-01-01", periods=140)
    # Flat, then one violent month, then flat: a cliff.
    vals = [100.0] * 80 + [100.0 - 2.0 * i for i in range(21)] + [58.0] * 39
    cliff = pd.Series(vals[:140], index=idx)
    tt = ta.compute(pd.DataFrame({"Close": cliff, "High": cliff, "Low": cliff,
                                  "Volume": [1e6] * 140}, index=idx))
    check("the worst month inside 6m is measured",
          round(tt["worst_month_in_6m"], 2) >= 0.35, True)
    rise = pd.Series([100.0 * (1.002 ** i) for i in range(140)], index=idx)
    tr = ta.compute(pd.DataFrame({"Close": rise, "High": rise, "Low": rise,
                                  "Volume": [1e6] * 140}, index=idx))
    check("a series that only rose has no down month",
          tr["worst_month_in_6m"], 0.0)



def test_events():
    """Structured event feeds: 8-K item codes, quakes, and what they override."""
    from src import events as ev, dislocation as ds
    cfg = yaml.safe_load(open("config/thresholds.yml"))["dislocation"]

    sub = {"filings": {"recent": {
        "filingDate": ["2026-07-02", "2026-06-11", "2026-05-30", "2026-01-04"],
        "form": ["8-K", "8-K", "10-Q", "8-K"],
        "items": ["5.02,9.01", "2.06", "", "4.02"],
        "accessionNumber": ["a1", "a2", "a3", "a4"]}}}

    recent = ev.parse_submissions(sub, "2026-06-01")
    check("only filings inside the window are returned", len(recent), 2)
    check("a 10-Q is not an 8-K", all(e["form"] == "8-K" for e in recent), True)
    check("5.02 is read as an officer departure",
          "Departure" in recent[0]["labels"][0], True)
    check("and mapped to cause #5", recent[0]["causes"], [5])
    check("9.01 (exhibits) is carried but not given a cause",
          "9.01" in recent[0]["codes"] and len(recent[0]["labels"]) == 1, True)
    check("2.06 maps to a physical shock", recent[1]["causes"], [2])

    # The most important code in the file. A company that has told the market
    # its own past accounts cannot be relied on has NOT been dislocated — the
    # "intact fundamentals" this screen measured are, by its own admission,
    # not intact.
    allf = ev.parse_submissions(sub, "2025-01-01")
    disq = next((e["disqualifies"] for e in allf if e["disqualifies"]), None)
    check("a 4.02 disqualifies", bool(disq), True)
    check("and says why", "cannot be relied on" in disq, True)
    for code in ("1.03", "3.01", "4.02"):
        check(f"{code} carries a disqualifier",
              bool(ev.ITEM_CODES[code].get("disqualifies")), True)
    check("a routine 8.01 does not",
          ev.ITEM_CODES["8.01"].get("disqualifies"), None)

    # Quakes are matched to a market by box, and small ones are ignored.
    q = ev.parse_quakes({"features": [
        {"properties": {"mag": 7.1, "place": "off Sumatra", "time": 1755000000000},
         "geometry": {"coordinates": [100.2, -2.1, 30]}},
        {"properties": {"mag": 5.2, "place": "minor", "time": 1755000000000},
         "geometry": {"coordinates": [139.0, 35.0, 10]}},
        {"properties": {"mag": 6.5, "place": "Honshu", "time": 1755000000000},
         "geometry": {"coordinates": [139.0, 35.0, 10]}}]})
    check("a magnitude 7.1 off Sumatra lands in Indonesia",
          [x["magnitude"] for x in q.get("ID", [])], [7.1])
    check("a 5.2 is below the bar and dropped", len(q.get("JP", [])), 1)
    check("a 6.5 in Honshu lands in Japan",
          q["JP"][0]["magnitude"], 6.5)
    check("quakes carry cause #2", q["JP"][0]["causes"], [2])

    # An OBSERVED event must collapse the inferred shortlist.
    base = dict(return_6m=-0.38, rs_vs_market_index_6m=-0.31,
                worst_month_in_6m=0.26, vol20_over_vol50=1.6,
                revenue_growth_1y=0.08, eps_growth_1y=0.04,
                free_cash_flow_ttm=120.0, net_debt_to_ebitda=1.2,
                loss_years_in_10=0, accruals_ratio=0.01)
    inferred = ds.assess(base, cfg)
    seen = ds.assess(dict(base, _events={"observed_causes": [5],
                                         "events": [{"source": "SEC 8-K"}]}), cfg)
    check("without a feed the app infers a shortlist",
          len(inferred["candidate_causes"]) > 1, True)
    check("with an observed 5.02 it collapses to one",
          [c["n"] for c in seen["candidate_causes"]], [5])
    check("and is graded as observed rather than inferred",
          seen["evidence_grade"], "observed")
    check("the inferred shortlist is kept as context",
          len(seen["inferred_shortlist"]) > 1, True)

    # A disqualifying filing must remove the name, not annotate it.
    bad = ds.assess(dict(base, _events={"observed_causes": [5],
                                        "disqualifies": "accounts unreliable"}), cfg)
    check("a disqualifying filing removes the name", bad["qualifies"], False)
    check("even though its fundamentals tested intact",
          bad["fundamentals"]["intact"], True)

    # Silence from the feeds is not evidence of nothing happening.
    quiet = ev.explain.__doc__
    none_ = ds.assess(dict(base, _events={"events": [], "note": "nothing seen"}), cfg)
    check("no events -> still inferred", none_["evidence_grade"], "inferred")
    check("and the feed note travels with it", none_["feed_note"], "nothing seen")

    # No key, no commercial feed — and that must not raise.
    import os as _os
    saved = _os.environ.pop("FINNHUB_API_KEY", None)
    check("the commercial feed is skipped without a key",
          ev.news_events(None, "AAPL"), [])
    if saved:
        _os.environ["FINNHUB_API_KEY"] = saved

    # The CIK regression: CompanyRecord has no `cik` attribute, so resolving it
    # from the record would silently disable the 8-K feed on every name.
    check("CompanyRecord still has no cik field",
          hasattr(CompanyRecord(ticker="X", market="US"), "cik"), False)
    src = open("src/run.py").read()
    check("run.py resolves CIK through the provider",
          "edgar.cik_for(t)" in src, True)
    check("and shouts if none resolve", "NO CIK resolved" in src, True)



def test_rsi_reading():
    """RSI read against the regime, not against a fixed 30/70 table."""
    # The same number means opposite things in different markets. This is the
    # whole reason the reading is contextual: a static table would call RSI 45
    # in a bull run "bearish zone" when it is the pullback you buy.
    check("45 in an uptrend is a pullback to buy",
          "pullback" in ta.rsi_reading(45, "uptrend")["label"], True)
    check("45 in a range is the bearish zone",
          ta.rsi_reading(45, "ranging")["label"], "bearish zone")
    check("45 in a downtrend is the trend intact",
          ta.rsi_reading(45, "downtrend")["label"], "downtrend intact")

    check("72 in a range is plainly overbought",
          ta.rsi_reading(72, "ranging")["label"], "overbought")
    check("72 in an uptrend is normal, and says so",
          "normal in an uptrend" in ta.rsi_reading(72, "uptrend")["label"], True)
    check("28 in a downtrend is normal, and says so",
          "normal in a downtrend" in ta.rsi_reading(28, "downtrend")["label"], True)
    check("28 in an uptrend is a warning, not a bargain",
          "unusually weak" in ta.rsi_reading(28, "uptrend")["label"], True)
    check("62 in a downtrend flags possible trend change",
          "unusually strong" in ta.rsi_reading(62, "downtrend")["label"], True)

    # The trap the user's own note calls out: the level is not the signal.
    check("oversold says to wait for the cross back above 30",
          "cross back ABOVE 30" in ta.rsi_reading(25, "ranging")["note"], True)
    check("overbought says to wait for the cross back below 70",
          "cross back BELOW 70" in ta.rsi_reading(75, "ranging")["note"], True)
    check("a deep downtrend reading refuses to call it a buy",
          "does not end it" in ta.rsi_reading(15, "downtrend")["note"], True)

    check("no RSI yields no label", ta.rsi_reading(None)["label"], None)

    # Regime detection from the moving-average structure.
    check("above both MAs with 50>200 is an uptrend",
          ta.rsi_regime(110, 105, 100), "uptrend")
    check("below both with 50<200 is a downtrend",
          ta.rsi_regime(90, 95, 100), "downtrend")
    check("mixed structure is ranging", ta.rsi_regime(101, 95, 100), "ranging")
    check("no 200-day means ranging, not a guess",
          ta.rsi_regime(100, 100, None), "ranging")

    # Divergence: price lower low, RSI higher low.
    # Bullish divergence needs the SECOND low to arrive gently: a violent fall
    # to a low, a bounce, then a slow drift to a marginally lower low. The slow
    # drift is what leaves RSI higher at the deeper price.
    idx = pd.bdate_range("2025-01-01", periods=120)
    shape = ([100.0] * 60                                    # quiet lead-in
             + [100.0 - 3.4 * i for i in range(15)]          # crash to ~50
             + [49.0 + 1.05 * i for i in range(20)]          # strong bounce
             + [70.0 - 0.95 * i for i in range(25)])         # slow drift under it
    p = pd.Series(shape[:120], index=idx)
    r = ta.rsi(p, 14)
    check("the second low really is lower",
          p.iloc[-1] < p.iloc[60:90].min(), True)
    check("a marginal new low reached slowly is bullish divergence",
          ta.rsi_divergence(p, r), "bullish")
    flat = pd.Series([100.0] * 120, index=idx)
    check("a flat series has no divergence",
          ta.rsi_divergence(flat, ta.rsi(flat, 14)), None)
    check("too little history yields no divergence",
          ta.rsi_divergence(p.iloc[:20], r.iloc[:20]), None)

    # And it must reach the page.
    d = ta.rsi_reading(38, "downtrend", "bullish")
    check("divergence is labelled", d["divergence"], "bullish divergence")
    check("and explained", "lower low" in d["divergence_note"], True)
    src = open("src/render.py").read()
    check("the label is shown beside the number in the drawer",
          'm.get("rsi_label")' in src, True)
    check("the regime and note are shown too", '"RSI context"' in src, True)
    check("divergence has its own row", '"RSI divergence"' in src, True)

    # End to end through compute().
    rise = pd.Series([100.0 * (1.004 ** i) for i in range(260)],
                     index=pd.bdate_range("2025-01-01", periods=260))
    t = ta.compute(pd.DataFrame({"Close": rise, "High": rise, "Low": rise,
                                 "Volume": [1e6] * 260}, index=rise.index))
    check("compute() classifies the regime", t["rsi_regime"], "uptrend")
    check("compute() attaches a label", bool(t["rsi_label"]), True)
    check("and a note", bool(t["rsi_note"]), True)



def test_display_metric_contract():
    """Every metric the renderer displays must also be persisted.

    A row merged in from the other region's last run has no live metrics dict,
    so it falls back to the stored `key_metrics`. If the display reads a key
    the store never wrote, that row shows "—" while an identical live row
    shows a number — and a dash is indistinguishable from genuinely missing
    data, which is the one thing this app is built never to do.

    This drifted twice (returns, then the RSI reading), so it is now checked
    structurally rather than by memory.
    """
    import inspect
    src = inspect.getsource(rn.build_payload)
    used = set(re.findall(r'm\.get\(\s*"([a-z0-9_]+)"', src))
    declared = set(rn.DISPLAY_METRICS)
    missing = sorted(used - declared)
    check("every displayed metric is persisted", missing, [])
    check("the contract is not empty", len(declared) > 20, True)

    # The two that actually drifted, pinned by name.
    for k in ("return_6m", "return_3m", "return_12m", "worst_month_in_6m",
              "rsi_label", "rsi_regime", "rsi_note", "rsi_divergence"):
        check(f"{k} is persisted", k in declared, True)

    # screens.py must use the shared list, not its own copy.
    sc_src = open("src/screens.py").read()
    check("screens.py imports the contract",
          "from .render import DISPLAY_METRICS" in sc_src, True)
    check("and no longer keeps a private list",
          'key_metrics = {k: m.get(k) for k in (\n' in sc_src, False)

    # End to end: a stored row must render the same fields as a live one.
    rec = CompanyRecord(ticker="Z1", market="HK", currency="HKD",
                        years=[mkyear(2025 - i, revenue=1000.0, net_income=100.0,
                                      eps_diluted=1.0, shares_diluted=100.0,
                                      total_equity=800.0, total_assets=1200.0,
                                      cfo=150.0, capex=30.0,
                                      current_assets=400.0,
                                      current_liabilities=200.0,
                                      operating_income=140.0,
                                      pretax_income=130.0) for i in range(4)])
    rec.price, rec.market_cap = 10.0, 1000.0
    rec.technicals = {"rsi_14": 48.6, "return_6m": -0.12, "return_3m": 0.04,
                      "return_12m": 0.15, "worst_month_in_6m": 0.09,
                      "rsi_label": "bearish zone", "rsi_regime": "ranging",
                      "rsi_note": "downward momentum, but moderating",
                      "price_above_sma200": 1, "rs_vs_market_index_6m": 0.15}
    m = mx.compute_metrics(rec)
    th = yaml.safe_load(open("config/thresholds.yml"))
    out = sc.screen_universe([rec], {"Z1": m}, th, {}, cycle=None)
    stored = out["results"]["Z1"]["metrics"]
    check("the stored copy carries the 6-month return",
          stored.get("return_6m"), -0.12)
    check("and the RSI label", stored.get("rsi_label"), "bearish zone")

    # Render from the STORED copy only, exactly as a merged row does.
    rows = rn.build_payload({"Z1": out["results"]["Z1"]}, {}, out)
    km = rows[0]["key_metrics"]
    check("a merged row shows its 6-month return, not a dash",
          km["Return 6m"], "-12.0%")
    check("a current stored row raises no staleness warning",
          any("earlier build" in w for w in rows[0]["warnings"]), False)

    # A row written by an older build has the key ABSENT rather than None.
    # That is the only way to tell "we never stored this" from "we stored it
    # and it was genuinely missing", and the row must say which it is.
    old_row = dict(out["results"]["Z1"])
    old_row["metrics"] = {k: v for k, v in stored.items()
                          if k not in ("return_6m", "rsi_label")}
    old_rows = rn.build_payload({"Z1": old_row}, {}, out)
    check("an older stored row is flagged, not silently blank",
          any("earlier build" in w for w in old_rows[0]["warnings"]), True)
    check("and the warning says how to fix it",
          any("re-run this market" in w for w in old_rows[0]["warnings"]), True)
    # A LIVE row must never be flagged, however sparse its metrics.
    live = rn.build_payload({"Z1": old_row}, {"Z1": {"pe_ttm": 9.0}}, out)
    check("a live row is never called stale",
          any("earlier build" in w for w in live[0]["warnings"]), False)
    check("and its RSI reading", "bearish zone" in km["RSI(14)"], True)
    check("and its RSI context", "ranging" in km["RSI context"], True)



def test_malaysia_market():
    """Bursa Malaysia joins the Asia job; the dislocation screen needs no change."""
    from src import run as rn_run, dislocation as ds
    uni = yaml.safe_load(open("config/universe.yml"))
    my = uni["markets"]["MY"]

    check("Malaysia is in the universe", bool(my), True)
    check("index is the KLCI", my["index_ticker"], "^KLSE")
    check("Yahoo suffix is .KL", my["ticker_suffix"], ".KL")
    check("Bursa codes are padded to 4 digits", my["ticker_pad"], 4)
    check("currency is MYR", my["currency"], "MYR")
    check("an FX pair exists for MYR", uni["fx_pairs"]["MYR"], "MYRUSD=X")
    check("fundamentals come from Yahoo, like the other Asian markets",
          my["fundamentals_provider"], "yahoo")
    check("min_expected suits a 30-name index", my["min_expected"] <= 30, True)
    check("it runs in the Asia job", "MY" in rn_run.REGION_MARKETS["asia"], True)
    check("and in a full rebuild", "MY" in rn_run.REGION_MARKETS["all"], True)
    check("the .KL suffix maps back to MY",
          rn_run.SUFFIX_MARKET[".KL"], "MY")
    check("the market has a display label", rn.MARKET_LABELS["MY"], "Bursa Malaysia")

    # The column-name regression. Bursa's Wikipedia table heads its ticker
    # column "Stock Code", which the generic fetcher did not recognise — it
    # would have returned nothing and fallen back to the 30-name seed while
    # LOOKING like a successful fetch. Same failure as the S&P 500 one.
    src = open("src/run.py").read()
    check("the fetcher accepts 'Stock Code'", '"stock code"' in src, True)

    import pandas as _pd
    from io import StringIO
    html = ("<table><tr><th>Constituent Name</th><th>Stock Code</th></tr>"
            + "".join(f"<tr><td>Co {i}</td><td>{1000 + i}</td></tr>"
                      for i in range(30)) + "</table>")
    got = rn_run.wikipedia_constituents.__wrapped__ if hasattr(
        rn_run.wikipedia_constituents, "__wrapped__") else None
    # Exercise the parsing directly against the table shape.
    tables = _pd.read_html(StringIO(html))
    cols = {str(c).strip().lower(): c for c in tables[0].columns}
    col = next((cols[k] for k in ("code", "ticker", "symbol", "ticker symbol",
                                  "stock code", "stock symbol", "sehk code",
                                  "scrip") if k in cols), None)
    check("a Bursa-shaped table resolves its ticker column", col is not None, True)
    syms = [f"{str(v).strip().zfill(4)}.KL" for v in tables[0][col].tolist()]
    check("codes are padded and suffixed", syms[0], "1000.KL")
    check("all 30 parse", len(syms), 30)

    # Every seed is a well-formed Yahoo symbol.
    for t in my["seed_fallback"]:
        check(f"{t} is a valid Bursa symbol",
              bool(re.match(r"^\d{4}\.KL$", str(t))), True)
    check("seeds are unique", len(set(my["seed_fallback"])),
          len(my["seed_fallback"]))
    check("Maybank is in the seeds", "1155.KL" in my["seed_fallback"], True)

    # The dislocation screen is market-agnostic — a Malaysian name works with
    # no code change at all, because it compares against its OWN index.
    m = dict(return_6m=-0.36, rs_vs_market_index_6m=-0.28,
             worst_month_in_6m=0.24, vol20_over_vol50=1.4,
             revenue_growth_1y=0.05, eps_growth_1y=0.02,
             free_cash_flow_ttm=80.0, net_debt_to_ebitda=1.5,
             loss_years_in_10=0, accruals_ratio=0.02)
    a = ds.assess(m, yaml.safe_load(open("config/thresholds.yml"))["dislocation"])
    check("a Malaysian faller is assessed like any other", a["qualifies"], True)
    check("and scoped against its own market", a["scope"], "name")


def test_synopsis():
    """A per-stock synopsis, and the persistence trap it could have walked into."""
    from src import synopsis as sy

    # ---- the contract ---------------------------------------------------
    # The synopsis is rendered for merged rows too, and a merged row carries
    # only DISPLAY_METRICS. Reading anything else would render fully for the
    # region that just ran and degrade silently everywhere else.
    undeclared = sorted(set(sy.SYNOPSIS_FIELDS) - set(rn.DISPLAY_METRICS))
    check("every field the synopsis reads is persisted", undeclared, [])

    try:
        sy._g({"revenue_growth_1y": 0.2}, "revenue_growth_1y")
        raised = False
    except KeyError:
        raised = True
    check("reading an undeclared metric fails loudly", raised, True)

    src = open("src/synopsis.py").read()
    body = src.split("def _pct", 1)[1]          # everything after the accessor
    check("no metric is read behind the accessor's back",
          re.findall(r'\bm\.get\(\s*"', body), [])

    # ---- trimming a feed blurb -------------------------------------------
    long = ("Acme Berhad operates as a plantation company in Malaysia. "
            "The company was incorporated in 1971 and is headquartered in "
            "Kuala Lumpur. It also engages in property development, "
            "manufacturing, and a great many other activities besides.")
    t = sy.trim_description(long, 120)
    check("a long blurb is cut", len(t) <= 121, True)
    check("and cut at a sentence end, not mid-word", t.endswith("."), True)
    check("the first fact survives the cut", t.startswith("Acme Berhad"), True)
    check("a short blurb is left alone",
          sy.trim_description("Makes things."), "Makes things.")
    check("no blurb is not an error", sy.trim_description(None), "")

    # ---- a realistic row --------------------------------------------------
    rec = CompanyRecord(ticker="SYN", market="HK", currency="HKD",
                        sector="Energy", industry="Oil & Gas Integrated",
                        years=[mkyear(2025 - i, revenue=1000.0, net_income=100.0,
                                      eps_diluted=1.0, shares_diluted=100.0,
                                      total_equity=800.0, total_assets=1200.0,
                                      cfo=150.0, capex=30.0, current_assets=400.0,
                                      current_liabilities=200.0,
                                      operating_income=140.0, pretax_income=130.0)
                               for i in range(6)])
    rec.price, rec.market_cap = 8.0, 800.0
    rec.business_summary = long
    rec.technicals = {"rsi_14": 31.4, "rsi_label": "oversold", "rsi_regime": "ranging",
                      "return_6m": -0.34, "return_12m": -0.21,
                      "pct_below_52w_high": 0.38, "price_above_sma200": 0,
                      "rs_vs_market_index_6m": -0.26}
    m = mx.compute_metrics(rec)
    th = yaml.safe_load(open("config/thresholds.yml"))
    out = sc.screen_universe([rec], {"SYN": m}, th, {}, cycle=None)
    res = out["results"]["SYN"]

    check("the business description is persisted with the row",
          res["business_summary"].startswith("Acme Berhad"), True)
    check("and trimmed before it is stored",
          len(res["business_summary"]) <= sy.MAX_DESCRIPTION_CHARS, True)
    check("the industry travels with it", res["industry"], "Oil & Gas Integrated")

    s = sy.build(res, m, dict(rn.FRAMEWORKS), len(rn.FRAMEWORKS))
    text = " ".join(s["numbers"])
    check("the synopsis leads with the framework verdict",
          "frameworks" in s["numbers"][0], True)
    check("it names the count out of the real total",
          f"of {len(rn.FRAMEWORKS)} value frameworks" in s["numbers"][0], True)
    check("it says whether the technicals agree",
          "technical timing test" in s["numbers"][0], True)
    check("it quotes the valuation", "priced at" in text, True)
    check("it quotes return on equity", "return on equity" in text, True)
    check("it reads the price action", "over six months" in text, True)
    check("and the RSI in words, not just a number",
          "RSI at 31" in text and "oversold" in text, True)
    check("the description is reported, not generated", s["what_source"], "feed")
    check("it is brief", len(s["numbers"]) <= 10, True)
    check("the one-liner fits a tooltip", len(s["one_liner"]) < 100, True)

    # ---- the parity test that matters -------------------------------------
    # Same row, rendered from the stored subset instead of the live metrics.
    # If these two ever differ, half the published table is reading a poorer
    # synopsis than the other half and nothing on the page would say so.
    stored_only = {k: res["metrics"].get(k) for k in rn.DISPLAY_METRICS}
    s2 = sy.build(res, stored_only, dict(rn.FRAMEWORKS), len(rn.FRAMEWORKS))
    check("a merged row gets exactly the same synopsis as a live one",
          s2["numbers"], s["numbers"])
    check("including the one-liner", s2["one_liner"], s["one_liner"])

    # ---- graceful degradation ---------------------------------------------
    bare = sy.build({"ticker": "X", "frameworks": {}}, {}, {}, 11)
    check("an empty row still produces a verdict", len(bare["numbers"]) >= 1, True)
    check("and says the ratios are missing rather than printing dashes",
          "none of the headline valuation ratios" in " ".join(bare["numbers"]), True)
    nodesc = sy.build({"sector": "Utilities", "frameworks": {}}, {}, {}, 11)
    check("with no blurb it falls back to the classification",
          nodesc["what_source"], "classification")
    check("and does not invent a description",
          "Utilities" in nodesc["what"], True)

    # ---- the flags ---------------------------------------------------------
    flagged = dict(res)
    flagged["reflexive"] = {"stage": "DE", "label": "the moment of truth",
                            "late": True}
    flagged["dislocation"] = {"return_6m": -0.34, "qualifies": True,
                              "evidence_grade": "observed",
                              "observed_causes": [{"n": 2,
                                                   "name": "Natural calamity or physical shock"}]}
    f = sy.build(flagged, m, dict(rn.FRAMEWORKS), len(rn.FRAMEWORKS))
    ftext = " ".join(f["numbers"])
    check("a late reflexive stage is called out", "Soros stage DE" in ftext, True)
    check("and named as risk, not opportunity",
          "risk rather than as an opportunity" in ftext, True)
    check("the 30% fall is stated in the prose", "It fell 34%" in ftext, True)
    check("an observed cause is distinguished from an inferred one",
          "a feed actually reported" in ftext, True)
    check("and the staleness caveat rides with it",
          "may simply be stale" in ftext, True)

    dq = dict(res)
    dq["dislocation"] = {"return_6m": -0.51, "qualifies": False,
                         "disqualified_by_filing": "8-K item 4.02"}
    dtext = " ".join(sy.build(dq, m, dict(rn.FRAMEWORKS), 11)["numbers"])
    check("a disqualifying filing is not dressed up as a bargain",
          "takes it off the dislocation list" in dtext, True)

    explained = dict(res)
    explained["dislocation"] = {"return_6m": -0.33, "qualifies": False}
    etext = " ".join(sy.build(explained, m, dict(rn.FRAMEWORKS), 11)["numbers"])
    check("a fall the accounts explain says so",
          "business problem rather than a price accident" in etext, True)

    # ---- it reaches the page ----------------------------------------------
    rows = rn.build_payload({"SYN": res}, {}, out)
    check("the payload carries the synopsis", bool(rows[0]["syn"]["numbers"]), True)
    html = rn.TEMPLATE
    check("the drawer renders it", 'class="syn"' in html, True)
    check("and says where the words came from", "No forecast" in html, True)
    check("the table tooltip carries the one-liner",
          "r.syn.one_liner" in html, True)


def test_lynch_categories():
    """Lynch's six categories, and the bars that follow from them."""
    from src import lynch as ly

    # --- the classifier ----------------------------------------------------
    # A cyclical stays a cyclical however fast it grew last cycle: that growth
    # rate is a position in the cycle, not a trend.
    c = ly.classify({"eps_cagr_5y": 0.28}, sector="Energy",
                    industry="Oil & Gas Integrated")
    check("an energy name is cyclical despite 28% growth", c["category"], "cyclical")
    c = ly.classify({"eps_cagr_5y": 0.30}, sector="Consumer Cyclical",
                    industry="Auto Manufacturers")
    check("so is a carmaker", c["category"], "cyclical")

    # Trouble that is recent AND visible in the price outranks the industry.
    c = ly.classify({"loss_years_in_3": 1, "pct_below_52w_high": 0.55,
                     "eps_cagr_5y": -0.4}, sector="Energy", industry="Oil")
    check("a recent loss plus a collapse is a turnaround", c["category"], "turnaround")
    check("and the reason is stated", "off the 52-week high" in c["why"], True)

    c = ly.classify({"eps_cagr_5y": 0.05, "price_to_tangible_book": 0.6},
                    sector="Industrials", industry="Conglomerates")
    check("priced under tangible book is an asset play", c["category"], "asset_play")
    c = ly.classify({"eps_cagr_5y": 0.02, "ncav_to_market_cap": 0.9},
                    sector="Technology", industry="Software")
    check("so is a name whose current assets cover the price",
          c["category"], "asset_play")

    check("20%+ growth is a fast grower",
          ly.classify({"eps_cagr_5y": 0.24}, "Technology", "Software")["category"],
          "fast_grower")
    check("10–20% is a stalwart",
          ly.classify({"eps_cagr_5y": 0.12}, "Consumer Defensive",
                      "Packaged Foods")["category"], "stalwart")
    check("under 10% is a slow grower",
          ly.classify({"eps_cagr_5y": 0.04, "dividend_yield": 0.05},
                      "Utilities", "Utilities - Regulated")["category"],
          "slow_grower")
    check("no growth rate at all is unclassified, not a pass",
          ly.classify({}, "Technology", "Software")["category"], "unclassified")
    check("and the strictest bands are applied to it",
          "strictest" in ly.classify({}, "T", "S")["why"], True)
    check("every category has a stated rationale",
          sorted(ly.RATIONALE) == sorted(ly.CATEGORIES), True)

    # --- the single most expensive mistake Lynch names ---------------------
    w = ly.peak_earnings_warning({"pe_ttm": 8.0, "eps_vs_5y_avg": 1.5}, "cyclical")
    check("a cyclical on peak earnings and a low P/E is flagged", bool(w), True)
    check("and flagged as the danger, not the bargain", "peak" in w, True)
    w2 = ly.peak_earnings_warning({"pe_ttm": 40.0, "eps_vs_5y_avg": 0.5}, "cyclical")
    check("the inversion also works the other way",
          "interesting end of the cycle" in (w2 or ""), True)
    check("and it says nothing about a non-cyclical",
          ly.peak_earnings_warning({"pe_ttm": 8.0, "eps_vs_5y_avg": 1.5},
                                   "stalwart"), None)

    # --- the config ---------------------------------------------------------
    th = yaml.safe_load(open("config/thresholds.yml"))
    lyn = th["lynch"]
    check("Lynch runs 9 tests", len(lyn["tests"]), 9)
    check("and needs 6 of them", lyn["min_tests_passed"], 6)
    check("the balance sheet bar is Lynch's 75/25 rule",
          lyn["tests"]["leverage"]["threshold"], 0.33)
    check("the growth floor is the sweet spot, not 10%",
          lyn["tests"]["growth_floor"]["threshold"], 0.15)
    check("and the ceiling is his 30% caution",
          lyn["tests"]["growth_ceiling"]["threshold"], 0.30)
    check("callable bank debt is tested separately from total debt",
          lyn["tests"]["debt_is_not_callable"]["metric"], "short_term_debt_share")
    check("cash generation has a floor",
          lyn["tests"]["cash_generation"]["metric"], "fcf_yield")
    val = lyn["tests"]["valuation"]
    check("the valuation test routes on the category", val["by"], "lynch_category")
    check("a stalwart is judged on growth plus yield",
          val["cases"]["stalwart"]["metric"], "growth_plus_yield_to_pe")
    check("a slow grower on its dividend",
          val["cases"]["slow_grower"]["metric"], "dividend_yield")
    check("a cyclical on where its earnings sit in the cycle",
          val["cases"]["cyclical"]["metric"], "eps_vs_5y_avg")
    check("a turnaround on whether it can pay what is due",
          val["cases"]["turnaround"]["metric"], "cash_to_short_term_debt")
    check("an asset play on what it owns",
          val["cases"]["asset_play"]["metric"], "price_to_tangible_book")
    for cat in ("cyclical", "turnaround", "asset_play"):
        check(f"the growth floor does not apply to a {cat}",
              lyn["tests"]["growth_floor"]["cases"][cat]["not_applicable"], True)

    # --- the Rule of 20 ------------------------------------------------------
    uni = {f"T{i}": {"pe_ttm": 12.0 + i} for i in range(9)}     # median 16
    r = ly.rule_of_20(uni, 3.0, market=None)
    check("the rule of 20 sums a median P/E and inflation", r["total"], 19.0)
    check("and reads under 20 as not expensive", r["verdict"], "cheap")
    check("it reports the median it used", r["median_pe"], 16.0)
    check("and how many names it was struck on", r["names"], 9)
    check("above 23 is expensive",
          ly.rule_of_20(uni, 8.0, market=None)["verdict"], "expensive")
    check("between the two is fair",
          ly.rule_of_20(uni, 5.0, market=None)["verdict"], "fair")
    # Half a sum is not a sum: with no inflation reading it must refuse rather
    # than quietly report the P/E as if it were the rule.
    nocpi = ly.rule_of_20(uni, None, market=None)
    check("with no inflation reading it declines to answer",
          nocpi["available"], False)
    check("but still says what it had", nocpi["median_pe"], 16.0)
    check("an empty universe is refused too",
          ly.rule_of_20({}, 3.0, market=None)["available"], False)
    # Absurd P/Es are excluded from the median rather than dragging it.
    junk = dict(uni); junk["JUNK"] = {"pe_ttm": 4000.0}; junk["NEG"] = {"pe_ttm": -8.0}
    check("a 4000x outlier does not move the median",
          ly.rule_of_20(junk, 3.0, market=None)["median_pe"], 16.0)
    panel20 = rn._lynch_panel({"stalwart": 3}, {},
                              ly.rule_of_20(uni, 3.0, market=None))
    check("the rule reaches the page", "Rule of 20: 19.0" in panel20, True)
    check("with the caveat that it is our median, not the index's",
          "cap-weighted" in panel20, True)

    # --- the metrics --------------------------------------------------------
    rec = CompanyRecord(ticker="LY", market="US", currency="USD",
                        sector="Consumer Defensive", industry="Packaged Foods")
    rec.years = [mkyear(2025 - i, revenue=1000.0, net_income=120.0 * (0.9 ** i),
                        eps_diluted=1.2 * (0.9 ** i), shares_diluted=100.0,
                        total_equity=800.0, total_assets=1400.0,
                        total_debt=200.0, short_term_debt=40.0,
                        long_term_debt=160.0, cash_and_equivalents=300.0,
                        current_assets=600.0, current_liabilities=250.0,
                        inventory=100.0, cfo=180.0, capex=40.0,
                        operating_income=170.0, pretax_income=160.0,
                        dividends_paid=-30.0) for i in range(8)]
    rec.price, rec.market_cap = 24.0, 2400.0
    rec.dividend_yield = 0.025
    rec.insider_ownership, rec.institutional_ownership = 0.04, 0.62
    rec.first_trade_date = "1986-03-13"
    m = mx.compute_metrics(rec)

    check("short-term debt is split out as a share of the total",
          m["short_term_debt_share"], 0.2)
    check("long-term debt is measured against equity",
          m["long_term_debt_to_equity"], 0.2)
    check("net cash per share is computed", m["net_cash_per_share"], 1.0)
    check("and the price of the business net of that cash",
          m["price_ex_cash"], 23.0)
    check("the P/E is restruck on the ex-cash price",
          round(m["pe_ex_cash"], 2), round(23.0 / 1.2, 2))
    check("cash is measured against the debt that falls due first",
          m["cash_to_short_term_debt"], 7.5)
    check("current earnings are placed against their own five-year average",
          round(m["eps_vs_5y_avg"], 3), 1.221)
    check("the cash-flow record is a share of the years evaluated",
          m["cfo_positive_share_10y"], 1.0)
    check("listing age comes from the first trade date",
          m["listing_age_years"] > 39, True)
    check("price to sales is available for the relative-value route",
          m["price_to_sales"], 2.4)
    check("PEGY divides the P/E by growth plus yield",
          m["pegy_ratio"] is not None, True)
    check("and Lynch's own multiple form is the inverse",
          round(m["pegy_ratio"] * m["growth_plus_yield_to_pe"], 6), 1.0)
    check("the category is attached to the metrics",
          m["lynch_category"] in ly.CATEGORIES, True)

    # A stalwart with a dividend: the plain PEG route and the PEGY route give
    # different answers, which is the entire reason the category exists.
    # The yield has to be substantial to flip the answer, which is itself worth
    # knowing: Lynch's 1.5 bar was written when P/Es were 10, and on a modern
    # large cap it is demanding. That is why the PEG route survives as an
    # alternative satisfying route rather than being replaced outright.
    stal = {"lynch_category": "stalwart", "history_years": 8,
            "pe_ttm": 12.0, "peg_ratio": 1.2, "eps_cagr_5y": 0.10,
            "growth_plus_yield_to_pe": (10.0 + 8.0) / 12.0,
            "dividend_yield": 0.08}
    t_pegy = sc.evaluate_test("valuation", val, stal)
    check("a stalwart is scored on growth plus yield, not on PEG",
          t_pegy["metric"], "growth_plus_yield_to_pe")
    check("and passes on it where a plain PEG of 1.5 would have failed",
          t_pegy["result"], True)
    check("the panel says which yardstick was used",
          "stalwart ratio" in (t_pegy.get("note") or ""), True)

    fast = {"lynch_category": "fast_grower", "history_years": 5,
            "peg_ratio_lynch": 0.8}
    t_fast = sc.evaluate_test("valuation", val, fast)
    check("a fast grower still runs on PEG", t_fast["metric"], "peg_ratio_lynch")
    check("and five years of statements is now enough to run it",
          t_fast["result"], True)

    # --- not applicable is not a failure ------------------------------------
    # Every non-category field is supplied, so the only tests leaving the
    # denominator are the ones Lynch does not ask of a cyclical.
    cyc_m = {"lynch_category": "cyclical", "history_years": 10,
             "eps_cagr_5y": 0.02, "short_term_debt_share": 0.2,
             "insider_ownership": 0.03, "institutional_ownership": 0.5}
    t_na = sc.evaluate_test("growth_floor", lyn["tests"]["growth_floor"], cyc_m)
    check("the growth floor on a cyclical is not applicable",
          t_na["not_applicable"], True)
    check("it is not scored as a failure", t_na["result"], None)
    check("and it leaves the denominator", t_na["insufficient"], True)
    check("with a reason a person can read",
          "cycle" in (t_na.get("note") or ""), True)

    fwr = sc.run_framework("lynch", lyn, cyc_m)
    check("the denominator drops for a cyclical", fwr["effective_total"] < 9, True)
    check("and the bar scales down with it", fwr["required"] < 6, True)
    check("the two growth-band tests do not apply to a cyclical",
          fwr["n_not_applicable"], 2)
    check("this is NOT reported as a short data history",
          fwr["limited_history"], False)

    # --- missing feed fields are not failures either ------------------------
    t_miss = sc.evaluate_test("skin_in_the_game",
                              lyn["tests"]["skin_in_the_game"], {"history_years": 8})
    check("an absent ownership figure is not scored against the company",
          t_miss["not_applicable"], True)
    check("and says whose gap it is",
          "feed" in (t_miss.get("note") or ""), True)


def test_schloss_deep_value():
    """Assets first: the balance sheet, the ten-year chart, and the regime."""
    th = yaml.safe_load(open("config/thresholds.yml"))
    sch = th["schloss"]
    check("Schloss runs 11 tests", len(sch["tests"]), 11)
    check("and needs 7 of them", sch["min_tests_passed"], 7)
    check("his hard rule is debt below equity",
          sch["tests"]["debt_below_equity"]["threshold"], 1.0)
    check("with a preference for minimal long-term debt",
          sch["tests"]["minimal_long_term_debt"]["threshold"], 0.30)
    check("book value must be real, not goodwill",
          sch["tests"]["book_is_real"]["metric"], "goodwill_to_assets")
    check("the false-bottom test uses the ten-year low",
          sch["tests"]["no_false_bottom"]["metric"], "pct_above_10y_low")
    check("the 20-year history bar is tested on listing age",
          sch["tests"]["long_operating_history"]["threshold"], 20)
    check("and is skipped rather than failed when the feed has no date",
          sch["tests"]["long_operating_history"]["skip_if_missing"], True)
    check("net-net satisfies the valuation test on its own",
          sch["tests"]["cheap_against_assets"]["alt_metric"], "ncav_to_market_cap")

    # --- the ten-year window ------------------------------------------------
    idx = pd.date_range("2016-01-01", periods=2600, freq="B")
    # 125 down to 60, but it traded at 20 early on: the classic false bottom.
    path = np.concatenate([
        np.linspace(20, 125, 1600), np.linspace(125, 60, 1000)])
    close = pd.Series(path[:len(idx)], index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99,
                       "Close": close, "Volume": np.full(len(idx), 1e6)})
    t = ta.compute(df)
    check("the ten-year low is found, far below the 52-week one",
          t["low_10y"] < 30, True)
    check("and the price is far above it", t["pct_above_10y_low"] > 1.0, True)
    check("so the false-bottom test fails it",
          t["pct_above_10y_low"] <= sch["tests"]["no_false_bottom"]["threshold"],
          False)
    check("while the 52-week view alone would have called it a collapse",
          t["pct_below_52w_high"] > 0.2, True)
    check("the position inside the decade is reported too",
          0.0 <= t["price_in_10y_range"] <= 1.0, True)

    # --- the regime switch ---------------------------------------------------
    rich = {f"T{i}": {"price_to_tangible_book": 3.0 + i} for i in range(40)}
    reg = sc.set_value_regime(rich, sch)
    check("with nothing below book, Schloss switches to relative value",
          reg["regime"], "relative_value")
    check("and every row carries the regime",
          rich["T0"]["value_regime"], "relative_value")
    t_rel = sc.evaluate_test("cheap_against_assets",
                             sch["tests"]["cheap_against_assets"],
                             {**rich["T0"], "price_to_sales": 0.4,
                              "history_years": 10})
    check("the valuation test switches to price to sales",
          t_rel["metric"], "price_to_sales")
    check("and passes a genuinely depressed one", t_rel["result"], True)
    check("the switch is explained on the row",
          "relative value" in (t_rel.get("note") or ""), True)

    cheap = {f"T{i}": {"price_to_tangible_book": 0.5 + (i % 4)} for i in range(40)}
    reg2 = sc.set_value_regime(cheap, sch)
    check("with book discounts on offer, he stays in deep-value mode",
          reg2["regime"], "deep_value_available")
    t_deep = sc.evaluate_test("cheap_against_assets",
                              sch["tests"]["cheap_against_assets"],
                              {**cheap["T0"], "history_years": 10})
    check("and the test stays on tangible book",
          t_deep["metric"], "price_to_tangible_book")

    # A net-net passes the valuation test even above book.
    t_nn = sc.evaluate_test("cheap_against_assets",
                            sch["tests"]["cheap_against_assets"],
                            {"price_to_tangible_book": 1.4,
                             "ncav_to_market_cap": 1.2,
                             "value_regime": "deep_value_available",
                             "history_years": 10})
    check("a net-net passes on the alternative route", t_nn["result"], True)
    check("and is marked as having done so", t_nn["via_alt"], True)

    # --- end to end ----------------------------------------------------------
    rec = CompanyRecord(ticker="SCH", market="US", currency="USD",
                        sector="Industrials", industry="Conglomerates")
    rec.years = [mkyear(2025 - i, revenue=2000.0, net_income=60.0,
                        eps_diluted=0.6, shares_diluted=100.0,
                        total_equity=1200.0, total_assets=1800.0,
                        total_debt=300.0, short_term_debt=50.0,
                        long_term_debt=250.0, goodwill=50.0,
                        cash_and_equivalents=200.0, current_assets=900.0,
                        current_liabilities=400.0, total_liabilities=600.0,
                        cfo=140.0, capex=40.0, operating_income=90.0,
                        pretax_income=80.0, dividends_paid=-20.0)
                 for i in range(10)]
    rec.price, rec.market_cap = 9.0, 900.0
    rec.insider_ownership, rec.dividend_yield = 0.11, 0.022
    rec.first_trade_date = "1978-06-01"
    rec.technicals = {"pct_above_5y_low": 0.20, "pct_above_10y_low": 0.55,
                      "pct_below_52w_high": 0.30}
    m = mx.compute_metrics(rec)
    out = sc.screen_universe([rec], {"SCH": m}, th, {}, cycle=None)
    f = out["results"]["SCH"]["frameworks"]["schloss"]
    check("a cheap, low-debt, long-listed name passes Schloss", f["passed"], True)
    names = {t["name"]: t for t in f["tests"]}
    check("the listing-age test actually ran",
          names["long_operating_history"]["result"], True)
    check("on a real number rather than a guess",
          names["long_operating_history"]["value"] > 45, True)
    check("insider alignment is scored", names["insider_alignment"]["result"], True)
    check("and the dividend is counted", names["pays_while_you_wait"]["result"], True)

    # The universe split and the regime both reach the page.
    check("the run reports which value regime it was in",
          out["value_regime"]["regime"] in
          ("deep_value_available", "relative_value"), True)
    check("and how the universe splits across Lynch's categories",
          sum(out["lynch_census"].values()), 1)

    html = rn.TEMPLATE
    check("the category badge is rendered on the row", 'class="cat ' in html, True)
    check("the drawer states the category", "Lynch category:" in html, True)
    check("and the cyclical warning has somewhere to appear",
          "Cyclical warning" in html, True)
    panel = rn._lynch_panel({"cyclical": 9, "stalwart": 1}, out["value_regime"])
    check("the panel warns when one category swallows the universe",
          "Check this" in panel, True)
    check("and states Lynch's real ownership bar against ours",
          "15–20%" in panel, True)


def test_buffett_owner_earnings_and_b_label():
    """Owner earnings, the DCF and its refusals, and the three business tenets."""
    from src import buffett as bf
    th = yaml.safe_load(open("config/thresholds.yml"))
    bcfg = th["buffett"]

    # --- owner earnings ------------------------------------------------------
    ys = [mkyear(2025 - i, revenue=1000.0 - 50 * i, net_income=100.0,
                 depreciation_amortization=60.0, capex=80.0,
                 current_assets=400.0, current_liabilities=200.0,
                 cash_and_equivalents=100.0, shares_diluted=100.0)
          for i in range(8)]
    oe = bf.owner_earnings(ys)
    check("owner earnings are computed", oe["available"], True)
    check("maintenance capex never exceeds total capex",
          oe["maintenance_capex"] <= oe["capex_total"], True)
    check("the more conservative estimate is the one used",
          oe["maintenance_capex"] >= min(60.0, 80.0), True)
    check("and the method is named", bool(oe["maintenance_method"]), True)
    check("Buffett's own caveat travels with the number",
          "must be a guess" in oe["caveat"], True)
    check("owner earnings sit below reported earnings plus depreciation",
          oe["owner_earnings"] < oe["net_income"] + oe["depreciation"], True)

    check("no depreciation line means no owner earnings, not a guess",
          bf.owner_earnings([mkyear(2025, net_income=100.0, capex=10.0)]
                            )["available"], False)
    check("and no capex line likewise",
          bf.owner_earnings([mkyear(2025, net_income=100.0,
                                    depreciation_amortization=10.0)
                             ])["available"], False)

    # --- the discounted cash flow, and what it refuses to do ----------------
    dcf = bcfg["intrinsic_value"]
    v = bf.intrinsic_value(2.0, 0.08, 0.10, dcf)
    check("a DCF runs on positive owner earnings", v["available"], True)
    check("and returns a per-share value above zero", v["value_per_share"] > 0, True)
    check("the discount rate used is reported", v["discount_rate"], 0.10)
    check("so is the share of value sitting in the terminal figure",
          0 < v["terminal_share"] < 1, True)
    check("and the caveat says so out loud",
          "terminal" in v["caveat"], True)
    # Growth is capped: an unsustainable rate must not be extrapolated for a
    # decade, which is the classic way a DCF produces a nonsense number.
    fast = bf.intrinsic_value(2.0, 0.60, 0.10, dcf)
    check("a 60% growth rate is capped, not projected",
          fast["growth_used"], dcf["growth_cap"])
    check("and the capping is flagged", fast["growth_capped"], True)
    check("a loss-making business gets no DCF at all",
          bf.intrinsic_value(-1.0, 0.08, 0.10, dcf)["available"], False)
    check("and the refusal explains itself",
          "not a valuation" in
          bf.intrinsic_value(-1.0, 0.08, 0.10, dcf)["reason"], True)
    check("no growth history means no projection",
          bf.intrinsic_value(2.0, None, 0.10, dcf)["available"], False)
    # A near-zero discount rate must not produce an infinite value.
    zero = bf.intrinsic_value(2.0, 0.08, 0.001, dcf)
    check("the discount rate is floored so the value stays finite",
          zero["discount_rate"], dcf["min_discount_rate"])
    check("and the terminal growth stays below it",
          zero["terminal_growth"] < zero["discount_rate"], True)
    check("a higher discount rate always gives a lower value",
          bf.intrinsic_value(2.0, 0.08, 0.14, dcf)["value_per_share"]
          < v["value_per_share"], True)

    check("margin of safety is the discount to value",
          round(bf.margin_of_safety(75.0, 100.0), 4), 0.25)
    check("a price above value gives a negative margin",
          bf.margin_of_safety(120.0, 100.0) < 0, True)
    check("and a nonsense value gives none at all",
          bf.margin_of_safety(75.0, 0.0), None)

    # --- the B label ---------------------------------------------------------
    tcfg = bcfg["business_tenets"]
    good = {"gross_margin_ttm": 0.62, "gross_margin_cv": 0.05,
            "roic_5y_avg": 0.22, "return_on_net_tangible_assets": 0.45,
            "roe_years_above_15": 10, "roe_years_evaluated": 10,
            "loss_years_in_10": 0, "loss_years_in_3": 0, "eps_cv_5y": 0.12,
            "listing_age_years": 42, "history_years": 10,
            "lynch_category": "stalwart"}
    t = bf.business_tenets(good, "Consumer Defensive", "Beverages", tcfg)
    check("a wide-moat, steady, in-circle business is labelled B", t["label"], "B")
    check("and the summary names all three tenets",
          all(k in t["summary"] for k in ("competence", "returns", "steady")), True)
    check("the moat evidence is itemised, not asserted",
          len(t["moat"]["evidence"]) >= 4, True)
    check("with the caveat that these are footprints, not the moat",
          "footprints" in t["moat"]["caveat"].lower(), True)

    # Each tenet can veto on its own.
    # Outside the list AND without a strong moat: no admission.
    weak = dict(good, gross_margin_ttm=0.22, roic_5y_avg=0.08,
                return_on_net_tangible_assets=0.09)
    out_of_circle = bf.business_tenets(
        weak, "Aerospace", "Aerospace & Defense",
        {**tcfg, "circle_of_competence": ["Consumer Defensive"]})
    check("outside the declared circle there is no B", out_of_circle["label"], None)
    check("and it says whose declaration that was",
          "declared in config" in out_of_circle["circle"]["why"], True)
    # "Aerospace" must not be caught by the "space" disruption word — that is a
    # substring, not an industry, and mature manufacturers were being excluded.
    check("aerospace is not mistaken for a disruption zone",
          out_of_circle["history"]["ok"], True)

    # The high-barrier route: outside the list, but every moat marker holds.
    via_moat = bf.business_tenets(
        good, "Aerospace", "Aerospace & Defense",
        {**tcfg, "circle_of_competence": ["Consumer Defensive"]})
    check("a genuine high-barrier business is admitted to the circle",
          via_moat["circle"]["ok"], True)
    check("and is labelled as having come in that way",
          via_moat["circle"]["via_moat"], True)
    check("the row says so in words",
          "high-barrier route" in via_moat["circle"]["why"], True)
    check("that route needs EVERY marker, not a majority",
          bf.business_tenets(
              dict(good, return_on_net_tangible_assets=0.09), "Aerospace",
              "Aerospace & Defense",
              {**tcfg, "circle_of_competence": ["Consumer Defensive"]}
          )["circle"]["ok"], False)
    check("and it can be switched off entirely",
          bf.business_tenets(
              good, "Aerospace", "Aerospace & Defense",
              {**tcfg, "circle_of_competence": ["Consumer Defensive"],
               "admit_on_strong_moat": False})["circle"]["ok"], False)

    thin = dict(good, gross_margin_ttm=0.18, roic_5y_avg=0.06,
                return_on_net_tangible_assets=0.05, roe_years_above_15=1)
    check("a commodity business gets no B",
          bf.business_tenets(thin, "Consumer Defensive", "Beverages",
                             tcfg)["label"], None)

    lossy = dict(good, loss_years_in_10=2, loss_years_in_3=1)
    r_lossy = bf.business_tenets(lossy, "Consumer Defensive", "Beverages", tcfg)
    check("loss years break the consistent-history tenet", r_lossy["label"], None)
    check("and the reason is specific",
          any("loss" in x for x in r_lossy["history"]["reasons"]), True)

    ta_ = dict(good, lynch_category="turnaround")
    check("a turnaround is excluded by name",
          bf.business_tenets(ta_, "Consumer Defensive", "Beverages",
                             tcfg)["label"], None)
    check("a disruption-zone industry is excluded too",
          bf.business_tenets(good, "Healthcare", "Biotechnology",
                             tcfg)["label"], None)
    young = dict(good, listing_age_years=4)
    check("and so is a business too young to have been tested",
          bf.business_tenets(young, "Consumer Defensive", "Beverages",
                             tcfg)["label"], None)

    # The circle of competence is never inferred. With nothing declared, the
    # tenet must stand aside and SAY it is standing aside.
    undeclared = bf.business_tenets(good, "Anything", "At all",
                                    {**tcfg, "circle_of_competence": []})
    check("with no circle declared the tenet does not bite",
          undeclared["circle"]["ok"], True)
    check("but the row says the tenet is untested",
          "not being tested" in undeclared["circle"]["why"], True)
    check("and points at the file to edit",
          "thresholds.yml" in undeclared["circle"]["why"], True)

    # --- the metrics behind the new tests ------------------------------------
    rec = CompanyRecord(ticker="BRK", market="US", currency="USD",
                        sector="Consumer Defensive", industry="Beverages")
    rec.years = [mkyear(2025 - i, revenue=1000.0, gross_profit=620.0,
                        sga_expense=200.0, net_income=180.0,
                        eps_diluted=1.8, shares_diluted=100.0,
                        total_equity=600.0, total_assets=1100.0,
                        total_debt=300.0, long_term_debt=250.0,
                        short_term_debt=50.0, cash_and_equivalents=200.0,
                        current_assets=500.0, current_liabilities=250.0,
                        operating_income=300.0, pretax_income=280.0,
                        depreciation_amortization=50.0, capex=40.0,
                        cfo=230.0, dividends_paid=-60.0) for i in range(8)]
    rec.price, rec.market_cap = 30.0, 3000.0
    rec.first_trade_date = "1980-01-02"
    rec.technicals = {"price_5y_ago": 15.0}
    m = mx.compute_metrics(rec)
    check("net margin is computed", m["net_margin_ttm"], 0.18)
    check("SG&A is measured against gross profit, not revenue",
          round(m["sga_to_gross_profit"], 4), round(200.0 / 620.0, 4))
    check("debt payoff is stated in years of earnings",
          round(m["debt_payoff_years"], 4), round(250.0 / 180.0, 4))
    check("capex is measured against net income",
          round(m["capex_to_net_income"], 4), round(40.0 / 180.0, 4))
    check("owner earnings reach the metrics", m["owner_earnings"] is not None, True)
    check("as a per-share figure too",
          m["owner_earnings_per_share"] is not None, True)
    # 5 years of retained earnings = 5 x (180 - 60) = 600; market cap went from
    # 15 x 100 = 1500 to 30 x 100 = 3000, so 1500 of value on 600 retained.
    check("retained earnings over five years are summed", m["retained_earnings_5y"], 600.0)
    check("against the change in market value", m["market_cap_change_5y"], 1500.0)
    check("giving the one-dollar premise", m["one_dollar_premise"], 2.5)

    # --- end to end ----------------------------------------------------------
    out = sc.screen_universe([rec], {"BRK": m}, th,
                             {"us_10y": 4.5, "_enabled": True}, cycle=None)
    res = out["results"]["BRK"]
    check("the discount rate is the long bond plus a premium",
          round(out["buffett_valuation"]["discount_rate"], 4), 0.09)
    check("the risk-free rate used is reported",
          out["buffett_valuation"]["risk_free_pct"], 4.5)
    check("the B label is attached in the pipeline",
          m["buffett_b_label"], "B")
    check("with its evidence", bool(m["business_tenets"]["moat"]["evidence"]), True)
    fw = res["frameworks"]["buffett"]
    names = {t["name"]: t for t in fw["tests"]}
    check("the margin-of-safety test is present", "margin_of_safety" in names, True)
    check("the one-dollar premise is scored", names["one_dollar_premise"]["value"], 2.5)
    check("and overheads against gross profit",
          names["overhead_discipline"]["value"] is not None, True)

    rows = rn.build_payload({"BRK": res}, {"BRK": m}, out)
    check("the row carries the B label", rows[0]["b"], True)
    check("and the evidence behind it", bool(rows[0]["b_detail"]), True)
    html = rn.TEMPLATE
    check("the badge is rendered", 'class="blab"' in html, True)
    check("there is a filter for it", 'id="bOnly"' in html, True)
    check("the drawer shows the three tenets",
          "Buffett business tenets" in html, True)
    check("and the B filter survives a copied link", "p.set('b','1')" in html, True)

    # The label must round-trip through the store, or merged rows lose it.
    stored = res["metrics"]
    check("the B label is persisted", stored.get("buffett_b_label"), "B")
    check("so is the evidence", bool(stored.get("business_tenets")), True)
    merged = rn.build_payload({"BRK": res}, {}, out)
    check("and a merged row still shows the badge", merged[0]["b"], True)


def test_buffett_indicator():
    """Two FRED series in different units — the classic way to be wrong by 1000x."""
    from src import buffett as bf

    class FakeFred:
        """Stands in for FRED. Units arrive from the metadata call, as in life."""
        enabled = True

        def __init__(self, eq=62_000_000.0, gdp=29_000.0, eq_units="Mil. of $",
                     gdp_units="Bil. of $", meta=True,
                     eq_date="2026-04-01", gdp_date="2026-04-01"):
            self.eq, self.gdp, self.meta = eq, gdp, meta
            self.eq_units, self.gdp_units = eq_units, gdp_units
            self.eq_date, self.gdp_date = eq_date, gdp_date

        def series_meta(self, sid):
            if not self.meta:
                return {}
            from src.providers.fred import UNIT_SCALE
            u = self.eq_units if sid == bf.EQUITIES_SERIES else self.gdp_units
            return {"id": sid, "units": u, "scale": UNIT_SCALE.get(u.lower())}

        def latest_observation(self, sid):
            return ((self.eq_date, self.eq) if sid == bf.EQUITIES_SERIES
                    else (self.gdp_date, self.gdp))

    r = bf.buffett_indicator(FakeFred())
    check("the indicator computes", r["available"], True)
    # 62,000,000 million = 62 trillion of equity; 29,000 billion = 29 trillion
    # of GDP. Only correct unit handling gets from those two numbers to 2.14.
    check("units are applied, not assumed", round(r["ratio"], 3), 2.138)
    check("and the verdict follows Buffett's own bands",
          r["verdict"], "substantially overvalued")
    check("which is quoted rather than paraphrased",
          "playing with fire" in r["reading"], True)
    check("the units used are reported", r["equities_units"], "Mil. of $")
    check("and the caveat says which version of the ratio this is",
          "Z.1" in r["caveat"], True)

    check("a cheap market reads as undervalued",
          bf.buffett_indicator(FakeFred(eq=20_000_000.0))["verdict"],
          "significantly undervalued")
    check("and a reading near 100% is fair",
          bf.buffett_indicator(FakeFred(eq=29_000_000.0))["verdict"],
          "fairly valued")

    # THE BUG THIS GUARDS AGAINST. Skip the scaling and the ratio comes out at
    # 2,138x — still a number, still renderable, completely wrong.
    check("without unit handling the ratio would be absurd",
          62_000_000.0 / 29_000.0 > 1000, True)
    refused = bf.buffett_indicator(FakeFred(gdp_units="Mil. of $"))
    check("such a value is refused, not published", refused["available"], False)
    check("with the raw numbers shown so the cause is visible",
          bool(refused["raw"]["equities"]), True)
    check("and a reason that names the plausible range",
          "plausible" in refused["reason"], True)

    assumed = bf.buffett_indicator(FakeFred(meta=False))
    check("with no metadata it falls back to the documented units",
          assumed["available"], True)
    check("and flags that it assumed them", assumed["units_assumed"], True)
    check("in the caveat a reader will actually see",
          "assumed" in assumed["caveat"], True)

    lagged = bf.buffett_indicator(FakeFred(eq_date="2025-10-01"))
    check("series published out of step are disclosed",
          "dated differently" in lagged["reading"], True)

    class NoKey:
        enabled = False

    nk = bf.buffett_indicator(NoKey())
    check("no API key means no number", nk["available"], False)
    check("and the fix is named", "FRED_API_KEY" in nk["reason"], True)

    class Empty(FakeFred):
        def latest_observation(self, sid):
            return None

    check("a missing series is reported, not guessed around",
          bf.buffett_indicator(Empty())["available"], False)

    line = rn._buffett_indicator_line(bf.buffett_indicator(FakeFred()))
    check("the panel renders the number", "Buffett Indicator: 214%" in line, True)
    check("and renders the refusal just as plainly",
          "not computed" in rn._buffett_indicator_line(nk), True)
    panel = rn._lynch_panel({"stalwart": 2}, {}, {},
                            bf.buffett_indicator(FakeFred()),
                            {"names": 10, "b_labelled": 3})
    check("it sits in the market-gauge panel", "Buffett Indicator" in panel, True)
    check("beside a count of the B-labelled names",
          "3 of 10 names" in panel, True)

    # The provider knows how to read units off FRED rather than assuming them.
    from src.providers import fred as fredmod
    check("millions scale to dollars", fredmod.UNIT_SCALE["mil. of $"], 1e6)
    check("and billions likewise", fredmod.UNIT_SCALE["bil. of $"], 1e9)
    check("the provider exposes a metadata call",
          hasattr(fredmod.FredProvider, "series_meta"), True)
    check("and a dated observation call",
          hasattr(fredmod.FredProvider, "latest_observation"), True)


def test_lynch_five_year_window():
    """Lynch runs on five years of statements, not six."""
    th = yaml.safe_load(open("config/thresholds.yml"))
    lyn = th["lynch"]
    for t in ("valuation", "growth_floor", "growth_ceiling"):
        check(f"{t} needs five years, not six",
              lyn["tests"][t]["min_history_years"], 5)
    check("the PEG used is the five-year one",
          lyn["tests"]["valuation"]["metric"], "peg_ratio_lynch")
    check("as is the growth rate",
          lyn["tests"]["growth_floor"]["metric"], "eps_cagr_lynch")

    # Exactly five statements: the case the fencepost used to make unevaluable.
    rec = CompanyRecord(ticker="FIVE", market="SG", currency="SGD",
                        sector="Industrials", industry="Specialty Machinery")
    rec.years = [mkyear(2025 - i, revenue=1000.0,
                        net_income=100.0 * (0.88 ** i),
                        eps_diluted=1.0 * (0.88 ** i), shares_diluted=100.0,
                        total_equity=700.0, total_assets=1200.0,
                        total_debt=150.0, short_term_debt=30.0,
                        long_term_debt=120.0, cash_and_equivalents=200.0,
                        current_assets=500.0, current_liabilities=250.0,
                        inventory=80.0, cfo=150.0, capex=30.0,
                        depreciation_amortization=40.0,
                        operating_income=140.0, pretax_income=130.0)
                 for i in range(5)]
    rec.price, rec.market_cap = 10.0, 1000.0
    m = mx.compute_metrics(rec)
    check("five statements give a growth rate",
          m["eps_cagr_lynch"] is not None, True)
    check("struck over the four-year span they actually cover",
          m["eps_cagr_lynch_years"], 4)
    check("and the span is reported, not silently called five years",
          m["eps_cagr_lynch_years"] < 5, True)
    check("the six-year metric stays empty, as it should", m["eps_cagr_5y"], None)
    check("the Lynch PEG is computed from the five-year rate",
          m["peg_ratio_lynch"] is not None, True)
    check("and PEGY with it", m["growth_plus_yield_to_pe"] is not None, True)

    out = sc.screen_universe([rec], {"FIVE": m}, th, {}, cycle=None)
    names = {t["name"]: t
             for t in out["results"]["FIVE"]["frameworks"]["lynch"]["tests"]}
    check("the growth floor now runs on five years of statements",
          names["growth_floor"]["result"] is not None, True)
    check("rather than dropping out for want of a sixth",
          bool(names["growth_floor"].get("insufficient")), False)
    check("and the valuation test runs too",
          bool(names["valuation"].get("insufficient")), False)

    # Four statements is still too few: the span would be three years.
    rec4 = CompanyRecord(ticker="FOUR", market="SG", currency="SGD")
    rec4.years = rec.years[:4]
    rec4.price, rec4.market_cap = 10.0, 1000.0
    check("four statements are still not enough",
          mx.compute_metrics(rec4)["eps_cagr_lynch"], None)


def test_schloss_five_year_window():
    """Schloss reads a five-year statement window, so Asia is judged on Asia's feed."""
    th = yaml.safe_load(open("config/thresholds.yml"))
    sch = th["schloss"]["tests"]
    check("earnings durability reads five years, not ten",
          sch["earnings_durability"]["metric"], "loss_years_in_5")
    check("at most one loss year in five",
          sch["earnings_durability"]["threshold"], 1)
    check("the cash-flow record reads five years too",
          sch["cash_generation_record"]["metric"], "cfo_positive_share_5y")
    # Prices are not statements: a ten-year price window costs nothing and the
    # false-bottom test needs it, so it stays.
    check("but the price-history tests still use ten years",
          sch["no_false_bottom"]["metric"], "pct_above_10y_low")

    def mk(ni_by_year, cfo_by_year):
        r = CompanyRecord(ticker="W5", market="TH", currency="THB")
        r.years = [mkyear(2025 - i, revenue=1000.0, net_income=ni_by_year[i],
                          eps_diluted=ni_by_year[i] / 100.0, shares_diluted=100.0,
                          total_equity=800.0, total_assets=1200.0,
                          total_debt=200.0, long_term_debt=150.0,
                          short_term_debt=50.0, cash_and_equivalents=150.0,
                          current_assets=500.0, current_liabilities=250.0,
                          total_liabilities=400.0, cfo=cfo_by_year[i],
                          capex=30.0, depreciation_amortization=40.0,
                          operating_income=140.0, pretax_income=130.0)
                   for i in range(len(ni_by_year))]
        r.price, r.market_cap = 8.0, 800.0
        return r

    # A company that lost money EIGHT and NINE years ago but has been steady
    # since. On a ten-year window those old losses still count; on the five-year
    # window Schloss now uses, the record being judged is the recent one.
    old_trouble = mk([100.0] * 8 + [-50.0, -60.0], [120.0] * 8 + [-20.0, -30.0])
    m = mx.compute_metrics(old_trouble)
    check("the ten-year count still sees the old losses", m["loss_years_in_10"], 2)
    check("the five-year count does not", m["loss_years_in_5"], 0)
    check("and the five-year cash record is clean",
          m["cfo_positive_share_5y"], 1.0)
    check("while the ten-year one is not",
          m["cfo_positive_share_10y"] < 1.0, True)

    # A four-year Asian feed: the share metric scales, so it is judged on the
    # same bar rather than being marked down for the provider's depth.
    thin = mk([100.0, 90.0, 95.0, 110.0], [120.0, 110.0, 115.0, 130.0])
    m4 = mx.compute_metrics(thin)
    check("four years of statements still produce a cash record",
          m4["cfo_positive_share_5y"], 1.0)
    check("computed on the years actually there", m4["cfo_years_evaluated_5"], 4)
    check("and the window used is reported",
          m4["statement_years_used_schloss"], 4)

    out = sc.screen_universe([thin], {"W5": m4}, th, {}, cycle=None)
    names = {t["name"]: t
             for t in out["results"]["W5"]["frameworks"]["schloss"]["tests"]}
    check("earnings durability is evaluated on a four-year feed",
          names["earnings_durability"]["result"], True)
    check("so is the cash-flow record",
          names["cash_generation_record"]["result"], True)
    check("neither drops out for want of a decade",
          bool(names["cash_generation_record"].get("insufficient")), False)

    # One loss in five passes; two do not.
    one = mx.compute_metrics(mk([100.0, -20.0, 95.0, 110.0, 105.0],
                                [120.0, 110.0, 115.0, 130.0, 125.0]))
    two = mx.compute_metrics(mk([-10.0, -20.0, 95.0, 110.0, 105.0],
                                [120.0, 110.0, 115.0, 130.0, 125.0]))
    check("one loss year in five is tolerated", one["loss_years_in_5"], 1)
    check("two are not", two["loss_years_in_5"], 2)
    check("and the threshold sits between them",
          one["loss_years_in_5"] <= sch["earnings_durability"]["threshold"]
          < two["loss_years_in_5"], True)

    # The five-year fields must survive the round trip through the store, or a
    # merged Asian row loses the very tests this change was made for.
    stored = out["results"]["W5"]["metrics"]
    for k in ("loss_years_in_5", "cfo_positive_share_5y",
              "statement_years_used_schloss"):
        check(f"{k} is persisted", k in stored, True)


def test_sentiment_gauges():
    """Six psychology gauges, all contrarian, none of them substituted."""
    from src import sentiment as sen
    scfg = yaml.safe_load(open("config/universe.yml")).get("sentiment", {})

    # --- VIX ----------------------------------------------------------------
    calm = pd.Series(np.full(300, 13.0) + np.random.default_rng(1).normal(0, .3, 300))
    v = sen.vix_state(calm, cfg=scfg)
    check("a low VIX reads as complacency", v["state"], "complacency")
    check("and is framed as a place to take profit", "profit" in v["reading"], True)
    panicky = pd.Series(list(np.full(280, 14.0)) + list(np.linspace(15, 42, 20)))
    p = sen.vix_state(panicky, cfg=scfg)
    check("a spike reads as panic", p["state"], "panic")
    check("and is framed as where buying has paid", "buying" in p["reading"], True)
    check("with the caution that it is never the first day",
          "first day" in p["reading"], True)
    check("the level is placed in its own year",
          0.9 <= p["percentile_1y"] <= 1.0, True)

    # Term structure: spot above three-month is the panic tell.
    back = sen.vix_state(panicky, pd.Series(np.full(300, 22.0)), scfg)
    check("backwardation is detected", back["term_structure"] > 1.0, True)
    check("and named as what a real panic looks like",
          "BACKWARDATED" in back["term_reading"], True)
    contango = sen.vix_state(calm, pd.Series(np.full(300, 18.0)), scfg)
    check("the normal shape is not called a panic",
          "not about this week" in contango["term_reading"], True)
    check("no history means no reading",
          sen.vix_state(pd.Series(dtype=float))["available"], False)

    # --- RSI ----------------------------------------------------------------
    r = sen.rsi_state(74.0, [72.0, 71.0, 75.0], scfg)
    check("RSI above 70 is euphoria", r["state"], "euphoric")
    check("RSI below 30 is capitulation",
          sen.rsi_state(24.0, [26.0], scfg)["state"], "capitulating")
    # The pair is the point: a hot index over a lukewarm median is a NARROW
    # market, not a euphoric one, and the gauge has to say which.
    narrow = sen.rsi_state(72.0, [50.0, 51.0, 49.0], scfg)
    check("an index far above its median stock is flagged",
          bool(narrow["divergence"]), True)
    check("and read as narrowness, not mood",
          "carried by a few large names" in narrow["divergence"], True)
    check("with no RSI at all there is no reading",
          sen.rsi_state(None, [])["available"], False)

    # --- put/call ------------------------------------------------------------
    check("a high put/call is contrarian bullish",
          sen.put_call_state(1.25, cfg=scfg)["state"], "fearful")
    check("stated as the worry already being paid for",
          "already paid for" in sen.put_call_state(1.25, cfg=scfg)["reading"], True)
    check("a low one is greed", sen.put_call_state(0.5, cfg=scfg)["state"], "greedy")
    off = sen.put_call_state(None, cfg=scfg)
    check("with no feed it reports itself off", off["available"], False)
    check("and names the setting that turns it on",
          "put_call_url" in off["reason"], True)

    # --- breadth: A/D line and McClellan -------------------------------------
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")

    def mkframes(n, drift, vol=0.01):
        out = {}
        for i in range(n):
            steps = rng.normal(drift, vol, len(idx))
            c = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)
            out[f"T{i}"] = pd.DataFrame({"Close": c, "Volume": 1e6})
        return out

    # What the McClellan Oscillator actually measures is the RATE OF CHANGE of
    # breadth — a fast average of net advances minus a slow one. A market where
    # the same wide majority advances every single day reads near ZERO, because
    # nothing is changing. So the fixtures below are turns, not trends: that is
    # the only shape the indicator has an opinion about.
    def turning(n, early, late, split=200):
        out = {}
        for i in range(n):
            steps = np.concatenate([
                rng.normal(early, 0.002, split),
                rng.normal(late, 0.002, len(idx) - split)])
            out[f"T{i}"] = pd.DataFrame(
                {"Close": pd.Series(100 * np.exp(np.cumsum(steps)), index=idx),
                 "Volume": 1e6})
        return out

    broad = mkframes(40, 0.004, vol=0.002)
    ad = sen.advance_decline(broad)
    check("the A/D line computes", ad["available"], True)
    check("on a stated number of names", ad["names"], 40)
    check("a uniformly advancing market reads near zero, as it should",
          abs(ad["mcclellan_oscillator"]) < 25, True)
    check("and the caveat explains that this is a rate of change",
          "RATE OF CHANGE" in ad["caveat"], True)
    check("and that it is not the NYSE figure", "not NYSE-wide" in ad["caveat"], True)

    up_turn = sen.advance_decline(turning(40, -0.004, 0.004))
    check("a market turning up gives a positive oscillator",
          up_turn["mcclellan_oscillator"] > 0, True)
    check("and is read as thrust", "positive" in up_turn["oscillator_state"], True)
    down_turn = sen.advance_decline(turning(40, 0.004, -0.004))
    check("a market rolling over gives a negative one",
          down_turn["mcclellan_oscillator"] < 0, True)
    check("read as distribution under the surface",
          "distribution" in down_turn["oscillator_state"], True)
    check("too few names means no gauge, not a guess",
          sen.advance_decline(mkframes(3, 0.001))["available"], False)

    # --- participation: the deceptive-rally test -----------------------------
    # An index up 20% over six months while the median constituent is down is
    # exactly the "few mega-caps" case the user asked to catch.
    flat = mkframes(30, 0.0)
    up_index = pd.Series(100 * np.exp(np.linspace(0, 0.20, 300)), index=idx)
    part = sen.participation(flat, up_index)
    check("participation computes", part["available"], True)
    check("the gap between index and median stock is measured",
          part["breadth_gap"] > 0.05, True)
    check("and named as deceptive strength",
          "deceptive strength" in part["reading"], True)
    together = sen.participation(mkframes(30, 0.0015), up_index)
    check("a broad rally is not accused of narrowness",
          "broadly consistent" in together["reading"], True)

    # --- COT ------------------------------------------------------------------
    def cotrow(d, ncl, ncs, oi=100000.0):
        return {"report_date_as_yyyy_mm_dd": d, "open_interest_all": oi,
                "noncomm_positions_long_all": ncl,
                "noncomm_positions_short_all": ncs,
                "comm_positions_long_all": ncs, "comm_positions_short_all": ncl}

    rows = [cotrow(f"2025-{(i % 12) + 1:02d}-0{(i % 8) + 1}", 20000 + i * 10,
                   20000) for i in range(100)]
    rows.insert(0, cotrow("2026-08-11", 60000, 10000))
    c = sen.cot_state(rows, "E-mini S&P 500")
    check("COT positioning computes", c["available"], True)
    check("normalised against open interest, not stated raw",
          round(c["spec_net_pct_oi"], 2), 0.50)
    check("an extreme is placed in its own history", c["percentile"] >= 0.9, True)
    check("and called crowded", c["state"], "crowded long")
    check("with the reason a crowd matters",
          "nobody left to buy" in c["reading"], True)
    check("the report date travels with it", c["report_date"], "2026-08-11")
    short = sen.cot_state([cotrow("2026-08-11", 5000, 40000)]
                          + [cotrow(f"2025-0{i+1}-01", 30000, 10000)
                             for i in range(9)], "Gold")
    check("the other extreme reads as crowded short", short["state"], "crowded short")
    check("no rows means no reading, not a zero",
          sen.cot_state([], "Copper")["available"], False)

    # --- Fear & Greed ---------------------------------------------------------
    fg = sen.fear_greed(index_close=up_index, high_low_ratio=0.9,
                        mcclellan=70.0, pcr=0.55, hy_spread=3.0,
                        vix_level=12.0, vix_ma50=16.0,
                        stock_20d=0.05, bond_20d=-0.01)
    check("the composite computes", fg["available"], True)
    check("on all seven factors", fg["inputs_used"], 7)
    check("and reads as greed", fg["score"] >= 55, True)
    check("every sub-score is published", len(fg["subscores"]), 7)
    check("and it is labelled a replication, not CNN's number",
          "replication" in fg["caveat"], True)

    fearful = sen.fear_greed(
        index_close=pd.Series(100 * np.exp(np.linspace(0, -0.20, 300))),
        high_low_ratio=0.05, mcclellan=-70.0, pcr=1.4, hy_spread=9.0,
        vix_level=38.0, vix_ma50=20.0, stock_20d=-0.09, bond_20d=0.02)
    check("the mirror case reads as fear", fearful["score"] <= 25, True)
    check("and is labelled extreme", fearful["label"], "extreme fear")

    # THE FAILURE MODE THIS GUARDS. Missing factors must be EXCLUDED, never
    # scored 50 — otherwise a composite drifts to neutral as its feeds die and
    # a broken gauge looks like a calm market.
    partial = sen.fear_greed(high_low_ratio=0.95, mcclellan=75.0)
    check("a partial composite says how many inputs it had",
          partial["inputs_used"], 2)
    check("and lists what was left out", len(partial["missing"]), 5)
    check("missing factors do not drag it toward neutral",
          partial["score"] > 80, True)
    check("with nothing at all it refuses",
          sen.fear_greed()["available"], False)

    # --- assembly -------------------------------------------------------------
    s = sen.build(sen.vix_state(calm, cfg=scfg), r,
                  sen.put_call_state(0.5, cfg=scfg), fg, [c], ad, part,
                  cycle_mode="defensive")
    check("the crowd verdict is greedy here", s["crowd"], "greedy")
    check("and the action is to take profit, not to sell everything",
          "take profit" in s["action"], True)
    check("agreement with the Marks gauge is stated",
          "agrees with the Marks cycle gauge" in s["cycle_note"], True)
    dis = sen.build(sen.vix_state(calm, cfg=scfg), r,
                    sen.put_call_state(0.5, cfg=scfg), fg, [c], ad, part,
                    cycle_mode="opportunistic")
    check("and so is disagreement", "DISAGREES" in dis["cycle_note"], True)
    check("with the point that the disagreement is the information",
          "the disagreement is the information" in dis["cycle_note"], True)
    check("sentiment is labelled as positioning, not value",
          "POSITIONING, not value" in s["caveat"], True)

    fearful_all = sen.build(p, sen.rsi_state(22.0, [25.0], scfg),
                            sen.put_call_state(1.4, cfg=scfg), fearful, [],
                            ad, part)
    check("the fearful case flips the verdict", fearful_all["crowd"], "fearful")
    check("and is not turned into an instruction to buy",
          "caveat that fear also persists" in fearful_all["action"], True)

    # --- it reaches the page ---------------------------------------------------
    html = rn._sentiment_panel(s)
    check("the panel renders", "Market psychology" in html, True)
    check("with the composite in the header", "Fear &amp; Greed" in html, True)
    check("the McClellan reading", "McClellan Oscillator" in html, True)
    check("the COT block", "Institutional positioning" in html, True)
    check("and the staleness of the COT report",
          "published on Friday" in html, True)
    check("an unavailable gauge shows its reason rather than a blank",
          "not available" in rn._sentiment_panel(
              sen.build(sen.vix_state(calm, cfg=scfg), r,
                        sen.put_call_state(None, cfg=scfg), fg, [], ad, part)),
          True)
    check("the panel is empty rather than wrong when nothing ran",
          rn._sentiment_panel({}), "")

    # Config and provider plumbing.
    uni = yaml.safe_load(open("config/universe.yml"))
    check("the three-month VIX series is configured",
          uni["macro_series"]["vix_3m"], "VXVCLS")
    check("COT contracts are keyed by code, not by name",
          all(str(v).isalnum() for v in uni["sentiment"]["cot_contracts"].values()),
          True)
    check("the put/call source is off by default",
          uni["sentiment"]["put_call_url"], "")
    from src.providers.cftc import CftcProvider, COT_URL
    check("the CFTC endpoint needs no API key", "api_key" not in COT_URL, True)
    check("and the provider looks up by contract code",
          "cftc_contract_market_code" in
          CftcProvider.history.__doc__ + str(CftcProvider.history.__code__.co_consts),
          True)


def test_greenblatt_magic_formula():
    """Greenblatt's own definitions of capital and value, and the list they make."""
    th = yaml.safe_load(open("config/thresholds.yml"))
    gcfg = th["greenblatt"]
    check("the market-cap floor is applied to the ranking",
          gcfg["min_market_cap_usd"], 100000000)
    check("excess cash is defined as a share of revenue",
          gcfg["operating_cash_pct_of_revenue"], 0.02)
    check("negative working capital is floored, and that is a stated choice",
          gcfg["floor_negative_working_capital"], True)
    check("the portfolio is the top 20-30", 20 <= gcfg["top_n"] <= 30, True)

    # --- the two metrics ----------------------------------------------------
    # Revenue 1000, cash 300 (excess = 300 - 20 = 280), ST debt 50, CL 250,
    # CA 600, net PPE 400, EBIT 200, debt 200, market cap 2000.
    #   Greenblatt capital = (600 - 280) - (250 - 50) + 400 = 520
    #   Greenblatt ROC     = 200 / 520 = 38.5%
    #   Greenblatt EV      = 2000 + 200 - 280 = 1920
    #   Greenblatt EY      = 200 / 1920 = 10.4%
    rec = CompanyRecord(ticker="GB", market="US", currency="USD",
                        sector="Industrials", industry="Machinery")
    rec.years = [mkyear(2025 - i, revenue=1000.0, operating_income=200.0,
                        pretax_income=190.0, net_income=150.0,
                        eps_diluted=1.5, shares_diluted=100.0,
                        total_equity=900.0, total_assets=1500.0,
                        total_debt=200.0, short_term_debt=50.0,
                        long_term_debt=150.0, cash_and_equivalents=300.0,
                        current_assets=600.0, current_liabilities=250.0,
                        total_liabilities=600.0, net_ppe=400.0, goodwill=100.0,
                        cfo=180.0, capex=40.0, depreciation_amortization=50.0)
                 for i in range(6)]
    rec.price, rec.market_cap = 20.0, 2000.0
    m = mx.compute_metrics(rec, greenblatt_cfg=gcfg)

    check("excess cash is cash above the operating need", m["excess_cash"], 280.0)
    check("and the operating need is 2% of revenue", m["operating_cash_need"], 20.0)
    check("capital employed nets off interest-bearing current liabilities",
          m["invested_capital_greenblatt"], 520.0)
    check("return on capital follows from it",
          round(m["ebit_to_invested_capital_greenblatt"], 4),
          round(200.0 / 520.0, 4))
    check("enterprise value subtracts EXCESS cash, not all cash",
          m["enterprise_value_greenblatt"], 1920.0)
    check("earnings yield follows from that",
          round(m["ebit_to_ev_greenblatt"], 4), round(200.0 / 1920.0, 4))
    # Goodwill is excluded from the capital base by construction: it is what
    # somebody once paid, not what the business needs to run.
    check("goodwill is not in the capital base",
          m["invested_capital_greenblatt"] < 520.0 + 100.0, True)

    # The general definitions must NOT have moved. Buffett, Munger, Klarman and
    # Templeton thresholds are calibrated against those, and shifting them
    # silently would change unrelated pass/fail results with no explanation.
    check("the general EV still subtracts all cash",
          m["enterprise_value"], 2000.0 + 200.0 - 300.0)
    check("and is a different number from Greenblatt's",
          m["enterprise_value"] != m["enterprise_value_greenblatt"], True)
    check("the general capital base is untouched too",
          m["ebit_to_invested_capital"] != m["ebit_to_invested_capital_greenblatt"],
          True)

    # Negative working capital: floored, and the fact recorded.
    supplier_funded = CompanyRecord(ticker="NWC", market="US", currency="USD",
                                    sector="Consumer Defensive",
                                    industry="Grocery")
    supplier_funded.years = [mkyear(2025 - i, revenue=1000.0,
                                    operating_income=100.0, net_income=70.0,
                                    eps_diluted=0.7, shares_diluted=100.0,
                                    total_equity=300.0, total_assets=900.0,
                                    total_debt=100.0, short_term_debt=20.0,
                                    cash_and_equivalents=50.0,
                                    current_assets=200.0,
                                    current_liabilities=500.0,
                                    net_ppe=400.0, cfo=90.0, capex=30.0,
                                    depreciation_amortization=25.0)
                             for i in range(6)]
    supplier_funded.price, supplier_funded.market_cap = 10.0, 1000.0
    mn = mx.compute_metrics(supplier_funded, greenblatt_cfg=gcfg)
    check("negative working capital is recorded as having been floored",
          mn["greenblatt_working_capital_floored"], True)
    check("so the capital base is just net fixed assets",
          mn["invested_capital_greenblatt"], 400.0)
    check("which keeps return on capital finite rather than absurd",
          mn["ebit_to_invested_capital_greenblatt"], 0.25)

    # --- the ranking ---------------------------------------------------------
    def mkrec(t, ebit, ev_cash, cap_ppe, mktcap, sector="Industrials",
              market="US"):
        r = CompanyRecord(ticker=t, market=market, currency="USD",
                          sector=sector, industry="Machinery")
        r.years = [mkyear(2025 - i, revenue=1000.0, operating_income=ebit,
                          pretax_income=ebit - 10, net_income=ebit * 0.7,
                          eps_diluted=ebit * 0.007, shares_diluted=100.0,
                          total_equity=900.0, total_assets=1500.0,
                          total_debt=100.0, short_term_debt=20.0,
                          cash_and_equivalents=ev_cash, current_assets=600.0,
                          current_liabilities=250.0, total_liabilities=600.0,
                          net_ppe=cap_ppe, cfo=ebit, capex=30.0,
                          depreciation_amortization=40.0) for i in range(6)]
        r.price, r.market_cap = mktcap / 100.0, mktcap
        return r

    recs = [mkrec("CHEAP", 300.0, 100.0, 200.0, 900.0),
            mkrec("MID", 200.0, 100.0, 500.0, 1800.0),
            mkrec("DEAR", 100.0, 100.0, 900.0, 4000.0),
            mkrec("BANK", 300.0, 100.0, 200.0, 900.0, sector="Financial Services"),
            mkrec("TINY", 300.0, 100.0, 200.0, 0.5)]
    mets = {r.ticker: mx.compute_metrics(r, greenblatt_cfg=gcfg) for r in recs}
    for t, mm in mets.items():
        mm["market_cap_usd"] = next(r.market_cap for r in recs if r.ticker == t)
    # The synthetic caps above are small numbers for readability, so the floor
    # is lowered for the ranking test and exercised separately by TINY.
    gcfg2 = {**gcfg, "min_market_cap_usd": 1.0}
    res = sc.run_greenblatt(recs, mets, gcfg2,
                            th["global"]["capital_intensive_excluded_sectors"])
    port = res.pop("_portfolio")

    check("the cheap, capital-light name ranks first",
          port["rows"][0]["ticker"], "CHEAP")
    check("and the expensive, capital-heavy one ranks last",
          port["rows"][-1]["ticker"], "DEAR")
    check("both component ranks are published",
          all(("ey_rank" in r and "roc_rank" in r) for r in port["rows"]), True)
    check("and the composite is their sum",
          port["rows"][0]["combined_score"],
          port["rows"][0]["ey_rank"] + port["rows"][0]["roc_rank"])
    check("financials are excluded from the ranking",
          "sector excluded" in res["BANK"]["ineligible_reason"], True)
    check("and so is anything under the market-cap floor",
          "market-cap floor" in res["TINY"]["ineligible_reason"], True)
    check("the excluded names are counted", port["excluded"], 2)
    check("the ranking says which basis it used",
          port["rows"][0]["basis"], "greenblatt")
    check("and reports how many names it could use it for",
          port["on_greenblatt_basis"], 3)

    # The global rank travels with every row even when ranking per market, so
    # this app's per-market deviation from the book stays visible.
    two_mkt = recs[:3] + [mkrec("HKCHEAP", 400.0, 100.0, 150.0, 800.0,
                                market="HK")]
    mets2 = {r.ticker: mx.compute_metrics(r, greenblatt_cfg=gcfg)
             for r in two_mkt}
    for t, mm in mets2.items():
        mm["market_cap_usd"] = next(r.market_cap for r in two_mkt if r.ticker == t)
    res2 = sc.run_greenblatt(
        two_mkt, mets2, {**gcfg2, "rank_scope": "market"},
        th["global"]["capital_intensive_excluded_sectors"])
    res2.pop("_portfolio")
    check("a lone name in its own market is top of that market",
          res2["HKCHEAP"]["combined_rank"], 1)
    check("but its GLOBAL rank is reported too",
          res2["HKCHEAP"]["global_rank"] is not None, True)
    check("against the whole eligible universe",
          res2["HKCHEAP"]["global_universe_size"], 4)
    check("so the per-market deviation is visible, not assumed",
          res2["MID"]["global_rank"] != res2["MID"]["combined_rank"]
          or res2["MID"]["scope"] == "US", True)

    # --- the panel ------------------------------------------------------------
    html = rn._magic_formula_panel(port)
    check("the portfolio panel renders", "Magic Formula portfolio" in html, True)
    check("with both component ranks in the table", "EY rank" in html, True)
    check("and the annual rebalance instruction",
          "hold for a year" in html, True)
    check("with the reason the holding period matters",
          "one year in three" in html, True)
    check("it states what was excluded and why",
          "balance sheet IS the business" in html, True)
    check("and flags the two different rankings",
          "this app&rsquo;s deviation" in html or "this app's deviation" in html,
          True)
    check("an empty portfolio renders nothing rather than an empty table",
          rn._magic_formula_panel({}), "")

    # End to end, and through the store.
    out = sc.screen_universe(
        recs, mets, {**th, "greenblatt": gcfg2}, {}, cycle=None)
    check("the portfolio reaches the run output",
          bool(out["magic_formula"]["rows"]), True)
    check("and is not left in the per-ticker results",
          "_portfolio" in out["results"], False)
    stored = out["results"]["CHEAP"]["metrics"]
    check("Greenblatt's earnings yield is persisted",
          stored.get("ebit_to_ev_greenblatt") is not None, True)
    check("so is his capital base",
          stored.get("invested_capital_greenblatt") is not None, True)
    rows = rn.build_payload({"CHEAP": out["results"]["CHEAP"]}, {}, out)
    check("and a merged row still shows the earnings yield",
          rows[0]["key_metrics"]["Earnings yield (EBIT/EV)"] != "—", True)


def test_value_growth_style():
    """The value–growth axis, and the '?' that made it necessary."""
    from src import style as st, lynch as ly
    th = yaml.safe_load(open("config/thresholds.yml"))
    scfg = th["style"]

    # --- the sector tilt ----------------------------------------------------
    check("technology tilts growth",
          st.sector_tilt("Technology", "Software - Infrastructure")["direction"],
          "growth")
    check("and so does an internet platform whatever its sector",
          st.sector_tilt("Consumer Cyclical", "Internet Retail")["direction"],
          "growth")
    check("banks tilt value",
          st.sector_tilt("Financial Services", "Banks")["direction"], "value")
    check("machinery tilts neither way",
          st.sector_tilt("Industrials", "Machinery")["direction"], "neutral")
    check("and the tilt explains itself",
          "priced on what it will earn" in
          st.sector_tilt("Technology", "Software")["why"], True)

    # --- the axis itself ----------------------------------------------------
    def rec(t, sector, industry, **mm):
        r = CompanyRecord(ticker=t, market="HK", currency="HKD",
                          sector=sector, industry=industry)
        return r, dict(mm)

    universe = []
    mets = {}
    # A spread of cheap, slow names and expensive, fast ones, so the
    # percentiles have something to rank.
    for i in range(10):
        r, mm = rec(f"CHEAP{i}", "Financial Services", "Banks",
                    ebit_to_ev=0.14 + i * 0.002, fcf_yield=0.11 + i * 0.002,
                    price_to_book=0.5 + i * 0.02, dividend_yield=0.06,
                    eps_cagr_lynch=0.01, revenue_cagr_5y=0.01,
                    revenue_growth_1y=0.01, gross_margin_ttm=0.30)
        universe.append(r)
        mets[r.ticker] = mm
    for i in range(10):
        r, mm = rec(f"FAST{i}", "Technology", "Software - Infrastructure",
                    ebit_to_ev=0.02 + i * 0.001, fcf_yield=0.01 + i * 0.001,
                    price_to_book=9.0 + i * 0.2, dividend_yield=0.0,
                    eps_cagr_lynch=0.30, revenue_cagr_5y=0.28,
                    revenue_growth_1y=0.25, gross_margin_ttm=0.75)
        universe.append(r)
        mets[r.ticker] = mm

    census = st.assign(universe, mets, scfg)
    check("cheap slow banks come out on the value side",
          mets["CHEAP0"]["style"] in ("value", "deep value"), True)
    check("fast expensive software comes out on the growth side",
          mets["FAST0"]["style"] in ("growth", "high growth"), True)
    check("the score is signed, growth positive",
          mets["FAST0"]["style_score"] > 0 > mets["CHEAP0"]["style_score"], True)
    check("book-to-price is used, not price-to-book",
          mets["CHEAP0"]["book_to_price"] > mets["FAST0"]["book_to_price"], True)
    check("every label carries its evidence",
          len(mets["FAST0"]["style_evidence"]) >= 5, True)
    check("including the sector tilt, named separately",
          any("sector tilt growth" in e for e in mets["FAST0"]["style_evidence"]),
          True)
    check("and a sentence a person can read",
          "percentile for growth" in mets["FAST0"]["style_why"], True)
    check("the census counts both sides",
          census["growth_names"] + census["value_names"] > 0, True)
    check("and says the labels are relative to this universe",
          "Relative to THIS universe" in census["caveat"], True)

    # Meituan's case, in the user's own words: a technology business is a
    # growth stock even when its multiples are unremarkable.
    mt, mtm = rec("3690.HK", "Technology", "Software - Application",
                  ebit_to_ev=0.05, fcf_yield=0.03, price_to_book=3.0,
                  dividend_yield=0.0, eps_cagr_lynch=0.10,
                  revenue_cagr_5y=0.15, revenue_growth_1y=0.12,
                  gross_margin_ttm=0.35)
    universe2 = universe + [mt]
    mets2 = {**{k: dict(v) for k, v in mets.items()}, "3690.HK": mtm}
    st.assign(universe2, mets2, scfg)
    check("a technology name lands on the growth side",
          mets2["3690.HK"]["style"] in ("growth", "high growth"), True)
    check("and the reason names its sector",
          "priced on what it will earn" in mets2["3690.HK"]["style_why"], True)

    # A name with nothing to score is UNSCORED, not quietly called blend — a
    # measured middle and an unmeasured one are different facts.
    blank, blankm = rec("EMPTY", "Industrials", "Machinery")
    st.assign(universe + [blank], {**mets, "EMPTY": blankm}, scfg)
    check("a name with no factors is not called blend", blankm["style"], None)
    check("it is labelled unscored", blankm["style_label"], "unscored")
    check("and says why", "no valuation or growth factors" in
          blankm["style_evidence"][0], True)

    # --- the '?' that prompted this -------------------------------------------
    # A company with exactly five statements used to classify as "unclassified"
    # because eps_cagr_5y needs a sixth. It should now find a growth rate.
    five = CompanyRecord(ticker="FIVE2", market="HK", currency="HKD",
                         sector="Industrials", industry="Machinery")
    five.years = [mkyear(2025 - i, revenue=1000.0,
                         net_income=100.0 * (0.9 ** i),
                         eps_diluted=1.0 * (0.9 ** i), shares_diluted=100.0,
                         total_equity=700.0, total_assets=1200.0,
                         total_debt=150.0, short_term_debt=30.0,
                         long_term_debt=120.0, cash_and_equivalents=200.0,
                         current_assets=500.0, current_liabilities=250.0,
                         inventory=80.0, cfo=150.0, capex=30.0,
                         depreciation_amortization=40.0,
                         operating_income=140.0, pretax_income=130.0)
                  for i in range(5)]
    five.price, five.market_cap = 10.0, 1000.0
    m5 = mx.compute_metrics(five)
    check("five statements no longer produce a question mark",
          ly.classify(m5, "Industrials", "Machinery")["category"] != "unclassified",
          True)
    check("because the classifier now reads the five-year rate",
          m5["eps_cagr_lynch"] is not None, True)

    # And where even that is missing, one year of revenue growth is used —
    # noisy, but labelled as such rather than shrugged at.
    one_year = ly.classify({"revenue_growth_1y": 0.22}, "Industrials", "Tools")
    check("a single year of revenue growth still yields a category",
          one_year["category"], "fast_grower")
    check("with the noise declared",
          "one year of revenue growth" in one_year["why"], True)
    check("and a name with truly nothing is still unclassified",
          ly.classify({}, "Industrials", "Tools")["category"], "unclassified")

    # --- it reaches the page ---------------------------------------------------
    html = rn.TEMPLATE
    check("the style badge is rendered", 'class="sty ' in html, True)
    check("there are value and growth filters",
          'id="valOnly"' in html and 'id="grwOnly"' in html, True)
    check("the two filters are mutually exclusive",
          "fVal=false;document.getElementById('valOnly')" in html, True)
    check("and survive a copied link", "p.set('style','growth')" in html, True)
    check("the drawer explains the label", "<b>Style: " in html, True)
    check("a blend carries no badge, to keep the column quiet",
          "r.sty !== 'blend'" in html, True)

    panel = rn._lynch_panel({"stalwart": 3}, {}, {}, {}, {},
                            {"census": {"value": 4, "growth": 6, "blend": 2},
                             "value_names": 4, "growth_names": 6,
                             "blend_names": 2, "caveat": "Relative to THIS universe."})
    check("the panel gains a value-or-growth card",
          "Value or growth" in panel, True)
    check("with the split stated", "4 value, 6 growth, 2 blend" in panel, True)
    check("and the Lynch card is named rather than 'universe split'",
          "Lynch categories</h4>" in panel, True)

    # Round trip through the store.
    out = sc.screen_universe(universe2, mets2, th, {}, cycle=None)
    check("the census reaches the run output",
          bool(out["style_census"]["census"]), True)
    stored = out["results"]["3690.HK"]["metrics"]
    check("the style label is persisted", stored.get("style") is not None, True)
    check("so is the evidence behind it",
          bool(stored.get("style_evidence")), True)
    rows = rn.build_payload({"3690.HK": out["results"]["3690.HK"]}, {}, out)
    check("and a merged row still carries the badge", bool(rows[0]["sty"]), True)


def test_technical_charts():
    """The technical panel is drawn, and the price series is paid for on purpose."""
    idx = pd.date_range("2023-01-01", periods=620, freq="B")
    rng = np.random.default_rng(5)
    close = pd.Series(30 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, len(idx)))),
                      index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99,
                       "Close": close, "Volume": np.full(len(idx), 1e6)})

    sp = ta.sparkline(df)
    check("a price series is produced", sp is not None, True)
    check("downsampled, not 500 daily bars", sp["points"] <= ta.SPARK_POINTS, True)
    check("and enough points left to draw a shape", sp["points"] >= 40, True)
    check("over about two years", 1.8 <= sp["years"] <= 2.1, True)
    check("normalised to 100 at the start", sp["px"][0], 100.0)
    check("and rounded to one decimal",
          all(round(v, 1) == v for v in sp["px"]), True)
    check("the real prices ride alongside for the axis labels",
          sp["lo"] < sp["hi"], True)
    check("the 200-day line has the same number of points",
          len(sp["ma"]), len(sp["px"]))
    # The moving average must be sampled from the DAILY series, never
    # recomputed from the sampled points — a 200-day mean of 80 weekly samples
    # is a different and wrong line.
    daily_ma = float(close.rolling(200).mean().iloc[-1])
    check("and is the true 200-day average, not a mean of the samples",
          abs(sp["ma"][-1] - daily_ma / sp["first"] * 100.0) < 0.2, True)
    check("a short history yields no chart rather than a misleading one",
          ta.sparkline(df.iloc[-30:]), None)
    check("and neither does an empty frame",
          ta.sparkline(pd.DataFrame()), None)

    # --- payload discipline --------------------------------------------------
    check("the series is in the display contract, so merged rows chart too",
          "spark" in rn.DISPLAY_METRICS, True)
    check("and there is a stated cap on how many the page may carry",
          rn.MAX_SPARKLINES > 0, True)

    def mkres(t, surfaced):
        return {"ticker": t, "name": t, "market": "US", "surfaced": surfaced,
                "frameworks": {}, "technical": {"n_passed": 4, "n_total": 6,
                                                "tests": []},
                "metrics": {"spark": sp, "rsi_14": 55.0}}

    results = {f"S{i}": mkres(f"S{i}", True) for i in range(3)}
    results.update({f"N{i}": mkres(f"N{i}", False) for i in range(4)})
    rows = {r["ticker"]: r for r in rn.build_payload(results, {}, {})}
    check("surfaced rows carry the price series",
          all(rows[f"S{i}"]["spark"] for i in range(3)), True)
    check("rows nobody is reviewing do not",
          any(rows[f"N{i}"]["spark"] for i in range(4)), False)
    check("but they still carry the numbers every other chart needs",
          rows["N0"]["tech_raw"]["rsi"], 55.0)

    # A row with a deep dive gets a series whether or not it surfaced.
    rows2 = {r["ticker"]: r for r in
             rn.build_payload(results, {}, {}, report_tickers={"N1"})}
    check("a name with a deep dive gets one too", bool(rows2["N1"]["spark"]), True)

    # The cap bounds the worst case rather than trusting the universe to be small.
    many = {f"M{i}": mkres(f"M{i}", True) for i in range(rn.MAX_SPARKLINES + 25)}
    capped = rn.build_payload(many, {}, {})
    check("the cap is enforced",
          sum(1 for r in capped if r["spark"]), rn.MAX_SPARKLINES)
    check("and the choice is deterministic, not dict order",
          [r["ticker"] for r in rn.build_payload(many, {}, {}) if r["spark"]],
          [r["ticker"] for r in capped if r["spark"]])

    # --- raw values, not formatted strings ------------------------------------
    # A chart cannot plot "12.5%". This is the mistake that would have made the
    # whole panel silently blank.
    raw = rows["S0"]["tech_raw"]
    check("the chart inputs are numbers", isinstance(raw["rsi"], float), True)
    check("not the formatted strings the metric grid uses",
          isinstance(rows["S0"]["key_metrics"]["RSI(14)"], str), True)

    # --- it reaches the page ---------------------------------------------------
    html = rn.TEMPLATE
    for fn in ("svgLine", "svgRsi", "svgRange", "svgReturns", "testBars",
               "technicalPanel"):
        check(f"{fn} is defined in the page", f"function {fn}(" in html, True)
    check("the drawer calls the chart panel", "technicalPanel(r)" in html, True)
    check("the numeric list survives underneath, collapsed",
          "the same tests as numbers" in html, True)
    check("a row without a series explains why rather than showing a gap",
          "No price series" in html, True)
    check("the threshold marker is the midpoint of every bar",
          "markAt=0.5" in html, True)
    check("and the price chart has a fixed height, not a third of the screen",
          ".chart.wide>svg{height:" in html, True)


if __name__ == "__main__":
    test_schema_identities()
    test_metrics_math()
    test_threshold_engine()
    test_greenblatt_ranking()
    test_macro_gate()
    test_technicals()
    test_currency_reconciliation()
    test_history_depth_parity()
    test_end_to_end_with_config()
    test_cycle_and_new_frameworks()
    test_macro_carry()
    test_debt_cycle()
    test_marks_from_source()
    test_soros_and_rogers_from_source()
    test_buffett_additions_and_graham()
    test_commodities_cnav_and_gauge()
    test_reflexive_stages()
    test_ownership_and_government_theme()
    test_dislocation()
    test_events()
    test_rsi_reading()
    test_display_metric_contract()
    test_malaysia_market()
    test_synopsis()
    test_lynch_categories()
    test_schloss_deep_value()
    test_buffett_owner_earnings_and_b_label()
    test_buffett_indicator()
    test_lynch_five_year_window()
    test_schloss_five_year_window()
    test_sentiment_gauges()
    test_greenblatt_magic_formula()
    test_value_growth_style()
    test_technical_charts()

    print("\n" + "=" * 62)
    print(f"  {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\n  Failures:")
        for f in FAILURES:
            print("   -", f)
    print("=" * 62)
    sys.exit(1 if FAIL else 0)
