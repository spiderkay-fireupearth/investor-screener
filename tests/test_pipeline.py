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
    check("Buffett now has 8 tests", len(th["buffett"]["tests"]), 8)
    check("Buffett bar is 6 of 8", th["buffett"]["min_tests_passed"], 6)
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
    check("Buffett reports 8 tests in the pipeline", fw["buffett"]["n_total"], 8)
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

    print("\n" + "=" * 62)
    print(f"  {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\n  Failures:")
        for f in FAILURES:
            print("   -", f)
    print("=" * 62)
    sys.exit(1 if FAIL else 0)
