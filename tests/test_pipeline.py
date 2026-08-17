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
          ["buffett", "greenblatt", "klarman", "lynch", "marks", "munger",
           "schloss", "soros", "templeton"])

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
    m_ok = {k: 99 for k in ("ev_to_ebit",)}
    metrics_pass4 = {"ev_to_ebit": 10.0, "net_debt_to_ebitda": 1.0,
                     "accruals_ratio": 0.01, "fcf_yield": 0.05,
                     "pct_below_52w_high": 0.02, "roic_5y_avg": 0.04,
                     "history_years": 10}
    base = sc.run_framework("marks", marks_cfg, metrics_pass4, "fail", 0)
    defn = sc.run_framework("marks", marks_cfg, metrics_pass4, "fail", +1)
    oppo = sc.run_framework("marks", marks_cfg, metrics_pass4, "fail", -1)
    check("4 tests pass on this fixture", base["n_passed"], 4)
    check("neutral bar is 4", base["required"], 4)
    check("defensive bar rises to 5", defn["required"], 5)
    check("opportunistic bar falls to 3", oppo["required"], 3)
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

    print("\n" + "=" * 62)
    print(f"  {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\n  Failures:")
        for f in FAILURES:
            print("   -", f)
    print("=" * 62)
    sys.exit(1 if FAIL else 0)
