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


def test_end_to_end_with_config():
    print("\n[8] End-to-end against the real config")
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
          ["buffett", "greenblatt", "klarman", "lynch", "munger", "schloss", "soros"])

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


if __name__ == "__main__":
    test_schema_identities()
    test_metrics_math()
    test_threshold_engine()
    test_greenblatt_ranking()
    test_macro_gate()
    test_technicals()
    test_currency_reconciliation()
    test_end_to_end_with_config()

    print("\n" + "=" * 62)
    print(f"  {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\n  Failures:")
        for f in FAILURES:
            print("   -", f)
    print("=" * 62)
    sys.exit(1 if FAIL else 0)
