"""On-demand deep dive for a single ticker.

    python -m src.deepdive --ticker 7203.T
    python -m src.deepdive --ticker AAPL --horizon 252

Produces out/deepdive/<TICKER>.html plus a JSON sibling. Reuses the screener's
providers, metrics and framework engine, so a name's Buffett verdict here is the
same one the nightly screen produced — there is one implementation, not two.

What this deliberately does NOT do: invent numbers it cannot source. Value Line
VLMAP, the AAII and BofA cash surveys, put/call ratios and news sentiment are
not freely available, and rather than approximate them the report states what is
missing. Free substitutes that carry similar signal — VIX term structure,
XLY/XLP, the Buffett Indicator, and breadth computed from our own universe —
are used instead and labelled as such.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .schema import CompanyRecord, FundamentalYear
from .providers.yahoo import YahooProvider
from .providers.edgar import EdgarProvider
from .providers.fred import FredProvider
from .store import Store
from . import technicals as ta
from . import metrics as mx
from . import screens as sc
from . import analytics as an
from . import deepdive_render as dr
from . import library as lib

log = logging.getLogger("deepdive")

# Free market-context tickers, standing in for the paywalled sentiment inputs.
CONTEXT_TICKERS = {
    "vix": "^VIX", "vix3m": "^VIX3M",
    "xly": "XLY", "xlp": "XLP", "spx": "^GSPC",
}
CONTEXT_FRED = {
    "us_10y": "DGS10", "us_2y": "DGS2",
    "bbb_yield": "BAMLC0A4CBBBEY", "hy_spread": "BAMLH0A0HYM2",
    "cpi": "CPIAUCSL", "gdp": "GDP", "wilshire": "WILL5000PR",
}


def find_market(ticker: str, universe_cfg: Dict) -> str:
    """Locate a ticker's market, falling back on its Yahoo suffix."""
    t = ticker.upper()
    for mkt, cfg in universe_cfg["markets"].items():
        pool = list(cfg.get("constituents", [])) + list(cfg.get("seed_fallback", []))
        if t in [p.upper() for p in pool]:
            return mkt
    for suffix, mkt in ((".T", "JP"), (".SI", "SG"), (".HK", "HK"),
                        (".BK", "TH"), (".JK", "ID")):
        if t.endswith(suffix):
            return mkt
    return "US"


def market_context(yahoo: YahooProvider, fred: FredProvider,
                   store: Store) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"unavailable": [
        "Value Line VLMAP (paid subscription)",
        "CBOE equity put/call ratio (no reliable free feed)",
        "AAII asset allocation survey (survey-published)",
        "BofA Global Fund Manager Survey (proprietary)",
        "Robinhood Investor Index / eToro Retail Investor Beat (proprietary)",
    ]}

    frames = {}
    for key, tk in CONTEXT_TICKERS.items():
        df = yahoo.prices(tk, period="1y")
        if df is None:
            df = store.load_prices(tk)
        else:
            store.save_prices(tk, df)
        frames[key] = df

    if frames.get("vix") is not None and not frames["vix"].empty:
        ctx["vix"] = float(frames["vix"]["Close"].iloc[-1])
    if frames.get("vix3m") is not None and not frames["vix3m"].empty:
        ctx["vix3m"] = float(frames["vix3m"]["Close"].iloc[-1])
    if an._n(ctx.get("vix")) and an._n(ctx.get("vix3m")) and ctx["vix3m"]:
        r = ctx["vix"] / ctx["vix3m"]
        ctx["vix_vix3m"] = r
        # Backwardation in the VIX term structure marks acute near-term fear.
        ctx["vix_reading"] = ("near-term fear elevated — spot above 3-month "
                              "(backwardation)" if r > 1.0 else
                              "calm — normal contango in the term structure")

    if all(frames.get(k) is not None and not frames[k].empty for k in ("xly", "xlp")):
        ratio = (frames["xly"]["Close"] / frames["xlp"]["Close"]).dropna()
        ctx["xly_xlp_roc_4w"] = an.roc(ratio, 4)
        if an._n(ctx["xly_xlp_roc_4w"]):
            ctx["xly_xlp_reading"] = (
                "discretionary leading staples — risk appetite improving"
                if ctx["xly_xlp_roc_4w"] > 0 else
                "staples leading discretionary — defensive rotation, weaker outlook")

    macro = fred.snapshot(CONTEXT_FRED)
    ctx["macro"] = macro
    if an._n(macro.get("wilshire")) and an._n(macro.get("gdp")) and macro.get("gdp"):
        # Wilshire is in $bn, GDP in $bn — directly comparable.
        ctx["buffett_indicator"] = macro["wilshire"] / macro["gdp"]
        b = ctx["buffett_indicator"]
        ctx["buffett_reading"] = ("richly valued vs GDP" if b > 1.5 else
                                  "moderately valued vs GDP" if b > 1.0 else
                                  "cheap vs GDP")

    # Breadth from our own stored S&P 500 series — no vendor feed needed.
    us_frames = {}
    try:
        cur = store.conn.execute(
            "SELECT DISTINCT ticker FROM prices WHERE ticker NOT LIKE '^%' "
            "AND ticker NOT LIKE '%=X' LIMIT 600")
        for row in cur.fetchall():
            t = row["ticker"]
            if any(t.endswith(s) for s in (".T", ".SI", ".HK", ".BK", ".JK")):
                continue
            df = store.load_prices(t)
            if df is not None and len(df) >= 200:
                us_frames[t] = df
    except Exception as e:                       # noqa: BLE001
        log.warning("breadth read failed: %s", e)
    if us_frames:
        ctx["breadth"] = an.breadth_from_universe(us_frames)
    return ctx


def peer_comparison(rec: CompanyRecord, store: Store, limit: int = 6) -> List[Dict]:
    """Same-sector peers from the screener's own universe.

    Honest about its bound: this can only surface names already tracked across
    your six indices. A genuinely global peer search needs judgment and a wider
    data source — ask for that separately rather than trusting this list to be
    exhaustive.
    """
    rows = store.latest_results()
    peers = []
    for t, r in rows.items():
        if t == rec.ticker or not r.get("sector"):
            continue
        if rec.sector and r["sector"].strip().lower() != rec.sector.strip().lower():
            continue
        m = r.get("metrics") or {}
        peers.append({
            "ticker": t, "name": r.get("name"), "market": r.get("market"),
            "n_passed": r.get("n_frameworks_passed", 0),
            "frameworks": r.get("frameworks_passed", []),
            "pe": m.get("pe_ttm"), "ev_ebit": m.get("ev_to_ebit"),
            "roe": m.get("roe_ttm"), "roic": m.get("roic_5y_avg"),
            "fcf_yield": m.get("fcf_yield"), "de": m.get("debt_to_equity"),
            "mcap_usd": r.get("market_cap_usd"),
        })
    peers.sort(key=lambda p: (-p["n_passed"], -(p["fcf_yield"] or -9)))
    return peers[:limit]


def synthesise(m: Dict, fw: Dict, gbm: Dict, tech: Dict, val: Dict,
               hurst: Dict) -> Dict[str, Any]:
    """Rule-based Buy / Hold / Avoid. Every input is stated so the call can be
    argued with rather than taken on faith."""
    reasons_for, reasons_against = [], []
    score = 0.0

    n_fw = sum(1 for v in fw.values() if v.get("passed"))
    score += n_fw * 1.0
    if n_fw >= 3:
        reasons_for.append(f"clears {n_fw} of 7 investor frameworks")
    elif n_fw <= 1:
        reasons_against.append(f"clears only {n_fw} of 7 frameworks")

    ev = m.get("ev_to_ebit")
    if an._n(ev):
        if ev < 10:
            score += 1.5; reasons_for.append(f"EV/EBIT of {ev:.1f} is cheap")
        elif ev > 25:
            score -= 1.5; reasons_against.append(f"EV/EBIT of {ev:.1f} is demanding")

    fcfy = m.get("fcf_yield")
    if an._n(fcfy):
        if fcfy > 0.06:
            score += 1.0; reasons_for.append(f"free cash flow yield {fcfy:.1%}")
        elif fcfy < 0:
            score -= 1.0; reasons_against.append("negative free cash flow")

    fedm = val.get("fed_model", {})
    if an._n(fedm.get("spread")):
        if fedm["spread"] > 0:
            score += 0.5
            reasons_for.append("earnings yield above the 10-year Treasury")
        else:
            score -= 0.5
            reasons_against.append("earnings yield below the 10-year Treasury")

    q = val.get("tobins_q", {}).get("q")
    if an._n(q):
        if q < 1:
            score += 1.0; reasons_for.append(f"Tobin's Q of {q:.2f} is below 1")
        elif q > 3:
            score -= 0.5; reasons_against.append(f"Tobin's Q of {q:.2f} is elevated")

    if tech.get("price_above_sma200") == 1:
        score += 0.75; reasons_for.append("trading above its 200-day average")
    elif tech.get("price_above_sma200") == 0:
        score -= 0.75; reasons_against.append("trading below its 200-day average")

    p_up = gbm.get("prob_above_spot")
    if an._n(p_up):
        score += (p_up - 0.5) * 4
        if p_up > 0.55:
            reasons_for.append(f"GBM puts {p_up:.0%} of outcomes above spot")
        elif p_up < 0.45:
            reasons_against.append(f"GBM puts only {p_up:.0%} of outcomes above spot")

    if score >= 6:
        call, conviction = "BUY", "high"
    elif score >= 3:
        call, conviction = "BUY", "moderate"
    elif score >= 0:
        call, conviction = "HOLD", "moderate"
    else:
        call, conviction = "AVOID", "moderate"

    # Position size from volatility, not from enthusiasm: a 60%-vol name gets a
    # smaller slice than a 15%-vol name at identical conviction.
    sigma = gbm.get("sigma_annual")
    if an._n(sigma) and sigma > 0:
        base = 0.02 / max(sigma, 0.10)          # ~2% portfolio vol contribution
        size = max(0.01, min(0.10, base * (1.0 if conviction == "moderate" else 1.4)))
    else:
        size = 0.03
    if call == "AVOID":
        size = 0.0

    h = hurst.get("hurst")
    if an._n(h) and h > 0.55:
        horizon = "medium to long term — the series shows trend persistence"
    elif an._n(h) and h < 0.45:
        horizon = "short to medium term — the series mean-reverts, so entries matter more"
    else:
        horizon = "long term — no exploitable short-horizon structure"

    return {"call": call, "conviction": conviction, "score": score,
            "position_size": size, "horizon": horizon,
            "reasons_for": reasons_for, "reasons_against": reasons_against}


def analyse(ticker: str, cfg_dir: str = "config", out_dir: str = "out",
            db_path: str = "data/screener.db", horizon: int = 252) -> Dict[str, Any]:
    with open(os.path.join(cfg_dir, "universe.yml")) as f:
        universe_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "thresholds.yml")) as f:
        thresholds = yaml.safe_load(f)

    ticker = ticker.strip().upper()
    market = find_market(ticker, universe_cfg)
    mcfg = universe_cfg["markets"][market]
    log.info("Deep dive: %s (market %s, %s)", ticker, market, mcfg["name"])

    store = Store(db_path)
    yahoo = YahooProvider()
    edgar = EdgarProvider() if market == "US" else None
    fred = FredProvider()

    # ---- prices
    df = yahoo.prices(ticker, period="10y")
    if df is not None and not df.empty:
        store.save_prices(ticker, df)
    px = store.load_prices(ticker)
    if px is None or px.empty:
        store.close()
        return {"error": f"no price history for {ticker}"}

    idx_df = yahoo.prices(mcfg["index_ticker"], period="5y")
    if idx_df is None or idx_df.empty:
        idx_df = store.load_prices(mcfg["index_ticker"])

    prof = yahoo.profile(ticker) or store.load_profile(ticker) or {}
    if prof:
        store.save_profile(ticker, prof)

    rec = CompanyRecord(ticker=ticker, market=market,
                        name=prof.get("name") or ticker,
                        sector=prof.get("sector"), industry=prof.get("industry"),
                        currency=prof.get("currency") or mcfg.get("currency", "USD"),
                        financial_currency=prof.get("financial_currency"),
                        standard="us-gaap" if market == "US" else "ifrs")
    rec.shares_outstanding = prof.get("shares_outstanding")
    rec.market_cap = prof.get("market_cap")

    fx_rates = {"USD": 1.0}
    for ccy in universe_cfg.get("fx_pairs", {}):
        r = store.latest_fx(ccy)
        if r:
            fx_rates[ccy.upper()] = r

    fx = store.latest_fx(rec.currency) or 1.0
    rec.technicals = ta.compute(px, index_df=idx_df, fx_to_usd=fx)
    rec.price = rec.technicals.get("price") or prof.get("price")
    if not rec.market_cap and rec.price and rec.shares_outstanding:
        rec.market_cap = rec.price * rec.shares_outstanding

    # ---- fundamentals
    years: List[FundamentalYear] = []
    if market == "US" and edgar:
        years = edgar.fetch(ticker, years=15)
    if not years:
        years = yahoo.fundamentals(ticker, currency=rec.currency)
    if years:
        store.save_fundamentals(ticker, years, years[0].source)
    rec.years = sorted(years, key=lambda y: y.fiscal_year, reverse=True)

    m = mx.compute_metrics(rec, fx_rates=fx_rates)
    m["market_cap_usd"] = rec.market_cap * fx if rec.market_cap else None

    # ---- frameworks (identical engine to the nightly screen)
    ctx = market_context(yahoo, fred, store)
    screened = sc.screen_universe([rec], {ticker: m}, thresholds, ctx.get("macro", {}))
    result = screened["results"][ticker]
    fw = result["frameworks"]

    # ---- quantitative models
    gbm = an.gbm_monte_carlo(px["Close"], ticker, horizon_days=horizon)
    hurst = an.hurst_exponent(px["Close"])
    probs = an.prob_over_years(gbm.get("mu_annual"), gbm.get("sigma_annual"), 3.0)
    swings = an.swing_levels(px)
    fib = an.fibonacci_levels(swings.get("swing_high"), swings.get("swing_low"))

    latest = rec.latest
    ten_y = ctx.get("macro", {}).get("us_10y")
    val: Dict[str, Any] = {}
    val["fed_model"] = an.fed_model(m.get("eps_ttm"), rec.price, ten_y)
    val["tobins_q"] = an.tobins_q(m.get("market_cap_stmt_ccy") or rec.market_cap,
                                  latest.total_liabilities if latest else None,
                                  latest.total_assets if latest else None)
    eps_by_year = {y.fiscal_year: y.eps_diluted for y in rec.years
                   if y.eps_diluted is not None}
    val["cape"] = an.cape_ratio(rec.price, eps_by_year)
    val["bbb_formula"] = an.bbb_implied_rate(m.get("earnings_yield")
                                             or (1 / m["pe_ttm"] if m.get("pe_ttm") else None))
    val["bbb_actual"] = ctx.get("macro", {}).get("bbb_yield")

    div_yield = None
    try:
        import yfinance as yf
        divs = yf.Ticker(ticker).dividends
        val["dividend"] = an.dividend_relevance(px["Close"], divs)
        if divs is not None and len(divs) and rec.price:
            ttm = float(divs.iloc[-4:].sum()) if len(divs) >= 4 else float(divs.iloc[-1])
            div_yield = ttm / rec.price
            g = 0.03
            r_req = ((ten_y or 4.0) / 100.0) + 0.045
            val["ggm"] = an.gordon_growth(ttm * (1 + g), r_req, g)
            val["dividend_yield"] = div_yield
    except Exception as e:                       # noqa: BLE001
        val["dividend"] = {"error": str(e)}

    peers = peer_comparison(rec, store)
    rx = synthesise(m, fw, gbm, rec.technicals, val, hurst)

    payload = {
        "ticker": ticker, "name": rec.name, "market": market,
        "market_name": mcfg["name"], "sector": rec.sector, "industry": rec.industry,
        "currency": rec.currency, "financial_currency": rec.financial_currency,
        "price": rec.price, "market_cap": rec.market_cap,
        "market_cap_usd": m.get("market_cap_usd"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": m, "frameworks": fw, "technical": result.get("technical"),
        "gbm": gbm, "probs_3y": probs, "hurst": hurst,
        "swings": swings, "fibonacci": fib,
        "valuation": val, "context": ctx, "peers": peers,
        "recommendation": rx, "warnings": rec.warnings,
        "history_years": m.get("history_years"),
        "price_series": {
            "dates": [d.date().isoformat() for d in px.index[-504:]],
            "close": [float(v) for v in px["Close"].iloc[-504:]],
            "sma50": [None if pd.isna(v) else float(v)
                      for v in px["Close"].rolling(50).mean().iloc[-504:]],
            "sma200": [None if pd.isna(v) else float(v)
                       for v in px["Close"].rolling(200).mean().iloc[-504:]],
        },
    }

    # Write into the persistent library first, then mirror the whole library
    # into out/. Reports accumulate instead of each publish wiping the last.
    outp = os.path.join(out_dir, "deepdive")
    os.makedirs(outp, exist_ok=True)
    html_path = dr.render_deepdive(payload, outp)
    with open(os.path.join(outp, f"{ticker}.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    data_dir = os.path.dirname(db_path) or "data"
    lib.save_report(ticker, open(html_path, encoding="utf-8").read(), {
        "ticker": ticker, "name": rec.name, "market": market,
        "market_label": {"US": "S&P 500", "JP": "Nikkei 225", "SG": "SGX",
                         "HK": "HKEX", "TH": "SET", "ID": "IDX"}.get(market, market),
        "call": rx["call"], "score": round(rx["score"], 1),
        "frameworks_passed": sum(1 for v in fw.values() if v.get("passed")),
        "price": round(rec.price, 2) if rec.price else None,
        "currency": rec.currency,
        "generated_utc": payload["generated_utc"],
    }, data_dir)
    n = lib.publish(data_dir, out_dir)
    log.info("Published %d report(s) to the deep-dive library", n)

    store.close()
    return {"ticker": ticker, "html": html_path,
            "call": rx["call"], "score": rx["score"]}


def main():
    p = argparse.ArgumentParser(description="Single-ticker deep dive")
    p.add_argument("--ticker", required=True)
    p.add_argument("--config", default="config")
    p.add_argument("--out", default="out")
    p.add_argument("--db", default="data/screener.db")
    p.add_argument("--horizon", type=int, default=252,
                   help="Monte Carlo horizon in trading days")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    res = analyse(a.ticker, a.config, a.out, a.db, a.horizon)
    if res.get("error"):
        print("ERROR:", res["error"])
        sys.exit(1)
    print(f"\n{res['ticker']}: {res['call']} (score {res['score']:.1f})")
    print(f"Output: {res['html']}")


if __name__ == "__main__":
    main()
