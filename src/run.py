"""Pipeline entry point.

    python -m src.run --region asia     # 18:30 SGT job: SG, HK, TH, ID
    python -m src.run --region us       # 07:00 SGT job: S&P 500
    python -m src.run --region all      # both, for a manual full rebuild

Regions are split because the markets close 11 hours apart. Running Asia at
18:30 SGT (after SET closes at 17:30) and the US at 07:00 SGT (after the NYSE
close lands at 04:00/05:00 SGT) means each screen reads that market's most
recent settled close rather than a stale one. Each job only touches its own
universe, so total API volume is unchanged versus a single combined run.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd
import requests

from .schema import CompanyRecord, FundamentalYear
from .providers.yahoo import YahooProvider
from .providers.edgar import EdgarProvider
from .providers.fred import FredProvider
from .store import Store
from . import technicals as ta
from . import metrics as mx
from . import screens as sc
from . import render as rn
from . import library as lib
from . import cycle as cyc
from . import debtcycle as dbt

log = logging.getLogger("screener")

REGION_MARKETS = {
    "us": ["US"],
    "asia": ["JP", "SG", "HK", "TH", "ID"],
    "all": ["US", "JP", "SG", "HK", "TH", "ID"],
}


def load_config(cfg_dir: str = "config"):
    with open(os.path.join(cfg_dir, "universe.yml")) as f:
        universe = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "thresholds.yml")) as f:
        thresholds = yaml.safe_load(f)
    return universe, thresholds


def _http_get(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch with a descriptive User-Agent.

    This matters more than it looks: pandas.read_html(url) fetches via urllib
    with a default Python user-agent, and Wikipedia returns 403 to those. The
    result is a silent fall-through to the 20-name seed list, a one-minute run,
    and a screener that looks like it covered the S&P 500 but covered 4% of it.
    """
    ua = os.environ.get("SEC_USER_AGENT") or "investor-screener/1.0 (+github actions)"
    try:
        r = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
        if r.status_code == 200:
            return r.text
        log.warning("GET %s -> HTTP %s", url, r.status_code)
    except Exception as e:                      # noqa: BLE001
        log.warning("GET %s failed: %s", url, e)
    return None


def sp500_constituents(fallback: List[str]) -> List[str]:
    """Resolve S&P 500 membership. Membership churns, so fetch rather than pin.

    Two independent sources are tried before giving up, because a single
    unauthenticated scrape is not something to hang a 500-name universe on.
    """
    # 1) Wikipedia, fetched properly and parsed from HTML text.
    html = _http_get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if html:
        try:
            from io import StringIO
            for t in pd.read_html(StringIO(html)):
                if "Symbol" in t.columns:
                    syms = [str(s).strip().upper().replace(".", "-")
                            for s in t["Symbol"].tolist()]
                    syms = [s for s in syms if s and s != "NAN"]
                    if len(syms) > 400:
                        log.info("S&P 500: %d constituents from Wikipedia", len(syms))
                        return syms
        except Exception as e:                  # noqa: BLE001
            log.warning("Wikipedia parse failed: %s", e)

    # 2) A plain CSV mirror — no scraping, no user-agent games.
    csv = _http_get("https://raw.githubusercontent.com/datasets/"
                    "s-and-p-500-companies/main/data/constituents.csv")
    if csv:
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv))
            col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
            if col:
                syms = [str(s).strip().upper().replace(".", "-") for s in df[col]]
                syms = [s for s in syms if s and s != "NAN"]
                if len(syms) > 400:
                    log.info("S&P 500: %d constituents from CSV mirror", len(syms))
                    return syms
        except Exception as e:                  # noqa: BLE001
            log.warning("CSV mirror parse failed: %s", e)

    # Loud, not a buried warning. A 20-name "S&P 500" screen is worse than none,
    # because it looks complete.
    log.error("=" * 72)
    log.error("S&P 500 CONSTITUENT FETCH FAILED ON ALL SOURCES.")
    log.error("Falling back to a %d-name seed list. The US screen is NOT the", len(fallback))
    log.error("S&P 500 this run — treat its results as a sample, not a universe.")
    log.error("=" * 72)
    return fallback


def wikipedia_constituents(url: str, suffix: str = "",
                           min_expected: int = 100,
                           pad_to: Optional[int] = None) -> List[str]:
    """Pull an index's members from a Wikipedia components table.

    Generic on purpose: adding a new index should be a config edit, not a code
    change. Returns [] on any failure so the caller can fall back — never a
    partial list, because a half-populated index reads as a complete one.
    """
    html = _http_get(url)
    if not html:
        return []
    from io import StringIO
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:                      # noqa: BLE001
        log.warning("Could not parse tables at %s: %s", url, e)
        return []

    for t in tables:
        cols = {str(c).strip().lower(): c for c in t.columns}
        col = next((cols[k] for k in ("code", "ticker", "symbol", "ticker symbol")
                    if k in cols), None)
        if col is None:
            continue
        syms = []
        for v in t[col].tolist():
            s = str(v).strip().upper().replace(".0", "")
            if not s or s == "NAN":
                continue
            if pad_to and s.isdigit():
                s = s.zfill(pad_to)
            syms.append(f"{s}{suffix}")
        if len(syms) >= min_expected:
            log.info("Fetched %d constituents from %s", len(syms), url)
            return syms
    log.warning("No usable constituent table found at %s", url)
    return []


SUFFIX_MARKET = {".T": "JP", ".SI": "SG", ".HK": "HK", ".BK": "TH", ".JK": "ID"}


def theme_map(universe_cfg: Dict) -> Dict[str, List[str]]:
    """ticker -> [theme names]. A ticker may carry several tags."""
    out: Dict[str, List[str]] = {}
    for name, block in (universe_cfg.get("themes") or {}).items():
        for t in (block.get("tickers") or []):
            out.setdefault(str(t).upper(), []).append(name)
    return out


def merge_themes(universe_cfg: Dict, resolved: Dict[str, List[str]],
                 markets: List[str]) -> Dict[str, List[str]]:
    """Union thematic tickers into whichever market they list on.

    Deduplicated: a name already in the S&P 500 or Nasdaq-100 gains a tag and
    is NOT fetched twice. Names whose market isn't in this run are skipped —
    the Asia job shouldn't pull US tickers.
    """
    tm = theme_map(universe_cfg)
    if not tm:
        return resolved
    added_total = 0
    for ticker in tm:
        mkt = next((m for suf, m in SUFFIX_MARKET.items()
                    if ticker.endswith(suf)), "US")
        if mkt not in markets:
            continue
        pool = resolved.setdefault(mkt, [])
        if ticker not in {t.upper() for t in pool}:
            pool.append(ticker)
            added_total += 1
    if added_total:
        log.info("Themes: %d tagged tickers, %d new after dedupe against the indices",
                 len(tm), added_total)
    return resolved


def resolve_universe(universe_cfg: Dict, markets: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mkt in markets:
        m = universe_cfg["markets"].get(mkt)
        if not m:
            continue
        src = m.get("constituents_source")
        seed = m.get("seed_fallback", []) or list(m.get("constituents", []))

        if src == "dynamic":            # S&P 500 — its own two-source resolver
            names = [str(t).upper() for t in sp500_constituents(seed)
                     if t is not None and not isinstance(t, bool)]
            # Union any extra indices, deduplicated and order-preserving. Most
            # of the Nasdaq-100 is already in the S&P 500; only the difference
            # is added, so nothing is fetched or screened twice.
            seen = {t.upper() for t in names}
            for extra in (m.get("extra_indices") or []):
                got = wikipedia_constituents(
                    extra["url"], suffix=extra.get("ticker_suffix", ""),
                    min_expected=extra.get("min_expected", 80),
                    pad_to=extra.get("ticker_pad"))
                if not got:
                    got = extra.get("seed_fallback", [])
                    log.warning("%s fetch failed — using %d seed names",
                                extra.get("name"), len(got))
                # YAML 1.1 turns bare ON/OFF/YES/NO/Y/N into booleans, so
                # ON Semiconductor arrives as True unless it was quoted in the
                # config. Coerce here as well — config should not be able to
                # put a non-string into a ticker list.
                got = [str(t).upper() for t in got
                       if t is not None and not isinstance(t, bool)]
                added = [t for t in got if t.upper() not in seen]
                seen.update(t.upper() for t in added)
                names.extend(added)
                log.info("%s: %d constituents, %d new after dedupe against the S&P 500",
                         extra.get("name"), len(got), len(added))
            out[mkt] = names
        elif src == "wikipedia":
            syms = wikipedia_constituents(
                m["constituents_url"],
                suffix=m.get("ticker_suffix", ""),
                min_expected=m.get("min_expected", 100),
                pad_to=m.get("ticker_pad"))
            if syms:
                out[mkt] = syms
            else:
                log.error("%s constituent fetch failed — falling back to %d "
                          "seed names. This market is a SAMPLE, not the index.",
                          mkt, len(seed))
                out[mkt] = seed
        else:
            out[mkt] = list(m.get("constituents", []))
    return out


def refresh_fx(yahoo: YahooProvider, store: Store, fx_pairs: Dict[str, Optional[str]]):
    """One conversion point for the whole system. Cached so a Yahoo FX blip
    doesn't invalidate the market-cap gates."""
    for ccy, pair in fx_pairs.items():
        if ccy == "USD" or not pair:
            store.save_fx("USD", 1.0)
            continue
        rate = yahoo.fx_rate(pair)
        if rate:
            store.save_fx(ccy, rate)
            log.info("FX %s -> USD = %.6f", ccy, rate)
        else:
            cached = store.latest_fx(ccy)
            log.warning("FX %s fetch failed; using cached %s", ccy, cached)


def build_record(ticker: str, market_cfg: Dict, market_key: str,
                 store: Store, yahoo: YahooProvider, edgar: Optional[EdgarProvider],
                 index_df, fx_to_usd: Optional[float],
                 refresh_fundamentals: bool = True) -> Optional[CompanyRecord]:
    ccy = market_cfg.get("currency", "USD")
    rec = CompanyRecord(ticker=ticker, market=market_key, currency=ccy,
                        standard="us-gaap" if market_key == "US" else "ifrs")

    # ---- prices: fetch, persist, then read back so we always screen on the
    # full stored history rather than only on what today's call returned.
    df = yahoo.prices(ticker, period="10y")
    if df is not None and not df.empty:
        store.save_prices(ticker, df)
    stored = store.load_prices(ticker)
    if stored is None or stored.empty:
        log.warning("%s: no price history available, skipping", ticker)
        return None
    if df is None:
        stale = store.price_staleness_days(ticker)
        rec.warnings.append(f"price fetch failed; using stored data ({stale}d old)")

    # ---- profile
    prof = yahoo.profile(ticker)
    if prof:
        store.save_profile(ticker, prof)
    else:
        prof = store.load_profile(ticker) or {}
        if prof:
            rec.warnings.append("profile fetch failed; using cached profile")
    rec.name = prof.get("name") or ticker
    rec.sector = prof.get("sector")
    rec.industry = prof.get("industry")
    if prof.get("currency"):
        rec.currency = prof["currency"]
    rec.financial_currency = prof.get("financial_currency") or rec.currency
    rec.quote_type = prof.get("quote_type")
    rec.shares_outstanding = prof.get("shares_outstanding")
    rec.market_cap = prof.get("market_cap")

    # ---- technicals on this market's own calendar
    rec.technicals = ta.compute(stored, index_df=index_df, fx_to_usd=fx_to_usd)
    rec.price = rec.technicals.get("price") or prof.get("price")
    if not rec.market_cap and rec.price and rec.shares_outstanding:
        rec.market_cap = rec.price * rec.shares_outstanding

    # ---- fundamentals: EDGAR for US (audited), Yahoo for Asia (only option)
    years: List[FundamentalYear] = []
    source = market_cfg.get("fundamentals_provider", "yahoo")
    if refresh_fundamentals:
        if source == "edgar" and edgar is not None:
            years = edgar.fetch(ticker, years=12)
            if not years:
                rec.warnings.append("no SEC filings found; falling back to Yahoo")
                years = yahoo.fundamentals(ticker, currency=rec.currency)
                source = "yahoo"
        else:
            years = yahoo.fundamentals(ticker, currency=rec.currency)
        if years:
            store.save_fundamentals(ticker, years, source)

    if not years:
        cached = store.load_fundamentals(ticker)
        if cached:
            age = store.fundamentals_age_days(ticker)
            rec.warnings.append(f"fundamentals fetch failed; using cache ({age}d old)")
            for c in cached:
                fy = FundamentalYear(
                    ticker=ticker, fiscal_year=c["fiscal_year"],
                    period_end=c.get("period_end", ""),
                    standard=c.get("standard", "ifrs"),
                    currency=c.get("currency", ccy), source=c.get("source", "cache"))
                for k, v in c.items():
                    if hasattr(fy, k) and k not in ("ticker", "fiscal_year"):
                        try:
                            setattr(fy, k, v)
                        except Exception:      # noqa: BLE001
                            pass
                years.append(fy)
        else:
            rec.warnings.append("no fundamentals available — value screens cannot run")

    years.sort(key=lambda y: y.fiscal_year, reverse=True)
    rec.years = years
    return rec


def run(region: str, cfg_dir: str = "config", out_dir: str = "out",
        db_path: str = "data/screener.db", limit: Optional[int] = None,
        skip_fundamentals: bool = False) -> Dict[str, Any]:
    universe_cfg, thresholds = load_config(cfg_dir)
    markets = REGION_MARKETS[region]

    store = Store(db_path)
    yahoo = YahooProvider()
    edgar = EdgarProvider() if "US" in markets else None
    fred = FredProvider()

    run_id = f"{region}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    store.start_run(run_id, region)
    t0 = time.time()

    refresh_fx(yahoo, store, universe_cfg.get("fx_pairs", {}))
    macro = fred.snapshot(universe_cfg.get("macro_series", {}))

    tickers_by_market = merge_themes(universe_cfg, 
                                     resolve_universe(universe_cfg, markets), markets)
    themes_by_ticker = theme_map(universe_cfg)
    fund_tickers = {str(t).upper() for t in (universe_cfg.get("etfs") or [])}

    # State the universe up front. A run that quietly covers 20 names instead of
    # 500 still finishes green, and the only way to notice is to read the size.
    total_universe = sum(len(v) for v in tickers_by_market.values())
    log.info("UNIVERSE for region '%s': %d tickers — %s", region, total_universe,
             ", ".join(f"{k}={len(v)}" for k, v in tickers_by_market.items()))
    if "US" in markets and len(tickers_by_market.get("US", [])) < 100 and not limit:
        log.error("US universe is only %d names — expected ~500. "
                  "Constituent fetch almost certainly failed.",
                  len(tickers_by_market.get("US", [])))

    records: List[CompanyRecord] = []
    ok = fail = 0

    for mkt in markets:
        mcfg = universe_cfg["markets"][mkt]
        tickers = tickers_by_market.get(mkt, [])
        if limit:
            tickers = tickers[:limit]
        log.info("=== %s (%s): %d tickers ===", mkt, mcfg.get("name"), len(tickers))

        idx_df = yahoo.prices(mcfg["index_ticker"], period="5y")
        if idx_df is not None:
            store.save_prices(mcfg["index_ticker"], idx_df)
        else:
            idx_df = store.load_prices(mcfg["index_ticker"])
            log.warning("%s index fetch failed; relative strength may be stale", mkt)

        fx = store.latest_fx(mcfg.get("currency", "USD"))

        for i, t in enumerate(tickers, 1):
            try:
                rec = build_record(t, mcfg, mkt, store, yahoo, edgar, idx_df, fx,
                                   refresh_fundamentals=not skip_fundamentals)
                if rec:
                    rec.themes = themes_by_ticker.get(t.upper(), [])
                    # Trust the config's ETF list over Yahoo's quoteType, which
                    # is occasionally absent; fall back to it when not listed.
                    if t.upper() in fund_tickers:
                        rec.quote_type = "ETF"
                    records.append(rec)
                    ok += 1
                else:
                    fail += 1
            except Exception as e:                     # noqa: BLE001
                log.exception("%s failed: %s", t, e)
                fail += 1
            if i % 25 == 0:
                log.info("  %s: %d/%d", mkt, i, len(tickers))

    # ---- metrics, then screens
    # One FX table for the whole run, so ratio reconciliation and the USD size
    # gates use identical rates. Includes reporting currencies (CNY, USD, ...)
    # that no market in the universe actually trades in.
    fx_rates: Dict[str, float] = {"USD": 1.0}
    for ccy in set(list(universe_cfg.get("fx_pairs", {}).keys())
                   + [m.get("currency") for m in universe_cfg["markets"].values()]):
        if not ccy:
            continue
        r = store.latest_fx(ccy)
        if r:
            fx_rates[ccy.upper()] = r

    metrics_by_ticker: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        m = mx.compute_metrics(rec, fx_rates=fx_rates)
        fx = fx_rates.get((rec.currency or "USD").upper()) or 1.0
        m["market_cap_usd"] = rec.market_cap * fx if rec.market_cap else None
        m["fx_to_usd"] = fx
        metrics_by_ticker[rec.ticker] = m

    # Where are we in the cycle? Computed from the primary index, our own
    # breadth, and the VIX — then it modulates the Marks bar.
    primary_idx = store.load_prices(universe_cfg["markets"]["US"]["index_ticker"])
    breadth = None
    try:
        from . import analytics as an
        frames = {r.ticker: store.load_prices(r.ticker) for r in records[:400]}
        frames = {k: v for k, v in frames.items() if v is not None and len(v) >= 200}
        if frames:
            breadth = an.breadth_from_universe(frames)
    except Exception as e:                        # noqa: BLE001
        log.warning("breadth for cycle gauge failed: %s", e)
    vix_df = store.load_prices("^VIX")
    vix = float(vix_df["Close"].iloc[-1]) if vix_df is not None and len(vix_df) else None
    cycle_state = cyc.assess(primary_idx, breadth, vix,
                             thresholds.get("market_cycle", {}))
    log.info("Market cycle: %s (%s)", cycle_state.get("mode"),
             cycle_state.get("evidence"))

    # Where are we in the Big Debt Cycle? This is slow-moving — quarterly
    # series dominate it — but it is recomputed every run so the page never
    # shows a stage that predates a policy change.
    debt_state: Dict[str, Any] = {"enabled": False, "reason": "not computed"}
    try:
        cpi_hist = fred.history("CPIAUCSL", limit=400) if fred.enabled else []
        cape_res = dbt.universe_cape(records, metrics_by_ticker, cpi_hist)
        cape_val = cape_res.get("cape")
        if cape_val:
            log.info("Universe CAPE (US, %d names): %.1f",
                     cape_res.get("names_used", 0), cape_val)
        else:
            log.warning("Universe CAPE unavailable: %s", cape_res.get("error"))
        debt_state = dbt.build(fred, cape=cape_val, vix=vix)
        debt_state["cape_detail"] = cape_res
        if debt_state.get("enabled"):
            log.info("Debt cycle: stage %s — %s (alert %s)",
                     debt_state.get("stage"), debt_state.get("stage_name"),
                     debt_state.get("checklist", {}).get("level"))
            if debt_state.get("missing_series"):
                log.warning("Debt cycle ran with %d series missing: %s",
                            len(debt_state["missing_series"]),
                            ", ".join(debt_state["missing_series"]))
        else:
            log.warning("Debt cycle skipped: %s", debt_state.get("reason"))
    except Exception as e:                        # noqa: BLE001
        log.warning("debt-cycle stage failed: %s", e)
        debt_state = {"enabled": False, "reason": f"failed: {e}"}

    screened = sc.screen_universe(records, metrics_by_ticker, thresholds, macro,
                                  cycle=cycle_state)
    screened["debt_cycle"] = debt_state
    store.save_screen_results(run_id, screened["results"])

    # Merge in the other region's most recent results so the published page
    # always shows all five markets, each as fresh as its own last run.
    merged = store.latest_results()
    merged.update(screened["results"])

    os.makedirs(out_dir, exist_ok=True)
    # Which names already have a deep dive, so their rows can link to it.
    data_dir = os.path.dirname(db_path) or "data"
    have_reports = {r["ticker"] for r in lib.list_reports(data_dir) if r.get("ticker")}
    html_path = rn.render(merged, metrics_by_ticker, screened, thresholds,
                          universe_cfg, out_dir=out_dir, region=region, run_id=run_id,
                          report_tickers=have_reports)

    # Keep a copy of the published page in the data store, so a deep-dive
    # deploy can restore it instead of wiping the site root.
    if lib.snapshot_site(out_dir, data_dir):
        log.info("Snapshotted the screener page into the data store")

    # Republish stored deep dives. Without this a nightly refresh publishes an
    # out/ with no deepdive/ folder and silently deletes every report.
    n_reports = lib.publish(data_dir, out_dir)
    if n_reports:
        log.info("Republished %d deep-dive report(s)", n_reports)

    elapsed = time.time() - t0
    notes = f"{len(records)} records, {elapsed:.0f}s, macro_gate={'open' if screened['macro_gate_open'] else 'CLOSED'}"
    store.end_run(run_id, ok, fail, notes)
    log.info("Run %s complete: %s", run_id, notes)
    log.info("Wrote %s", html_path)

    store.close()
    return {"run_id": run_id, "ok": ok, "fail": fail, "html": html_path,
            "surfaced": sum(1 for v in merged.values() if v.get("surfaced"))}


def main():
    p = argparse.ArgumentParser(description="Multi-market value + technical screener")
    p.add_argument("--region", choices=["us", "asia", "all"], default="all")
    p.add_argument("--config", default="config")
    p.add_argument("--out", default="out")
    p.add_argument("--db", default="data/screener.db")
    p.add_argument("--limit", type=int, default=None,
                   help="cap tickers per market (for testing)")
    p.add_argument("--skip-fundamentals", action="store_true",
                   help="prices/technicals only; reuse stored fundamentals")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout)
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    res = run(a.region, a.config, a.out, a.db, a.limit, a.skip_fundamentals)
    print(f"\n{res['ok']} ok / {res['fail']} failed · {res['surfaced']} names surfaced")
    print(f"Output: {res['html']}")


if __name__ == "__main__":
    main()
