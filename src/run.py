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
import re
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
from . import commodities as cmd
from . import reflexivity as rfx
from . import dislocation as dis
from . import events as evt
from . import buffett as bf
from . import sentiment as sen
from .providers.cftc import CftcProvider

log = logging.getLogger("screener")

REGION_MARKETS = {
    "us": ["US"],
    "asia": ["JP", "SG", "HK", "TH", "ID", "MY"],
    "all": ["US", "JP", "SG", "HK", "TH", "ID", "MY"],
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
        # Column naming is not standardised across Wikipedia's index pages.
        # Bursa Malaysia's table calls it "Stock Code"; without that spelling
        # the fetch returns nothing and silently falls back to the seed list,
        # which is exactly the S&P failure mode from earlier in this project.
        col = next((cols[k] for k in ("code", "ticker", "symbol",
                                      "ticker symbol", "stock code",
                                      "stock symbol", "sehk code", "scrip")
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



# ---------------------------------------------------------------------------
# Nasdaq-listed coverage beyond the Nasdaq-100.
#
# Nasdaq publishes its own symbol directory as a pipe-delimited text file, free
# and without a key. It carries every Nasdaq-listed security — roughly four
# thousand — which is far more than this pipeline can fetch fundamentals for,
# so two filters do the narrowing and both are honest about what they drop:
#
#   1. STRUCTURAL, from the file itself: test issues, ETFs, and the fifth-letter
#      suffixes that mark warrants, rights, units and preferred shares. These
#      are not companies and screening them would be noise.
#   2. LIQUIDITY, from prices we have to fetch anyway. The file carries no
#      market cap, so the alternative to ranking on turnover is an alphabetical
#      cut — which would give a universe of names beginning with A. Median
#      dollar turnover needs only the batch price download and is a better
#      proxy for "worth screening" than size alone.
# ---------------------------------------------------------------------------
NASDAQ_SYMBOL_FILE = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

# What a security IS, read from its name rather than from its ticker. The
# fifth-letter convention is the traditional shortcut and it is wrong often
# enough to matter: GOOGL ends in L, which the convention reserves for
# "miscellaneous", and it is Alphabet's ordinary class A stock. The security
# name says what the instrument is in words, so that is what gets parsed.
#
# Two rules, both learned the hard way:
#
#   * Match WHOLE WORDS. As a bare substring, "unit" is inside "United" and
#     inside "Community", which quietly dropped United Therapeutics, United
#     Natural Foods, United Community Banks and Community Trust Bancorp — four
#     ordinary Nasdaq common stocks — from every run.
#   * Match only the INSTRUMENT DESCRIPTION. Nasdaq formats the field as
#     "Company Name - What It Is", and the disqualifying words belong to the
#     second half. "Preferred Bank - Common Stock" is a bank called Preferred,
#     not a preferred share, and testing the whole string dropped it too.
NON_COMMON_NAME_WORDS = (
    "warrant", "warrants", "right", "rights", "unit", "units",
    "preferred", "convertible", "notes", "debenture", "debentures",
    "subordinated", "when issued", "depositary receipt",
)
_NON_COMMON_RE = re.compile(
    r"\b(" + "|".join(w.replace(" ", r"\s+") for w in NON_COMMON_NAME_WORDS)
    + r")\b", re.I)


def _instrument_desc(name: str) -> str:
    """The 'what it is' half of a Nasdaq security name.

    "Apple Inc. - Common Stock" describes the instrument after the dash. Where
    there is no dash the whole string is returned, and the word-boundary match
    is then doing the work on its own.
    """
    parts = re.split(r"\s+-\s+", name or "", maxsplit=1)
    return parts[1] if len(parts) > 1 else (name or "")
# "Depositary" alone is NOT disqualifying: every foreign issuer on Nasdaq —
# PDD, BIDU, JD, NTES, ASML — trades as American Depositary Shares, and they
# are ordinary equity. What disqualifies is a depositary share representing a
# PREFERRED series, which the name marks with a series letter or a coupon.
# Dropping on the word alone would have removed the foreign listings this whole
# layer exists to reach.
DEPOSITARY_PREFERRED_MARKERS = ("series", "%", "pfd")
# Kept as a narrow backstop for the three suffixes that are unambiguous even
# when the name is terse. A and B are share CLASSES and are never dropped.
NON_COMMON_SUFFIXES = set("WRU")


def nasdaq_listed(cfg: Dict[str, Any]) -> List[str]:
    """Candidate Nasdaq common stocks from the official symbol directory."""
    url = cfg.get("url") or NASDAQ_SYMBOL_FILE
    text = _http_get(url)
    if not text:
        log.warning("Nasdaq symbol directory fetch failed — no extra Nasdaq "
                    "coverage this run")
        return []
    cats = {str(c).upper() for c in (cfg.get("market_categories") or ["Q"])}
    out, seen_header, dropped = [], False, {"cat": 0, "etf": 0, "test": 0,
                                            "suffix": 0, "status": 0}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        if not seen_header:
            seen_header = True          # first row is the column header
            continue
        if parts[0].startswith("File Creation Time"):
            continue
        sym, name, cat, test, status, _lot = (parts[0].strip(), parts[1],
                                              parts[2].strip(), parts[3].strip(),
                                              parts[4].strip(), parts[5])
        etf = parts[6].strip() if len(parts) > 6 else "N"
        if not sym or not sym.isalpha():
            dropped["suffix"] += 1
            continue
        if test.upper() == "Y":
            dropped["test"] += 1
            continue
        if cfg.get("exclude_etfs", True) and etf.upper() == "Y":
            dropped["etf"] += 1
            continue
        if cats and cat.upper() not in cats:
            dropped["cat"] += 1
            continue
        # Financial status: N is normal. D (deficient), E (delinquent),
        # Q (bankrupt) and G/H/J/K are companies already in trouble with the
        # exchange, which is a different screen from this one.
        if cfg.get("normal_status_only", True) and status.upper() not in ("N", ""):
            dropped["status"] += 1
            continue
        desc = _instrument_desc(name)
        if _NON_COMMON_RE.search(desc):
            dropped["suffix"] += 1
            continue
        low = desc.lower()
        if "deposit" in low and any(x in low for x in DEPOSITARY_PREFERRED_MARKERS):
            dropped["suffix"] += 1
            continue
        if len(sym) == 5 and sym[4].upper() in NON_COMMON_SUFFIXES:
            dropped["suffix"] += 1
            continue
        out.append(sym.upper())
    log.info("Nasdaq directory: %d candidates after filters (dropped "
             "%d wrong tier, %d ETFs, %d test issues, %d non-common, "
             "%d exchange-flagged)", len(out), dropped["cat"], dropped["etf"],
             dropped["test"], dropped["suffix"], dropped["status"])
    return out


def rank_by_liquidity(yahoo: YahooProvider, symbols: List[str],
                      cfg: Dict[str, Any]) -> List[str]:
    """Keep the most traded candidates, measured on prices we fetch anyway.

    Returns [] rather than an arbitrary slice when the price download fails:
    an alphabetical cut of a four-thousand-name file is a universe of companies
    beginning with A, which would look like coverage and be nothing of the kind.
    """
    if not symbols:
        return []
    want = int(cfg.get("max_names", 150))
    floor = float(cfg.get("min_median_turnover_usd", 5_000_000))
    log.info("Ranking %d Nasdaq candidates by liquidity (keeping %d above "
             "USD %s median daily turnover)", len(symbols), want, f"{floor:,.0f}")
    frames = yahoo.prices_batch(symbols, period="6mo")
    scored = []
    for sym, df in (frames or {}).items():
        if df is None or len(df) < 40 or "Volume" not in df:
            continue
        try:
            turnover = float((df["Close"].astype(float)
                              * df["Volume"].astype(float)).iloc[-60:].median())
        except Exception:                        # noqa: BLE001
            continue
        if turnover >= floor:
            scored.append((sym, turnover))
    if not scored:
        log.warning("No Nasdaq candidate cleared the turnover floor — either "
                    "the price download failed or the floor is set too high; "
                    "no extra names added")
        return []
    scored.sort(key=lambda x: -x[1])
    kept = [s for s, _v in scored[:want]]
    log.info("Nasdaq liquidity pass: %d of %d priced, %d above the floor, "
             "%d kept (smallest kept trades USD %s a day)",
             len(frames or {}), len(symbols), len(scored), len(kept),
             f"{scored[min(len(scored), want) - 1][1]:,.0f}")
    return kept


SUFFIX_MARKET = {".T": "JP", ".SI": "SG", ".HK": "HK", ".BK": "TH",
                 ".JK": "ID", ".KL": "MY"}


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


def resolve_universe(universe_cfg: Dict, markets: List[str],
                     tags: Optional[Dict[str, str]] = None
                     ) -> Dict[str, List[str]]:
    """Resolve each market's ticker list, and record WHERE each name came from.

    `tags` is filled in with ticker -> listing label ("S&P 500", "Nasdaq-100",
    ...). Without it the extra coverage is invisible: a Nasdaq name unioned
    into the US list is indistinguishable from an S&P 500 one on the page, so
    a reader asking "did the Nasdaq names arrive?" has no way to answer.
    """
    out: Dict[str, List[str]] = {}
    tags = tags if tags is not None else {}
    for mkt in markets:
        m = universe_cfg["markets"].get(mkt)
        if not m:
            continue
        src = m.get("constituents_source")
        seed = m.get("seed_fallback", []) or list(m.get("constituents", []))

        if src == "dynamic":            # S&P 500 — its own two-source resolver
            names = [str(t).upper() for t in sp500_constituents(seed)
                     if t is not None and not isinstance(t, bool)]
            for _t in names:
                tags.setdefault(_t, "S&P 500")
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
                for _t in added:
                    tags.setdefault(_t.upper(), extra.get("name") or "index")
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
        # Every other market gets its own index name, so the label is present
        # on every row rather than only on the US ones.
        for _t in out.get(mkt, []):
            tags.setdefault(str(_t).upper(), m.get("index_name") or mkt)
    return out


def fetch_put_call(url: str):
    """Daily put/call ratio from a configured CSV, or nothing at all.

    Deliberately unforgiving: it takes the last numeric column of each row and
    refuses anything outside a plausible 0.2–3.0 band. A put/call ratio is a
    small number with a narrow range, so a parse that lands outside it has read
    the wrong column — and a wrong column here would feed the composite a
    number that looks like sentiment and is not.
    """
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            log.warning("put/call source -> HTTP %s", r.status_code)
            return None, []
        vals = []
        for line in r.text.splitlines():
            parts = [p.strip() for p in line.replace("\t", ",").split(",")]
            for p in reversed(parts):
                try:
                    v = float(p)
                except (TypeError, ValueError):
                    continue
                if 0.2 <= v <= 3.0:
                    vals.append(v)
                break
        if not vals:
            log.warning("put/call source parsed to nothing usable — the "
                        "composite will run without it")
            return None, []
        return vals[0], vals[:260]
    except Exception as e:                           # noqa: BLE001
        log.warning("put/call fetch failed: %s", e)
    return None, []


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
    rec.business_summary = prof.get("business_summary")
    rec.insider_ownership = prof.get("insider_ownership")
    rec.institutional_ownership = prof.get("institutional_ownership")
    rec.dividend_yield = prof.get("dividend_yield")
    rec.first_trade_date = prof.get("first_trade_date")
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
    # CPI comes back as an index level. Lynch's Rule of 20 wants a rate, and
    # every other reader of this field wants one too, so the conversion happens
    # once, here, rather than in each consumer.
    macro["us_cpi_yoy"] = None
    _cpi_id = (universe_cfg.get("macro_series", {}) or {}).get("us_cpi")
    if _cpi_id:
        try:
            _cpi = fred.history(_cpi_id, limit=24)
            if len(_cpi) >= 13 and _cpi[12][1]:
                macro["us_cpi_yoy"] = (_cpi[0][1] / _cpi[12][1] - 1) * 100.0
                log.info("US CPI year on year: %.2f%%", macro["us_cpi_yoy"])
            else:
                log.warning("CPI history too short for a year-on-year rate — "
                            "Lynch's Rule of 20 will show as unavailable")
        except Exception as e:                       # noqa: BLE001
            log.warning("CPI history failed: %s", e)

    listing_by_ticker: Dict[str, str] = {}
    tickers_by_market = merge_themes(
        universe_cfg,
        resolve_universe(universe_cfg, markets, listing_by_ticker), markets)

    # Nasdaq coverage beyond the Nasdaq-100, deduplicated against everything
    # already resolved. Done here rather than in resolve_universe because the
    # liquidity ranking needs the price provider, and ranking is what keeps
    # this from being an alphabetical slice of a four-thousand-name file.
    _nas = ((universe_cfg.get("markets", {}).get("US") or {})
            .get("nasdaq_listed") or {})
    if _nas.get("enabled") and "US" in tickers_by_market:
        try:
            have = {t.upper() for t in tickers_by_market["US"]}
            cands = [t for t in nasdaq_listed(_nas) if t.upper() not in have]
            log.info("Nasdaq extra: %d candidates after removing the %d names "
                     "already in the universe", len(cands), len(have))
            added = [t for t in rank_by_liquidity(yahoo, cands, _nas)
                     if t.upper() not in have]
            tickers_by_market["US"].extend(added)
            for _t in added:
                listing_by_ticker[_t.upper()] = "Nasdaq listed"
            log.info("Nasdaq extra: %d names added — US universe is now %d",
                     len(added), len(tickers_by_market["US"]))
        except Exception as e:                    # noqa: BLE001
            log.warning("Nasdaq extra coverage failed, continuing without it: %s", e)
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
                    rec.listing = listing_by_ticker.get(t.upper())
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
        m = mx.compute_metrics(
            rec, fx_rates=fx_rates,
            greenblatt_cfg=thresholds.get("greenblatt", {}))
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
    # Marks's gauge now votes on five markers, not three. Credit spreads and
    # CAPE are computed below for the debt cycle anyway, so feeding them in
    # costs nothing and removes the two biggest blind spots in the first
    # version — what credit charges for risk, and what equities cost.
    # The key in universe.yml is `hy_credit_spread`. An earlier version of this
    # line asked for `hy_spread`, got None, and the credit signal silently
    # stopped voting — the gauge ran on four signals while reporting five, and
    # the only visible symptom was a missing phrase in one banner. Accept both
    # spellings and say so loudly if neither is present.
    _hy = macro.get("hy_credit_spread")
    if _hy is None:
        _hy = macro.get("hy_spread")
    if _hy is None:
        log.warning("Credit spread absent from the macro snapshot (looked for "
                    "hy_credit_spread and hy_spread) — the cycle gauge will "
                    "run WITHOUT its credit vote")
    _hy_bp = _hy * 100.0 if isinstance(_hy, (int, float)) else None
    _cape_pre = dbt.universe_cape(
        records, metrics_by_ticker,
        fred.history("CPIAUCSL", limit=400) if fred.enabled else []).get("cape")
    cycle_state = cyc.assess(primary_idx, breadth, vix,
                             thresholds.get("market_cycle", {}),
                             hy_oas=_hy_bp, cape=_cape_pre)
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

    # Buffett's own market-level gauge. Two FRED series with different units,
    # so the provider reads the units rather than assuming them, and the whole
    # thing refuses to publish a figure it cannot defend.
    try:
        buf_ind = bf.buffett_indicator(fred)
        if buf_ind.get("available"):
            log.info("Buffett Indicator: %.1f%% — %s", buf_ind["pct"],
                     buf_ind["verdict"])
            if buf_ind.get("units_assumed"):
                log.warning("FRED units metadata unavailable — the Buffett "
                            "Indicator used assumed millions/billions scaling")
        else:
            log.warning("Buffett Indicator unavailable: %s",
                        buf_ind.get("reason"))
    except Exception as e:                           # noqa: BLE001
        log.warning("Buffett Indicator failed: %s", e)
        buf_ind = {"available": False, "reason": f"failed: {e}"}

    screened = sc.screen_universe(records, metrics_by_ticker, thresholds, macro,
                                  cycle=cycle_state)
    screened["buffett_indicator"] = buf_ind

    # ---- market psychology --------------------------------------------------
    # Six gauges of what the crowd is doing, all contrarian in application. Any
    # that cannot be computed says so; none is substituted with a proxy.
    try:
        scfg = universe_cfg.get("sentiment", {}) or {}
        vix_series = (vix_df["Close"].astype(float)
                      if vix_df is not None and len(vix_df) else None)
        vix3m_hist = fred.history("VXVCLS", limit=400) if fred.enabled else []
        vix3m = (pd.Series([v for _d, v in reversed(vix3m_hist)])
                 if vix3m_hist else None)
        sen_vix = sen.vix_state(vix_series, vix3m, scfg) if vix_series is not None \
            else {"available": False, "reason": "no VIX price history"}

        idx_close = (primary_idx["Close"].astype(float)
                     if primary_idx is not None and len(primary_idx) else None)
        idx_rsi = ta._last(ta.rsi(idx_close, 14)) if idx_close is not None else None
        sen_rsi = sen.rsi_state(
            idx_rsi, [m.get("rsi_14") for m in metrics_by_ticker.values()], scfg)

        ad = sen.advance_decline(frames)
        part = sen.participation(frames, idx_close)

        # Put/call: off unless a source is configured, and the composite says so.
        pcr_val, pcr_hist = None, []
        if scfg.get("put_call_url"):
            pcr_val, pcr_hist = fetch_put_call(scfg["put_call_url"])
        sen_pcr = sen.put_call_state(pcr_val, pcr_hist, scfg)

        hy_hist = ([v for _d, v in fred.history("BAMLH0A0HYM2", limit=300)]
                   if fred.enabled else [])
        bond_20d = None
        tnx = store.load_prices("^TNX")
        if tnx is not None and len(tnx) > 21:
            tc = tnx["Close"].astype(float)
            # Falling yields = rising bond prices; the sign is inverted on purpose.
            bond_20d = -float(tc.iloc[-1] / tc.iloc[-21] - 1)
        stock_20d = (float(idx_close.iloc[-1] / idx_close.iloc[-21] - 1)
                     if idx_close is not None and len(idx_close) > 21 else None)
        fg = sen.fear_greed(
            index_close=idx_close,
            high_low_ratio=part.get("high_low_ratio"),
            mcclellan=ad.get("mcclellan_oscillator"),
            pcr=pcr_val, hy_spread=_hy, hy_history=hy_hist,
            vix_level=sen_vix.get("level"), vix_ma50=sen_vix.get("ma50"),
            stock_20d=stock_20d, bond_20d=bond_20d)

        cot_rows = []
        try:
            cftc = CftcProvider()
            for label, code in (scfg.get("cot_contracts") or {}).items():
                st = sen.cot_state(
                    cftc.history(str(code), scfg.get("cot_weeks", 160)), label)
                cot_rows.append(st)
                if st.get("available"):
                    log.info("COT %s: %s (%.0fth percentile, week of %s)",
                             label, st["state"], st["percentile"] * 100,
                             st["report_date"])
                else:
                    log.warning("COT %s unavailable: %s", label, st.get("reason"))
        except Exception as e:                       # noqa: BLE001
            log.warning("COT feed failed: %s", e)

        screened["sentiment"] = sen.build(
            sen_vix, sen_rsi, sen_pcr, fg, cot_rows, ad, part,
            cycle_state.get("mode") if cycle_state else None)
        log.info("Sentiment: crowd=%s, fear&greed=%s on %s of 7 factors",
                 screened["sentiment"]["crowd"],
                 f"{fg['score']:.0f}" if fg.get("available") else "n/a",
                 fg.get("inputs_used", 0))
    except Exception as e:                           # noqa: BLE001
        log.warning("sentiment panel failed: %s", e)
        screened["sentiment"] = {"error": str(e)}
    screened["debt_cycle"] = debt_state

    # Rogers: "Why not just stop after that analysis and buy or sell the
    # commodity itself?" The board shows the underlying beside the instrument.
    try:
        screened["commodity_board"] = cmd.build(yahoo, store)
        log.info("Commodity board: %d rows, %d symbols unavailable",
                 len(screened["commodity_board"]["rows"]),
                 len(screened["commodity_board"]["missing"]))
    except Exception as e:                        # noqa: BLE001
        log.warning("commodity board failed: %s", e)
        screened["commodity_board"] = {}
    store.save_screen_results(run_id, screened["results"])

    # Merge in the other region's most recent results so the published page
    # always shows all five markets, each as fresh as its own last run.
    merged = store.latest_results()
    merged.update(screened["results"])

    # Where each name sits on Soros's path. Done after the merge so the other
    # region's stored rows are labelled too, and the census covers the whole
    # published table rather than just this run's half of it.
    census = rfx.annotate(merged, metrics_by_ticker)
    # Only the names that actually fell get an event lookup — three feeds
    # across 900 names would be thousands of requests for no purpose.
    dis_cfg = thresholds.get("dislocation") or {}
    disl = dis.scan(merged, metrics_by_ticker, dis_cfg)
    try:
        fallers = [t for t, r in merged.items() if r.get("dislocation")]
        if fallers:
            quakes = evt.quake_events(requests.Session())
            log.info("Quake feed: %s", ", ".join(
                f"{k}={len(v)}" for k, v in quakes.items()) or "nothing significant")
            sess = requests.Session()
            n_cik = 0
            for t in fallers[:dis_cfg.get("max_event_lookups", 60)]:
                m = metrics_by_ticker.get(t)
                if m is None:
                    continue
                # CompanyRecord carries no CIK — an earlier version of this
                # line read `getattr(rec, "cik", None)`, which is None on every
                # record, and the 8-K feed would have silently never fired.
                # The provider is the only thing that knows the mapping.
                cik = None
                if (merged[t].get("market") or "") == "US":
                    try:
                        cik = edgar.cik_for(t)
                    except Exception:              # noqa: BLE001
                        cik = None
                    n_cik += bool(cik)
                m["_events"] = evt.explain(
                    sess, t, cik, merged[t].get("market") or "",
                    quakes, use_news=dis_cfg.get("use_news", True))
            us_fallers = sum(1 for t in fallers
                             if (merged[t].get("market") or "") == "US")
            if us_fallers and not n_cik:
                log.error("NO CIK resolved for any of the %d US fallers — the "
                          "8-K feed produced nothing and every name will show "
                          "an inferred shortlist only", us_fallers)
            else:
                log.info("8-K lookup: %d of %d US fallers resolved to a CIK",
                         n_cik, us_fallers)
            if len(fallers) > dis_cfg.get("max_event_lookups", 60):
                log.warning("Event lookup capped at %d of %d fallers — the "
                            "rest show an inferred shortlist only",
                            dis_cfg.get("max_event_lookups", 60), len(fallers))
            disl = dis.scan(merged, metrics_by_ticker, dis_cfg)
    except Exception as e:                        # noqa: BLE001
        log.warning("event lookup failed, falling back to inference only: %s", e)
    screened["dislocation_summary"] = disl
    log.info("Dislocation: %d names fell >%.0f%% in 6m, %d with intact "
             "fundamentals", disl["fell_30pct"], abs(disl["threshold"]) * 100,
             disl["fundamentals_intact"])
    screened["reflexive_census"] = census
    if census:
        log.info("Reflexive stages: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(census.items())))

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
