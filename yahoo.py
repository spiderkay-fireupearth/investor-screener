"""Yahoo Finance provider — daily prices for all five markets, fundamentals for Asia.

Yahoo is the only free source that reaches SGX, HKEX, SET and IDX on both
prices and fundamentals. It is also unofficial, personal-use-only under Yahoo's
terms, and throttles aggressively. Everything here is built on that assumption:

  * every successful fetch is written to the local store before anything else
    happens, so a failed refresh costs one day of freshness, never history;
  * requests back off exponentially on 429 and never retry more than `retries`;
  * a market that returns nothing raises a staleness flag rather than silently
    producing an empty screen.

If Yahoo's Asian fundamentals prove too patchy, swap this class for an EODHD
adapter implementing the same three methods. Nothing downstream changes.
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
import random
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from ..schema import FundamentalYear

log = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:                       # pragma: no cover
    yf = None


# yfinance row label -> our schema field. Yahoo renames rows between versions,
# so each field lists fallbacks in priority order.
INCOME_MAP: Dict[str, List[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "sga_expense": ["Selling General And Administration",
                    "Selling General Administrative",
                    "General And Administrative Expense"],
    "operating_income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "pretax_income": ["Pretax Income"],
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "eps_diluted": ["Diluted EPS", "Basic EPS"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "shares_diluted": ["Diluted Average Shares", "Basic Average Shares"],
}

BALANCE_MAP: Dict[str, List[str]] = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "cash_and_equivalents": ["Cash And Cash Equivalents",
                             "Cash Cash Equivalents And Short Term Investments"],
    "short_term_investments": ["Other Short Term Investments", "Short Term Investments"],
    "total_debt": ["Total Debt", "Net Debt"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation",
                        "Short Term Debt", "Other Current Borrowings"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest",
                     "Common Stock Equity"],
    "minority_interest": ["Minority Interest"],
    "goodwill": ["Goodwill"],
    "intangibles": ["Other Intangible Assets", "Goodwill And Other Intangible Assets"],
    "inventory": ["Inventory"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "payables": ["Accounts Payable", "Payables"],
    "net_ppe": ["Net PPE", "Net Property Plant And Equipment"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
}

CASHFLOW_MAP: Dict[str, List[str]] = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "depreciation_amortization": ["Depreciation And Amortization",
                                  "Depreciation Amortization Depletion",
                                  "Depreciation"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "stock_based_compensation": ["Stock Based Compensation",
                                 "Share Based Compensation"],
}


def _pick(df: Optional[pd.DataFrame], labels: List[str], col) -> Optional[float]:
    if df is None or df.empty:
        return None
    for lab in labels:
        if lab in df.index:
            try:
                v = df.loc[lab, col]
            except (KeyError, IndexError):
                continue
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if v is not None and not pd.isna(v):
                return float(v)
    return None


class YahooProvider:
    """Prices for every market; fundamentals for the non-US markets."""

    def __init__(self, retries: int = 4, base_delay: float = 1.0,
                 polite_delay: float = 0.35):
        if yf is None:
            raise ImportError("yfinance is required: pip install yfinance")
        self.retries = retries
        self.base_delay = base_delay
        self.polite_delay = polite_delay
        self._last = 0.0

    # ------------------------------------------------------------------ utils
    def _pace(self):
        gap = time.time() - self._last
        if gap < self.polite_delay:
            time.sleep(self.polite_delay - gap)
        self._last = time.time()

    def _with_retry(self, fn, what: str):
        for attempt in range(self.retries):
            self._pace()
            try:
                return fn()
            except Exception as e:                      # noqa: BLE001
                msg = str(e).lower()
                transient = ("429" in msg or "too many" in msg or "timed out" in msg
                             or "connection" in msg or "unavailable" in msg)
                if attempt == self.retries - 1 or not transient:
                    log.warning("yahoo %s failed: %s", what, e)
                    return None
                sleep = self.base_delay * (2 ** attempt) + random.uniform(0, 0.6)
                log.info("yahoo %s throttled, backing off %.1fs", what, sleep)
                time.sleep(sleep)
        return None

    # ----------------------------------------------------------------- prices
    def prices(self, ticker: str, period: str = "10y") -> Optional[pd.DataFrame]:
        """Split- and dividend-adjusted daily OHLCV, newest last."""
        def _fetch():
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval="1d", auto_adjust=True,
                           raise_errors=True)
            if df is None or df.empty:
                raise ValueError("empty price frame")
            return df

        df = self._with_retry(_fetch, f"prices({ticker})")
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["Close"])
        # Yahoo occasionally returns a tz-aware index; normalise to naive dates.
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except (TypeError, AttributeError):
            df.index = pd.to_datetime(df.index)
        return df

    def prices_batch(self, tickers: List[str], period: str = "10y") -> Dict[str, pd.DataFrame]:
        """Batch download is far kinder to Yahoo's rate limits than N single calls."""
        out: Dict[str, pd.DataFrame] = {}
        CHUNK = 40
        for i in range(0, len(tickers), CHUNK):
            chunk = tickers[i:i + CHUNK]

            def _fetch():
                return yf.download(chunk, period=period, interval="1d",
                                   auto_adjust=True, group_by="ticker",
                                   progress=False, threads=True)

            raw = self._with_retry(_fetch, f"batch[{i}:{i+len(chunk)}]")
            if raw is None or raw.empty:
                continue
            for t in chunk:
                try:
                    sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                    sub = sub.dropna(subset=["Close"])
                    if sub.empty:
                        continue
                    try:
                        sub.index = pd.to_datetime(sub.index).tz_localize(None)
                    except (TypeError, AttributeError):
                        sub.index = pd.to_datetime(sub.index)
                    out[t] = sub
                except (KeyError, TypeError):
                    continue
            time.sleep(0.8)     # breathe between chunks
        return out

    # ------------------------------------------------------------------ meta
    def profile(self, ticker: str) -> Dict[str, Any]:
        def _fetch():
            t = yf.Ticker(ticker)
            info = {}
            try:
                fi = t.fast_info
                info["price"] = getattr(fi, "last_price", None)
                info["market_cap"] = getattr(fi, "market_cap", None)
                info["shares_outstanding"] = getattr(fi, "shares", None)
                info["currency"] = getattr(fi, "currency", None)
            except Exception:                            # noqa: BLE001
                pass
            try:
                full = t.get_info()
                info.setdefault("price", full.get("currentPrice"))
                info.setdefault("market_cap", full.get("marketCap"))
                info.setdefault("shares_outstanding", full.get("sharesOutstanding"))
                info.setdefault("currency", full.get("currency"))
                # The currency the FINANCIAL STATEMENTS are reported in, which
                # is frequently NOT the currency the shares trade in. Many
                # HKEX-listed mainland issuers report in CNY but trade in HKD;
                # several SGX names report in USD but trade in SGD. Comparing a
                # HKD market cap against CNY book value silently corrupts every
                # valuation ratio, so this field is load-bearing.
                info["financial_currency"] = full.get("financialCurrency")
                info["name"] = full.get("longName") or full.get("shortName")
                info["sector"] = full.get("sector")
                info["industry"] = full.get("industry")
                # What the company actually does. Everything else on this page
                # is a ratio; without one line of description a screener asks
                # you to judge a business you cannot name.
                info["business_summary"] = full.get("longBusinessSummary")
                info["trailing_pe"] = full.get("trailingPE")
                # ETF / MUTUALFUND / EQUITY. A fund has no revenue, equity or
                # ROE, so the value frameworks must be skipped rather than
                # failed on missing data.
                info["quote_type"] = full.get("quoteType")
                # Ownership. Lynch bought what the institutions had not found;
                # Schloss bought where the managers had their own money. Both
                # come free with the profile call that already runs.
                info["insider_ownership"] = full.get("heldPercentInsiders")
                info["institutional_ownership"] = full.get("heldPercentInstitutions")
                dy = full.get("dividendYield")
                if dy is None:
                    dy = full.get("trailingAnnualDividendYield")
                # Yahoo is inconsistent: some rows carry 0.031, others 3.1 for
                # the same 3.1%. Anything above 1 is a percentage, not a ratio —
                # a genuine 100%+ yield does not exist outside a data error.
                if isinstance(dy, (int, float)) and dy > 1:
                    dy = dy / 100.0
                info["dividend_yield"] = dy
                # First quote date — the only listing-age signal available for
                # every market. Schloss wanted 20+ years of operating history,
                # which no 10-year statement feed can confirm on its own.
                ft = (full.get("firstTradeDateEpochUtc")
                      or full.get("firstTradeDateMilliseconds"))
                if isinstance(ft, (int, float)) and ft > 0:
                    secs = ft / 1000.0 if ft > 4e10 else float(ft)
                    try:
                        info["first_trade_date"] = _dt.datetime.fromtimestamp(
                            secs, _dt.timezone.utc).date().isoformat()
                    except (OverflowError, OSError, ValueError):
                        pass
            except Exception:                            # noqa: BLE001
                pass
            return info

        return self._with_retry(_fetch, f"profile({ticker})") or {}

    # ----------------------------------------------------------- fundamentals
    def fundamentals(self, ticker: str, currency: str = "USD",
                     years: int = 10) -> List[FundamentalYear]:
        def _fetch():
            t = yf.Ticker(ticker)
            return t.income_stmt, t.balance_sheet, t.cashflow

        res = self._with_retry(_fetch, f"fundamentals({ticker})")
        if res is None:
            return []
        inc, bal, cf = res
        if (inc is None or inc.empty) and (bal is None or bal.empty):
            return []

        cols = []
        for df in (inc, bal, cf):
            if df is not None and not df.empty:
                cols.extend(list(df.columns))
        cols = sorted(set(cols), reverse=True)[:years]

        out: List[FundamentalYear] = []
        for col in cols:
            ts = pd.Timestamp(col)
            fy = FundamentalYear(
                ticker=ticker,
                fiscal_year=int(ts.year),
                period_end=ts.date().isoformat(),
                standard="ifrs",
                currency=currency,
                source="yahoo",
            )
            missing: List[str] = []
            for target, (df, mapping) in {
                **{k: (inc, INCOME_MAP) for k in INCOME_MAP},
                **{k: (bal, BALANCE_MAP) for k in BALANCE_MAP},
                **{k: (cf, CASHFLOW_MAP) for k in CASHFLOW_MAP},
            }.items():
                val = _pick(df, mapping[target], col)
                if val is None:
                    missing.append(target)
                else:
                    setattr(fy, target, val)

            # Yahoo reports capex as a negative number; the schema wants magnitude.
            if fy.capex is not None:
                fy.capex = abs(fy.capex)
            # "Cash Cash Equivalents And Short Term Investments" double-counts if
            # we also picked up short_term_investments — prefer the narrower tag.
            if (fy.cash_and_equivalents and fy.short_term_investments
                    and fy.cash_and_equivalents < fy.short_term_investments):
                fy.short_term_investments = None
            if fy.total_liabilities is None and fy.total_assets and fy.total_equity:
                fy.total_liabilities = fy.total_assets - fy.total_equity
                if "total_liabilities" in missing:
                    missing.remove("total_liabilities")
            fy.missing_fields = missing
            out.append(fy)

        out.sort(key=lambda x: x.fiscal_year, reverse=True)
        return out

    # --------------------------------------------------------------------- fx
    def fx_rate(self, pair_ticker: str) -> Optional[float]:
        df = self.prices(pair_ticker, period="1mo")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
