"""SEC EDGAR XBRL provider — US fundamentals from the audited filings.

Why this exists rather than pulling US fundamentals from Yahoo too:
Greenblatt's return-on-capital needs net working capital and net fixed assets,
and Klarman's NCAV needs current assets against TOTAL liabilities. Yahoo's
summarised statements are unreliable on exactly those lines. EDGAR has them,
free, permanently, with no quota to outgrow.

The cost is tag normalisation, which is what most of this file is.

SEC requires a descriptive User-Agent with contact info. Set SEC_USER_AGENT,
e.g. "MyScreener/1.0 (you@example.com)". Requests are throttled to <10/sec.
"""
from __future__ import annotations

import json
import os
import time
import logging
from typing import Dict, List, Optional, Any

import requests

from ..schema import FundamentalYear

log = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_UA = "investor-screener/1.0 (contact: set-SEC_USER_AGENT-env-var)"

# SEC asks for <10 requests/second. We stay well under.
_MIN_INTERVAL = 0.12
_last_call = [0.0]


def _throttle():
    delta = time.time() - _last_call[0]
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_call[0] = time.time()


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_UA),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _get(url: str, host: str = "data.sec.gov", retries: int = 3) -> Optional[dict]:
    h = _headers()
    h["Host"] = host
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(url, headers=h, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            log.warning("EDGAR %s -> HTTP %s", url, r.status_code)
            return None
        except Exception as e:            # noqa: BLE001
            log.warning("EDGAR %s failed (%s/%s): %s", url, attempt + 1, retries, e)
            time.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Tag mapping. Order matters — first tag that yields data wins.
# ---------------------------------------------------------------------------
TAGS: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "interest_expense": [
        "InterestExpense",
        "InterestIncomeExpenseNet",
        "InterestExpenseDebt",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "minority_interest": ["MinorityInterest"],
    "goodwill": ["Goodwill"],
    "intangibles": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "inventory": ["InventoryNet", "InventoryGross"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "payables": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "CommonStockSharesIssued"],
}

# Debt is a sum of several tags rather than a single lookup.
DEBT_TAGS_LONG = ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermNotesPayable",
                  "OperatingLeaseLiabilityNoncurrent"]
DEBT_TAGS_SHORT = ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent",
                   "CommercialPaper", "OperatingLeaseLiabilityCurrent"]


class EdgarProvider:
    def __init__(self, cache_dir: str = "data/edgar_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._ticker_map: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------ CIK
    def _load_ticker_map(self) -> Dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        cache = os.path.join(self.cache_dir, "company_tickers.json")
        data = None
        if os.path.exists(cache) and (time.time() - os.path.getmtime(cache) < 7 * 86400):
            try:
                with open(cache) as f:
                    data = json.load(f)
            except Exception:       # noqa: BLE001
                data = None
        if data is None:
            data = _get(TICKER_MAP_URL, host="www.sec.gov")
            if data:
                with open(cache, "w") as f:
                    json.dump(data, f)
        mapping: Dict[str, str] = {}
        if data:
            for row in data.values():
                mapping[str(row["ticker"]).upper()] = str(row["cik_str"]).zfill(10)
        self._ticker_map = mapping
        return mapping

    def cik_for(self, ticker: str) -> Optional[str]:
        m = self._load_ticker_map()
        t = ticker.upper()
        # Yahoo writes BRK-B where SEC writes BRK.B
        for candidate in (t, t.replace("-", "."), t.replace(".", "-")):
            if candidate in m:
                return m[candidate]
        return None

    # ----------------------------------------------------------- companyfacts
    def _company_facts(self, cik: str) -> Optional[dict]:
        cache = os.path.join(self.cache_dir, f"CIK{cik}.json")
        if os.path.exists(cache) and (time.time() - os.path.getmtime(cache) < 86400):
            try:
                with open(cache) as f:
                    return json.load(f)
            except Exception:       # noqa: BLE001
                pass
        data = _get(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        if data:
            with open(cache, "w") as f:
                json.dump(data, f)
        return data

    # ---------------------------------------------------------------- extract
    @staticmethod
    def _annual_values(facts: dict, tags: List[str]) -> Dict[int, float]:
        """Return {fiscal_year: value} using the first tag that has annual data.

        Handles restatements by preferring the most recently FILED figure for
        each fiscal year — which is what an investor reading today's 10-K sees.
        """
        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei = facts.get("facts", {}).get("dei", {})
        for tag in tags:
            node = gaap.get(tag) or dei.get(tag)
            if not node:
                continue
            best: Dict[int, tuple] = {}     # fy -> (filed_date, value)
            for unit_rows in node.get("units", {}).values():
                for row in unit_rows:
                    if row.get("form") not in ("10-K", "10-K/A", "20-F", "40-F"):
                        continue
                    if row.get("fp") != "FY":
                        continue
                    fy = row.get("fy")
                    val = row.get("val")
                    filed = row.get("filed", "")
                    if fy is None or val is None:
                        continue
                    # Income/cashflow items must cover a full year, not a quarter.
                    start, end = row.get("start"), row.get("end")
                    if start and end:
                        try:
                            from datetime import date
                            d0 = date.fromisoformat(start)
                            d1 = date.fromisoformat(end)
                            if (d1 - d0).days < 300:
                                continue
                        except Exception:   # noqa: BLE001
                            pass
                    prev = best.get(fy)
                    if prev is None or filed > prev[0]:
                        best[fy] = (filed, float(val))
            if best:
                return {fy: v for fy, (_, v) in best.items()}
        return {}

    @staticmethod
    def _period_ends(facts: dict) -> Dict[int, str]:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        node = gaap.get("Assets")
        out: Dict[int, str] = {}
        if node:
            for unit_rows in node.get("units", {}).values():
                for row in unit_rows:
                    if row.get("fp") == "FY" and row.get("fy") and row.get("end"):
                        out.setdefault(row["fy"], row["end"])
        return out

    def _sum_tags(self, facts: dict, tags: List[str]) -> Dict[int, float]:
        """Sum several tags per fiscal year (used for total debt)."""
        totals: Dict[int, float] = {}
        for tag in tags:
            vals = self._annual_values(facts, [tag])
            for fy, v in vals.items():
                totals[fy] = totals.get(fy, 0.0) + v
        return totals

    # ------------------------------------------------------------------ public
    def fetch(self, ticker: str, years: int = 12) -> List[FundamentalYear]:
        cik = self.cik_for(ticker)
        if not cik:
            log.info("No CIK for %s — not an SEC filer", ticker)
            return []
        facts = self._company_facts(cik)
        if not facts:
            return []

        series: Dict[str, Dict[int, float]] = {
            k: self._annual_values(facts, tags) for k, tags in TAGS.items()
        }
        debt_long = self._sum_tags(facts, DEBT_TAGS_LONG)
        debt_short = self._sum_tags(facts, DEBT_TAGS_SHORT)
        ends = self._period_ends(facts)

        all_years = set()
        for d in series.values():
            all_years |= set(d.keys())
        all_years |= set(debt_long) | set(debt_short)
        ordered = sorted(all_years, reverse=True)[:years]

        out: List[FundamentalYear] = []
        for fy in ordered:
            fy_obj = FundamentalYear(
                ticker=ticker,
                fiscal_year=int(fy),
                period_end=ends.get(fy, f"{fy}-12-31"),
                standard="us-gaap",
                currency="USD",
                source="edgar",
            )
            missing = []
            for fieldname in TAGS:
                val = series.get(fieldname, {}).get(fy)
                if val is None:
                    missing.append(fieldname)
                else:
                    setattr(fy_obj, fieldname, val)
            dl, ds = debt_long.get(fy), debt_short.get(fy)
            td = (dl or 0.0) + (ds or 0.0)
            fy_obj.total_debt = td if td > 0 else None
            # Kept apart as well as summed: Lynch's bank-debt-versus-funded-debt
            # test needs the split, and EDGAR has already tagged it.
            fy_obj.long_term_debt = dl
            fy_obj.short_term_debt = ds
            if fy_obj.total_debt is None:
                missing.append("total_debt")
            # Total liabilities is often omitted; derive from assets - equity.
            if fy_obj.total_liabilities is None and fy_obj.total_assets and fy_obj.total_equity:
                fy_obj.total_liabilities = fy_obj.total_assets - fy_obj.total_equity
                if "total_liabilities" in missing:
                    missing.remove("total_liabilities")
            # Same for gross profit when only revenue and COGS-derived data exist.
            fy_obj.missing_fields = missing
            out.append(fy_obj)
        return out
