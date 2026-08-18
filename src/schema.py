"""Common fundamental schema.

The whole point of this module: US filers report US GAAP, and SGX/HKEX/SET/IDX
filers report IFRS or a local variant. "Operating income", "book value" and
"total debt" are not the same line item in both. Every provider must map its
native output into `FundamentalYear` so that downstream screens compare like
with like, and must declare which `standard` it used so screens can adjust.

All monetary values are in the filing's REPORTING currency. Conversion to USD
happens once, in fx.py, at ranking time — never here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import math


def _isnum(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


@dataclass
class FundamentalYear:
    """One fiscal year of normalised financials for one company."""

    ticker: str
    fiscal_year: int
    period_end: str                      # ISO date
    standard: str                        # 'us-gaap' | 'ifrs'
    currency: str
    source: str                          # 'edgar' | 'yahoo'

    # --- Income statement -------------------------------------------------
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None      # EBIT
    pretax_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    interest_expense: Optional[float] = None

    # --- Balance sheet ----------------------------------------------------
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    minority_interest: Optional[float] = None
    goodwill: Optional[float] = None
    intangibles: Optional[float] = None
    inventory: Optional[float] = None
    receivables: Optional[float] = None
    payables: Optional[float] = None
    net_ppe: Optional[float] = None

    # --- Cash flow --------------------------------------------------------
    cfo: Optional[float] = None
    capex: Optional[float] = None                 # stored positive
    depreciation_amortization: Optional[float] = None
    dividends_paid: Optional[float] = None

    # --- Share counts -----------------------------------------------------
    shares_diluted: Optional[float] = None
    shares_outstanding: Optional[float] = None

    # --- Provenance -------------------------------------------------------
    missing_fields: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- derived
    @property
    def ebit(self) -> Optional[float]:
        if _isnum(self.operating_income):
            return self.operating_income
        # IFRS filers often omit a clean operating income line. Reconstruct.
        if _isnum(self.pretax_income) and _isnum(self.interest_expense):
            return self.pretax_income + abs(self.interest_expense)
        return None

    @property
    def ebitda(self) -> Optional[float]:
        e, da = self.ebit, self.depreciation_amortization
        if _isnum(e) and _isnum(da):
            return e + da
        return None

    @property
    def free_cash_flow(self) -> Optional[float]:
        if _isnum(self.cfo) and _isnum(self.capex):
            return self.cfo - abs(self.capex)
        return None

    @property
    def net_debt(self) -> Optional[float]:
        if not _isnum(self.total_debt):
            return None
        cash = (self.cash_and_equivalents or 0) + (self.short_term_investments or 0)
        return self.total_debt - cash

    @property
    def tangible_book_value(self) -> Optional[float]:
        if not _isnum(self.total_equity):
            return None
        return self.total_equity - (self.goodwill or 0) - (self.intangibles or 0)

    @property
    def net_working_capital(self) -> Optional[float]:
        """Greenblatt's NWC: excludes excess cash and interest-bearing debt."""
        if not (_isnum(self.current_assets) and _isnum(self.current_liabilities)):
            return None
        # Strip cash from current assets — it isn't capital the business employs.
        ca = self.current_assets - (self.cash_and_equivalents or 0)
        return max(ca - self.current_liabilities, 0.0)

    @property
    def invested_capital(self) -> Optional[float]:
        """Greenblatt's denominator: net working capital + net fixed assets."""
        nwc, ppe = self.net_working_capital, self.net_ppe
        if nwc is None or not _isnum(ppe):
            return None
        ic = nwc + ppe
        return ic if ic > 0 else None

    @property
    def ncav(self) -> Optional[float]:
        """Graham/Klarman net current asset value: current assets less ALL liabilities."""
        if not (_isnum(self.current_assets) and _isnum(self.total_liabilities)):
            return None
        return self.current_assets - self.total_liabilities

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "ebit": self.ebit,
            "ebitda": self.ebitda,
            "free_cash_flow": self.free_cash_flow,
            "net_debt": self.net_debt,
            "tangible_book_value": self.tangible_book_value,
            "invested_capital": self.invested_capital,
            "ncav": self.ncav,
        })
        return d


@dataclass
class CompanyRecord:
    """Everything the screens need about one company."""

    ticker: str
    market: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    business_summary: Optional[str] = None    # one-paragraph blurb from the feed
    currency: str = "USD"            # currency the SHARES TRADE in
    financial_currency: Optional[str] = None   # currency the STATEMENTS use
    standard: str = "ifrs"
    quote_type: Optional[str] = None      # EQUITY | ETF | MUTUALFUND ...
    themes: List[str] = field(default_factory=list)

    years: List[FundamentalYear] = field(default_factory=list)   # newest first

    # Market data (reporting currency)
    price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    median_turnover: Optional[float] = None

    # Technicals, populated by technicals.py
    technicals: Dict[str, Any] = field(default_factory=dict)

    # Data-quality flags surfaced in the UI
    warnings: List[str] = field(default_factory=list)

    @property
    def latest(self) -> Optional[FundamentalYear]:
        return self.years[0] if self.years else None

    def year_series(self, attr: str, n: int = 10) -> List[Optional[float]]:
        """Newest-first series of one field or derived property."""
        out = []
        for y in self.years[:n]:
            out.append(getattr(y, attr, None))
        return out
