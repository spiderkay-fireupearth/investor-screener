"""CFTC Commitments of Traders — free, no API key.

The CFTC publishes the weekly COT report through a Socrata open-data endpoint.
No key, no registration, generous rate limits. The legacy futures-only dataset
carries the split this app wants: non-commercial (speculators) against
commercial (hedgers who handle the physical), plus open interest to normalise
by.

Two things this module refuses to do:

  * GUESS AT A CONTRACT. Market names in the CFTC file are long, punctuated and
    inconsistent ("GOLD - COMMODITY EXCHANGE INC."). The contract codes are
    stable; the names are not. So lookups go by CODE, configured in
    universe.yml, and a code that returns nothing is reported as returning
    nothing rather than being fuzzily matched to something else.

  * PRETEND TO BE CURRENT. The report is published Friday for the preceding
    Tuesday, so it is always at least three days stale and often more. The
    report date travels with every row, and the panel prints it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# Socrata: legacy COT, futures only. Public, keyless.
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

FIELDS = ("report_date_as_yyyy_mm_dd,open_interest_all,"
          "noncomm_positions_long_all,noncomm_positions_short_all,"
          "comm_positions_long_all,comm_positions_short_all,"
          "market_and_exchange_names,cftc_contract_market_code")


class CftcProvider:
    def __init__(self, session: Optional[Any] = None, timeout: int = 30):
        self.session = session or requests.Session()
        self.timeout = timeout

    def history(self, contract_code: str, weeks: int = 160) -> List[Dict[str, Any]]:
        """Newest-first weekly rows for one contract market code."""
        try:
            r = self.session.get(COT_URL, timeout=self.timeout, params={
                "cftc_contract_market_code": contract_code,
                "$select": FIELDS,
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": weeks,
            })
            if r.status_code != 200:
                log.warning("CFTC %s -> HTTP %s", contract_code, r.status_code)
                return []
            rows = r.json()
            if not isinstance(rows, list):
                log.warning("CFTC %s returned %s, not a list",
                            contract_code, type(rows).__name__)
                return []
            if not rows:
                log.warning("CFTC returned no rows for contract code %s — check "
                            "the code in universe.yml against the CFTC's own "
                            "list; codes are stable, market names are not",
                            contract_code)
            return rows
        except Exception as e:                       # noqa: BLE001
            log.warning("CFTC %s failed: %s", contract_code, e)
        return []
