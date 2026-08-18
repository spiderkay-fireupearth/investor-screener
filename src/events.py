"""What actually happened, from feeds that say so in words or codes.

The dislocation scan can see that a price fell and the accounts did not. It
cannot see WHY, because neither of its feeds contains any text. This module adds
the missing half — and it is deliberately built around *structured* sources
first and headlines last, because a coded event is evidence and a headline is
an impression.

Three feeds, in descending order of how much they can be trusted:

1.  **SEC 8-K item codes.** A US issuer must file within four business days of a
    material event, tagged with a numbered item. Item 5.02 IS "departure of
    directors or certain officers" — it is not a model's guess that a headline
    sounds like a resignation. Free, no key, and already the endpoint this app
    uses for fundamentals.

    Two of the codes matter more than the rest, and they point the opposite way
    to what you would expect from a "dislocation" screen: **4.02** (previously
    issued financials should no longer be relied upon) and **1.03**
    (bankruptcy). A name that fell 30% while its accounts looked healthy, and
    which has filed a 4.02, has not been dislocated — its accounts are fiction
    and the market worked it out first. Those codes DISQUALIFY.

2.  **USGS earthquakes.** Free, no key, structured, global. Matters because
    three of the six markets in this universe sit on active margins, and it is
    the only one of the fifteen causes with a definitive public register.

3.  **GDELT and, optionally, a commercial news API.** Words. Useful for
    narrowing, never for concluding. GDELT is free and global; the commercial
    feeds are US-skewed, which is the wrong shape for a universe that is
    five-sixths non-US — a feed that is rich on the S&P and thin on the SET
    makes the app look most confident exactly where it knows least.

Nothing here decides anything. It attaches evidence to a name so that the
fifteen candidate causes can collapse to one observed event, or stay a
shortlist when no feed saw anything.

TESTING NOTE: the sandbox has no route to sec.gov, usgs.gov or gdeltproject.org,
so the PARSING here is unit tested against fixtures and the FETCHING is not.
Every network path fails soft and reports what it could not reach.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8-K item codes, mapped to the fifteen causes. Only codes that say something
# about WHY a price moved are listed; routine ones (2.02 results, 7.01 Reg FD,
# 9.01 exhibits) are deliberately absent.
# ---------------------------------------------------------------------------
ITEM_CODES: Dict[str, Dict[str, Any]] = {
    "1.01": {"label": "Entry into a material agreement", "cause": None},
    "1.02": {"label": "Termination of a material agreement", "cause": None},
    "1.03": {"label": "Bankruptcy or receivership", "cause": None,
             "disqualifies": "in bankruptcy — the fall is not a dislocation"},
    "2.04": {"label": "Debt acceleration or increased obligation", "cause": 7},
    "2.06": {"label": "Material impairment", "cause": 2},
    "3.01": {"label": "Delisting notice or listing-rule failure", "cause": 6,
             "disqualifies": "facing delisting — a governance failure, not a "
                             "panic"},
    "4.01": {"label": "Change of auditor", "cause": 5},
    "4.02": {"label": "Prior financial statements no longer reliable",
             "cause": 5,
             "disqualifies": "has told the market its own past accounts cannot "
                             "be relied on. The 'intact fundamentals' this "
                             "screen tested are, by the company's own "
                             "admission, not intact"},
    "5.02": {"label": "Departure or appointment of directors or officers",
             "cause": 5},
    "5.03": {"label": "Change in fiscal year or by-laws", "cause": 6},
    "7.01": {"label": "Regulation FD disclosure", "cause": 13},
    "8.01": {"label": "Other material event", "cause": None},
}

# Market bounding boxes, for matching a quake to an exchange. Deliberately
# generous — a quake offshore of a country still closes its factories.
MARKET_BOX = {
    "JP": (24.0, 46.0, 122.0, 154.0),
    "ID": (-11.0, 6.0, 95.0, 141.0),
    "TH": (5.0, 21.0, 97.0, 106.0),
    "HK": (21.0, 23.5, 113.0, 115.0),
    "SG": (0.5, 2.0, 103.0, 105.0),
    "US": (24.0, 50.0, -125.0, -66.0),
}


def _ua() -> str:
    return (os.environ.get("SEC_USER_AGENT")
            or "investor-screener/1.0 (contact via github)")


def _json(session, url: str, timeout: int = 30, headers: Optional[Dict] = None):
    try:
        r = session.get(url, timeout=timeout,
                        headers=headers or {"User-Agent": _ua()})
        if r.status_code != 200:
            log.warning("%s -> HTTP %s", url.split("?")[0], r.status_code)
            return None
        return r.json()
    except Exception as e:                          # noqa: BLE001
        log.warning("%s failed: %s", url.split("?")[0], e)
        return None


# ---------------------------------------------------------------------------
# 1. SEC 8-K
# ---------------------------------------------------------------------------

def parse_submissions(doc: Dict[str, Any], since: str,
                      forms=("8-K",)) -> List[Dict[str, Any]]:
    """Pull recent filings of the given forms out of a submissions JSON.

    The `items` field is what makes this worth doing: EDGAR already tells us
    which numbered item each 8-K reports, so the cause arrives as a code rather
    than as prose to be interpreted.
    """
    recent = ((doc or {}).get("filings") or {}).get("recent") or {}
    dates = recent.get("filingDate") or []
    types = recent.get("form") or []
    items = recent.get("items") or []
    accs = recent.get("accessionNumber") or []
    out = []
    for i, d in enumerate(dates):
        if d < since:
            continue
        if i < len(types) and types[i] not in forms:
            continue
        raw = items[i] if i < len(items) else ""
        codes = [c.strip() for c in str(raw).split(",") if c.strip()]
        known = [c for c in codes if c in ITEM_CODES]
        out.append({
            "source": "SEC 8-K",
            "date": d,
            "form": types[i] if i < len(types) else "",
            "accession": accs[i] if i < len(accs) else "",
            "codes": codes,
            "labels": [ITEM_CODES[c]["label"] for c in known],
            "causes": sorted({ITEM_CODES[c]["cause"] for c in known
                              if ITEM_CODES[c].get("cause")}),
            "disqualifies": next((ITEM_CODES[c]["disqualifies"] for c in known
                                  if ITEM_CODES[c].get("disqualifies")), None),
        })
    return out


def sec_events(session, cik: str, days: int = 180) -> List[Dict[str, Any]]:
    if not cik:
        return []
    cik10 = str(cik).lstrip("0").zfill(10)
    doc = _json(session, f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if not doc:
        return []
    since = (date.today() - timedelta(days=days)).isoformat()
    return parse_submissions(doc, since)


# ---------------------------------------------------------------------------
# 2. USGS
# ---------------------------------------------------------------------------

def parse_quakes(geojson: Dict[str, Any], min_mag: float = 6.0
                 ) -> Dict[str, List[Dict[str, Any]]]:
    """Group significant quakes by the market whose box they fall in."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for f in (geojson or {}).get("features", []):
        props = f.get("properties") or {}
        geom = (f.get("geometry") or {}).get("coordinates") or []
        mag = props.get("mag")
        if mag is None or mag < min_mag or len(geom) < 2:
            continue
        lon, lat = float(geom[0]), float(geom[1])
        for mkt, (lo_lat, hi_lat, lo_lon, hi_lon) in MARKET_BOX.items():
            if lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon:
                ts = props.get("time")
                out.setdefault(mkt, []).append({
                    "source": "USGS",
                    "magnitude": float(mag),
                    "place": props.get("place"),
                    "date": (date.fromtimestamp(ts / 1000).isoformat()
                             if ts else None),
                    "causes": [2],
                })
    return out


def quake_events(session, days: int = 180, min_mag: float = 6.0
                 ) -> Dict[str, List[Dict[str, Any]]]:
    start = (date.today() - timedelta(days=days)).isoformat()
    doc = _json(session,
                "https://earthquake.usgs.gov/fdsnws/event/1/query"
                f"?format=geojson&starttime={start}&minmagnitude={min_mag}",
                timeout=45, headers={"User-Agent": _ua()})
    return parse_quakes(doc, min_mag) if doc else {}


# ---------------------------------------------------------------------------
# 3. GDELT, and an optional commercial feed
# ---------------------------------------------------------------------------

def parse_gdelt(doc: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    arts = (doc or {}).get("articles") or []
    return [{"source": "GDELT", "title": a.get("title"),
             "url": a.get("url"), "date": (a.get("seendate") or "")[:8],
             "domain": a.get("domain")}
            for a in arts[:limit]]


def gdelt_events(session, query: str, days: int = 90,
                 limit: int = 5) -> List[Dict[str, Any]]:
    """Free, no key. Used for context, never for a conclusion."""
    span = f"{min(days, 365) * 24}H"
    doc = _json(session,
                "https://api.gdeltproject.org/api/v2/doc/doc"
                f"?query={query}&mode=artlist&maxrecords={limit}"
                f"&timespan={span}&format=json", timeout=30)
    return parse_gdelt(doc, limit)


def news_events(session, symbol: str, days: int = 90,
                limit: int = 5) -> List[Dict[str, Any]]:
    """Finnhub company news. Optional: skipped entirely without a key.

    Kept last and kept small on purpose. Coverage is US-skewed, and this
    universe is five-sixths non-US, so an absent result here says more about
    the vendor than about the company.
    """
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    frm = (date.today() - timedelta(days=days)).isoformat()
    doc = _json(session, "https://finnhub.io/api/v1/company-news"
                         f"?symbol={symbol}&from={frm}&to={date.today()}"
                         f"&token={key}", timeout=30)
    if not isinstance(doc, list):
        return []
    return [{"source": "Finnhub", "title": a.get("headline"),
             "url": a.get("url"), "date": a.get("datetime"),
             "publisher": a.get("source")} for a in doc[:limit]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def explain(session, ticker: str, cik: Optional[str], market: str,
            quakes_by_market: Optional[Dict[str, list]] = None,
            days: int = 180, use_news: bool = True) -> Dict[str, Any]:
    """Everything the feeds know about why this name might have fallen."""
    ev: List[Dict[str, Any]] = []
    reached: List[str] = []
    missed: List[str] = []

    filings = sec_events(session, cik, days) if cik else []
    if cik:
        (reached if filings is not None else missed).append("SEC 8-K")
    ev.extend(filings or [])

    for q in (quakes_by_market or {}).get(market, []):
        ev.append(q)

    if use_news:
        g = gdelt_events(session, f'"{ticker}"', min(days, 90))
        ev.extend(g)
        ev.extend(news_events(session, ticker, min(days, 90)))

    disq = next((e["disqualifies"] for e in ev if e.get("disqualifies")), None)
    causes = sorted({c for e in ev for c in (e.get("causes") or [])})
    return {"ticker": ticker, "events": ev, "observed_causes": causes,
            "disqualifies": disq,
            "feeds_reached": reached, "feeds_missed": missed,
            "found_nothing": not ev,
            "note": ("No feed reported anything for this name. That is NOT "
                     "evidence that nothing happened — 8-K covers US issuers "
                     "only, and news coverage of SGX, SET and IDX names is "
                     "thin. Absence here means the app did not see it."
                     if not ev else None)}
