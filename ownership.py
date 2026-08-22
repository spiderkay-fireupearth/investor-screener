"""Who owns the shares: institutions (13F) and insiders (Form 4).

Both feeds are free from the SEC and both have a limitation worth stating
before any number derived from them is trusted.

**13F is 45 days stale and long-only.** Managers over $100m in US equities file
within 45 days of quarter end, and the form covers US-listed long positions
only — no shorts, no bonds, no foreign listings. So a "new position" may be
five months old and already sold, and nothing here says anything at all about
the SGX, HKEX, SET, IDX or Nikkei names in this universe. It is a census of the
past, not a signal about the present.

**Form 4 is fast but small.** Insiders report within two business days, so it
is the freshest ownership data that exists. Buffett's own test is specific and
it is about purchases, not grants:

    "I looked at the proxy material of a large American company and found that
     eight directors had never purchased a share of the company's stock using
     their own money."

That distinction — an open-market PURCHASE with the insider's own cash, versus
an option exercise or a stock grant — is the entire point, and it is why this
module reads the transaction codes rather than counting filings. Code P is a
purchase. Code A is an award. Treating them alike would turn routine
compensation into a bullish signal, which is exactly backwards.

NOTE ON TESTING: this module cannot be exercised against the live SEC endpoints
from the build sandbox, which has no route to sec.gov. The parsing is unit
tested against fixtures; the FETCHING is not. The first production run is the
real test, and every network path here fails soft and reports loudly rather
than failing the pipeline.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import re
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SEC_13F_INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
SEC_ARCHIVES = "https://www.sec.gov/Archives"

# Form 4 transaction codes. Only P is money out of the insider's own pocket.
BUY_CODES = {"P"}
SELL_CODES = {"S"}
NEUTRAL_CODES = {"A", "M", "F", "G", "C", "D", "I", "J", "K", "U", "W", "X", "Z"}


def _ua() -> str:
    return (os.environ.get("SEC_USER_AGENT")
            or "investor-screener/1.0 (contact via github)")


def _get(session, url: str, timeout: int = 60, binary: bool = False):
    """Single fetch. Returns None on any failure — never raises upward."""
    try:
        r = session.get(url, headers={"User-Agent": _ua(),
                                      "Accept-Encoding": "gzip, deflate"},
                        timeout=timeout)
        if r.status_code != 200:
            log.warning("SEC %s -> HTTP %s", url, r.status_code)
            return None
        return r.content if binary else r.text
    except Exception as e:                          # noqa: BLE001
        log.warning("SEC %s failed: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# 13F
# ---------------------------------------------------------------------------

def _quarter_slugs(n: int = 2) -> List[str]:
    """The last `n` completed quarters, newest first, as SEC data-set slugs.

    13F is due 45 days after quarter end, so the most recent quarter is not
    published for six weeks. Asking for it early returns a 404, which is why
    this steps back a full quarter before starting.
    """
    today = date.today()
    q = (today.month - 1) // 3 + 1
    y = today.year
    # Step back one quarter for the filing lag, then one more per requested.
    out = []
    for _ in range(n + 1):
        q -= 1
        if q == 0:
            q, y = 4, y - 1
        out.append(f"{y}q{q}")
    return out[1:] if len(out) > n else out


def cusip_ticker_map(session, store=None) -> Dict[str, str]:
    """CUSIP -> ticker, built from the SEC's own fails-to-deliver file.

    13F reports CUSIPs; this app speaks tickers, and the SEC publishes no
    direct crosswalk. The fails-to-deliver data does carry both columns for
    every security that had a settlement failure — which, over a fortnight,
    is most liquid US names. It is an odd source for a mapping and it is
    incomplete by construction, so coverage is reported rather than assumed.
    """
    if store is not None:
        cached = store.load_blob("cusip_map") if hasattr(store, "load_blob") else None
        if cached:
            return cached
    out: Dict[str, str] = {}
    today = date.today()
    for back in (0, 1):
        d = today.replace(day=1) - timedelta(days=back * 15)
        url = (f"https://www.sec.gov/files/data/fails-deliver-data/"
               f"cnsfails{d.year}{d.month:02d}a.zip")
        blob = _get(session, url, binary=True)
        if not blob:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for nm in z.namelist():
                    with z.open(nm) as fh:
                        txt = io.TextIOWrapper(fh, encoding="latin-1")
                        for row in csv.DictReader(txt, delimiter="|"):
                            c = (row.get("CUSIP") or "").strip().upper()
                            t = (row.get("SYMBOL") or "").strip().upper()
                            if len(c) == 9 and t and t.isalpha():
                                out.setdefault(c, t)
            break
        except Exception as e:                      # noqa: BLE001
            log.warning("fails-to-deliver parse failed: %s", e)
    log.info("CUSIP map: %d entries", len(out))
    return out


def parse_infotable(text: str) -> Dict[str, Dict[str, float]]:
    """Aggregate one quarter's INFOTABLE.tsv by CUSIP.

    Returns cusip -> {holders, value, shares}. `holders` counts distinct
    accession numbers, which is one per filing manager — counting rows would
    multiply-count a manager that reports the same holding across several
    accounts.
    """
    agg: Dict[str, Dict[str, Any]] = {}
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        cusip = (row.get("CUSIP") or "").strip().upper()
        if len(cusip) != 9:
            continue
        rec = agg.setdefault(cusip, {"accessions": set(), "value": 0.0,
                                     "shares": 0.0})
        acc = (row.get("ACCESSION_NUMBER") or "").strip()
        if acc:
            rec["accessions"].add(acc)
        for key, field in (("value", "VALUE"), ("shares", "SSHPRNAMT")):
            try:
                rec[key] += float(row.get(field) or 0)
            except (TypeError, ValueError):
                pass
    return {c: {"holders": len(v["accessions"]), "value": v["value"],
                "shares": v["shares"]} for c, v in agg.items()}


def institutional(session, tickers: List[str], store=None) -> Dict[str, Any]:
    """Per-ticker 13F ownership for the two most recent published quarters."""
    slugs = _quarter_slugs(2)
    quarters: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    for slug in slugs:
        blob = _get(session, f"https://www.sec.gov/files/structureddata/data/"
                             f"form-13f-data-sets/{slug}_form13f.zip",
                    binary=True, timeout=180)
        if not blob:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                nm = next((n for n in z.namelist()
                           if n.upper().endswith("INFOTABLE.TSV")), None)
                if not nm:
                    log.warning("13F %s: no INFOTABLE.tsv in archive", slug)
                    continue
                with z.open(nm) as fh:
                    quarters.append((slug, parse_infotable(
                        io.TextIOWrapper(fh, encoding="latin-1").read())))
        except Exception as e:                      # noqa: BLE001
            log.warning("13F %s parse failed: %s", slug, e)

    if not quarters:
        return {"enabled": False,
                "reason": f"no 13F data set could be downloaded (tried "
                          f"{', '.join(slugs)}) — the quarter may not be "
                          f"published yet, or sec.gov refused the request"}

    cmap = cusip_ticker_map(session, store)
    if not cmap:
        return {"enabled": False,
                "reason": "13F downloaded but the CUSIP-to-ticker map is "
                          "empty, so nothing could be attributed to a ticker"}

    by_ticker: Dict[str, Dict[str, Any]] = {}
    want = {t.upper() for t in tickers}
    for i, (slug, table) in enumerate(quarters):
        for cusip, v in table.items():
            tk = cmap.get(cusip)
            if not tk or tk not in want:
                continue
            slot = "current" if i == 0 else "prior"
            by_ticker.setdefault(tk, {})[slot] = v

    out: Dict[str, Dict[str, Any]] = {}
    for tk, d in by_ticker.items():
        cur, pri = d.get("current"), d.get("prior")
        if not cur:
            continue
        rec = {"institutional_holders": cur["holders"],
               "institutional_value_usd": cur["value"] * 1000.0,
               "institutional_shares": cur["shares"]}
        if pri and pri["holders"]:
            rec["institutional_holders_change"] = cur["holders"] - pri["holders"]
            rec["institutional_holders_change_pct"] = (
                cur["holders"] / pri["holders"] - 1.0)
        if pri and pri["shares"]:
            rec["institutional_share_change_pct"] = (
                cur["shares"] / pri["shares"] - 1.0)
        out[tk] = rec

    return {"enabled": True, "quarters": [s for s, _ in quarters],
            "coverage": len(out), "of": len(want), "by_ticker": out,
            "caveat": ("13F is filed 45 days after quarter end and covers "
                       "US-listed LONG positions only. These numbers describe "
                       "where institutions were, not where they are, and they "
                       "say nothing about any non-US name in this universe.")}


# ---------------------------------------------------------------------------
# Form 4
# ---------------------------------------------------------------------------

_TX = re.compile(r"<transactionCode>\s*([A-Z])\s*</transactionCode>")
_ACQ = re.compile(r"<transactionAcquiredDisposedCode>.*?<value>\s*([AD])\s*</value>",
                  re.S)
_SHARES = re.compile(r"<transactionShares>.*?<value>\s*([\d.]+)\s*</value>", re.S)
_PRICE = re.compile(r"<transactionPricePerShare>.*?<value>\s*([\d.]+)\s*</value>", re.S)


def parse_form4(xml: str) -> Dict[str, Any]:
    """Reduce one Form 4 to open-market purchases and sales.

    Only transaction code P counts as a purchase. Buffett's test is about
    directors buying "using their own money" — an option exercise (M) or a
    stock award (A) is not that, and counting them would make every
    compensation event look like insider conviction.
    """
    codes = _TX.findall(xml or "")
    if not codes:
        return {"buys": 0, "sells": 0, "buy_value": 0.0, "other": 0}
    shares = [float(x) for x in _SHARES.findall(xml or "")]
    prices = [float(x) for x in _PRICE.findall(xml or "")]
    buys = sells = other = 0
    buy_value = 0.0
    for i, c in enumerate(codes):
        if c in BUY_CODES:
            buys += 1
            if i < len(shares) and i < len(prices):
                buy_value += shares[i] * prices[i]
        elif c in SELL_CODES:
            sells += 1
        else:
            other += 1
    return {"buys": buys, "sells": sells, "buy_value": buy_value, "other": other}


def insider_activity(session, cik_by_ticker: Dict[str, str],
                     days: int = 120, max_filings: int = 400) -> Dict[str, Any]:
    """Recent open-market insider purchases, per ticker.

    Bounded on purpose. Reading every Form 4 for a 900-name universe is
    thousands of requests against a rate-limited endpoint; `max_filings` caps
    the work and whatever is dropped is REPORTED rather than silently omitted,
    because "no insider buying" and "we stopped looking" must not look alike.
    """
    if not cik_by_ticker:
        return {"enabled": False, "reason": "no CIK map available"}

    by_cik = {str(v).lstrip("0"): k for k, v in cik_by_ticker.items() if v}
    found: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"buys": 0, "sells": 0, "buy_value": 0.0, "filings": 0})
    seen = 0
    truncated = False

    today = date.today()
    for back in range(days):
        d = today - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        q = (d.month - 1) // 3 + 1
        idx = _get(session, f"{SEC_ARCHIVES}/edgar/daily-index/{d.year}/QTR{q}/"
                            f"form.{d:%Y%m%d}.idx", timeout=30)
        if not idx:
            continue
        for line in idx.splitlines():
            if not line.startswith("4 "):
                continue
            parts = [p for p in line.split("  ") if p.strip()]
            if len(parts) < 4:
                continue
            cik = parts[2].strip().lstrip("0") if len(parts) > 2 else ""
            tk = by_cik.get(cik)
            if not tk:
                continue
            if seen >= max_filings:
                truncated = True
                break
            path = parts[-1].strip()
            doc = _get(session, f"https://www.sec.gov/Archives/{path}", timeout=20)
            seen += 1
            if not doc:
                continue
            r = parse_form4(doc)
            slot = found[tk]
            slot["buys"] += r["buys"]
            slot["sells"] += r["sells"]
            slot["buy_value"] += r["buy_value"]
            slot["filings"] += 1
        if truncated:
            break

    if truncated:
        log.warning("Insider scan stopped at the %d-filing cap — names later "
                    "in the window were NOT checked and must not be read as "
                    "having no insider buying", max_filings)

    return {"enabled": True, "window_days": days, "filings_read": seen,
            "truncated": truncated, "by_ticker": dict(found),
            "caveat": ("Only transaction code P — an open-market purchase with "
                       "the insider's own money — counts as a buy. Option "
                       "exercises and stock awards are excluded deliberately.")}
