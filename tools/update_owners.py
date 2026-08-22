#!/usr/bin/env python3
"""Rebuild config/owners.yml from the latest SEC filings, unattended.

    python -m tools.update_owners --check     # report, write nothing
    python -m tools.update_owners --write     # rewrite the config

This runs on its own schedule (.github/workflows/refresh-owners.yml), separate
from the two screener refreshes. That separation is the point: the screener
jobs already run close to their timeouts, and a 13F changes four times a year.
If this tool fails, the badges go stale — and every badge on the page is dated,
so the page says so — but nothing else stops working.

WHAT IT WRITES, AND WHAT IT REFUSES TO
--------------------------------------
The output is committed without human review, so the design rule is:

    a wrong badge must be impossible; a missing badge is acceptable.

Everything below follows from that.

  * IT NEVER INVENTS A TICKER. A 13F information table contains no symbols, and
    there is no free authoritative CUSIP crosswalk. The mapping lives in the
    `cusips:` block of the config and is carried forward. A CUSIP that is not
    in it is written to `unmapped:` — the position is reported on the page as
    unmapped and is NOT badged. Tickers really do change (Barrick GOLD -> B,
    UScellular -> AD, both in the last year), and a guess here would silently
    stamp a manager's name on the wrong company.

  * IT RECONCILES BEFORE IT WRITES. The summed value of the parsed rows must
    match the filing's own cover-page `tableValueTotal`, and the row count must
    match `tableEntryTotal`. A truncated download or a parser regression fails
    both, and the tool aborts rather than writing a half-read filing that would
    look like a manager selling half the book.

  * IT WILL NOT GO BACKWARDS. A filing whose period is not newer than the one
    already in the config is ignored. Amended filings (13F-HR/A) for the SAME
    period are accepted, because they supersede — Oaktree amended its Q1 2026
    filing twice within a week.

  * IT REFUSES IMPLAUSIBLE SWINGS. If more than `--max-churn` percent of the
    prior quarter's positions would vanish at once, it aborts and reports.
    That catches a filing read against the wrong manager, and a parse that
    silently dropped a section.

Alongside the 13F it records recent SCHEDULE 13D/13G filings. Those are filed
within days of crossing 5% of a company rather than 45 days after a quarter, so
they are the only signal here that is close to current. They cover large stakes
only, so they supplement the quarterly picture rather than replacing it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"

# Rows we badge. See src/owners.py for why converts and options must not be.
COMMON_CLASS_HINTS = ("com", "common", "cl a", "cl b", "class a", "class b",
                      "ordinary", "shs", "share", "adr", "ads", "sponsored",
                      "ser a", "ser c", "series a", "series c", "cap stk")
EXCLUDE_CLASS_HINTS = ("bond", "note", "deb", "conv", "warrant", "right",
                       "unit", "pfd", "preferred", "etf")

# The SEC asks for a contact address and throttles without one. The workflow
# passes the repo secret through; a local run without it still works, slowly.
UA_ENV = "SEC_USER_AGENT"
THROTTLE_S = 0.2                      # SEC asks for <= 10 requests/second


def _ua() -> str:
    ua = os.environ.get(UA_ENV)
    if not ua:
        print(f"  warning: ${UA_ENV} is unset; the SEC throttles anonymous "
              f"requests", file=sys.stderr)
    return ua or "investor-screener/1.0 (contact via github)"


def fetch(url: str, timeout: int = 60, tries: int = 3) -> Optional[str]:
    """One GET, retried on transient failure. Returns None, never raises."""
    import requests
    for attempt in range(tries):
        try:
            time.sleep(THROTTLE_S)
            r = requests.get(url, headers={"User-Agent": _ua(),
                                           "Accept-Encoding": "gzip, deflate"},
                             timeout=timeout)
            if r.status_code == 200:
                return r.text
            # 403 from the SEC almost always means the User-Agent was refused,
            # and retrying will not fix it. Say which, so the log is actionable.
            if r.status_code == 403:
                print(f"  SEC {url} -> 403. This is usually a rejected "
                      f"User-Agent: set ${UA_ENV} to 'name contact@example.com'.",
                      file=sys.stderr)
                return None
            print(f"  SEC {url} -> HTTP {r.status_code} "
                  f"(attempt {attempt + 1}/{tries})", file=sys.stderr)
        except Exception as e:                          # noqa: BLE001
            print(f"  SEC {url} failed: {e} (attempt {attempt + 1}/{tries})",
                  file=sys.stderr)
        time.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Finding the filing
# ---------------------------------------------------------------------------

def recent_filings(cik: str) -> List[Dict[str, str]]:
    """Every filing in the submissions feed, as flat dicts.

    The feed stores columns as parallel arrays rather than rows, which is
    compact and easy to misread — zipping them here means the rest of the tool
    never has to think about index alignment.
    """
    cik10 = str(cik).strip().lstrip("0").zfill(10)
    doc = fetch(SUBMISSIONS.format(cik10=cik10))
    if not doc:
        return []
    try:
        data = json.loads(doc)
    except ValueError:
        print("  submissions feed was not valid JSON", file=sys.stderr)
        return []
    rec = (data.get("filings") or {}).get("recent") or {}
    keys = ("form", "accessionNumber", "filingDate", "reportDate",
            "primaryDocument")
    cols = {k: rec.get(k) or [] for k in keys}
    n = min((len(v) for v in cols.values() if v), default=0)
    return [{k: (cols[k][i] if i < len(cols[k]) else "") for k in keys}
            for i in range(n)]


def latest_13f(filings: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """The newest 13F-HR, preferring an amendment for the same period.

    Amendments matter: Oaktree filed Q1 2026 on 15 May and amended it on the
    19th and again on the 20th. Taking the original would publish numbers the
    manager has already corrected.
    """
    hr = [f for f in filings if f["form"] in ("13F-HR", "13F-HR/A")]
    if not hr:
        return None
    newest_period = max(f["reportDate"] for f in hr if f["reportDate"])
    same = [f for f in hr if f["reportDate"] == newest_period]
    # Newest filingDate wins, so an amendment supersedes its original.
    return sorted(same, key=lambda f: (f["filingDate"], f["accessionNumber"]))[-1]


def prior_13f(filings: List[Dict[str, str]], newest_period: str
              ) -> Optional[Dict[str, str]]:
    hr = [f for f in filings
          if f["form"] in ("13F-HR", "13F-HR/A")
          and f["reportDate"] and f["reportDate"] < newest_period]
    if not hr:
        return None
    prev_period = max(f["reportDate"] for f in hr)
    same = [f for f in hr if f["reportDate"] == prev_period]
    return sorted(same, key=lambda f: (f["filingDate"], f["accessionNumber"]))[-1]


def filing_files(cik: str, accession: str) -> List[str]:
    acc = accession.replace("-", "")
    url = ARCHIVES.format(cik=str(cik).lstrip("0"), acc=acc) + "/index.json"
    doc = fetch(url)
    if not doc:
        return []
    try:
        items = json.loads(doc).get("directory", {}).get("item", [])
    except ValueError:
        return []
    return [i.get("name", "") for i in items]


def infotable_url(cik: str, accession: str) -> Optional[str]:
    """The information table XML, which is never named the same way twice.

    Filers name it whatever they like — `13F_OCMLP_2Q2026.xml`, `56757.xml`,
    `infotable.xml`. The only reliable rule is: an XML file that is not
    primary_doc.xml. Where several qualify, the largest wins, because the
    information table is always the biggest XML in a 13F submission.
    """
    acc = accession.replace("-", "")
    base = ARCHIVES.format(cik=str(cik).lstrip("0"), acc=acc)
    names = [n for n in filing_files(cik, accession)
             if n.lower().endswith(".xml") and n.lower() != "primary_doc.xml"]
    if not names:
        return None
    if len(names) == 1:
        return f"{base}/{names[0]}"
    # Prefer an obviously-named one; otherwise let the caller try each.
    for n in names:
        if "infotable" in n.lower() or "13f" in n.lower():
            return f"{base}/{n}"
    return f"{base}/{names[0]}"


def cover_totals(cik: str, accession: str) -> Dict[str, Any]:
    """tableEntryTotal / tableValueTotal from primary_doc.xml.

    These are the manager's own count and sum. They are what makes an
    unattended run safe: parse the table, add it up, and if it does not equal
    what the filer said, do not write anything.
    """
    acc = accession.replace("-", "")
    base = ARCHIVES.format(cik=str(cik).lstrip("0"), acc=acc)
    doc = fetch(f"{base}/primary_doc.xml")
    if not doc:
        return {}
    out: Dict[str, Any] = {}
    for tag in ("tableEntryTotal", "tableValueTotal", "isConfidentialOmitted",
                "otherIncludedManagersCount", "periodOfReport", "reportType"):
        m = re.search(rf"<(?:\w+:)?{tag}>([^<]+)</(?:\w+:)?{tag}>", doc)
        if m:
            out[tag] = m.group(1).strip()
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_infotable(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "infoTable":
            continue
        row: Dict[str, Any] = {}
        for child in el.iter():
            name = _strip_ns(child.tag)
            if name == "infoTable":
                continue
            txt = (child.text or "").strip()
            if txt:
                row[name] = txt
        out.append(row)
    return out


def classify(row: Dict[str, Any]) -> Tuple[str, str]:
    """('common' | 'excluded' | 'unsure', reason).

    putCall is checked FIRST. A put on common stock has a titleOfClass that
    looks exactly like a holding, and it is the opposite of one — Oaktree's
    largest reported line is a put on the S&P 500.
    """
    pc = (row.get("putCall") or "").strip().lower()
    if pc in ("put", "call"):
        return "excluded", f"{pc} option"
    if (row.get("sshPrnamtType") or "").strip().upper() == "PRN":
        return "excluded", "reported in principal — a bond, not shares"
    cls = (row.get("titleOfClass") or "").strip().lower()
    for hint in EXCLUDE_CLASS_HINTS:
        if hint in cls:
            return "excluded", f"class contains {hint!r}"
    for hint in COMMON_CLASS_HINTS:
        if hint in cls:
            return "common", cls
    return "unsure", f"unrecognised class {cls!r}"


def aggregate(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]],
                                                   Dict[str, int], List[Dict]]:
    """Sum common-stock rows by CUSIP; also return the exclusion tally.

    A 13F has one row per included manager per security — Berkshire's Q2 2026
    filing is 89 rows describing 29 securities — so rows must be summed by
    CUSIP or every insurance subsidiary becomes a phantom position.
    """
    out: Dict[str, Dict[str, Any]] = {}
    tally: Dict[str, int] = {"convertible_bonds": 0, "options": 0,
                             "warrants": 0, "etfs": 0, "other": 0}
    unsure: List[Dict] = []
    for r in rows:
        kind, why = classify(r)
        if kind == "unsure":
            unsure.append({**r, "_why": why})
            continue
        if kind == "excluded":
            if "option" in why:
                tally["options"] += 1
            elif "principal" in why or "bond" in why or "note" in why or "deb" in why:
                tally["convertible_bonds"] += 1
            elif "warrant" in why:
                tally["warrants"] += 1
            elif "etf" in why:
                tally["etfs"] += 1
            else:
                tally["other"] += 1
            continue
        cusip = (r.get("cusip") or "").strip().upper()
        if not cusip:
            continue
        slot = out.setdefault(cusip, {"shares": 0, "value": 0,
                                      "name": r.get("nameOfIssuer") or "",
                                      "class": r.get("titleOfClass") or ""})
        try:
            slot["shares"] += int(float(r.get("sshPrnamt") or 0))
            slot["value"] += int(float(r.get("value") or 0))
        except ValueError:
            print(f"  unparseable amount on {cusip}", file=sys.stderr)
    return out, tally, unsure


def sum_all_rows(rows: List[Dict[str, Any]]) -> int:
    """Every row's value, for reconciliation against the cover page."""
    total = 0
    for r in rows:
        try:
            total += int(float(r.get("value") or 0))
        except ValueError:
            pass
    return total


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

def diff(prior: Dict[str, Dict[str, Any]], latest: Dict[str, Dict[str, Any]]
         ) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cusip, cur in latest.items():
        was = prior.get(cusip)
        if not was:
            out[cusip] = {**cur, "change": "new", "delta_pct": None}
            continue
        a, b = was["shares"], cur["shares"]
        if b == a:
            out[cusip] = {**cur, "change": "hold", "delta_pct": None}
        else:
            out[cusip] = {**cur, "change": "add" if b > a else "trim",
                          "delta_pct": (round((b - a) / a * 100.0, 1)
                                        if a else None)}
    return out


# ---------------------------------------------------------------------------
# Schedule 13D / 13G — the fast signal
# ---------------------------------------------------------------------------

SUBJECT_RE = re.compile(
    r"SUBJECT\s+COMPANY.*?COMPANY\s+CONFORMED\s+NAME:\s*(.+?)\s*\n"
    r".*?CENTRAL\s+INDEX\s+KEY:\s*(\d+)", re.S | re.I)


def stake_filings(cik: str, filings: List[Dict[str, str]], limit: int = 12,
                  since: str = "") -> List[Dict[str, str]]:
    """Recent SC 13D/13G filings, with the company each one is about.

    These are filed within days of crossing 5% of a company, so they are the
    only near-current signal available here. The subject company is not in the
    submissions feed — it lives in the filing's SGML header — so each one costs
    an extra fetch. Capped at `limit`, newest first, because this is a
    "what changed lately" panel and not an archive.
    """
    forms = [f for f in filings
             if f["form"].upper().startswith(("SC 13D", "SC 13G"))
             and (not since or f["filingDate"] >= since)]
    forms.sort(key=lambda f: f["filingDate"], reverse=True)
    out: List[Dict[str, str]] = []
    for f in forms[:limit]:
        acc = f["accessionNumber"]
        base = ARCHIVES.format(cik=str(cik).lstrip("0"), acc=acc.replace("-", ""))
        head = fetch(f"{base}/{acc}-index-headers.html")
        subject, subj_cik = "", ""
        if head:
            m = SUBJECT_RE.search(head)
            if m:
                subject = re.sub(r"\s+", " ", m.group(1)).strip()
                subj_cik = m.group(2)
        out.append({
            "form": f["form"], "filed": f["filingDate"],
            "accession": acc, "subject": subject, "subject_cik": subj_cik,
            "url": f"{base}/{acc}-index.htm",
        })
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_owner_block(old: Dict[str, Any], filing: Dict[str, str],
                      cover: Dict[str, Any], changed: Dict[str, Dict[str, Any]],
                      gone: Dict[str, Dict[str, Any]], tally: Dict[str, int],
                      stakes: List[Dict[str, str]], cik: str,
                      total_value: int) -> Tuple[Dict[str, Any], List[str]]:
    """The new YAML block for one owner, plus the CUSIPs it could not map."""
    cus = {str(k).upper(): str(v).upper()
           for k, v in (old.get("cusips") or {}).items()}
    acc = filing["accessionNumber"]
    positions: Dict[str, Any] = {}
    unmapped: List[str] = []
    for cusip, row in sorted(changed.items(), key=lambda kv: -kv[1]["value"]):
        tkr = cus.get(cusip)
        if not tkr:
            unmapped.append(cusip)
            continue
        rec: Dict[str, Any] = {
            "shares": row["shares"],
            "value": round(row["value"] / 1000),          # USD thousands
            "pct": round(row["value"] / total_value * 100, 2) if total_value else 0.0,
            "change": row["change"],
        }
        if row.get("delta_pct") is not None:
            rec["delta_pct"] = row["delta_pct"]
        positions[tkr] = rec

    exited: Dict[str, Any] = {}
    for cusip, row in gone.items():
        tkr = cus.get(cusip)
        if not tkr:
            continue
        exited[tkr] = {"shares_prior": row["shares"],
                       "value_prior": round(row["value"] / 1000),
                       "note": f"{row['name']}, exited in the quarter to "
                               f"{cover.get('periodOfReport', '')}"}

    block = dict(old)
    block.update({
        "accession": acc,
        "period": filing.get("reportDate") or cover.get("periodOfReport") or "",
        "filed": filing.get("filingDate") or "",
        "source": filing.get("form") or "13F-HR",
        "url": (ARCHIVES.format(cik=str(cik).lstrip("0"),
                                acc=acc.replace("-", ""))
                + f"/{acc}-index.htm"),
        "portfolio_value_usd": int(cover.get("tableValueTotal") or total_value),
        "reported_rows": int(cover.get("tableEntryTotal") or 0),
        "distinct_positions": len(positions),
        "confidential_omitted":
            str(cover.get("isConfidentialOmitted", "")).lower() == "true",
        "excluded": {k: v for k, v in tally.items() if v},
        "positions": positions,
        "exited": exited,
        "cusips": {k: v for k, v in sorted(cus.items())},
        "unmapped": sorted(unmapped),
        "stake_filings": stakes,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return block, unmapped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/owners.yml")
    ap.add_argument("--check", action="store_true",
                    help="report only; write nothing")
    ap.add_argument("--write", action="store_true", help="rewrite the config")
    ap.add_argument("--only", help="update just this owner key (BK, OT)")
    ap.add_argument("--max-churn", type=float, default=60.0,
                    help="abort if more than this %% of prior positions vanish")
    ap.add_argument("--stake-limit", type=int, default=12,
                    help="how many recent SC 13D/G filings to record")
    args = ap.parse_args()

    doc = load_config(args.config)
    owners = doc.get("owners") or {}
    if not owners:
        print(f"No owners in {args.config}. Nothing to do.", file=sys.stderr)
        return 1

    changed_any = False
    problems: List[str] = []

    for key, old in owners.items():
        if args.only and key != args.only:
            continue
        cik = str(old.get("cik") or "").strip()
        name = old.get("name") or key
        print(f"\n=== {key} · {name} (CIK {cik}) ===")
        if not cik:
            problems.append(f"{key}: no CIK in the config")
            continue

        filings = recent_filings(cik)
        if not filings:
            problems.append(f"{key}: could not read the submissions feed")
            continue

        latest = latest_13f(filings)
        if not latest:
            problems.append(f"{key}: no 13F-HR in the submissions feed")
            continue
        print(f"  latest 13F: {latest['form']} {latest['accessionNumber']} "
              f"period {latest['reportDate']} filed {latest['filingDate']}")

        # Never go backwards. An amendment for the SAME period is allowed
        # through (it supersedes); an older period is not.
        have_period = str(old.get("period") or "")
        have_acc = str(old.get("accession") or "")
        if have_period and latest["reportDate"] < have_period:
            print(f"  config already has {have_period}, which is newer. Skipped.")
            continue
        if latest["accessionNumber"] == have_acc:
            # The 13F has not moved — which is the normal state for eleven
            # weeks out of thirteen. That is NOT a reason to stop: the weekly
            # run exists for the Schedule 13D/G sweep, and those are filed
            # within days of a stake crossing 5% rather than on the quarterly
            # clock. An earlier version of this returned here, which meant the
            # weekly cron did nothing at all except print this line.
            print("  13F unchanged; refreshing the 13D/G sweep and metadata.")
            block = dict(old)
            touched = []

            stakes = stake_filings(cik, filings, limit=args.stake_limit)
            if stakes != (old.get("stake_filings") or []):
                block["stake_filings"] = stakes
                touched.append(f"{len(stakes)} 13D/G filing(s)")
            if stakes:
                for s in stakes[:6]:
                    print(f"    {s['filed']}  {s['form']:<12} "
                          f"{s['subject'] or '(subject not parsed)'}")

            # The submissions feed is authoritative for the filing date. The
            # archive folder's timestamps are not: a submission accepted after
            # 17:30 ET is disseminated with the NEXT business day's date, so
            # reading the mtime off the directory listing can be a day early.
            # This is how the Oaktree date in the hand-written config came to
            # say the 13th when EDGAR says the 14th.
            feed_filed = latest.get("filingDate") or ""
            if feed_filed and feed_filed != str(old.get("filed") or ""):
                print(f"    corrected filing date "
                      f"{old.get('filed')!r} -> {feed_filed!r} (EDGAR is "
                      f"authoritative; the archive mtime is not)")
                block["filed"] = feed_filed
                touched.append("filing date")

            if touched:
                block["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())
                owners[key] = block
                changed_any = True
                print(f"  updated: {', '.join(touched)}")
            else:
                print("  nothing to change.")
            continue

        url = infotable_url(cik, latest["accessionNumber"])
        if not url:
            problems.append(f"{key}: no information table XML in "
                            f"{latest['accessionNumber']}")
            continue
        xml = fetch(url)
        if not xml:
            problems.append(f"{key}: could not download {url}")
            continue
        try:
            rows = parse_infotable(xml)
        except ET.ParseError as e:
            problems.append(f"{key}: information table would not parse ({e})")
            continue

        cover = cover_totals(cik, latest["accessionNumber"])
        # --- reconciliation gate ------------------------------------------
        want_rows = int(cover.get("tableEntryTotal") or 0)
        want_val = int(cover.get("tableValueTotal") or 0)
        got_val = sum_all_rows(rows)
        print(f"  parsed {len(rows)} rows totalling ${got_val:,}")
        print(f"  cover page says {want_rows} rows totalling ${want_val:,}")
        if want_rows and len(rows) != want_rows:
            problems.append(f"{key}: read {len(rows)} rows but the filing says "
                            f"{want_rows} — refusing to write a partial filing")
            continue
        if want_val and abs(got_val - want_val) > max(1000, want_val * 0.0001):
            problems.append(f"{key}: rows sum to ${got_val:,} but the filing "
                            f"says ${want_val:,} — refusing to write")
            continue
        print("  reconciles to the cover page ✓")

        latest_agg, tally, unsure = aggregate(rows)
        print(f"  {len(latest_agg)} common-stock positions; excluded "
              + ", ".join(f"{v} {k}" for k, v in tally.items() if v))
        for u in unsure:
            print(f"  UNCLASSIFIED: {u.get('nameOfIssuer')} | {u.get('cusip')} "
                  f"| {u['_why']} — left unbadged")

        prior_f = prior_13f(filings, latest["reportDate"])
        prior_agg: Dict[str, Dict[str, Any]] = {}
        if prior_f:
            purl = infotable_url(cik, prior_f["accessionNumber"])
            pxml = fetch(purl) if purl else None
            if pxml:
                try:
                    prior_agg, _, _ = aggregate(parse_infotable(pxml))
                    print(f"  prior quarter {prior_f['reportDate']}: "
                          f"{len(prior_agg)} positions")
                except ET.ParseError:
                    pass
        if not prior_agg:
            problems.append(f"{key}: no readable prior quarter — every position "
                            f"would read as 'new', which is wrong. Not writing.")
            continue

        # --- churn gate ----------------------------------------------------
        vanished = [c for c in prior_agg if c not in latest_agg]
        churn = len(vanished) / len(prior_agg) * 100 if prior_agg else 0
        print(f"  churn: {len(vanished)}/{len(prior_agg)} prior positions gone "
              f"({churn:.0f}%)")
        if churn > args.max_churn:
            problems.append(f"{key}: {churn:.0f}% of positions vanished, above "
                            f"the {args.max_churn:.0f}% limit — this usually "
                            f"means a bad parse. Not writing.")
            continue

        changed = diff(prior_agg, latest_agg)
        gone = {c: v for c, v in prior_agg.items() if c not in latest_agg}
        total_value = want_val or got_val

        stakes = stake_filings(cik, filings, limit=args.stake_limit)
        if stakes:
            print(f"  {len(stakes)} recent SC 13D/G filing(s):")
            for s in stakes[:6]:
                print(f"    {s['filed']}  {s['form']:<12} "
                      f"{s['subject'] or '(subject not parsed)'}")

        block, unmapped = build_owner_block(
            old, latest, cover, changed, gone, tally, stakes, cik, total_value)
        if unmapped:
            print(f"  {len(unmapped)} CUSIP(s) have no ticker mapping and are "
                  f"NOT badged:")
            for c in unmapped:
                print(f"    {c}  {changed[c]['name']}  "
                      f"${changed[c]['value'] / 1000:,.0f}k")
            print("    -> add them to the `cusips:` block to badge them.")

        # Summary of what actually moved, for the commit message / run log.
        for ch in ("new", "add", "trim", "hold"):
            ts = sorted(t for t, r in block["positions"].items()
                        if r["change"] == ch)
            if ts:
                print(f"  {ch:<5} {len(ts):>3}  {', '.join(ts)}")
        if block["exited"]:
            print(f"  exit  {len(block['exited']):>3}  "
                  f"{', '.join(sorted(block['exited']))}")

        owners[key] = block
        changed_any = True

    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)

    if not changed_any:
        print("\nNothing changed.")
        return 2 if problems else 0

    if args.write:
        import yaml
        doc["owners"] = owners
        header = ""
        if os.path.exists(args.config):
            with open(args.config, encoding="utf-8") as f:
                text = f.read()
            # Keep the hand-written explanation at the top of the file. It is
            # the only place the exclusion rule is written down for a reader,
            # and a dumped YAML file would silently drop it.
            cut = text.find("\nowners:")
            if cut > 0:
                header = text[:cut + 1]
        with open(args.config, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump({"owners": owners}, f, sort_keys=False,
                           default_flow_style=False, allow_unicode=True,
                           width=100)
        print(f"\nWrote {args.config}")
    else:
        print("\n(--check: nothing was written. Pass --write to apply.)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
