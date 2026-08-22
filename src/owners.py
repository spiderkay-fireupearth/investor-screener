"""Disclosed institutional owners, as a tag on a row.

This module answers one question — "does a named investor hold this?" — and it
is deliberately the dumbest module in the project. It reads a static YAML file
and returns dictionaries. It makes no network calls, has no failure modes worth
retrying, and cannot slow a refresh down.

That is a design choice, not laziness. The alternative is to fetch both 13Fs
from EDGAR on every run, which would add two network dependencies and a parsing
step to a pipeline that already had to have its workflow timeouts raised to 300
minutes. The data changes four times a year. Paying a per-run cost for a
per-quarter fact is the wrong trade, so `tools/update_owners.py` does the fetch
once, offline, and writes the result into `config/owners.yml`.

WHAT A BADGE MEANS, precisely, because the whole module is worthless if this is
fuzzy:

    A badge says: on the report date of the filing named in the config, this
    manager disclosed a long position in the common stock of this company.

It does not say the manager holds it now (a 13F is filed up to 45 days after
quarter end, and describes a date up to 45 days before that). It does not say
the manager likes it — index-hugging and legacy restructuring stakes both show
up. And it says nothing at all about companies outside US-listed equity, which
is why `manual` entries exist and why they are labelled differently on the page.

THE EXCLUSION THAT MATTERS MOST is handled upstream, in the config: convertible
bonds and options are not positions in this sense. Oaktree's largest reported
line is a PUT on the S&P 500. A module that tagged "appears in the filing"
would have put an Oaktree badge on SPY, meaning the exact opposite of what a
reader would take from it. `config/owners.yml` carries common-stock longs only,
and records the excluded counts so the page can state what was dropped.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# The changes a position can show against the prior quarter. Anything else in
# the config is a typo, and a typo that silently renders as a blank chip is the
# kind of bug that survives for a year.
CHANGES = ("new", "add", "trim", "hold")

CHANGE_LABEL = {
    "new": "new position",
    "add": "added",
    "trim": "trimmed",
    "hold": "unchanged",
}

# What each change is worth knowing for. Deliberately not "bullish"/"bearish":
# a trim can be risk management, a hold can be inattention, and the page should
# not launder a share count into a recommendation.
CHANGE_NOTE = {
    "new": "First appeared in this filing — the manager did not hold it a "
           "quarter ago.",
    "add": "The share count rose versus the prior quarter.",
    "trim": "The share count fell versus the prior quarter, but the position "
            "is still open.",
    "hold": "The share count is unchanged from the prior quarter.",
}


def load(path: str = "config/owners.yml") -> Dict[str, Any]:
    """Read the owners config. A missing file is not an error.

    The feature is additive — every row on the page renders correctly with no
    owners at all — so a missing or unreadable config degrades to "no badges"
    rather than taking the run down with it.
    """
    if not os.path.exists(path):
        log.info("No owners config at %s — owner badges are off this run", path)
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:                              # noqa: BLE001
        log.warning("Could not read %s (%s) — owner badges are off this run",
                    path, e)
        return {}
    owners = doc.get("owners") or {}
    if not isinstance(owners, dict):
        log.warning("owners.yml: `owners` is not a mapping — ignoring it")
        return {}
    return _validate(owners)


def _validate(owners: Dict[str, Any]) -> Dict[str, Any]:
    """Drop what cannot be rendered honestly, and say what was dropped.

    A bad `change` value would render as an empty chip next to a real share
    count, which reads as "no change" rather than "we do not know". Rejecting
    it here means the page can never show that.
    """
    out: Dict[str, Any] = {}
    for key, block in owners.items():
        if not isinstance(block, dict):
            log.warning("owners.yml: %s is not a mapping — skipped", key)
            continue
        pos = block.get("positions") or {}
        clean: Dict[str, Any] = {}
        for tkr, row in pos.items():
            if not isinstance(row, dict):
                log.warning("owners.yml: %s/%s is not a mapping — skipped", key, tkr)
                continue
            ch = str(row.get("change") or "").strip().lower()
            if ch not in CHANGES:
                log.warning("owners.yml: %s/%s has change=%r, which is not one "
                            "of %s — skipped rather than rendered blank",
                            key, tkr, row.get("change"), ", ".join(CHANGES))
                continue
            r = dict(row)
            r["change"] = ch
            clean[str(tkr).upper()] = r
        b = dict(block)
        b["positions"] = clean
        b["exited"] = {str(k).upper(): v
                       for k, v in (block.get("exited") or {}).items()}
        b["manual"] = {str(k).upper(): v
                       for k, v in (block.get("manual") or {}).items()}
        b.setdefault("badge", key)
        out[key] = b
    return out


def by_ticker(owners: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """ticker -> [owner tags], the lookup the run loop needs.

    A ticker can carry more than one owner, and the order is the order the
    owners appear in the config rather than anything derived from the numbers,
    so a row never reshuffles its badges between runs.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, block in owners.items():
        base = {
            "key": key,
            "badge": block.get("badge") or key,
            "name": block.get("name") or key,
            "manager": block.get("manager") or "",
            "period": block.get("period") or "",
            "filed": block.get("filed") or "",
            "url": block.get("url") or "",
            "source": block.get("source") or "",
        }
        for tkr, row in (block.get("positions") or {}).items():
            tag = dict(base)
            tag.update({
                "shares": row.get("shares"),
                "value": row.get("value"),        # USD thousands
                "pct": row.get("pct"),            # % of the manager's portfolio
                "change": row.get("change"),
                "change_label": CHANGE_LABEL.get(row.get("change"), ""),
                "change_note": CHANGE_NOTE.get(row.get("change"), ""),
                "delta_pct": row.get("delta_pct"),
                "disclosure": "13F",
            })
            out.setdefault(tkr, []).append(tag)
        # Manual entries carry the same badge but a visibly different
        # provenance: they are not in any filing, so they get no share count
        # and the page labels where they came from.
        for tkr, row in (block.get("manual") or {}).items():
            tag = dict(base)
            tag.update({
                "shares": None, "value": None, "pct": None,
                "change": None, "change_label": "", "change_note": "",
                "delta_pct": None,
                "disclosure": "manual",
                "manual_source": (row or {}).get("source") or "company disclosure",
                "manual_note": (row or {}).get("note") or "",
            })
            out.setdefault(tkr, []).append(tag)
    return out


def exits_by_ticker(owners: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """ticker -> [owners who sold out of it last quarter].

    Rendered differently from a holding, because it is the opposite fact. It
    exists so a name that carried a badge last quarter does not simply lose it
    with no explanation — a disappearing badge reads as a bug, and an exit is
    worth more to a reader than a hold.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, block in owners.items():
        for tkr, row in (block.get("exited") or {}).items():
            out.setdefault(tkr, []).append({
                "key": key,
                "badge": block.get("badge") or key,
                "name": block.get("name") or key,
                "period": block.get("period") or "",
                "url": block.get("url") or "",
                "shares_prior": (row or {}).get("shares_prior"),
                "value_prior": (row or {}).get("value_prior"),
                "note": (row or {}).get("note") or "",
            })
    return out


def coverage(owners: Dict[str, Any], seen: Dict[str, List[str]]
             ) -> List[Dict[str, Any]]:
    """Which pinned owner tickers actually reached the page, and which did not.

    This is the same discipline the `always_include` ledger uses, and for the
    same reason: a badge that silently fails to appear is indistinguishable
    from a manager who does not hold the name. Without this, "why is there no
    BK on Apple?" has no answer visible from the page.

    `seen` is ticker -> [owner keys] for the rows that actually rendered.
    """
    out = []
    for key, block in owners.items():
        want = set((block.get("positions") or {}).keys())
        want |= set((block.get("manual") or {}).keys())
        got = {t for t, keys in seen.items() if key in keys}
        missing = sorted(want - got)
        out.append({
            "key": key,
            "name": block.get("name") or key,
            "badge": block.get("badge") or key,
            "asked": len(want),
            "present": len(got),
            "missing": missing,
            "period": block.get("period") or "",
            "filed": block.get("filed") or "",
            "url": block.get("url") or "",
            "source": block.get("source") or "",
            "portfolio_value_usd": block.get("portfolio_value_usd"),
            "reported_rows": block.get("reported_rows"),
            "distinct_positions": block.get("distinct_positions"),
            "excluded": block.get("excluded") or {},
            "note": block.get("note") or "",
            "manager": block.get("manager") or "",
            "long_name": block.get("long_name") or block.get("name") or key,
            # Positions the updater found but could not name. These are NOT
            # badged, by design — it will not guess a ticker. Printing the
            # count is what stops that safety rule from turning into a silent
            # coverage hole: "3 positions could not be mapped" is a fact a
            # reader can act on, an absent badge is not.
            "unmapped": list(block.get("unmapped") or []),
            # Schedule 13D/G filings, which move between quarters. The 13F is
            # 45 days stale by construction; these are days old, and they are
            # the only thing here that can contradict it.
            "stake_filings": list(block.get("stake_filings") or []),
            "updated_utc": block.get("updated_utc") or "",
        })
    return out


def tags_for(rec_ticker: str, index: Dict[str, List[Dict[str, Any]]]
             ) -> List[Dict[str, Any]]:
    """The owner tags for one ticker, matched case-insensitively.

    Also tries the dash/dot variants of a share-class ticker. Berkshire's
    Lennar B shows as LEN.B in the filing, LEN-B on Yahoo, and LEN/B in a few
    places; without this the second-largest homebuilder position on the list
    would quietly go unbadged.
    """
    t = (rec_ticker or "").upper()
    if not t:
        return []
    for cand in (t, t.replace(".", "-"), t.replace("-", "."),
                 t.replace(".", ""), t.replace("-", "")):
        if cand in index:
            return index[cand]
    return []
