#!/usr/bin/env python3
"""Check that config/owners.yml is safe to publish. Exit 0 = safe.

    python -m tools.verify_owners                 # check the default config
    python -m tools.verify_owners --config X.yml

This is the gate between an unattended EDGAR update and the live page. It runs
after the updater writes and before the workflow commits; a non-zero exit makes
the workflow revert the file and commit nothing.

DEPENDENCIES ARE THE POINT. This imports only PyYAML and src/owners.py, and
nothing else. The first version of this check ran the project's pytest suite,
which imports numpy and pandas at module level — so a job whose real work is a
dozen HTTP requests failed on missing scientific-computing libraries it had no
use for. A verification step that needs the whole application installed is a
verification step that will be switched off the first time it is inconvenient.

WHAT IT CHECKS, and why each one is here rather than left to a human:

  * The file loads through the same code path the screener uses. A YAML file
    that parses but that src/owners.py rejects would turn every badge off at
    once, silently.
  * Every owner still has positions. An empty positions block is the signature
    of a parser that "succeeded" against a filing it did not understand.
  * Every position has a change value the renderer knows. An unrecognised one
    renders as a blank chip, which reads as "unchanged" rather than "unknown".
  * Every badged ticker has a CUSIP behind it. This is what makes the mapping
    auditable: a ticker with no CUSIP came from somewhere unaccountable.
  * Percentages and share counts are sane. A negative share count or a position
    claiming 300% of a portfolio means the arithmetic went wrong upstream.
  * The filing is identified and dated. A badge whose provenance is missing
    cannot be checked by a reader, which is the whole contract of the feature.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import owners as ow                                # noqa: E402

# A single manager holding more than this share of one company's float, or of
# its own book, is possible but is far more often an arithmetic error. Berkshire
# has run ~50% of its book in its top three names, so the bar is deliberately
# high — this is a tripwire for nonsense, not a view on concentration.
MAX_POSITION_PCT = 60.0
MAX_TOTAL_PCT = 130.0          # rounding across many rows, plus slack


def verify(path: str) -> List[str]:
    """Return a list of problems. Empty list means the file is safe."""
    problems: List[str] = []

    if not os.path.exists(path):
        return [f"{path} does not exist"]

    # 1. It must load through the code the screener actually uses.
    cfg = ow.load(path)
    if not cfg:
        return [f"{path} did not load through src/owners.py — the screener "
                f"would render with no badges at all"]

    # 1a. Anything the LOADER dropped on the way in.
    #
    # src/owners.py deliberately discards rows it cannot render honestly — a
    # position whose `change` is not one of the four known values would
    # otherwise draw a blank chip that reads as "unchanged" rather than
    # "unknown". That silent drop is right at render time and wrong here: a
    # file where thirty rows were discarded still "loads", and every one of
    # those positions would simply be missing from the page with nothing
    # anywhere saying so. Compare the raw file with what survived.
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = (yaml.safe_load(f) or {}).get("owners") or {}
    except Exception as e:                                  # noqa: BLE001
        return [f"{path} is not readable as YAML: {e}"]
    for key, rawblock in raw.items():
        # Checked BEFORE the shape guard: an owner block that is not a mapping
        # at all is rejected wholesale by the loader, and skipping past it here
        # would let the most complete failure of the three go unreported.
        if key not in cfg:
            problems.append(f"{key}: the whole owner block was rejected by the "
                            f"loader and would show no badges at all")
            continue
        if not isinstance(rawblock, dict):
            continue
        rawpos = set(str(k).upper() for k in (rawblock.get("positions") or {}))
        gotpos = set((cfg.get(key) or {}).get("positions") or {})
        dropped = sorted(rawpos - gotpos)
        if dropped:
            problems.append(
                f"{key}: the loader discarded {len(dropped)} position(s) that "
                f"are in the file but would not render — "
                f"{', '.join(dropped[:8])}"
                + ("…" if len(dropped) > 8 else "")
                + ". Check their `change` values against "
                + "/".join(ow.CHANGES) + ".")

    for key, block in cfg.items():
        pos: Dict[str, Any] = block.get("positions") or {}
        man: Dict[str, Any] = block.get("manual") or {}
        where = f"{key}"

        # 2. Something must be badged.
        if not pos and not man:
            problems.append(f"{where}: no positions and no manual holdings — "
                            f"this is what a failed parse looks like")
            continue

        # 3. Provenance. A badge that cannot be traced to a filing is not a
        #    badge, it is a rumour.
        for field in ("period", "accession", "cik"):
            if not block.get(field):
                problems.append(f"{where}: missing `{field}` — the page could "
                                f"not cite where these holdings came from")

        # 4. Every change value must be one the renderer knows about.
        cus = {str(v).upper() for v in (block.get("cusips") or {}).values()}
        total_pct = 0.0
        for tkr, row in pos.items():
            if row.get("change") not in ow.CHANGES:
                problems.append(f"{where}/{tkr}: change={row.get('change')!r} "
                                f"is not one of {', '.join(ow.CHANGES)}")
            shares = row.get("shares")
            if shares is not None and (not isinstance(shares, (int, float))
                                       or shares < 0):
                problems.append(f"{where}/{tkr}: shares={shares!r} is not a "
                                f"non-negative number")
            value = row.get("value")
            if value is not None and (not isinstance(value, (int, float))
                                      or value < 0):
                problems.append(f"{where}/{tkr}: value={value!r} is not a "
                                f"non-negative number")
            pct = row.get("pct")
            if isinstance(pct, (int, float)):
                total_pct += pct
                if pct > MAX_POSITION_PCT:
                    problems.append(f"{where}/{tkr}: claims {pct}% of the "
                                    f"portfolio, above the {MAX_POSITION_PCT}% "
                                    f"sanity limit")
                if pct < 0:
                    problems.append(f"{where}/{tkr}: negative portfolio "
                                    f"weight {pct}")
            # 5. Auditability: a badged ticker with no CUSIP behind it came
            #    from nowhere traceable.
            if cus and str(tkr).upper() not in cus:
                problems.append(f"{where}/{tkr}: badged but has no CUSIP in "
                                f"the `cusips:` map — its identity cannot be "
                                f"checked against the filing")
        if total_pct > MAX_TOTAL_PCT:
            problems.append(f"{where}: position weights sum to {total_pct:.0f}%, "
                            f"above the {MAX_TOTAL_PCT}% limit — the "
                            f"denominator is probably wrong")

        # 6. Manual holdings must be marked as such, or they masquerade as
        #    filed positions.
        for tkr in man:
            if tkr in pos:
                problems.append(f"{where}/{tkr}: appears as both a filed "
                                f"position and a manual one")

        # 7. An exit and a holding are mutually exclusive facts.
        for tkr in (block.get("exited") or {}):
            if tkr in pos:
                problems.append(f"{where}/{tkr}: recorded as both held and "
                                f"exited in the same quarter")

    # 8. The index and ledger the page builds from must actually build.
    try:
        idx = ow.by_ticker(cfg)
        ow.exits_by_ticker(cfg)
        ow.coverage(cfg, {})
    except Exception as e:                                  # noqa: BLE001
        problems.append(f"the page's own index could not be built: {e}")
        return problems
    if not idx:
        problems.append("no ticker carries a badge — the feature would be "
                        "invisible on the page")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/owners.yml")
    args = ap.parse_args()

    problems = verify(args.config)
    cfg = ow.load(args.config)
    for key, block in (cfg or {}).items():
        n = len(block.get("positions") or {})
        m = len(block.get("manual") or {})
        u = len(block.get("unmapped") or [])
        print(f"  {key}: {n} filed position(s)"
              + (f" + {m} manual" if m else "")
              + f" as at {block.get('period') or '?'}"
              + (f" · {u} unmapped and unbadged" if u else ""))

    if problems:
        print(f"\nFAILED — {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nOK — the config is safe to publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
