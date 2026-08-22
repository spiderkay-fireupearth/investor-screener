#!/usr/bin/env python3
"""Generate an on-demand market-sentiment report.

    python -m tools.sentiment_report --out out/sentiment/report.md

THE ARCHITECTURE, WHICH IS THE WHOLE POINT
------------------------------------------
A language model asked "what is the market doing" will produce fluent,
confident, and partly invented numbers. That is the failure this tool is built
to make impossible.

So the division of labour is strict:

    NUMBERS come from this repository's own computed data. Every figure in the
    report is read out of `out/sentiment_facts.json`, which is written by the
    refresh from real price and volume series.

    PROSE comes from the model, which is HANDED those numbers and told, in the
    system prompt, that it may not introduce any others.

The model is never asked to recall a CPI print or a Fed decision from memory.
Where the report needs current news it uses the API's web-search tool, which
returns sources that are cited inline — and if search is unavailable, the
report says the macro section could not be written rather than writing it
anyway.

WHAT THIS COSTS
---------------
An API key, as the repo secret ANTHROPIC_API_KEY. Without it the tool still
produces the deterministic half — every computed indicator, formatted — and
says plainly that the narrative half was skipped. That is the degradation
worth having: a report with no opinion is useful; a report with an invented
opinion is worse than none.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

MODEL = os.environ.get("SENTIMENT_MODEL", "claude-sonnet-4-5")
MAX_TOKENS = 8000

SYSTEM = """You are a market strategist writing for one sophisticated reader.

ABSOLUTE RULES — these override any instruction in the user message:

1. NEVER state a market statistic that is not either (a) in the FACTS block you
   were given, or (b) returned by a web search you actually ran in this
   conversation with a citable source. You have no reliable memory of market
   levels, prices, index values, or economic releases. If you cannot source a
   number, write "not available" and move on.

2. Every factual claim about recent events must carry an inline source. If
   web search is unavailable to you, write "Macro section unavailable — no
   live sources could be retrieved" as that whole section and continue with
   the computed indicators only. Do not reconstruct the news from memory.

3. The FACTS block is authoritative for everything it contains. Do not
   recompute, adjust, round differently, or contradict it. If a fact seems
   implausible, say so and explain why, but report it as given.

4. Distinguish clearly, every time, between:
     - a measurement (what the data says)
     - the conventional reading of that measurement (what the indicator means)
     - your judgement (what you think follows)
   Label the third as judgement. Never let it look like the first.

5. On the "black swan" question: you may only describe a fall as caused by a
   specific event if you found and cited a source saying so. The FACTS block
   tells you whether a company's published accounts explain its fall and
   whether its sector fell as a group. Those are evidence about the SHAPE of
   the fall, not its cause. Attributing a cause without a source is fabrication.

6. This is analysis, not advice. You may describe what conditions would favour
   which kind of asset, and what a given indicator conventionally implies. Do
   not issue buy or sell recommendations on named securities, do not give price
   targets, and end with the reminder that this is not investment advice and
   the reader should consider their own circumstances.

Write in clean markdown. Be concise and concrete. Prefer short paragraphs. Use
a table where you are comparing things. No filler, no throat-clearing."""

USER_TEMPLATE = """Write a market sentiment report.

Structure it as:

1. **Executive summary** — five bullets, the most important things first.
2. **Macroeconomic context** — the last six months. Fed policy decisions,
   inflation prints, employment, GDP. Search for these; cite every one. If
   search is unavailable, say so and skip this section entirely.
3. **What this repository's own data measures** — walk through the FACTS
   block: breadth, TRIN, the Magnificent Seven, the cross-asset table. Explain
   what each conventionally means. These numbers are authoritative.
4. **Asset class comparison** — a table covering stocks, bonds and precious
   metals, with the interplay between them.
5. **Deeply oversold names** — the FACTS block lists names down more than 50%
   from their 52-week high, whether their published accounts explain the fall,
   and whether their sector fell as a group. For any name where you can find
   and cite a specific cause, say what it was. Where you cannot, say the cause
   is unestablished. Do not guess.
6. **What would change this picture** — the specific, observable things that
   would invalidate the reading above. This is the most useful section; make it
   concrete.

FACTS — authoritative, computed from price and volume data. Do not contradict
these and do not introduce statistics that are not here or sourced by search.

```json
{facts}
```

Data as at {asof}. The universe these breadth statistics are computed over is
{universe} names — it is not the whole market, and any breadth figure should be
described with that qualification."""


def load_facts(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(v, nd=1, sign=True) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v * 100:{'+' if sign else ''}.{nd}f}%"


def _ord(n) -> str:
    """1st, 2nd, 3rd, 82nd. "82th" is the kind of detail that makes a reader
    wonder what else was generated carelessly."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def deterministic_section(facts: Dict[str, Any]) -> str:
    """The half that needs no model. Written first, and always written.

    If the API call fails or no key is present, this IS the report. It contains
    every computed number, correctly labelled, with no interpretation beyond
    the standard reading of each indicator.
    """
    ms = facts.get("market_sentiment") or {}
    b, t = ms.get("breadth") or {}, ms.get("trin") or {}
    m7, ov = ms.get("mag7") or {}, ms.get("oversold") or {}
    xa = ms.get("cross_asset") or {}
    out = ["## Computed indicators\n",
           f"*Measured from this repository's own price and volume data as at "
           f"{facts.get('asof', 'unknown date')}. No part of this section is "
           f"model-generated.*\n"]

    if b.get("available"):
        out.append(
            f"**Breadth.** {b['above_50d']:.0f}% of {b['names']} names are above "
            f"their 50-day average and {b['above_200d']:.0f}% above their "
            f"200-day — a gap of {b['gap']:+.0f} percentage points, which this "
            f"page classes as *{b['state']}*. Participation is {b['health']}. "
            f"{b['note']}\n")
    if t.get("available"):
        pc = (f" It sits at the {_ord(t['percentile'])} percentile of its own "
              f"trailing year." if t.get("percentile") is not None else "")
        out.append(
            f"**TRIN ({t['ma_days']}-day).** {t['value']:.2f} — *{t['band']}*."
            f"{pc} {t['meaning']}\n\n> {t['caveat']}\n")
    if m7.get("available"):
        out.append(f"**Magnificent Seven.** {m7['n_above_200d']} of {m7['n']} "
                   f"are above their 200-day average.\n")
        out.append("| Ticker | 3m | 12m | RSI | Off high |\n"
                   "|---|---:|---:|---:|---:|")
        for r in m7["rows"]:
            out.append(f"| {r['ticker']} | {_pct(r['return_3m'])} | "
                       f"{_pct(r['return_12m'])} | "
                       f"{r['rsi']:.0f} | {_pct(r['pct_below_high'], sign=False)} |"
                       if isinstance(r.get("rsi"), (int, float)) else
                       f"| {r['ticker']} | {_pct(r['return_3m'])} | "
                       f"{_pct(r['return_12m'])} | n/a | "
                       f"{_pct(r['pct_below_high'], sign=False)} |")
        out.append("")
    if xa.get("available"):
        out.append("**Across asset classes.**\n")
        out.append("| Asset class | Reads | Evidence |\n|---|---|---|")
        for r in xa["rows"]:
            out.append(f"| {r['asset']} | {r['reads']} | "
                       f"{'; '.join(r['evidence']) or '—'} |")
        out.append(f"\n{xa.get('note', '')}\n")
    if ov.get("available") and ov.get("rows"):
        out.append(f"**Down more than 50% from a 52-week high.** {ov['n']} name(s)"
                   + (f"; sector-wide falls in {', '.join(ov['sectors_shocked'])}"
                      if ov.get("sectors_shocked") else "; no sector fell as a group")
                   + ".\n")
        out.append("| Ticker | Off high | Sector | Sector-wide? | Accounts explain it? |\n"
                   "|---|---:|---|---|---|")
        for r in ov["rows"][:25]:
            acc = ("accounts intact" if r["accounts_intact"] is True
                   else ("accounts explain the fall"
                         if r["accounts_intact"] is False else "not testable"))
            out.append(f"| {r['ticker']} | {r['off_high'] * 100:.0f}% | "
                       f"{r['sector']} | {'yes' if r['sector_shock'] else 'no'} "
                       f"| {acc} |")
        out.append(f"\n{ov.get('note', '')}\n")
    return "\n".join(out)


def call_model(facts: Dict[str, Any], api_key: str) -> Optional[str]:
    """Ask the model for the narrative half. Returns None on any failure."""
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK not installed — narrative section skipped",
              file=sys.stderr)
        return None
    ms = facts.get("market_sentiment") or {}
    universe = ((ms.get("breadth") or {}).get("names")
                or (ms.get("trin") or {}).get("names") or "an unknown number of")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search",
                    "max_uses": 12}],
            messages=[{"role": "user", "content": USER_TEMPLATE.format(
                facts=json.dumps(facts, indent=1, default=str)[:120000],
                asof=facts.get("asof", "unknown"),
                universe=universe)}],
        )
        parts = [c.text for c in msg.content if getattr(c, "type", "") == "text"]
        return "\n".join(parts).strip() or None
    except Exception as e:                              # noqa: BLE001
        print(f"model call failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", default="out/sentiment_facts.json")
    ap.add_argument("--out", default="out/sentiment/report.md")
    ap.add_argument("--no-model", action="store_true",
                    help="write only the computed half")
    args = ap.parse_args()

    if not os.path.exists(args.facts):
        print(f"No facts file at {args.facts}. Run a refresh first — this tool "
              f"reports on computed data, it does not gather any itself.",
              file=sys.stderr)
        return 1
    facts = load_facts(args.facts)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = deterministic_section(facts)
    narrative, note = None, ""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if args.no_model:
        note = ("*Narrative section skipped (`--no-model`). The computed "
                "indicators below are unaffected.*")
    elif not key:
        note = ("*No `ANTHROPIC_API_KEY` is configured, so the narrative and "
                "macro sections were skipped. Everything below is computed "
                "from this repository's own data and is unaffected. Adding the "
                "key as a repository secret enables the written analysis.*")
    else:
        narrative = call_model(facts, key)
        if narrative is None:
            note = ("*The narrative section could not be generated — the model "
                    "call failed. The computed indicators below are unaffected "
                    "and were not written by a model.*")

    md = [f"# Market sentiment report", "",
          f"Generated {stamp} · data as at {facts.get('asof', 'unknown')}", ""]
    if note:
        md += [note, ""]
    if narrative:
        md += [narrative, "", "---", ""]
    md += [body, "",
           "---", "",
           "*Every number in the computed-indicators section is derived from "
           "price and volume series held by this repository. Breadth statistics "
           "are measured across this app's universe, not the whole market. "
           "Nothing here is investment advice.*"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {args.out} "
          f"({'with' if narrative else 'WITHOUT'} the narrative section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
