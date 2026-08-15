# Multi-market value + technical screener

Screens **S&P 500 · SGX · HKEX · SET · IDX** against seven investment
frameworks plus a technical timing overlay, and publishes a static pass/fail
screener. One refresh per market region per day.

```
Fundamentals   SEC EDGAR XBRL (US, audited filings)  +  Yahoo Finance (SG/HK/TH/ID)
Prices         Yahoo Finance, split & dividend adjusted, all five markets
Macro          FRED (rates, credit spreads, USD)
Cost           Free
```

---

## Quick start

```bash
pip install -r requirements.txt

export SEC_USER_AGENT="yourapp/1.0 (you@example.com)"   # SEC requires this
export FRED_API_KEY="..."                               # optional, macro gate

python -m src.run --region asia --limit 5      # smoke test, 5 names per market
python -m src.run --region us                  # S&P 500
python -m src.run --region all                 # full rebuild

open out/index.html
```

`--limit N` caps tickers per market and is the right way to try things without
spending 40 minutes of Yahoo rate limit. `--skip-fundamentals` refreshes prices
and technicals only, reusing stored fundamentals — useful because fundamentals
change quarterly and prices change daily.

## Layout

```
config/universe.yml      markets, index constituents, FX pairs, macro series
config/thresholds.yml    every pass/fail threshold — tune here, never in code
src/providers/           edgar.py · yahoo.py · fred.py  (swap-in adapters)
src/schema.py            common fundamental record across US GAAP and IFRS
src/metrics.py           all derived metrics + currency reconciliation
src/technicals.py        indicators, per-market trading calendars
src/screens.py           the seven frameworks as pass/fail rules
src/store.py             SQLite: every fetch persisted before anything computed
src/render.py            static HTML screener
tests/test_pipeline.py   101 assertions, no network needed
```

## Schedule

| Job | Cron (UTC) | Singapore | Covers |
|---|---|---|---|
| Asia | `30 10 * * 1-5` | Mon–Fri 18:30 | SGX, HKEX, SET, IDX |
| US | `0 23 * * 0-4` | Mon–Fri 07:00 | S&P 500 |

The split exists because the markets close eleven hours apart. SET is the last
Asian close at 17:30 SGT; the NYSE close lands at 04:00 SGT in summer and 05:00
in winter. A single midnight-SGT run would catch Asia fine but read New York
mid-session, leaving every S&P 500 name a full trading day stale. Each job only
touches its own universe, so total API volume is the same as one combined run.

The published page always shows all five markets: each run merges its fresh
results over the other region's stored ones.

### Deploying

Both GitHub Actions workflows publish `out/` to GitHub Pages. Set repo secrets
`SEC_USER_AGENT` and `FRED_API_KEY`, enable Pages with source "GitHub Actions",
and the schedule takes over. The SQLite store is carried between runs by
`actions/cache`; if you would rather have hard durability, commit `data/` or
push it to a release artifact instead.

## Tuning the screens

Everything lives in `config/thresholds.yml`. Each framework has a set of tests
and a `min_tests_passed` count — set it equal to the number of tests for a
strict AND, or lower it to let near-misses through.

```yaml
buffett:
  min_tests_passed: 5      # of 6
  tests:
    roe_consistency: {metric: roe, threshold: 0.15, lookback_years: 10, min_years_above: 8}
    leverage:        {metric: debt_to_equity, threshold: 0.5, operator: lte}
```

`global.unknown_counts_as` decides what happens when a metric is missing. The
default is `fail`, deliberately — a value screen that passes a company because
its balance sheet didn't load is worse than one that rejects it. Set it to
`skip` to scale the requirement to the tests that could actually be evaluated.

Greenblatt is rank-based rather than threshold-based: it ranks earnings yield
(EBIT/EV) and return on capital (EBIT / (net working capital + net fixed
assets)) separately and passes the top `top_n` on the combined rank. Ranking is
per-market by default so that one structurally cheap market — Hong Kong,
usually — doesn't monopolise a global top 30.

The Soros `macro_gate` can fail every name at once when high-yield spreads blow
out or the curve inverts deeply. That is the point: in a credit-stress regime,
single-name momentum stops meaning what it normally means.

## Things that will bite you

**Reporting currency ≠ trading currency.** Many HKEX-listed mainland issuers
report financials in CNY while the shares trade in HKD; several SGX names
report in USD. Every valuation ratio divides a market figure by a statement
figure, so `metrics.reconcile_currency` restates market cap into the statement
currency before any ratio is computed. Wrong by 8% is more dangerous than wrong
by 800% — nobody notices. `sanity_check` catches the ones that slip through and
flags the row rather than printing a P/B of 7,000,000 as if it were a number.

**Five trading calendars.** Chinese New Year, Songkran, Hari Raya, Golden Week,
Thanksgiving. Indicators are computed on each market's own bar series and
relative strength is measured against that market's own index — an Indonesian
name measured against the S&P tells you about the rupiah, not the company.

**Deep value in Asia is often a governance discount, not a mispricing.** The
Schloss and Klarman screens will surface Hong Kong and Indonesian names trading
below book for structural reasons: controlling-shareholder discounts, thin free
float, weak minority protection. The universe is capped to index constituents
and gated on market cap and turnover for exactly this reason. The screens are
working; the market is pricing something the screens can't see.

**Yahoo is unofficial.** Personal-use-only under Yahoo's terms, no SLA, and it
throttles. The store persists every successful fetch before anything is
computed, so an outage costs one day of freshness rather than the history, and
stale rows carry a visible warning in the detail drawer. If Asian fundamentals
prove too patchy, write an `EODHDProvider` with the same three methods
(`prices`, `profile`, `fundamentals`) and change one line in `run.py` —
roughly $20/mo for prices only, $80–100/mo for prices and fundamentals.

**Index membership churns.** S&P 500 constituents are fetched at runtime; the
four Asian lists in `config/universe.yml` are static and should be refreshed
each quarter after the index reviews.

## Tests

```bash
python -m tests.test_pipeline
```

101 assertions covering accounting identities, metric maths, threshold and
unknown handling, Greenblatt ranking and sector exclusion, the macro gate,
indicators across mismatched calendars, currency reconciliation, and the
renderer. No network required — every fixture has a hand-computed expected
value, so a regression in the maths fails loudly rather than quietly.

---

Not investment advice. A screen is a starting point for research, not a
conclusion.
