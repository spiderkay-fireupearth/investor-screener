"""The commodity board: what the thing costs, and what you can actually buy.

Rogers's position is that the equity is the wrong instrument:

    "The Yale study found that investing in commodities companies is not
     necessarily a substitute for commodities futures. The authors found that
     from 1962 to 2003, 'the cumulative performance of futures has been triple
     the cumulative performance of "matching" equities'."

    "The smart investor looking into a copper company first has to examine the
     supply-demand dynamics of copper. Why not just stop after that analysis
     and buy or sell the copper itself?"

So this module puts the underlying in front of the equity screen. For each
commodity it shows the futures price — the thing Rogers means — beside the
listed ETF you would have to buy to express it, and the gap between their
twelve-month returns.

**That gap is the point of this module.** A commodity ETF does not hold the
commodity; it holds futures and rolls them. In a market where the far month
costs more than the near one, every roll sells low and buys high, and the fund
bleeds regardless of what the price does. Rogers describes contango in the book
("futures often sell at a higher price farther away in time... those sugar
contracts sell at contango") but never quantifies its cost to a fund, because
he is not proposing you buy a fund.

Rather than model roll yield from a futures curve we do not have, this measures
the drag EMPIRICALLY: ETF 12-month return minus futures 12-month return. That
number already contains roll cost, fees, tracking error and, for the physically
backed metals, storage. It is what you actually got versus what the headline
price did — which is the only version of the question that matters.

Everything here comes from Yahoo, which is free and already the app's price
provider. No new key, no new dependency.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Each entry: the commodity, its Yahoo futures symbol, the ETF a retail
# investor would actually use, and how that ETF is structured — because the
# structure is what decides whether the gap above is small or ruinous.
#
#   physical  — holds the metal in a vault. No roll. Fee drag only.
#   futures   — holds and rolls futures. Exposed to contango.
#   equity    — holds MINERS, not the commodity. Included deliberately for
#               uranium, where no retail physical or futures vehicle is
#               generally available, and flagged so it is never mistaken for
#               the commodity itself.
#   etn       — an unsecured note. Credit risk of the issuer on top.
BOARD: List[Dict[str, str]] = [
    # --- energy
    {"name": "WTI crude",   "future": "CL=F", "etf": "USO",  "kind": "futures",
     "note": "front-month WTI, rolled monthly — the classic contango casualty"},
    {"name": "Brent crude",  "future": "BZ=F", "etf": "BNO",  "kind": "futures"},
    {"name": "Natural gas",  "future": "NG=F", "etf": "UNG",  "kind": "futures",
     "note": "steepest contango of anything on this board; historically the "
             "worst tracking of any commodity fund"},
    # --- industrial metals
    {"name": "Copper",       "future": "HG=F", "etf": "CPER", "kind": "futures"},
    # --- precious metals
    {"name": "Gold",         "future": "GC=F", "etf": "GLD",  "kind": "physical"},
    {"name": "Silver",       "future": "SI=F", "etf": "SLV",  "kind": "physical"},
    {"name": "Platinum",     "future": "PL=F", "etf": "PPLT", "kind": "physical"},
    {"name": "Palladium",    "future": "PA=F", "etf": "PALL", "kind": "physical"},
    # --- nuclear fuel. No retail futures vehicle; SRUUF holds physical U3O8
    # but trades OTC, so URA is what most people can buy — and it is miners.
    {"name": "Uranium",      "future": None,   "etf": "URA",  "kind": "equity",
     "note": "URA holds uranium MINERS, not uranium. Rogers's whole argument "
             "is that this is a different asset — treat it as an equity "
             "position with a commodity theme, not as the commodity"},
    # --- agriculture
    {"name": "Wheat",        "future": "ZW=F", "etf": "WEAT", "kind": "futures"},
    {"name": "Corn",         "future": "ZC=F", "etf": "CORN", "kind": "futures"},
    {"name": "Soybeans",     "future": "ZS=F", "etf": "SOYB", "kind": "futures"},
    {"name": "Sugar",        "future": "SB=F", "etf": "CANE", "kind": "futures",
     "note": "Rogers's own example: 1.4 cents to 66 cents, 1966-1974"},
    {"name": "Coffee",       "future": "KC=F", "etf": "JO",   "kind": "etn",
     "note": "an ETN — unsecured debt of the issuer, not a fund holding assets"},
    # --- broad baskets
    {"name": "Broad basket", "future": None,   "etf": "DBC",  "kind": "futures",
     "note": "optimised roll — designed to reduce exactly the drag measured "
             "in the tracking column"},
    {"name": "Broad, no K-1", "future": None,  "etf": "PDBC", "kind": "futures",
     "note": "same exposure without the K-1 tax form"},
    {"name": "Rogers index",  "future": None,  "etf": "RJI",  "kind": "etn",
     "note": "tracks the Rogers International Commodity Index, which he "
             "designed. An ETN, so it carries issuer credit risk"},
]


def all_symbols() -> List[str]:
    """Every Yahoo symbol this board needs, deduplicated."""
    out = []
    for row in BOARD:
        for k in ("future", "etf"):
            s = row.get(k)
            if s and s not in out:
                out.append(s)
    return out


def _ret(close: pd.Series, days: int) -> Optional[float]:
    s = close.dropna().astype(float)
    if len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1)


def _frame(store, yahoo, symbol: str, period: str = "2y"):
    """Live price if Yahoo answers, otherwise the stored copy."""
    df = None
    if yahoo is not None:
        try:
            df = yahoo.prices(symbol, period=period)
        except Exception as e:                       # noqa: BLE001
            log.warning("commodity fetch %s failed: %s", symbol, e)
            df = None
    if df is not None and not df.empty:
        if store is not None:
            store.save_prices(symbol, df)
        return df
    return store.load_prices(symbol) if store is not None else None


def assess(row: Dict[str, Any]) -> Dict[str, Any]:
    """Two separate verdicts, deliberately never merged into one.

    Rogers's whole method is a two-stage process — analyse the COMMODITY, then
    the instrument — and the app can only do half of it. So this returns half
    an answer twice rather than one whole-looking answer:

      * `commodity_call` is a TREND reading off price alone. It is not a
        supply-and-demand call and must never be read as one. Rogers's actual
        buy and sell rules run on inventories, rig and mine counts, project
        pipelines and days of consumption, none of which is in a free feed.
        His sell rule — "when you discover that stockpiles of all kinds of
        commodities are rising... the bull market will be over" — cannot fire
        here, and a trend that looks strong on price can be a commodity whose
        warehouses are filling.

      * `instrument_grade` is a judgment about the VEHICLE, and this one the
        app can make properly, because the tracking gap is measured rather
        than modelled.

    A strong commodity held through a poor instrument is a losing trade, and
    that combination is the one worth surfacing.
    """
    out: Dict[str, Any] = {"not_a_supply_call": True}

    # --- the commodity, on price only ------------------------------------
    price_role = "future" if row.get("future_12m") is not None else "etf"
    r12 = row.get(f"{price_role}_12m")
    vs200 = row.get(f"{price_role}_vs_200dma")
    off_hi = row.get(f"{price_role}_off_52w_high")
    if r12 is None and vs200 is None:
        out["commodity_call"] = "no price history"
    else:
        up = (vs200 is not None and vs200 > 0)
        strong = (r12 is not None and r12 > 0.10)
        weak = (r12 is not None and r12 < -0.10)
        if up and strong:
            out["commodity_call"] = ("uptrend, near the highs"
                                     if (off_hi is not None and off_hi < 0.10)
                                     else "uptrend")
        elif not up and weak:
            out["commodity_call"] = "downtrend"
        else:
            out["commodity_call"] = "no clear trend"
    out["measured_on"] = ("the futures price" if price_role == "future"
                          else "the fund price — no futures line for this one")

    # --- the vehicle ------------------------------------------------------
    gap, kind = row.get("tracking_gap_12m"), row.get("kind")
    if kind == "equity":
        out["instrument_grade"] = "not the commodity"
        out["instrument_note"] = ("this fund holds producers. Rogers's central "
                                  "claim is that producers underperformed the "
                                  "commodity threefold over 41 years, so this "
                                  "is an equity position, not a commodity one")
    elif gap is None:
        out["instrument_grade"] = "unmeasured"
        out["instrument_note"] = ("no futures line to compare against, so the "
                                  "drag on this vehicle is not measurable here")
    elif gap > -0.02:
        out["instrument_grade"] = "clean"
        out["instrument_note"] = "kept up with the commodity over the year"
    elif gap > -0.06:
        out["instrument_grade"] = "mild drag"
        out["instrument_note"] = "fees and roll cost a little"
    elif gap > -0.15:
        out["instrument_grade"] = "heavy drag"
        out["instrument_note"] = ("a material part of the move was given back "
                                  "to the roll")
    else:
        out["instrument_grade"] = "poor"
        out["instrument_note"] = ("on this evidence the fund is not a way to "
                                  "own this commodity for any length of time")
    if kind == "etn":
        out["instrument_note"] = ((out.get("instrument_note") or "")
                                  + "; and it is an ETN, so you also hold the "
                                    "issuer's credit")

    # --- the combination worth seeing ------------------------------------
    if (out["commodity_call"].startswith("uptrend")
            and out["instrument_grade"] in ("heavy drag", "poor")):
        out["flag"] = ("the commodity is rising and this vehicle is not "
                       "capturing it — the right call expressed the wrong way")
    elif (out["commodity_call"].startswith("uptrend")
            and out["instrument_grade"] == "not the commodity"):
        out["flag"] = ("the commodity is rising, but you would be buying "
                       "producers, whose result depends on things the "
                       "commodity price does not control")
    return out


def build(yahoo, store) -> Dict[str, Any]:
    """Assemble the board. Missing symbols are reported, never interpolated."""
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for spec in BOARD:
        row: Dict[str, Any] = {"name": spec["name"], "kind": spec["kind"],
                               "etf": spec.get("etf"),
                               "future": spec.get("future"),
                               "note": spec.get("note")}
        for role in ("future", "etf"):
            sym = spec.get(role)
            if not sym:
                continue
            df = _frame(store, yahoo, sym)
            if df is None or df.empty:
                missing.append(sym)
                continue
            c = df["Close"].astype(float)
            row[f"{role}_price"] = float(c.iloc[-1])
            row[f"{role}_1m"] = _ret(c, 21)
            row[f"{role}_3m"] = _ret(c, 63)
            row[f"{role}_12m"] = _ret(c, 252)
            if len(c) > 200:
                sma = c.rolling(200).mean().iloc[-1]
                row[f"{role}_vs_200dma"] = (float(c.iloc[-1] / sma - 1)
                                            if sma else None)
            if len(c) > 252:
                w = c.iloc[-252:]
                hi = float(w.max())
                row[f"{role}_off_52w_high"] = (hi - float(c.iloc[-1])) / hi if hi else None

        # The number this module exists for.
        f12, e12 = row.get("future_12m"), row.get("etf_12m")
        if f12 is not None and e12 is not None:
            row["tracking_gap_12m"] = e12 - f12
            g = row["tracking_gap_12m"]
            row["tracking_reading"] = (
                "the fund kept up with the commodity" if g > -0.02 else
                "mild drag — fees and roll" if g > -0.06 else
                "heavy drag: the fund gave back a material part of the move, "
                "which is what contango does to a rolling fund" if g > -0.15 else
                "severe drag — on this evidence the fund is not a way to own "
                "this commodity")
        row["assessment"] = assess(row)
        rows.append(row)

    if missing:
        log.warning("commodity board: %d symbols unavailable: %s",
                    len(missing), ", ".join(sorted(set(missing))))

    return {"rows": rows, "missing": sorted(set(missing)),
            "caveat": (
                "Futures prices are the commodity; ETF prices are the "
                "instrument. Rogers argues for the former and this app can "
                "only screen the latter, so both are shown with the gap "
                "between them made explicit rather than assumed away.")}
