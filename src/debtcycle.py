"""Ray Dalio's Big Debt Cycle, reduced to observable series.

The seven-stage archetype is a narrative model. To run it daily without
inventing conviction, every stage here is expressed as a set of *predicates over
published series*, each of which is True, False, or — importantly — None when
the data isn't there. A stage's score is (passed / evaluable), so a missing
series shrinks the denominator rather than silently voting "no". That is the
same discipline the investor screens use for missing fundamental years.

Two departures from the textbook, both deliberate:

1.  **The cycle is staged per sector, not for "the economy".** Dalio's archetype
    describes whichever sector carries the leverage. In 2026 the US household
    sector and the US federal government are in completely different places, and
    a single blended number would hide that. `classify()` therefore returns a
    public-sector stage, a private-sector stage, and a headline stage that is
    the more advanced of the two — with the disagreement stated, not smoothed.

2.  **The bubble checklist is scored separately from the stage.** A stage is a
    judgment about where you are; the checklist is a judgment about how much
    damage is available if you are wrong. They can and do diverge, and the alert
    level is driven by the checklist alone.

Nothing here emits a recommendation. It emits a stage, the evidence for it, and
the evidence against it.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Series. Every one is optional; absence is reported by name, never guessed.
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # Debt stocks and the income they are measured against
    "fed_debt_gdp":     "GFDEGDQ188S",       # federal debt, % of GDP, quarterly
    "hh_debt_gdp":      "HDTGPDUSQ163N",     # household debt, % of GDP (IMF)
    "corp_debt":        "NCBDBIQ027S",       # nonfin corporate debt, $bn
    "gdp":              "GDP",               # nominal GDP, $bn SAAR
    # Sustainability — is new borrowing paying interest on old borrowing?
    "fed_interest":     "A091RC1Q027SBEA",   # federal interest outlays, $bn SAAR
    "fed_deficit":      "MTSDS133FMS",       # monthly Treasury surplus/deficit
    # The tipping point
    "curve_10y2y":      "T10Y2Y",
    "curve_10y3m":      "T10Y3M",
    "policy":           "DFF",
    "cpi":              "CPIAUCSL",
    # Early warnings
    "hy_oas":           "BAMLH0A0HYM2",
    "bbb_oas":          "BAMLC0A4CBBB",
    "cc_delinq":        "DRCCLACBS",         # credit-card delinquency, %
    "all_delinq":       "DRALACBN",          # all loans, %
    # The printing lever
    "fed_assets":       "WALCL",             # Fed total assets, $mn weekly
    "m2":               "M2SL",
    # The wealth-gap lever
    "top1_wealth":      "WFRBST01134",       # top 1% share of net worth, %
    "bottom50_wealth":  "WFRBSN40188",
    # Activity
    "unemployment":     "UNRATE",
    "payrolls":         "PAYEMS",
}

# Things Dalio's checklist wants that have no reliable free daily series.
# Named rather than approximated, so their absence stays visible on the page.
UNAVAILABLE = [
    "FINRA margin debt — monthly, not on FRED; the single best 'amateur "
    "leverage' gauge and it has to be read manually",
    "Retail brokerage account openings and options volume — the 'new buyers' "
    "test; no free series",
    "Shiller CAPE — computed here from the index instead; the official series "
    "is not free in machine-readable form",
]

# Reference points. These are judgments, and are here so they can be argued
# with rather than buried in an if-statement.
CAPE_LONG_MEAN = 17.6          # 1881-present arithmetic mean
CAPE_MODERN_MEAN = 27.0        # 1990-present, the regime most people trade in
HY_OAS_CALM = 350.0            # bp; below this credit is not pricing risk
HY_OAS_STRESS = 600.0
ZERO_BOUND = 0.75              # % — "at or near 0"
VELOCITY_TEST_PP = 20.0        # Dalio's 3-year debt/GDP rise threshold


def _n(x) -> bool:
    """True if x is a usable number."""
    return (x is not None and isinstance(x, (int, float))
            and not (isinstance(x, float) and math.isnan(x)))


def _chg(series: Optional[List[Tuple[str, float]]], periods: int
         ) -> Optional[float]:
    """Absolute change over `periods` observations, newest-first ordering."""
    if not series or len(series) <= periods:
        return None
    return float(series[0][1] - series[periods][1])


def _pct_chg(series: Optional[List[Tuple[str, float]]], periods: int
             ) -> Optional[float]:
    if not series or len(series) <= periods:
        return None
    old = series[periods][1]
    if not old:
        return None
    return float(series[0][1] / old - 1)


def _latest(series: Optional[List[Tuple[str, float]]]) -> Optional[float]:
    return float(series[0][1]) if series else None


# ---------------------------------------------------------------------------
# The monitoring checklist
# ---------------------------------------------------------------------------

def bubble_checklist(h: Dict[str, list], cape: Optional[float],
                     vix: Optional[float]) -> Dict[str, Any]:
    """Dalio's bubble tests, each scored 0 (cool) / 1 (warm) / 2 (hot).

    The output is deliberately a score out of the tests that could actually be
    evaluated, so a missing feed lowers confidence instead of lowering the
    reading. A 'warm' is not a half-truth — it is the honest middle of a gauge
    whose ends are far apart.
    """
    tests: List[Dict[str, Any]] = []

    def add(key, label, score, detail):
        tests.append({"key": key, "label": label, "score": score,
                      "detail": detail})

    # 1. Prices high relative to earnings
    if _n(cape):
        s = 2 if cape > CAPE_MODERN_MEAN * 1.35 else 1 if cape > CAPE_MODERN_MEAN else 0
        add("valuation", "Prices high relative to earnings", s,
            f"CAPE {cape:.1f} vs modern mean {CAPE_MODERN_MEAN:.0f}, "
            f"long mean {CAPE_LONG_MEAN:.1f}")
    else:
        add("valuation", "Prices high relative to earnings", None,
            "CAPE unavailable")

    # 2. Broad bullish sentiment. Low volatility priced against high valuation
    #    is the observable form of "nobody thinks this can go wrong".
    if _n(vix):
        s = 2 if vix < 15 else 1 if vix < 20 else 0
        add("sentiment", "Broad bullish sentiment", s,
            f"VIX {vix:.1f}")
    else:
        add("sentiment", "Broad bullish sentiment", None, "VIX unavailable")

    # 3. New buyers entering with leverage. Margin debt is not free; the
    #    honest proxy is credit priced as though default is impossible while
    #    equity is priced for perfection.
    hy = _latest(h.get("hy_oas"))
    if _n(hy):
        s = 2 if hy < 300 else 1 if hy < HY_OAS_CALM else 0
        add("leverage", "Leverage cheap and abundant", s,
            f"US high-yield OAS {hy:.0f}bp (calm line {HY_OAS_CALM:.0f}bp)")
    else:
        add("leverage", "Leverage cheap and abundant", None,
            "high-yield OAS unavailable")

    # 4. Debt growing faster than income
    v = debt_velocity(h)
    if _n(v.get("fed_3y_pp")):
        pp = v["fed_3y_pp"]
        s = 2 if pp > VELOCITY_TEST_PP else 1 if pp > VELOCITY_TEST_PP / 2 else 0
        add("velocity", "Debt growing faster than income", s,
            f"federal debt/GDP {pp:+.1f}pp over 3y "
            f"(Dalio's test: >{VELOCITY_TEST_PP:.0f}pp)")
    else:
        add("velocity", "Debt growing faster than income", None,
            "federal debt/GDP history unavailable")

    # 5. Borrowing to pay interest
    sus = sustainability(h)
    if _n(sus.get("interest_to_deficit")):
        r = sus["interest_to_deficit"]
        s = 2 if r > 0.80 else 1 if r > 0.50 else 0
        add("sustainability", "New borrowing funding old interest", s,
            f"federal interest is {r * 100:.0f}% of the deficit")
    else:
        add("sustainability", "New borrowing funding old interest", None,
            "interest or deficit series unavailable")

    # 6. Central bank tightening into it
    real = real_policy_rate(h)
    if _n(real):
        s = 2 if real > 1.5 else 1 if real > 0 else 0
        add("tightening", "Policy restrictive in real terms", s,
            f"real policy rate {real:+.2f}%")
    else:
        add("tightening", "Policy restrictive in real terms", None,
            "policy rate or CPI unavailable")

    scored = [t for t in tests if t["score"] is not None]
    total = sum(t["score"] for t in scored)
    maxi = 2 * len(scored)
    pct = (total / maxi) if maxi else None

    if pct is None:
        level, reason = "GREY", "not enough data to score the checklist"
    elif pct >= 0.66:
        level = "RED"
        reason = ("most bubble conditions are present simultaneously — this is "
                  "the configuration in which a shock does structural damage")
    elif pct >= 0.40:
        level = "YELLOW"
        reason = ("several bubble conditions present but not all — damage from "
                  "a shock would be contained rather than systemic")
    else:
        level = "GREEN"
        reason = "bubble conditions largely absent"

    return {"tests": tests, "score": total, "max": maxi, "pct": pct,
            "level": level, "reason": reason,
            "evaluable": len(scored), "of": len(tests)}


def debt_velocity(h: Dict[str, list]) -> Dict[str, Any]:
    """Is debt outrunning income? Dalio's threshold is +20pp over three years."""
    out: Dict[str, Any] = {}
    fed = h.get("fed_debt_gdp")
    out["fed_level"] = _latest(fed)
    out["fed_3y_pp"] = _chg(fed, 12)     # quarterly series
    out["fed_1y_pp"] = _chg(fed, 4)
    hh = h.get("hh_debt_gdp")
    out["hh_level"] = _latest(hh)
    out["hh_3y_pp"] = _chg(hh, 12)
    corp, gdp = h.get("corp_debt"), h.get("gdp")
    if corp and gdp:
        c, g = _latest(corp), _latest(gdp)
        if _n(c) and _n(g) and g:
            out["corp_gdp"] = 100.0 * c / g
    if _n(out.get("fed_3y_pp")):
        out["fails_velocity_test"] = out["fed_3y_pp"] > VELOCITY_TEST_PP
    return out


def sustainability(h: Dict[str, list]) -> Dict[str, Any]:
    """Is new borrowing mainly servicing old borrowing?

    This is the test that separates a large debt from an unstable one. A debt
    stock is only a problem when the interest on it consumes the new issuance,
    because at that point the borrowing is self-referential: you issue to pay
    the coupon on what you already issued.
    """
    out: Dict[str, Any] = {}
    interest = _latest(h.get("fed_interest"))          # $bn, annual rate
    gdp = _latest(h.get("gdp"))
    out["interest_saar_bn"] = interest
    if _n(interest) and _n(gdp) and gdp:
        out["interest_to_gdp"] = interest / gdp

    # MTSDS133FMS is a monthly surplus/deficit in $mn; annualise the last 12.
    d = h.get("fed_deficit")
    if d and len(d) >= 12:
        twelve = sum(x[1] for x in d[:12]) / 1000.0    # $mn -> $bn
        out["deficit_ttm_bn"] = -twelve                # deficit reported negative
        if _n(interest) and out["deficit_ttm_bn"]:
            out["interest_to_deficit"] = interest / out["deficit_ttm_bn"]
    return out


def real_policy_rate(h: Dict[str, list]) -> Optional[float]:
    pol = _latest(h.get("policy"))
    cpi = h.get("cpi")
    if not _n(pol) or not cpi or len(cpi) < 13:
        return None
    yoy = (cpi[0][1] / cpi[12][1] - 1) * 100.0
    return float(pol - yoy)


def tipping_point(h: Dict[str, list]) -> Dict[str, Any]:
    """Curve shape and whether policy is at its peak relative to inflation."""
    out: Dict[str, Any] = {}
    c2 = _latest(h.get("curve_10y2y"))
    c3 = _latest(h.get("curve_10y3m"))
    out["curve_10y2y"] = c2
    out["curve_10y3m"] = c3
    inv = [x for x in (c2, c3) if _n(x) and x < 0]
    out["inverted"] = bool(inv)
    if _n(c2) and _n(c3):
        out["shape"] = ("inverted" if (c2 < 0 or c3 < 0) else
                        "flat" if (c2 < 0.25 and c3 < 0.25) else
                        "positively sloped")
        # Direction matters more than level: a curve that has *re-steepened*
        # after inverting is the classic late signature, not an all-clear.
        d2 = _chg(h.get("curve_10y2y"), 126)
        out["steepening_6m"] = (d2 > 0.15) if _n(d2) else None
    out["real_policy_rate"] = real_policy_rate(h)
    return out


def early_warnings(h: Dict[str, list]) -> Dict[str, Any]:
    """Spreads widening, and the riskiest debtors starting to miss payments."""
    out: Dict[str, Any] = {}
    hy = h.get("hy_oas")
    out["hy_oas"] = _latest(hy)
    out["hy_3m_bp"] = _chg(hy, 63)
    out["bbb_oas"] = _latest(h.get("bbb_oas"))
    if _n(out.get("hy_3m_bp")):
        out["spreads_widening"] = out["hy_3m_bp"] > 50

    cc = h.get("cc_delinq")
    out["cc_delinq"] = _latest(cc)
    out["cc_delinq_4q_pp"] = _chg(cc, 4)
    al = h.get("all_delinq")
    out["all_delinq"] = _latest(al)
    out["all_delinq_4q_pp"] = _chg(al, 4)
    if _n(out.get("cc_delinq_4q_pp")):
        out["riskiest_deteriorating"] = out["cc_delinq_4q_pp"] > 0.20
    return out


def tug_of_war(h: Dict[str, list]) -> Dict[str, Any]:
    """The four levers, and which way each is currently being pulled.

    Dalio's point is that the four must sum to a deleveraging; the question is
    only which combination, and who pays. Each lever is scored -1 (being pulled
    the deflationary way), 0 (neutral), +1 (being pulled the inflationary way),
    with the reasoning attached.
    """
    levers: List[Dict[str, Any]] = []

    # Austerity — is the primary balance improving?
    sus = sustainability(h)
    d = sus.get("deficit_ttm_bn")
    gdp = _latest(h.get("gdp"))
    if _n(d) and _n(gdp) and gdp:
        ratio = d / gdp
        pull = -1 if ratio < 0.03 else 0 if ratio < 0.05 else 1
        levers.append({"lever": "Austerity", "pull": pull,
                       "detail": f"deficit running {ratio * 100:.1f}% of GDP",
                       "reading": ("being applied" if pull < 0 else
                                   "absent — fiscal is expansionary" if pull > 0
                                   else "neutral")})
    else:
        levers.append({"lever": "Austerity", "pull": None,
                       "detail": "deficit or GDP unavailable",
                       "reading": "not evaluable"})

    # Defaults and restructuring — are losses being taken?
    ew = early_warnings(h)
    if _n(ew.get("cc_delinq_4q_pp")) or _n(ew.get("hy_3m_bp")):
        rising = bool(ew.get("riskiest_deteriorating")) or bool(
            ew.get("spreads_widening"))
        pull = -1 if rising else 0
        bits = []
        if _n(ew.get("cc_delinq")):
            bits.append(f"card delinquency {ew['cc_delinq']:.2f}% "
                        f"({ew['cc_delinq_4q_pp']:+.2f}pp y/y)"
                        if _n(ew.get("cc_delinq_4q_pp")) else
                        f"card delinquency {ew['cc_delinq']:.2f}%")
        if _n(ew.get("hy_oas")):
            bits.append(f"HY OAS {ew['hy_oas']:.0f}bp")
        levers.append({"lever": "Defaults / restructuring", "pull": pull,
                       "detail": "; ".join(bits),
                       "reading": ("losses being taken" if rising else
                                   "credit losses are not being realised — "
                                   "the lever is idle")})
    else:
        levers.append({"lever": "Defaults / restructuring", "pull": None,
                       "detail": "delinquency and spread history unavailable",
                       "reading": "not evaluable"})

    # Money printing — balance sheet and money supply direction
    fa = h.get("fed_assets")
    m2 = h.get("m2")
    fa_1y = _pct_chg(fa, 52)
    m2_1y = _pct_chg(m2, 12)
    if _n(fa_1y) or _n(m2_1y):
        expanding = (_n(fa_1y) and fa_1y > 0.01) or (_n(m2_1y) and m2_1y > 0.05)
        contracting = (_n(fa_1y) and fa_1y < -0.03)
        pull = 1 if expanding else -1 if contracting else 0
        bits = []
        if _n(fa_1y):
            bits.append(f"Fed balance sheet {fa_1y * 100:+.1f}% y/y")
        if _n(m2_1y):
            bits.append(f"M2 {m2_1y * 100:+.1f}% y/y")
        levers.append({"lever": "Money printing", "pull": pull,
                       "detail": "; ".join(bits),
                       "reading": ("being applied" if pull > 0 else
                                   "being withdrawn" if pull < 0 else
                                   "on hold — neither printing nor draining")})
    else:
        levers.append({"lever": "Money printing", "pull": None,
                       "detail": "balance sheet and M2 history unavailable",
                       "reading": "not evaluable"})

    # Wealth redistribution — is the gap closing or widening?
    t1 = h.get("top1_wealth")
    lvl = _latest(t1)
    chg3 = _chg(t1, 12)
    if _n(lvl):
        pull = -1 if (_n(chg3) and chg3 < -0.5) else 1 if (
            _n(chg3) and chg3 > 0.5) else 0
        levers.append({"lever": "Wealth redistribution", "pull": pull,
                       "detail": (f"top 1% hold {lvl:.1f}% of net worth"
                                  + (f", {chg3:+.1f}pp over 3y"
                                     if _n(chg3) else "")),
                       "reading": ("gap narrowing — redistribution occurring"
                                   if pull < 0 else
                                   "gap widening — redistribution pressure "
                                   "building, not releasing" if pull > 0 else
                                   "gap stable")})
    else:
        levers.append({"lever": "Wealth redistribution", "pull": None,
                       "detail": "top 1% net worth share unavailable",
                       "reading": "not evaluable"})

    pulls = [l["pull"] for l in levers if l["pull"] is not None]
    net = sum(pulls) if pulls else None
    if net is None:
        balance = "not evaluable"
    elif net >= 2:
        balance = ("tilted hard toward inflation — the adjustment is being made "
                   "through the currency and the price level, not the debt")
    elif net >= 1:
        balance = "tilted toward inflation"
    elif net <= -2:
        balance = ("tilted hard toward deflation — losses are being realised "
                   "rather than monetised")
    elif net <= -1:
        balance = "tilted toward deflation"
    else:
        balance = ("balanced on paper, but note *which* levers are idle — a "
                   "balance struck by two idle levers is not a policy, it is a "
                   "postponement")
    return {"levers": levers, "net": net, "balance": balance}


# ---------------------------------------------------------------------------
# Stage classification
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Early part of the cycle",
    2: "The bubble",
    3: "The top",
    4: "The depression",
    5: "The beautiful deleveraging",
    6: "Pushing on a string",
    7: "Normalisation",
}


# How much damage is available in each stage, which is NOT the stage number.
# Used only to decide which sector gets the headline when the two disagree.
STAGE_SEVERITY = {4: 6, 3: 5, 2: 4, 6: 3, 5: 2, 7: 1, 1: 0}


def _score(preds: List[Tuple[str, Optional[bool]]]) -> Tuple[Optional[float], list]:
    ev = [(n, p) for n, p in preds if p is not None]
    if not ev:
        return None, []
    return sum(1 for _n_, p in ev if p) / len(ev), ev


def classify(h: Dict[str, list], cape: Optional[float],
             vix: Optional[float]) -> Dict[str, Any]:
    """Score all seven stages, per sector, and report the disagreement."""
    v = debt_velocity(h)
    sus = sustainability(h)
    tp = tipping_point(h)
    ew = early_warnings(h)
    real = tp.get("real_policy_rate")
    pol = _latest(h.get("policy"))
    fa_1y = _pct_chg(h.get("fed_assets"), 52)
    t1_chg = _chg(h.get("top1_wealth"), 12)

    hot_val = (cape > CAPE_MODERN_MEAN) if _n(cape) else None
    tight_credit = (ew["hy_oas"] < HY_OAS_CALM) if _n(ew.get("hy_oas")) else None
    wide_credit = (ew["hy_oas"] > HY_OAS_STRESS) if _n(ew.get("hy_oas")) else None
    at_zero = (pol < ZERO_BOUND) if _n(pol) else None
    restrictive = (real > 0) if _n(real) else None
    inverted = tp.get("inverted")
    fed_rising = (v["fed_3y_pp"] > 0) if _n(v.get("fed_3y_pp")) else None
    fed_fast = v.get("fails_velocity_test")
    hh_rising = (v["hh_3y_pp"] > 0) if _n(v.get("hh_3y_pp")) else None
    self_funding = ((sus["interest_to_deficit"] > 0.5)
                    if _n(sus.get("interest_to_deficit")) else None)
    printing = (fa_1y > 0.01) if _n(fa_1y) else None
    gap_widening = (t1_chg > 0.5) if _n(t1_chg) else None

    def inv(x):
        return None if x is None else (not x)

    public = {
        1: [("debt/GDP not rising", inv(fed_rising)),
            ("interest not self-funding", inv(self_funding)),
            ("credit spreads calm", tight_credit)],
        2: [("debt/GDP rising", fed_rising),
            ("asset prices rich", hot_val),
            ("credit still cheap", tight_credit),
            ("not yet at the zero bound", inv(at_zero))],
        3: [("policy restrictive in real terms", restrictive),
            ("curve inverted or flat", inverted),
            ("interest now self-funding", self_funding),
            ("asset prices rich", hot_val)],
        4: [("policy at the zero bound", at_zero),
            ("spreads at stress levels", wide_credit),
            ("riskiest debtors deteriorating",
             ew.get("riskiest_deteriorating"))],
        5: [("printing under way", printing),
            ("debt/GDP falling", inv(fed_rising)),
            ("spreads normalising", inv(wide_credit))],
        # Stage 6 shares "zero rates + printing" with stage 4. What separates
        # them is that the defaults have already happened: pushing on a string
        # is the *quiet* aftermath, so calm spreads are a required condition,
        # not an incidental one. Without this the two stages tie and the call
        # falls to a tiebreak rather than to evidence.
        6: [("policy at the zero bound", at_zero),
            ("printing under way", printing),
            ("credit already calm — defaults are behind, not ahead",
             inv(wide_credit)),
            ("wealth gap widening", gap_widening)],
        7: [("policy off the zero bound", inv(at_zero)),
            ("debt/GDP stable or falling", inv(fed_rising)),
            ("spreads normal", tight_credit)],
    }
    private = {
        1: [("household debt/GDP not rising", inv(hh_rising)),
            ("delinquencies not deteriorating",
             inv(ew.get("riskiest_deteriorating"))),
            ("spreads calm", tight_credit)],
        2: [("household debt/GDP rising", hh_rising),
            ("asset prices rich", hot_val),
            ("credit cheap", tight_credit)],
        3: [("policy restrictive in real terms", restrictive),
            ("curve inverted or flat", inverted),
            ("delinquencies turning up", ew.get("riskiest_deteriorating"))],
        4: [("policy at the zero bound", at_zero),
            ("spreads at stress levels", wide_credit),
            ("delinquencies deteriorating", ew.get("riskiest_deteriorating"))],
        5: [("printing under way", printing),
            ("household debt/GDP falling", inv(hh_rising)),
            ("spreads normalising", inv(wide_credit))],
        # Stage 6 shares "zero rates + printing" with stage 4. What separates
        # them is that the defaults have already happened: pushing on a string
        # is the *quiet* aftermath, so calm spreads are a required condition,
        # not an incidental one. Without this the two stages tie and the call
        # falls to a tiebreak rather than to evidence.
        6: [("policy at the zero bound", at_zero),
            ("printing under way", printing),
            ("credit already calm — defaults are behind, not ahead",
             inv(wide_credit)),
            ("wealth gap widening", gap_widening)],
        7: [("policy off the zero bound", inv(at_zero)),
            ("household debt/GDP stable or falling", inv(hh_rising)),
            ("spreads normal", tight_credit)],
    }

    def pick(book):
        scored = {}
        for stage, preds in book.items():
            s, ev = _score(preds)
            if s is not None:
                scored[stage] = {"score": s, "evidence": ev}
        if not scored:
            return None, {}
        # Ties break toward the riskier stage, not the higher number, for the
        # same reason the headline does.
        best = max(scored, key=lambda k: (scored[k]["score"], STAGE_SEVERITY[k]))
        return best, scored

    pub_stage, pub_all = pick(public)
    pri_stage, pri_all = pick(private)

    # The headline is the *riskier* sector, not the higher stage number. The
    # stages are a sequence, not a severity scale — stage 7 (normalisation) is
    # a calmer place than stage 2 (a bubble), so taking max() on the number
    # would headline the safe sector and bury the dangerous one.
    if pub_stage and pri_stage:
        head = max((pub_stage, pri_stage), key=lambda s: STAGE_SEVERITY[s])
    else:
        head = pub_stage or pri_stage

    note = None
    if pub_stage and pri_stage and pub_stage != pri_stage:
        risk_side = "public" if head == pub_stage else "private"
        note = (f"The two sectors are not in the same place: the public sector "
                f"reads stage {pub_stage} ({STAGE_NAMES[pub_stage]}) while the "
                f"private sector reads stage {pri_stage} "
                f"({STAGE_NAMES[pri_stage]}). The headline follows the "
                f"{risk_side} sector because that is where the accident would "
                f"come from. The divergence is the story, not a rounding "
                f"error — a levered sovereign alongside an unlevered household "
                f"sector is a different problem from 2008, and it resolves "
                f"through the currency and the bond market rather than through "
                f"foreclosures.")

    # Report the runner-up whenever it is close. A stage call that wins by two
    # percentage points and is displayed as a single confident number is worse
    # than no call at all.
    runner = None
    book = pub_all if head == pub_stage else pri_all
    side = "public" if head == pub_stage else "private"
    if book:
        ranked = sorted(book.items(),
                        key=lambda kv: (-kv[1]["score"], -STAGE_SEVERITY[kv[0]]))
        if len(ranked) > 1 and ranked[1][1]["score"] >= ranked[0][1]["score"] - 0.15:
            tie = "ties with" if ranked[1][1]["score"] == ranked[0][1]["score"] \
                else "scores nearly as highly as"
            runner = (f"On the {side} sector, stage {ranked[1][0]} "
                      f"({STAGE_NAMES[ranked[1][0]]}) {tie} the headline call "
                      f"({ranked[1][1]['score']:.0%} vs "
                      f"{ranked[0][1]['score']:.0%}) — treat the stage as "
                      f"contested rather than settled.")

    return {
        "stage": head,
        "stage_name": STAGE_NAMES.get(head) if head else None,
        "public_stage": pub_stage,
        "public_stage_name": STAGE_NAMES.get(pub_stage) if pub_stage else None,
        "private_stage": pri_stage,
        "private_stage_name": STAGE_NAMES.get(pri_stage) if pri_stage else None,
        "sector_note": note,
        "contested": runner,
        "public_scores": {k: round(x["score"], 3) for k, x in pub_all.items()},
        "private_scores": {k: round(x["score"], 3) for k, x in pri_all.items()},
        "evidence": pub_all.get(pub_stage, {}).get("evidence", []),
    }


def asset_implications(h: Dict[str, list], stage: Optional[int],
                       tug: Dict[str, Any]) -> Dict[str, Any]:
    """Real assets versus financial assets, driven by two variables only.

    Dalio's rule of thumb reduces to: real assets win when the real rate is
    low or negative *and* the printing lever is engaged; financial assets win
    when the real rate is positive and the currency is being defended. Almost
    everything else is commentary.
    """
    real = real_policy_rate(h)
    net = tug.get("net")
    out: Dict[str, Any] = {"real_policy_rate": real, "lever_net": net}

    if not _n(real):
        out["reading"] = "real policy rate unavailable — cannot call this"
        return out

    inflationary = _n(net) and net >= 1
    if real < 0 and inflationary:
        out["favours"] = "real assets"
        out["reading"] = ("negative real rates with the printing lever engaged "
                          "— the configuration in which gold and commodities "
                          "have historically outrun stocks and bonds")
    elif real > 1.5 and not inflationary:
        out["favours"] = "financial assets"
        out["reading"] = ("positive real rates with no monetisation — cash and "
                          "bonds are being paid to exist, which is the "
                          "environment real assets struggle in")
    else:
        out["favours"] = "contested"
        out["reading"] = ("real rates are positive but the fiscal and "
                          "redistribution levers are pulling the other way — "
                          "this is the zone where both real and financial "
                          "assets can rise together and the resolution comes "
                          "later, through the currency")
    return out


def universe_cape(records, metrics_by_ticker: Dict[str, dict],
                  cpi_hist: Optional[list] = None,
                  market: str = "US", min_years: int = 7) -> Dict[str, Any]:
    """Index-level CAPE built from our own constituents.

    This is an *aggregate*, not an average of ratios: total market value over
    total ten-year-average real earnings, which is how an index CAPE is
    properly constructed. Averaging per-company CAPEs would let one loss-making
    small cap with a near-zero denominator dominate the number.

    It is not the official Shiller series — that isn't free in machine-readable
    form — and it is labelled as ours everywhere it is shown. Because EDGAR
    gives US filers a full decade and Yahoo gives Asian filers about four
    years, this is only computed for the US market. Extending it elsewhere
    would compare a ten-year mean against a four-year one and call the
    difference valuation.
    """
    cpi_by_year: Dict[int, float] = {}
    if cpi_hist:
        for d, v in cpi_hist:
            try:
                y = int(str(d)[:4])
            except (TypeError, ValueError):
                continue
            cpi_by_year.setdefault(y, v)          # first seen = latest in year
    latest_cpi = cpi_by_year.get(max(cpi_by_year)) if cpi_by_year else None

    tot_cap = 0.0
    tot_earn = 0.0
    used = 0
    # Count *why* names drop out. A single "skipped" tally cannot distinguish
    # "the market genuinely lacks history" from "the code is reading the wrong
    # attribute", and those need completely different fixes.
    why = {"in_market": 0, "no_market_cap": 0, "too_few_years": 0,
           "decade_loss_maker": 0}
    for rec in records or []:
        if getattr(rec, "market", None) != market:
            continue
        why["in_market"] += 1
        m = metrics_by_ticker.get(getattr(rec, "ticker", None)) or {}
        cap = m.get("market_cap_usd")
        fx = m.get("fx_to_usd")
        if not _n(cap) or not _n(fx):
            why["no_market_cap"] += 1
            continue
        # CompanyRecord stores its normalised statements in `.years`. Reading
        # any other attribute name yields an empty list on every single record
        # and produces a confident "no history available" that is really a
        # typo — so this must stay pinned to the dataclass field.
        years = {f.fiscal_year: f.net_income
                 for f in (getattr(rec, "years", None) or [])
                 if _n(getattr(f, "net_income", None))}
        recent = sorted(years, reverse=True)[:10]
        if len(recent) < min_years:
            why["too_few_years"] += 1
            continue
        vals = []
        for y in recent:
            e = years[y]
            if latest_cpi and cpi_by_year.get(y):
                e = e * (latest_cpi / cpi_by_year[y])
            vals.append(e * fx)                   # into USD
        avg = sum(vals) / len(vals)
        if avg <= 0:                              # a decade-average loss maker
            why["decade_loss_maker"] += 1
            continue
        tot_cap += cap
        tot_earn += avg
        used += 1

    # The Asia refresh carries no US constituents at all. That is not a data
    # failure and must not read like one on the page, or every evening run
    # would look broken.
    if why["in_market"] == 0:
        return {"error": f"no {market} names in this run — CAPE is computed on "
                         f"the US refresh only",
                "names_used": 0, "breakdown": why, "not_this_region": True}

    if used < 30 or tot_earn <= 0:
        return {"error": f"only {used} of {why['in_market']} {market} names "
                         f"were usable — {why['no_market_cap']} had no USD "
                         f"market cap, {why['too_few_years']} had fewer than "
                         f"{min_years} years of earnings, "
                         f"{why['decade_loss_maker']} lost money on a "
                         f"ten-year average",
                "names_used": used, "breakdown": why}
    return {"cape": tot_cap / tot_earn, "names_used": used,
            "breakdown": why,
            "inflation_adjusted": bool(latest_cpi),
            "note": "aggregate CAPE of our own US constituents, not the "
                    "official Shiller series"}


def build(fred, cape: Optional[float] = None,
          vix: Optional[float] = None) -> Dict[str, Any]:
    """Fetch the series and assemble the daily Dalio read."""
    if not getattr(fred, "enabled", False):
        return {"enabled": False,
                "reason": "FRED_API_KEY not set — debt-cycle stage skipped"}

    h: Dict[str, list] = {}
    missing: List[str] = []
    for name, sid in FRED_SERIES.items():
        obs = fred.history(sid, limit=800)
        if obs:
            h[name] = obs
        else:
            missing.append(f"{name} ({sid})")
    if missing:
        log.warning("debt cycle: %d series unavailable: %s",
                    len(missing), ", ".join(missing))

    cls = classify(h, cape, vix)
    tug = tug_of_war(h)
    return {
        "enabled": True,
        "stage": cls["stage"],
        "stage_name": cls["stage_name"],
        "classification": cls,
        "checklist": bubble_checklist(h, cape, vix),
        "velocity": debt_velocity(h),
        "sustainability": sustainability(h),
        "tipping_point": tipping_point(h),
        "early_warnings": early_warnings(h),
        "tug_of_war": tug,
        "assets": asset_implications(h, cls["stage"], tug),
        "missing_series": missing,
        "unavailable": UNAVAILABLE,
    }
