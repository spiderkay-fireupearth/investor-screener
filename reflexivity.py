"""Where a single name sits on Soros's boom/bust path.

The Soros framework in `thresholds.yml` answers "does this name pass". That is
the wrong question to ask of this book. Soros's path has nine stages and the
investment implication *inverts* along it — the same evidence that makes BC a
buy makes DE a sell. Collapsing that into pass/fail throws away the only thing
the framework is for.

So this module labels the stage instead, from his own description:

    "At first, recognition of an underlying trend is lagging but the trend is
     strong enough to manifest itself in earnings per share (AB). When the
     underlying trend is finally recognized, it is reinforced by rising
     expectations (BC). Doubts arise, but the trend survives... Such testing
     may be repeated several times (CD). Eventually, conviction develops and it
     is no longer shaken by a setback in the earning trend (DE). Expectations
     become excessive, and fail to be sustained by reality (EF). The bias is
     recognized as such and expectations are lowered (FG). Stock prices lose
     their last prop and plunge (G). The underlying trend is reversed,
     reinforcing the decline (GH). Eventually, the pessimism becomes overdone
     and the market stabilizes (HI)."

Two design decisions worth defending.

**The framework is OFF by default.** Soros's subject is a company whose share
price is an *input* to its own fundamentals — the mortgage trusts that funded
growth by issuing stock above book, the conglomerates that bought earnings with
their own multiple. A company that neither issues, retires nor acquires has no
such channel: its price reflects its fundamentals and does not feed them. For
those names the stage is reported as `near-equilibrium` and no reflexive claim
is made. That is most names, most of the time, and saying so is the honest
result rather than a gap.

**DE and HI look identical on the crude numbers** — earnings falling, price
rising — and they are opposite trades. They are separated by where the price
sits relative to its own range: conviction at the highs is DE, pessimism
exhausting near the lows is HI. Get that backwards and the framework tells you
to sell the bottom.

Nothing here is a recommendation. It is a label, its evidence, and what Soros
says that stage usually precedes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# Stage -> (short label, what he says happens next)
STAGES: Dict[str, Dict[str, str]] = {
    "AB": {"label": "Trend unrecognised",
           "note": "earnings are moving and the price has not noticed. Soros: "
                   "'recognition of an underlying trend is lagging but the "
                   "trend is strong enough to manifest itself in earnings per "
                   "share'. The stage before the crowd arrives.",
           "rule": 'earnings up more than 5%, price up less than 5%',
           "group": 'opportunity',
           "action": 'The business is improving and the market has not repriced it. The best risk/reward in the sequence — you are paid for noticing early. The risk is that you are early BECAUSE the earnings gain is not durable, so the question to answer is whether the improvement is structural.'},
    "BC": {"label": "Recognition",
           "note": "'the underlying trend is finally recognized, it is "
                   "reinforced by rising expectations'. Price and earnings "
                   "rising together, the multiple expanding. The self-"
                   "reinforcing phase.",
           "rule": 'earnings up, price up',
           "group": 'neutral',
           "action": 'The healthy phase: both moving together, the multiple expanding on real news. Nothing to act on — this is simply what owning a good business looks like while it works.'},
    "CD": {"label": "Tested and held",
           "note": "'Doubts arise, but the trend survives... Such testing may "
                   "be repeated several times.' A correction that was bought. "
                   "He notes the bias is harder to shake after each test.",
           "rule": 'earnings up, price up more, survived a drawdown of 10% or worse, still above its 200-day average',
           "group": 'neutral',
           "action": 'A correction that got bought. The trend is intact, but conviction hardens with each test — the holders are now less willing to hear bad news than they were before it.'},
    "DE": {"label": "Conviction through a setback",
           "note": "'conviction develops and it is no longer shaken by a "
                   "setback in the earning trend'. THE diagnostic stage: the "
                   "price is rising while the earnings are not. The market has "
                   "stopped listening to what it was originally responding to.",
           "rule": 'earnings flat or falling, price up more than 10%, still near the highs',
           "group": 'caution',
           "action": 'The price no longer rests on the earnings. Not a sell signal on its own, but the point at which momentum rather than results is carrying the quote. Raise your bar here; do not START a position on this stage.'},
    "EF": {"label": "Expectations excessive",
           "note": "'Expectations become excessive, and fail to be sustained "
                   "by reality.' The gap between what the price has done and "
                   "what the business has done is now wide.",
           "rule": 'price has outrun earnings by more than 60% over the year',
           "group": 'caution',
           "action": 'The gap is now wide and explicit. Late. If you own it, this is the stage where trimming costs least in regret — the thing that has to happen for the price to be right keeps getting larger.'},
    "FG": {"label": "De-rating",
           "note": "'The bias is recognized as such and expectations are "
                   "lowered.' Price falling on earnings that are still "
                   "growing — the multiple is coming out, not the business.",
           "rule": 'earnings still growing, price down more than 10%',
           "group": 'opportunity',
           "action": 'Often the most interesting badge for a value buyer: the MULTIPLE is coming out, not the business. De-rating without deterioration is what a cheap price on an intact company actually looks like.'},
    "GH": {"label": "Break, and the fundamentals follow",
           "note": "'Stock prices lose their last prop and plunge... The "
                   "underlying trend is reversed, reinforcing the decline.' "
                   "This is where reflexivity is CONFIRMED rather than "
                   "asserted: the earnings deteriorated after the price did.",
           "rule": 'earnings falling and price down more than 20%',
           "group": 'caution',
           "action": 'The bust, confirmed — and the only stage where reflexivity is PROVEN rather than asserted, because the earnings deteriorated after the price did. Not yet a bottom: the feedback is still running downhill.'},
    "HI": {"label": "Pessimism overdone",
           "note": "'Eventually, the pessimism becomes overdone and the market "
                   "stabilizes.' Price recovering while earnings are still "
                   "bad. The mirror of DE, and the opposite trade.",
           "rule": 'earnings falling, price rising, and more than 25% below the 52-week high',
           "group": 'opportunity',
           "action": "The mirror of DE and the opposite trade: price has stopped making lows while the accounts are still bad. Early-cycle recovery — with Soros's own warning that this can be early for a very long time."},
    "EQ": {"label": "Near-equilibrium",
           "note": "no observable channel from this company's share price back "
                   "into its own fundamentals — it neither issues, retires nor "
                   "acquires materially. Soros's framework does not apply, and "
                   "forcing it would be the error the framework exists to "
                   "avoid.",
           "rule": 'the company does not issue, retire or acquire in its own shares to any material degree',
           "group": 'off',
           "action": 'The framework is switched OFF, not failed. This price does not feed these fundamentals, so a stage label would be forcing a model onto a company it was not built for.'},
}

# Stages where the book says the risk is on the downside. Used to raise a
# visible flag regardless of which framework surfaced the name.
LATE_STAGES = ("DE", "EF")

# How the stages group for someone deciding whether to buy. This is the part
# that turns a taxonomy into something usable: the sequence AB->HI is a story,
# but an investor only needs to know which end of it they are standing at.
GROUPS = {
    "opportunity": ("AB", "FG", "HI"),
    "caution": ("DE", "EF", "GH"),
    "neutral": ("BC", "CD"),
    "off": ("EQ",),
}
GROUP_LABELS = {
    "opportunity": "Where to look",
    "caution": "Where to be careful",
    "neutral": "Trend intact — nothing to do",
    "off": "Framework does not apply",
}
GROUP_NOTES = {
    "opportunity": "The price has moved against a business that either has not "
                   "deteriorated or has already deteriorated as far as it is "
                   "going to.",
    "caution": "The price is being supported by something other than the "
               "business, or the business is still following the price down.",
    "neutral": "Price and earnings are moving together. The framework has "
               "nothing to add beyond what the value screens already say.",
    "off": "No channel from this share price back into its own fundamentals.",
}


def _n(x) -> bool:
    return (x is not None and isinstance(x, (int, float))
            and not (isinstance(x, float) and math.isnan(x)))


def channel_open(m: Dict[str, Any]) -> Dict[str, Any]:
    """Is there a live path from this share price to these fundamentals?

    Three observable channels, all from the book:
      * issuance — the mortgage trusts, "selling additional shares at a
        premium over book value"
      * retirement — the same mechanism run backwards
      * acquisition — the conglomerates, offering "their own highly priced
        stock in acquiring other companies"; carried goodwill is the residue
    """
    shares = m.get("share_count_change_1y")
    gw = m.get("goodwill_to_assets")
    reasons = []
    if _n(shares) and abs(shares) >= 0.02:
        reasons.append("share count moved {:+.1%} in a year — the company is "
                       "transacting in its own equity".format(shares))
    if _n(gw) and gw >= 0.10:
        reasons.append("goodwill is {:.0%} of assets — it has bought "
                       "businesses, and can buy more with paper".format(gw))
    if not _n(shares) and not _n(gw):
        return {"open": None, "reasons": ["share count and goodwill both "
                                          "unavailable — cannot tell"]}
    return {"open": bool(reasons), "reasons": reasons or [
        "share count static and little goodwill — no observable channel from "
        "the price back into the business"]}


def stage(m: Dict[str, Any]) -> Dict[str, Any]:
    """Label the stage, with the evidence that produced it."""
    ch = channel_open(m)
    eps_g = m.get("eps_growth_1y")
    r12 = m.get("return_12m")
    div = m.get("reflexive_divergence")
    dd1 = m.get("max_drawdown_1y")
    above200 = m.get("price_above_sma200")
    off_high = m.get("pct_below_52w_high")

    ev = []
    for k, v, fmt in (("1y EPS growth", eps_g, "{:+.1%}"),
                      ("12m price", r12, "{:+.1%}"),
                      ("price minus earnings", div, "{:+.1%}"),
                      ("worst 1y drawdown", dd1, "{:.1%}"),
                      ("below its 52w high", off_high, "{:.1%}")):
        if _n(v):
            ev.append(f"{k} {fmt.format(v)}")

    if not (_n(eps_g) and _n(r12)):
        return {"stage": None, "label": "not enough data",
                "note": "needs one year of earnings and one year of price",
                "channel": ch, "evidence": ev}

    if ch["open"] is False:
        return {"stage": "EQ", **STAGES["EQ"], "channel": ch, "evidence": ev}

    near_high = _n(off_high) and off_high < 0.10
    far_off_high = _n(off_high) and off_high > 0.25

    # Order matters: the diagnostic stages are tested before the benign ones,
    # so a name that qualifies for both is reported as the riskier.
    if eps_g <= 0 and r12 > 0.10:
        # Earnings not confirming, price rising anyway. Which one depends
        # entirely on where in its range the price is doing it.
        s = "DE" if not far_off_high else "HI"
    elif _n(div) and div > 0.60:
        s = "EF"
    elif eps_g > 0 and r12 < -0.10:
        s = "FG"
    elif eps_g < 0 and r12 < -0.20:
        s = "GH"
    elif eps_g > 0.05 and r12 < 0.05:
        s = "AB"
    elif (eps_g > 0 and r12 > eps_g and _n(dd1) and dd1 >= 0.10
          and above200 == 1):
        s = "CD"
    elif eps_g > 0 and r12 > 0:
        s = "BC"
    else:
        s = "EQ"

    out = {"stage": s, **STAGES[s], "channel": ch, "evidence": ev}
    out["late"] = s in LATE_STAGES
    if s == "DE":
        out["warning"] = ("price rising through an earnings setback near the "
                          "highs — the configuration Soros identifies as the "
                          "point at which the loop detaches from its own "
                          "justification")
    elif s == "EF":
        out["warning"] = ("the price has run far ahead of the earnings; "
                          "'expectations become excessive, and fail to be "
                          "sustained by reality'")
    elif s == "HI":
        out["warning"] = ("earnings still falling but the price has stopped "
                          "making lows well off its high — the mirror of DE, "
                          "and the opposite trade. Soros's own warning "
                          "applies: this can be early for a long time")
    return out


def annotate(results: Dict[str, Any],
             metrics_by_ticker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Attach a stage to every result, in place. Returns a stage census."""
    census: Dict[str, int] = {}
    for ticker, r in (results or {}).items():
        m = metrics_by_ticker.get(ticker) or r.get("metrics") or {}
        if not m:
            continue
        st = stage(m)
        r["reflexive"] = st
        # A name with no stage is counted too, under "n/a". The share of the
        # universe that could not be classified is the single most important
        # number for reading this panel: a census of 300 badges over 900 rows
        # describes a third of the table, and a legend that showed only the
        # 300 would imply it described all of it.
        key = st.get("stage") or "n/a"
        census[key] = census.get(key, 0) + 1
    return census
