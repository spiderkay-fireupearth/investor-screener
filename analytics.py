"""Quantitative models for the single-ticker deep dive.

Everything here is deterministic and reproducible: the Monte Carlo is seeded
from the ticker, so the same input always yields the same fan chart. Nothing in
this module fetches anything — callers pass data in — which keeps the maths
testable without a network.

A note on what is NOT here. The brief asked for a "fractal" model built on
Zn+1 = f(Zn, C). That is the Mandelbrot set iteration; it has no established
mapping onto price levels, and numbers produced from it would look rigorous
while meaning nothing. In its place this module computes the **Hurst exponent**,
which answers the question that formula was presumably reaching for — does this
series trend, mean-revert, or wander? — and is a real, checkable statistic.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd


def _seed_from(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16)


def _n(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (
        isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# 3. Geometric Brownian Motion
# ---------------------------------------------------------------------------
def gbm_monte_carlo(close: pd.Series, ticker: str, horizon_days: int = 252,
                    n_paths: int = 20000, lookback: int = 756,
                    drift_cap: float = 0.25) -> Dict[str, Any]:
    """dS = mu*S*dt + sigma*S*dW, solved in log space and simulated.

    Estimated on up to three years of daily log returns. The drift is CAPPED:
    an unconstrained historical mu extrapolates a three-year run straight into
    the forecast, which is how GBM models end up promising 40% a year forever.
    The cap is disclosed in the output so the reader knows when it bound.
    """
    px = close.dropna().astype(float)
    if len(px) < 60:
        return {"error": "need at least 60 trading days"}

    px = px.iloc[-lookback:]
    logret = np.log(px / px.shift(1)).dropna().values
    if len(logret) < 50:
        return {"error": "insufficient return history"}

    mu_d, sd_d = float(np.mean(logret)), float(np.std(logret, ddof=1))
    mu_a_raw, sigma_a = mu_d * 252.0, sd_d * math.sqrt(252.0)
    mu_a = max(-drift_cap, min(drift_cap, mu_a_raw))
    capped = abs(mu_a_raw) > drift_cap

    s0 = float(px.iloc[-1])
    T = horizon_days / 252.0
    rng = np.random.default_rng(_seed_from(ticker))
    z = rng.standard_normal(n_paths)
    # Closed-form terminal distribution; mu_a here is the log drift already.
    terminal = s0 * np.exp(mu_a * T + sigma_a * math.sqrt(T) * z)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q = {f"p{p}": float(np.percentile(terminal, p)) for p in pcts}

    # Paths for the fan chart. 800 keeps the percentile bands smooth — at 200
    # the outer bands are visibly noisy and read as structure that isn't there.
    steps = min(horizon_days, 252)
    zp = rng.standard_normal((800, steps))
    incr = (mu_a / 252.0 - 0.5 * (sigma_a ** 2) / 252.0) + \
           sigma_a / math.sqrt(252.0) * zp
    paths = s0 * np.exp(np.cumsum(incr, axis=1))
    band = {f"p{p}": np.percentile(paths, p, axis=0).tolist()
            for p in (10, 25, 50, 75, 90)}

    return {
        "s0": s0,
        "mu_annual": mu_a, "mu_annual_raw": mu_a_raw, "drift_capped": capped,
        "sigma_annual": sigma_a,
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "percentiles": q,
        "prob_above_spot": float(np.mean(terminal > s0)),
        "prob_up_20pct": float(np.mean(terminal > s0 * 1.20)),
        "prob_down_20pct": float(np.mean(terminal < s0 * 0.80)),
        "band": band,
        # Entry at the 25th percentile, exit at the 75th: buy into the cheap
        # tail of the modelled distribution, trim into the rich one.
        "entry": q["p25"], "exit": q["p75"],
    }


def prob_over_years(mu_a: float, sigma_a: float, years: float = 3.0) -> Dict[str, float]:
    """Closed-form P(S_T > S_0) and friends under GBM — no simulation needed."""
    if not (_n(mu_a) and _n(sigma_a)) or sigma_a <= 0:
        return {}
    from statistics import NormalDist
    nd = NormalDist()
    denom = sigma_a * math.sqrt(years)
    up = 1 - nd.cdf((0 - mu_a * years) / denom)
    return {
        "p_up": up, "p_down": 1 - up,
        "p_up_50": 1 - nd.cdf((math.log(1.5) - mu_a * years) / denom),
        "p_down_30": nd.cdf((math.log(0.7) - mu_a * years) / denom),
    }


# ---------------------------------------------------------------------------
# 8. Fractal-adjacent: the Hurst exponent (honest replacement for Zn+1=f(Zn,C))
# ---------------------------------------------------------------------------
def hurst_exponent(close: pd.Series, min_chunk: int = 8) -> Dict[str, Any]:
    """Rescaled-range (R/S) estimate of long-memory in the return series.

    H ≈ 0.5  random walk — past direction says nothing about the next move
    H > 0.55 persistent/trending — momentum has an edge
    H < 0.45 anti-persistent/mean-reverting — fade extremes
    """
    px = close.dropna().astype(float).values
    if len(px) < 128:
        return {"error": "need 128+ observations"}
    ts = np.diff(np.log(px))
    n = len(ts)
    sizes, rs = [], []
    size = min_chunk
    while size <= n // 2:
        chunks = n // size
        vals = []
        for i in range(chunks):
            c = ts[i * size:(i + 1) * size]
            if len(c) < 2:
                continue
            dev = np.cumsum(c - c.mean())
            r = dev.max() - dev.min()
            s = c.std(ddof=1)
            if s > 0 and r > 0:
                vals.append(r / s)
        if vals:
            sizes.append(size)
            rs.append(float(np.mean(vals)))
        size *= 2
    if len(sizes) < 3:
        return {"error": "not enough scales"}
    h = float(np.polyfit(np.log(sizes), np.log(rs), 1)[0])
    if h > 0.55:
        regime, reading = "persistent", ("Trends persist — momentum signals carry "
                                         "more weight than mean-reversion here.")
    elif h < 0.45:
        regime, reading = "mean-reverting", ("Moves tend to reverse — extremes are "
                                             "more likely to fade than to extend.")
    else:
        regime, reading = "random walk", ("Close to a random walk — neither momentum "
                                          "nor mean-reversion has a demonstrable edge.")
    return {"hurst": h, "regime": regime, "interpretation": reading,
            "scales": sizes, "rs": rs,
            # R/S is known to be biased upward on finite samples: a true random
            # walk often estimates near 0.55-0.60. Read values just over the
            # threshold as "not distinguishable from random", not as trend.
            "caveat": "R/S estimation is biased upward on short samples — treat "
                      "0.50-0.60 as indistinguishable from a random walk"}


# ---------------------------------------------------------------------------
# 4. Technical levels
# ---------------------------------------------------------------------------
def fibonacci_levels(high: float, low: float, uptrend: bool = True) -> Dict[str, float]:
    if not (_n(high) and _n(low)) or high <= low:
        return {}
    span = high - low
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    if uptrend:      # retracements DOWN from the swing high
        return {f"{r:.1%}": high - span * r for r in ratios}
    return {f"{r:.1%}": low + span * r for r in ratios}


def swing_levels(df: pd.DataFrame, window: int = 20,
                 lookback: int = 504) -> Dict[str, Any]:
    """Support and resistance from local extrema, clustered into bands."""
    d = df.iloc[-lookback:]
    if len(d) < window * 3:
        return {}
    hi, lo, close = d["High"], d["Low"], d["Close"]
    price = float(close.iloc[-1])

    peaks = [float(hi.iloc[i]) for i in range(window, len(d) - window)
             if hi.iloc[i] == hi.iloc[i - window:i + window + 1].max()]
    troughs = [float(lo.iloc[i]) for i in range(window, len(d) - window)
               if lo.iloc[i] == lo.iloc[i - window:i + window + 1].min()]

    def cluster(levels, tol=0.02):
        levels = sorted(levels)
        out = []
        for lv in levels:
            if out and abs(lv - out[-1][-1]) / max(out[-1][-1], 1e-9) < tol:
                out[-1].append(lv)
            else:
                out.append([lv])
        # Bands touched more than once are the ones that matter.
        return sorted(((float(np.mean(g)), len(g)) for g in out),
                      key=lambda x: -x[1])

    res = [(lv, k) for lv, k in cluster(peaks) if lv > price]
    sup = [(lv, k) for lv, k in cluster(troughs) if lv < price]
    return {
        "price": price,
        "resistance": sorted(res, key=lambda x: x[0])[:3],
        "support": sorted(sup, key=lambda x: -x[0])[:3],
        "swing_high": float(hi.max()), "swing_low": float(lo.min()),
    }


# ---------------------------------------------------------------------------
# 2.1.3 – 2.1.6 Valuation frameworks
# ---------------------------------------------------------------------------
def fed_model(eps: Optional[float], price: Optional[float],
              treasury_10y_pct: Optional[float]) -> Dict[str, Any]:
    """Equity earnings yield against the 10-year. Positive spread = equities
    cheap relative to bonds on this (much-criticised) measure."""
    if not (_n(eps) and _n(price)) or price <= 0:
        return {"error": "need EPS and price"}
    ey = eps / price
    out = {"earnings_yield": ey}
    if _n(treasury_10y_pct):
        out["treasury_10y"] = treasury_10y_pct / 100.0
        out["spread"] = ey - treasury_10y_pct / 100.0
        out["verdict"] = ("equities cheap vs bonds" if out["spread"] > 0
                          else "bonds cheap vs equities")
    return out


def bbb_implied_rate(sp_earnings_yield: Optional[float],
                     risk_premium: float = 0.028,
                     earnings_growth: float = 0.067) -> Optional[float]:
    """The brief's formula: S&P earnings yield − 2.8% risk premium + 6.7% growth."""
    if not _n(sp_earnings_yield):
        return None
    return sp_earnings_yield - risk_premium + earnings_growth


def tobins_q(market_cap: Optional[float], total_liabilities: Optional[float],
             total_assets: Optional[float]) -> Dict[str, Any]:
    """Approximate Tobin's Q = (equity market value + liabilities) / total assets.

    Book value stands in for replacement cost, which is the standard practical
    approximation — and its main weakness. Q < 1 suggests the market values the
    firm below the cost of rebuilding its asset base.
    """
    if not (_n(market_cap) and _n(total_assets)) or total_assets <= 0:
        return {"error": "need market cap and total assets"}
    q = (market_cap + (total_liabilities or 0.0)) / total_assets
    return {"q": q,
            "verdict": ("below replacement cost — cheap on this measure" if q < 1
                        else "at or above replacement cost")}


def cape_ratio(price: Optional[float], eps_by_year: Dict[int, float],
               cpi_by_year: Optional[Dict[int, float]] = None,
               min_years: int = 7) -> Dict[str, Any]:
    """Shiller CAPE: price ÷ the 10-year average of inflation-adjusted EPS.

    Needs a decade of earnings. EDGAR supplies that for US filers; Yahoo's ~4
    years does not, so this returns an explicit shortfall rather than a number
    computed from too little data.
    """
    yrs = sorted(eps_by_year.keys(), reverse=True)[:10]
    vals = [eps_by_year[y] for y in yrs if _n(eps_by_year.get(y))]
    if not _n(price) or len(vals) < min_years:
        return {"error": f"needs {min_years}+ years of EPS, have {len(vals)}",
                "years_available": len(vals)}
    if cpi_by_year:
        latest_cpi = cpi_by_year.get(max(cpi_by_year), None)
        if latest_cpi:
            vals = [eps_by_year[y] * (latest_cpi / cpi_by_year[y])
                    for y in yrs if _n(eps_by_year.get(y)) and cpi_by_year.get(y)]
    avg = float(np.mean(vals))
    if avg <= 0:
        return {"error": "average real EPS is not positive"}
    return {"cape": price / avg, "avg_real_eps": avg,
            "years_used": len(vals),
            "inflation_adjusted": bool(cpi_by_year)}


def gordon_growth(dps_next: Optional[float], required_return: Optional[float],
                  growth: Optional[float]) -> Dict[str, Any]:
    """V = D1 / (r − g). Only meaningful for a stable dividend payer."""
    if not all(_n(x) for x in (dps_next, required_return, growth)):
        return {"error": "needs dividend, required return and growth"}
    if growth >= required_return:
        return {"error": "growth >= required return — model undefined "
                         "(the formula diverges; treat as not applicable)"}
    return {"value": dps_next / (required_return - growth),
            "required_return": required_return, "growth": growth}


def dividend_relevance(close: pd.Series, dividends: Optional[pd.Series],
                       min_obs: int = 8) -> Dict[str, Any]:
    """R² of next-year return on starting dividend yield.

    Reported with its sample size because with a decade of annual observations
    this statistic has almost no power. A high R² on 8 points is noise, and
    presenting it without that caveat would be misleading.
    """
    if dividends is None or len(dividends) == 0:
        return {"error": "no dividend history"}
    try:
        ann_div = dividends.resample("YE").sum()
        ann_px = close.resample("YE").last()
        df = pd.concat([ann_div.rename("d"), ann_px.rename("p")],
                       axis=1, join="inner").dropna()
        df["yield"] = df["d"] / df["p"]
        df["fwd_ret"] = df["p"].shift(-1) / df["p"] - 1
        df = df.dropna()
        if len(df) < min_obs:
            return {"error": f"only {len(df)} annual observations",
                    "n_obs": len(df)}
        x, y = df["yield"].values, df["fwd_ret"].values
        r = float(np.corrcoef(x, y)[0, 1])
        return {"r_squared": r ** 2, "correlation": r, "n_obs": len(df),
                "current_yield": float(df["yield"].iloc[-1]),
                "caveat": f"{len(df)} annual observations — very low statistical "
                          f"power; treat as descriptive, not predictive"}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 17. Market context, computed from series we already hold
# ---------------------------------------------------------------------------
def breadth_from_universe(price_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """% of a universe above its 50- and 200-day averages, plus a TRIN proxy.

    Computed from our own S&P 500 price history rather than a vendor feed. The
    TRIN figure is an approximation over this universe, NOT the NYSE index —
    labelled as such wherever it is displayed.
    """
    above50 = above200 = adv = dec = 0
    advol = decvol = 0.0
    total = 0
    for _t, df in price_frames.items():
        if df is None or len(df) < 200:
            continue
        c = df["Close"].astype(float)
        total += 1
        if c.iloc[-1] > c.rolling(50).mean().iloc[-1]:
            above50 += 1
        if c.iloc[-1] > c.rolling(200).mean().iloc[-1]:
            above200 += 1
        chg = c.iloc[-1] - c.iloc[-2]
        v = float(df["Volume"].iloc[-1]) if "Volume" in df else 0.0
        if chg > 0:
            adv += 1
            advol += v
        elif chg < 0:
            dec += 1
            decvol += v
    if not total:
        return {"error": "no price frames"}
    out = {
        "universe_size": total,
        "pct_above_50dma": above50 / total,
        "pct_above_200dma": above200 / total,
        "pct_below_50dma": 1 - above50 / total,
        "pct_below_200dma": 1 - above200 / total,
        "advancers": adv, "decliners": dec,
    }
    if dec and decvol and adv and advol:
        trin = (adv / dec) / (advol / decvol)
        out["trin_proxy"] = trin
        out["trin_reading"] = ("overbought" if trin < 0.8 else
                               "oversold" if trin > 1.2 else "neutral")
    # Internals diverging from the index is the signal worth having.
    if out["pct_above_50dma"] < out["pct_above_200dma"] - 0.15:
        out["internals"] = ("weakening — short-term participation is falling "
                            "behind the longer-term trend")
    elif out["pct_above_50dma"] > out["pct_above_200dma"] + 0.15:
        out["internals"] = ("improving — more names are reclaiming short-term "
                            "averages than hold long-term ones")
    else:
        out["internals"] = "broadly consistent across time frames"
    return out


def roc(series: pd.Series, weeks: int = 4) -> Optional[float]:
    s = series.dropna()
    days = weeks * 5
    if len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1)
