"""FRED provider — macro series for the Soros reflexivity/regime overlay.

Free API key from https://fredaccount.stlouisfed.org/apikeys — set FRED_API_KEY.
If the key is absent the macro gate is skipped rather than failing the run:
a missing macro feed should degrade the Soros screen, not kill the pipeline.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Any

import requests

log = logging.getLogger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_META_URL = "https://api.stlouisfed.org/fred/series"

# What FRED's `units_short` strings mean as a multiplier into plain dollars.
# This exists because the Buffett Indicator divides one FRED series by another
# and the two are NOT published in the same units: the Z.1 equities series is
# in millions, GDP is in billions. Getting that wrong produces a ratio off by
# a factor of a thousand that still looks like a number.
UNIT_SCALE = {
    "mil. of $": 1e6, "millions of dollars": 1e6,
    "bil. of $": 1e9, "billions of dollars": 1e9,
    "thous. of $": 1e3, "thousands of dollars": 1e3,
    "$": 1.0, "dollars": 1.0,
}


class FredProvider:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            log.warning("FRED_API_KEY not set — macro gate will be skipped")

    def latest(self, series_id: str) -> Optional[float]:
        if not self.enabled:
            return None
        try:
            r = requests.get(FRED_URL, timeout=20, params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 10,
            })
            if r.status_code != 200:
                log.warning("FRED %s -> HTTP %s", series_id, r.status_code)
                return None
            for obs in r.json().get("observations", []):
                if obs.get("value") not in (".", "", None):
                    return float(obs["value"])
        except Exception as e:                       # noqa: BLE001
            log.warning("FRED %s failed: %s", series_id, e)
        return None

    def series_meta(self, series_id: str) -> Dict[str, Any]:
        """Title, units and frequency for one series.

        Used to SCALE rather than to assume. Two series that look comparable on
        a chart can be published in different units, and a ratio built on that
        assumption is wrong by orders of magnitude while remaining perfectly
        plausible on screen.
        """
        if not self.enabled:
            return {}
        try:
            r = requests.get(FRED_META_URL, timeout=20, params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            })
            if r.status_code != 200:
                log.warning("FRED meta %s -> HTTP %s", series_id, r.status_code)
                return {}
            rows = r.json().get("seriess") or []
            if not rows:
                return {}
            s = rows[0]
            units = str(s.get("units_short") or s.get("units") or "").strip()
            return {"id": series_id, "title": s.get("title"), "units": units,
                    "scale": UNIT_SCALE.get(units.lower()),
                    "frequency": s.get("frequency_short"),
                    "last_updated": s.get("last_updated")}
        except Exception as e:                       # noqa: BLE001
            log.warning("FRED meta %s failed: %s", series_id, e)
        return {}

    def latest_observation(self, series_id: str):
        """(date, value) of the most recent real observation, or None.

        The date matters for the Buffett Indicator: both halves of the ratio
        are quarterly and they are not always published together, so a stale
        numerator over a fresh denominator would quietly misstate the level.
        """
        if not self.enabled:
            return None
        try:
            r = requests.get(FRED_URL, timeout=20, params={
                "series_id": series_id, "api_key": self.api_key,
                "file_type": "json", "sort_order": "desc", "limit": 10,
            })
            if r.status_code != 200:
                return None
            for obs in r.json().get("observations", []):
                if obs.get("value") not in (".", "", None):
                    return obs["date"], float(obs["value"])
        except Exception as e:                       # noqa: BLE001
            log.warning("FRED %s failed: %s", series_id, e)
        return None

    def history(self, series_id: str, limit: int = 800):
        """Newest-first list of (date, value), skipping FRED's '.' placeholders.

        The debt-cycle stage is about *changes* — debt/GDP over three years,
        spreads over three months, the balance sheet over a year — so a
        latest-value-only feed cannot support it. Missing observations are
        dropped rather than forward-filled: an interpolated value would make a
        3-year change look real when part of it was invented.
        """
        if not self.enabled:
            return []
        try:
            r = requests.get(FRED_URL, timeout=30, params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            })
            if r.status_code != 200:
                log.warning("FRED %s history -> HTTP %s", series_id,
                            r.status_code)
                return []
            out = []
            for obs in r.json().get("observations", []):
                v = obs.get("value")
                if v in (".", "", None):
                    continue
                try:
                    out.append((obs["date"], float(v)))
                except (TypeError, ValueError):
                    continue
            return out
        except Exception as e:                       # noqa: BLE001
            log.warning("FRED %s history failed: %s", series_id, e)
        return []

    def snapshot(self, series_map: Dict[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, sid in series_map.items():
            out[name] = self.latest(sid)
        if out.get("us_10y") is not None and out.get("us_2y") is not None:
            out["yield_curve_10y2y"] = out["us_10y"] - out["us_2y"]
        else:
            out["yield_curve_10y2y"] = None
        out["_enabled"] = self.enabled
        return out
