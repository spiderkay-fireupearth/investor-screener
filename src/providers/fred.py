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
