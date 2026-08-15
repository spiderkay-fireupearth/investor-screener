"""SQLite persistence.

The rule this enforces: every successful fetch is written before anything is
computed. Yahoo is an unofficial feed that will fail on some days. When it does,
the run falls back to the last good snapshot per ticker, so a bad refresh costs
one day of freshness rather than the whole history. `staleness_days` surfaces in
the UI so a silently-degrading feed shows up as a visible number instead of as
quietly wrong screens.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    d      TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, d)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker      TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (ticker, fiscal_year, source)
);
CREATE TABLE IF NOT EXISTS profiles (
    ticker     TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fx (
    ccy TEXT NOT NULL, d TEXT NOT NULL, rate_to_usd REAL NOT NULL,
    PRIMARY KEY (ccy, d)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    region     TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    ok_count   INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    notes      TEXT
);
CREATE TABLE IF NOT EXISTS screen_results (
    run_id TEXT NOT NULL, ticker TEXT NOT NULL, d TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_screen_d ON screen_results(d);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = "data/screener.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -------------------------------------------------------------- prices
    def save_prices(self, ticker: str, df) -> int:
        if df is None or df.empty:
            return 0
        rows = []
        for idx, r in df.iterrows():
            rows.append((
                ticker, idx.date().isoformat() if hasattr(idx, "date") else str(idx),
                float(r.get("Open")) if r.get("Open") == r.get("Open") else None,
                float(r.get("High")) if r.get("High") == r.get("High") else None,
                float(r.get("Low")) if r.get("Low") == r.get("Low") else None,
                float(r.get("Close")) if r.get("Close") == r.get("Close") else None,
                float(r.get("Volume")) if r.get("Volume") == r.get("Volume") else None,
            ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker,d,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load_prices(self, ticker: str):
        import pandas as pd
        cur = self.conn.execute(
            "SELECT d,open,high,low,close,volume FROM prices WHERE ticker=? ORDER BY d",
            (ticker,))
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        df["d"] = pd.to_datetime(df["d"])
        df = df.set_index("d")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def price_staleness_days(self, ticker: str) -> Optional[int]:
        cur = self.conn.execute("SELECT MAX(d) AS m FROM prices WHERE ticker=?", (ticker,))
        row = cur.fetchone()
        if not row or not row["m"]:
            return None
        return (date.today() - date.fromisoformat(row["m"])).days

    # -------------------------------------------------------- fundamentals
    def save_fundamentals(self, ticker: str, years: List[Any], source: str) -> int:
        rows = [(ticker, y.fiscal_year, source, _now(), json.dumps(y.to_dict(), default=str))
                for y in years]
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO fundamentals "
                "(ticker,fiscal_year,source,fetched_at,payload) VALUES (?,?,?,?,?)", rows)
            self.conn.commit()
        return len(rows)

    def load_fundamentals(self, ticker: str, source: Optional[str] = None) -> List[Dict]:
        q = "SELECT payload FROM fundamentals WHERE ticker=?"
        args: List[Any] = [ticker]
        if source:
            q += " AND source=?"
            args.append(source)
        q += " ORDER BY fiscal_year DESC"
        return [json.loads(r["payload"]) for r in self.conn.execute(q, args).fetchall()]

    def fundamentals_age_days(self, ticker: str) -> Optional[int]:
        cur = self.conn.execute(
            "SELECT MAX(fetched_at) AS m FROM fundamentals WHERE ticker=?", (ticker,))
        row = cur.fetchone()
        if not row or not row["m"]:
            return None
        try:
            return (datetime.now(timezone.utc) - datetime.fromisoformat(row["m"])).days
        except ValueError:
            return None

    # ------------------------------------------------------------ profiles
    def save_profile(self, ticker: str, payload: Dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO profiles (ticker,fetched_at,payload) VALUES (?,?,?)",
            (ticker, _now(), json.dumps(payload, default=str)))
        self.conn.commit()

    def load_profile(self, ticker: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT payload FROM profiles WHERE ticker=?", (ticker,)).fetchone()
        return json.loads(row["payload"]) if row else None

    # ------------------------------------------------------------------ fx
    def save_fx(self, ccy: str, rate: float, d: Optional[str] = None):
        self.conn.execute("INSERT OR REPLACE INTO fx (ccy,d,rate_to_usd) VALUES (?,?,?)",
                          (ccy, d or date.today().isoformat(), rate))
        self.conn.commit()

    def latest_fx(self, ccy: str) -> Optional[float]:
        if ccy == "USD":
            return 1.0
        row = self.conn.execute(
            "SELECT rate_to_usd FROM fx WHERE ccy=? ORDER BY d DESC LIMIT 1",
            (ccy,)).fetchone()
        return float(row["rate_to_usd"]) if row else None

    # ----------------------------------------------------------------- runs
    def start_run(self, run_id: str, region: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id,region,started_at) VALUES (?,?,?)",
            (run_id, region, _now()))
        self.conn.commit()

    def end_run(self, run_id: str, ok: int, fail: int, notes: str = ""):
        self.conn.execute(
            "UPDATE runs SET ended_at=?, ok_count=?, fail_count=?, notes=? WHERE run_id=?",
            (_now(), ok, fail, notes, run_id))
        self.conn.commit()

    def save_screen_results(self, run_id: str, results: Dict[str, Any]):
        d = date.today().isoformat()
        rows = [(run_id, t, d, json.dumps(v, default=str)) for t, v in results.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO screen_results (run_id,ticker,d,payload) "
            "VALUES (?,?,?,?)", rows)
        self.conn.commit()

    def latest_results(self, region: Optional[str] = None) -> Dict[str, Any]:
        """Most recent screen result per ticker, across all runs."""
        q = """
        SELECT s.ticker, s.payload FROM screen_results s
        JOIN (SELECT ticker, MAX(d) AS md FROM screen_results GROUP BY ticker) x
          ON s.ticker = x.ticker AND s.d = x.md
        """
        out = {}
        for r in self.conn.execute(q).fetchall():
            payload = json.loads(r["payload"])
            if region and payload.get("market") != region:
                continue
            out[r["ticker"]] = payload
        return out
