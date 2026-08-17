"""Persistent library of deep-dive reports.

The problem this solves: GitHub Pages publishes whatever is in `out/` at the end
of a run, and each workflow builds `out/` from scratch. So a deep dive generated
on Monday disappears the moment Tuesday's refresh publishes its own `out/`.

The fix is to treat `data/deepdive/` as the library — it lives in the Actions
cache that every workflow restores — and to mirror it into `out/deepdive/` on
every publish. Reports then accumulate rather than replacing each other, and any
run republishes the whole set.
"""
from __future__ import annotations

import glob
import html
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List

LIB_SUBDIR = "deepdive"


def library_dir(data_dir: str = "data") -> str:
    p = os.path.join(data_dir, LIB_SUBDIR)
    os.makedirs(p, exist_ok=True)
    return p


def save_report(ticker: str, html_text: str, meta: Dict[str, Any],
                data_dir: str = "data") -> str:
    lib = library_dir(data_dir)
    with open(os.path.join(lib, f"{ticker}.html"), "w", encoding="utf-8") as f:
        f.write(html_text)
    with open(os.path.join(lib, f"{ticker}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return os.path.join(lib, f"{ticker}.html")


def list_reports(data_dir: str = "data") -> List[Dict[str, Any]]:
    lib = library_dir(data_dir)
    out = []
    for mp in glob.glob(os.path.join(lib, "*.meta.json")):
        try:
            with open(mp) as f:
                out.append(json.load(f))
        except Exception:                        # noqa: BLE001
            continue
    out.sort(key=lambda r: r.get("generated_utc", ""), reverse=True)
    return out


def publish(data_dir: str = "data", out_dir: str = "out") -> int:
    """Mirror the library into out/deepdive/ and write its index."""
    lib = library_dir(data_dir)
    dest = os.path.join(out_dir, LIB_SUBDIR)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for src in glob.glob(os.path.join(lib, "*.html")):
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
        n += 1
    for src in glob.glob(os.path.join(lib, "*.meta.json")):
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    _write_index(list_reports(data_dir), dest)
    return n


CALL_CLASS = {"BUY": "buy", "HOLD": "hold", "AVOID": "avoid"}


def _write_index(reports: List[Dict[str, Any]], dest: str):
    def e(s):
        return html.escape(str(s if s is not None else ""))

    rows = ""
    for r in reports:
        call = r.get("call", "")
        age = ""
        try:
            g = datetime.fromisoformat(str(r.get("generated_utc")).replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - g).days
            age = "today" if days == 0 else f"{days}d ago"
        except Exception:                        # noqa: BLE001
            age = e(str(r.get("generated_utc"))[:10])
        rows += (
            f'<tr onclick="location.href=\'{e(r["ticker"])}.html\'">'
            f'<td class="tk">{e(r["ticker"])}</td>'
            f'<td class="nm">{e(r.get("name"))}</td>'
            f'<td><span class="mk">{e(r.get("market_label") or r.get("market"))}</span></td>'
            f'<td><span class="call {CALL_CLASS.get(call, "")}">{e(call)}</span></td>'
            f'<td class="num">{e(r.get("frameworks_passed", "—"))}/7</td>'
            f'<td class="num">{e(r.get("price", "—"))}</td>'
            f'<td class="muted">{e(age)}</td></tr>')
    if not rows:
        rows = ('<tr><td colspan="7" class="muted" style="text-align:center;padding:40px">'
                'No reports yet — run the Deep dive workflow with a ticker.</td></tr>')

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep dives</title><style>
:root{{--bg:#f9f9f7;--surface:#fff;--line:rgba(11,11,11,.10);--tx:#0b0b0b;
--tx2:#52514e;--tx3:#898781;--acc:#2a78d6;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--surface:#1a1a19;
--line:rgba(255,255,255,.10);--tx:#fff;--tx2:#c3c2b7;--tx3:#898781;--acc:#3987e5}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font:14px/1.55 system-ui,
-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:900px;margin:0 auto;padding:34px 20px 70px}}
h1{{font-size:24px;margin:0 0 4px;letter-spacing:-.02em}}
.sub{{color:var(--tx2);font-size:13.5px;margin-bottom:20px}}
a{{color:var(--acc)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--surface);
border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th{{text-align:left;padding:10px;color:var(--tx3);font-size:10.5px;letter-spacing:.06em;
text-transform:uppercase;border-bottom:1px solid var(--line);font-weight:600}}
td{{padding:11px 10px;border-bottom:1px solid var(--line)}}
tr:last-child td{{border-bottom:none}}
tbody tr{{cursor:pointer}} tbody tr:hover{{background:var(--bg)}}
.tk{{font-weight:650;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}}
.nm{{color:var(--tx2)}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.muted{{color:var(--tx3);font-size:12px}}
.mk{{font-size:10.5px;padding:2px 7px;border-radius:4px;background:var(--bg);color:var(--tx2);font-weight:600}}
.call{{font-size:11px;font-weight:700;padding:3px 9px;border-radius:5px;letter-spacing:.04em}}
.call.buy{{background:rgba(12,163,12,.15);color:var(--good)}}
.call.hold{{background:rgba(250,178,25,.18);color:#8a6d10}}
.call.avoid{{background:rgba(208,59,59,.13);color:var(--bad)}}
@media(prefers-color-scheme:dark){{.call.hold{{color:var(--warn)}}}}
.foot{{color:var(--tx3);font-size:12px;margin-top:26px}}
</style></head><body><div class="wrap">
<h1>Deep dives</h1>
<div class="sub">{len(reports)} report{"" if len(reports) == 1 else "s"} ·
<a href="../">back to the screener</a> · generate more from the
<b>Deep dive (on demand)</b> workflow in GitHub Actions</div>
<table><thead><tr><th>Ticker</th><th>Name</th><th>Market</th><th>Call</th>
<th class="num">Passed</th><th class="num">Price</th><th>Generated</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="foot">Reports persist across nightly refreshes. Re-running a ticker
replaces its report. Not investment advice.</div>
</div></body></html>"""
    with open(os.path.join(dest, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


# ---------------------------------------------------------------------------
# Keeping the screener page alive across deep-dive publishes.
#
# GitHub Pages replaces the ENTIRE site with whatever artifact is uploaded.
# The deep-dive workflow builds an out/ containing only the report and its
# index, so deploying it wipes the screener's index.html and the site root
# starts returning 404 — the deep dive succeeds and the site disappears.
#
# The screener page is therefore snapshotted into the (cached) data store on
# every refresh, and restored into out/ before any deep-dive deploy. Cheap,
# and it does not require re-running an 11-minute pipeline to republish a page
# that has not changed.
# ---------------------------------------------------------------------------

def snapshot_site(out_dir: str = "out", data_dir: str = "data") -> bool:
    """Copy the rendered screener page into the data store."""
    src = os.path.join(out_dir, "index.html")
    if not os.path.exists(src):
        return False
    dest_dir = os.path.join(data_dir, "site")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_dir, "index.html"))
    return True


def restore_site(data_dir: str = "data", out_dir: str = "out") -> bool:
    """Put the last screener page back into out/ if it isn't already there.

    Returns False when there is no snapshot — which is the important case to
    report loudly, because deploying without one means the site root goes to
    404 and the only symptom is a page that used to work.
    """
    dest = os.path.join(out_dir, "index.html")
    if os.path.exists(dest):
        return True
    src = os.path.join(data_dir, "site", "index.html")
    if not os.path.exists(src):
        return False
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(src, dest)
    return True
