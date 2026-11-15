"""Spectre live dashboard server. Standard library only.

  python3 ui/server.py                      # reads runs/events.jsonl, port 8765
  SPECTRE_SAMPLE=1 python3 ui/server.py     # reads ui/sample_events.jsonl (sample data)
  PORT=9000 python3 ui/server.py

GET /                     -> ui/index.html
GET /events?tag=&since=   -> {"events": [...], "now": ts, "source": "...", "sample": bool}
    since: only events with ts > since. tag: only that tag, plus untagged events
    (referee verdicts carry no tag and are joined to their experiment in the browser).
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

UI = Path(__file__).resolve().parent
ROOT = UI.parent
SAMPLE = os.environ.get("SPECTRE_SAMPLE", "") not in ("", "0", "false")
EVENTS = UI / "sample_events.jsonl" if SAMPLE else Path(
    os.environ.get("SPECTRE_EVENTS", ROOT / "runs" / "events.jsonl"))

_cache = {"mtime": None, "size": None, "events": []}


def load_events() -> list[dict]:
    """Parse the jsonl file, re-reading only when it changed. Skips partial or bad lines."""
    try:
        st = EVENTS.stat()
    except FileNotFoundError:
        return []
    if (st.st_mtime, st.st_size) == (_cache["mtime"], _cache["size"]):
        return _cache["events"]
    out = []
    with open(EVENTS, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                out.append(ev)
    _cache.update(mtime=st.st_mtime, size=st.st_size, events=out)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if u.path == "/events":
            q = parse_qs(u.query)
            tag = (q.get("tag") or [""])[0]
            try:
                since = float((q.get("since") or ["0"])[0] or 0)
            except ValueError:
                since = 0.0
            evs = load_events()
            sel = [e for e in evs
                   if float(e.get("ts") or 0) > since and (not tag or e.get("tag") in (tag, None))]
            now = max((float(e.get("ts") or 0) for e in evs), default=since)
            body = json.dumps({"events": sel, "now": now, "source": str(EVENTS.relative_to(ROOT))
                               if EVENTS.is_relative_to(ROOT) else str(EVENTS),
                               "sample": SAMPLE, "server_time": time.time()})
            self._send(200, body.encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main():
    port = int(os.environ.get("PORT", "8765"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Spectre dashboard on http://localhost:{port}  (events: {EVENTS}{', SAMPLE DATA' if SAMPLE else ''})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
