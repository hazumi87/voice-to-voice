"""Local stand-in for the briefing-table engine's POST /api/voice/inbound (P2), so the
v2v dev-mode branch (P3) can be exercised before the engine half exists.

Usage:  .venv\\Scripts\\python.exe tools\\engine_inbound_stub.py [port] [behaviour]
  behaviour: ok (default) | held | nochannel | forbidden | unauth | boom
Point voice_devices.json -> voice_engine.inbound_url at http://127.0.0.1:<port>/api/voice/inbound
(hot-reloaded). Every inbound body is printed so you can see exactly what the engine
would receive. Bearer is checked against secrets/briefing-table-engine.token if present.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8299
MODE = sys.argv[2] if len(sys.argv) > 2 else "ok"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(HERE, "secrets", "briefing-table-engine.token")
TOKEN = open(TOK_PATH, encoding="utf-8").read().strip() if os.path.exists(TOK_PATH) else ""
SEEN = set()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/voice/channels":
            return self._send(200, {"focus": "pane:stub#1", "channels": [
                {"channelId": "pane:stub#1", "name": "Briefing Table", "aliases": ["briefing-table"],
                 "slug": "briefing-table", "vendor": "claude-code", "state": "live"}]})
        self._send(404, {"error": "not-found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            return self._send(400, {"error": "bad-json", "field": "body"})
        auth = self.headers.get("Authorization", "")
        print(f"[stub] {self.path} auth={'ok' if (TOKEN and auth == 'Bearer ' + TOKEN) else auth[:12]!r} "
              f"body={json.dumps(body)}", flush=True)
        if self.path != "/api/voice/inbound":
            return self._send(404, {"error": "not-found"})
        if TOKEN and auth != "Bearer " + TOKEN:
            return self._send(401, {"error": "unauthorized"})
        if MODE == "unauth":
            return self._send(401, {"error": "unauthorized"})
        if MODE == "boom":
            return self._send(500, {"error": "boom"})
        for f in ("text", "device", "sid", "utteranceId"):
            if not body.get(f):
                return self._send(400, {"error": "missing", "field": f})
        if body["utteranceId"] in SEEN:
            print("[stub] duplicate utteranceId ignored", flush=True)
        SEEN.add(body["utteranceId"])
        if MODE == "forbidden" or body.get("sid") != "ef01f7ee2166":
            return self._send(403, {"error": "speaker-not-allowed"})
        if MODE == "nochannel":
            return self._send(409, {"error": "no-channel"})
        if MODE == "held":
            return self._send(202, {"delivered": False, "held": "draft", "channelId": "pane:stub#1",
                                    "name": "Briefing Table", "vendor": "claude-code"})
        return self._send(202, {"delivered": True, "channelId": "pane:stub#1",
                                "name": "Briefing Table", "vendor": "claude-code"})


print(f"[stub] engine inbound stub on 127.0.0.1:{PORT} behaviour={MODE} token={'set' if TOKEN else 'none'}",
      flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
