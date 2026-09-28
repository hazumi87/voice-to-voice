"""Fake Dot bridge for silent end-to-end tests (stdlib only).

Speaks the announce ingress contract of working/atom_echo/esphome/va_bridge.py without a
device: POST /announce/<id>?start_conversation=&timeout=&busy_wait=&reply_id= "plays" the WAV
for its real duration (capped) and answers {delivered:true, played_ms}; GET /devices reports
connected/busy; POST /__mode {"busy": seconds} makes the device busy for a while so the
8 s busy-wait and ifIdle paths can be exercised; GET /__log lists every announce received
(no audio kept). Nothing is ever heard.

  .venv\\Scripts\\python.exe tools\\fake_bridge.py --port 8294 --device fixture-dot
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"device": "fixture-dot", "busy_until": 0.0, "log": [], "play_cap_s": 3.0}
LOCK = threading.Lock()


def _wav_seconds(b: bytes) -> float:
    try:
        import soundfile as sf  # available in the venv; optional here
        info = sf.info(io.BytesIO(b))
        return float(info.frames) / float(info.samplerate or 24000)
    except Exception:  # noqa: BLE001
        return max(0.2, len(b) / (24000 * 2))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/devices":
            with LOCK:
                busy = time.time() < STATE["busy_until"]
            return self._json(200, {"devices": [{"id": STATE["device"], "enabled": True,
                                                 "connected": True, "busy": busy}]})
        if u.path == "/__log":
            with LOCK:
                return self._json(200, {"seen": STATE["log"]})
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if u.path == "/__mode":
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            with LOCK:
                STATE["busy_until"] = time.time() + float(body.get("busy") or 0)
            return self._json(200, {"busy_until": STATE["busy_until"]})
        if u.path.startswith("/announce/"):
            dev = u.path.split("/announce/", 1)[1]
            if dev != STATE["device"]:
                return self._json(404, {"delivered": False, "error": "device_unknown"})
            if len(raw) < 100:
                return self._json(400, {"delivered": False, "error": "empty_wav"})
            timeout = float(q.get("timeout", "30") or 30)
            busy_wait = q.get("busy_wait")
            wait_cap = timeout if busy_wait is None else float(busy_wait)
            t0 = time.time()
            while True:
                with LOCK:
                    busy = time.time() < STATE["busy_until"]
                if not busy:
                    break
                if time.time() - t0 >= wait_cap:
                    with LOCK:
                        STATE["log"].append({"ts": time.time(), "device": dev, "bytes": len(raw),
                                             "q": q, "result": "device_busy"})
                    return self._json(409, {"delivered": False, "error": "device_busy", "device": dev})
                time.sleep(0.1)
            secs = min(_wav_seconds(raw), STATE["play_cap_s"])
            time.sleep(secs)
            played = int(secs * 1000)
            with LOCK:
                STATE["log"].append({"ts": time.time(), "device": dev, "bytes": len(raw), "q": q,
                                     "result": "delivered", "played_ms": played,
                                     "wav_s": round(_wav_seconds(raw), 2)})
            return self._json(200, {"delivered": True, "played_ms": played, "device": dev})
        return self._json(404, {"error": "not_found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8294)
    ap.add_argument("--device", default="fixture-dot")
    a = ap.parse_args()
    STATE["device"] = a.device
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[fake-bridge] {a.device} on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
