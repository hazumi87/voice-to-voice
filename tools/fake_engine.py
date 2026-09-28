"""Fake briefing-table engine for room-voice V1 tests (stdlib only).

Implements just enough of E1's contract for v2v to be exercised end to end without the
NUC: connections, channels open/close/state, inbound (with handle resolution + fallback),
/said, context, lines, and a `reply` helper the test uses to push a seat reply back to
v2v's /api/voice/deliver. Every call is recorded in `calls` and served at GET /_calls.

  .venv\\Scripts\\python.exe tools\\fake_engine.py --port 8299 --bearer test-token \\
        --v2v http://127.0.0.1:8221 --v2v-bearer <token v2v accepts>
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {
    "bearer": "",
    "v2v": "",
    "v2v_bearer": "",
    "rooms": {
        "r1": {"name": "Voice Interaction", "lead": "briefing-table", "aliases": ["voice", "briefing table"],
               "seats": [
                   {"handle": "briefing-table", "spokenName": "briefing table", "lead": True, "state": "working", "activity": "E1", "lastLineAgoS": 180},
                   {"handle": "vrpc-briefing-table", "spokenName": "vrpc briefing table", "lead": False, "state": "idle", "activity": "", "lastLineAgoS": 1200},
                   {"handle": "aurora-design", "spokenName": "aurora", "lead": False, "state": "idle", "activity": "", "lastLineAgoS": 3900}],
               "tools": [{"title": "Tasks", "kind": "tasks", "seat": "briefing-table"}],
               "settings": {"voice": {"enabled": True, "character": None, "paraphrase": "off", "wordCap": 40, "idleMinutes": 30}}},
        "r2": {"name": "Workrooms build", "lead": "briefing-table", "aliases": ["workrooms"], "seats": [], "tools": [],
               "settings": {"voice": {"enabled": True, "character": None, "paraphrase": "off", "wordCap": 40, "idleMinutes": 30}}},
    },
    "channels": {},      # channelId -> {device, muted, waiting, openedAt}
    "lines": {},         # roomId -> [line]
    "open": {},          # roomId -> [{vid,...}]
    "calls": [],
    "next_line": 700,
}
LOCK = threading.Lock()


def _resolve(room: dict, to: str) -> tuple[str, str]:
    if not to or to.lower() in ("lead", "the lead"):
        return room["lead"], "lead"
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
    for s in room["seats"]:
        if norm(s["handle"]) == norm(to):
            return s["handle"], "exact"
    for s in room["seats"]:
        if norm(s.get("spokenName") or "") == norm(to) or norm(to) in norm(s["handle"]):
            return s["handle"], "alias"
    return room["lead"], "fallback-lead"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self) -> bool:
        if not STATE["bearer"]:
            return True
        return self.headers.get("Authorization", "") == "Bearer " + STATE["bearer"]

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/_calls":
            with LOCK:
                return self._json(200, {"calls": STATE["calls"], "channels": STATE["channels"],
                                        "lines": STATE["lines"], "open": STATE["open"]})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        with LOCK:
            STATE["calls"].append({"m": "GET", "path": u.path, "q": q, "t": time.time()})
        if u.path == "/api/voice/connections":
            conns = []
            with LOCK:
                for rid, r in STATE["rooms"].items():
                    ch = STATE["channels"].get("room:" + rid)
                    conns.append({"channelId": "room:" + rid, "name": r["name"], "aliases": r["aliases"],
                                  "kind": "room", "state": "open", "lead": r["lead"], "seats": len(r["seats"]),
                                  "connected": {"device": ch["device"], "since": ch["openedAt"]} if ch else None})
            return self._json(200, {"provider": "briefing-table", "connections": conns})
        parts = u.path.split("/")
        if len(parts) >= 6 and parts[1:4] == ["api", "voice", "rooms"]:
            rid = parts[4]
            with LOCK:
                r = STATE["rooms"].get(rid)
                if not r:
                    return self._json(404, {"error": "room_unknown"})
                ch = STATE["channels"].get("room:" + rid)
                if parts[5] == "context":
                    if not ch:
                        return self._json(409, {"error": "no_channel"})
                    lines = STATE["lines"].get(rid, [])
                    recent = [{"id": l["id"], "kind": l["kind"], "author": l["author"], "target": l.get("target"),
                               "body": l["body"][:300], "agoS": int(time.time() - l["t"])} for l in lines[-15:]]
                    spoken = [l for l in lines if l.get("spoken")]
                    last = spoken[-1] if spoken else None
                    return self._json(200, {
                        "user": {"name": "Eric"}, "room": {"name": r["name"], "mode": "open", "lead": r["lead"]},
                        "seats": r["seats"], "tools": r["tools"] + [{"title": "Voice", "kind": "voice", "seat": None}],
                        "connection": {"device": ch["device"], "since": ch["openedAt"], "muted": ch["muted"], "waiting": ch["waiting"]},
                        "recent": recent,
                        "open": [{**o, "agoS": int(time.time() - o["t"])} for o in STATE["open"].get(rid, [])],
                        "lastSpoken": ({"vid": last.get("vid"), "author": last["author"], "body": last["body"],
                                        "agoS": int(time.time() - last["t"])} if last else None)})
                if parts[5] == "lines":
                    words = (q.get("q") or "").lower().split()
                    hits = [l for l in STATE["lines"].get(rid, []) if all(w in l["body"].lower() for w in words)]
                    return self._json(200, {"lines": hits[-int(q.get("limit", 5)):]})
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = self._body()
        if u.path == "/_reply":
            return self._reply(body)
        if u.path == "/_reset":
            with LOCK:
                STATE["channels"].clear()
                STATE["lines"].clear()
                STATE["open"].clear()
                STATE["calls"].clear()
            return self._json(200, {"ok": True})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        with LOCK:
            STATE["calls"].append({"m": "POST", "path": u.path, "body": body, "t": time.time()})
        if u.path == "/api/voice/channels/open":
            cid = body.get("channelId") or ""
            rid = cid[5:] if cid.startswith("room:") else None
            with LOCK:
                r = STATE["rooms"].get(rid)
                if not r:
                    return self._json(404, {"error": "channel_unknown", "field": "channelId"})
                # one connection per device: close any other channel this device holds
                for k, ch in list(STATE["channels"].items()):
                    if ch["device"] == body.get("device") and k != cid:
                        STATE["channels"].pop(k)
                STATE["channels"][cid] = {"device": body.get("device"), "muted": False, "waiting": False,
                                          "openedAt": time.time()}
                STATE["lines"].setdefault(rid, []).append(self._line(rid, "fact", "system", None,
                                                                     f"voice connected · {body.get('device')}"))
            return self._json(200, {"channelId": cid, "name": r["name"], "lead": r["lead"],
                                    "settings": {"voice": r["settings"]["voice"]}})
        if u.path == "/api/voice/channels/close":
            with LOCK:
                ch = STATE["channels"].pop(body.get("channelId") or "", None)
            return self._json(200, {"ok": True, "closed": bool(ch)})
        parts = u.path.split("/")
        if len(parts) == 6 and parts[1:4] == ["api", "voice", "rooms"] and parts[5] in ("mute", "disconnect"):
            rid = parts[4]
            with LOCK:
                ch = STATE["channels"].get("room:" + rid)
                if not ch:
                    return self._json(409, {"error": "no_channel"})
                if parts[5] == "disconnect":
                    STATE["channels"].pop("room:" + rid, None)
                    return self._json(200, {"voice": None})
                ch["muted"] = bool(body.get("muted"))
                return self._json(200, {"voice": {"connected": True, "device": ch["device"],
                                                  "muted": ch["muted"], "waiting": ch["waiting"]}})
        if u.path == "/api/voice/inbound":
            cid = body.get("channelId") or ""
            rid = cid[5:] if cid.startswith("room:") else None
            with LOCK:
                ch = STATE["channels"].get(cid)
                if not ch:
                    return self._json(409, {"error": "no_channel"})
                r = STATE["rooms"][rid]
                to, resolved = _resolve(r, (body.get("route") or {}).get("to") or "lead")
                line = self._line(rid, "voice", "user", to, body.get("text") or "",
                                  meta={"via": "voice", "dir": "in", "utteranceId": body.get("utteranceId"),
                                        "route": body.get("route"), "resolved": resolved,
                                        "followupTo": body.get("followupTo")})
                vid = "v" + format(line["id"], "x")
                line["vid"] = vid
                STATE["lines"].setdefault(rid, []).append(line)
                STATE["open"].setdefault(rid, []).append({"vid": vid, "to": to, "t": time.time(),
                                                          "delivered": "channel", "ackAgoS": None, "resultAgoS": None})
                if body.get("followupTo"):
                    ch["waiting"] = False
            return self._json(202, {"vid": vid, "lineId": line["id"], "to": to, "delivered": "channel",
                                    "resolved": resolved, "name": r["name"]})
        parts = u.path.split("/")
        if len(parts) == 6 and parts[1:4] == ["api", "voice", "rooms"] and parts[5] == "said":
            rid = parts[4]
            with LOCK:
                if "room:" + rid not in STATE["channels"]:
                    return self._json(409, {"error": "no_channel"})
                if body.get("action") == "answer":
                    STATE["lines"].setdefault(rid, []).append(self._line(rid, "voice", "user", None, body.get("text") or ""))
                    STATE["lines"][rid].append(self._line(rid, "voice", "voice", None, body.get("answer") or "", spoken=True))
                if body.get("followupTo"):
                    ch = STATE["channels"]["room:" + rid]
                    ch["waiting"] = False
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not_found"})

    def _line(self, rid, kind, author, target, body, meta=None, spoken=False) -> dict:
        STATE["next_line"] += 1
        return {"id": STATE["next_line"], "kind": kind, "author": author, "target": target, "body": body,
                "meta": meta or {}, "t": time.time(), "spoken": spoken}

    def _reply(self, body: dict):
        """Test helper: a seat replies -> the fake engine delivers to v2v like E1 would."""
        rid = body.get("room") or "r1"
        cid = "room:" + rid
        with LOCK:
            ch = STATE["channels"].get(cid)
            r = STATE["rooms"][rid]
            kind = body.get("kind") or "result"
            text = body.get("text") or ""
            author = body.get("from") or r["lead"]
            cap = r["settings"]["voice"]["wordCap"]
            over = len(text.split()) > cap
            line = self._line(rid, "voice", author, None, text, meta={"dir": "out", "kind": kind, "re": body.get("re")})
            STATE["lines"].setdefault(rid, []).append(line)
            for o in STATE["open"].get(rid, []):
                if body.get("re") and o["vid"] in body["re"]:
                    if kind == "ack" and o["ackAgoS"] is None:
                        o["ackAgoS"] = 0
                    if kind in ("result", "question"):
                        o["resultAgoS"] = 0
        if not ch:
            return self._json(200, {"lineId": line["id"], "spoken": False, "reason": "not-connected"})
        if ch["muted"]:
            return self._json(200, {"lineId": line["id"], "spoken": False, "reason": "muted"})
        deliver = {"device": ch["device"], "text": text, "channelId": cid, "name": author,
                   "replyId": f"fake-{line['id']}", "inReplyTo": (body.get("re") or [None])[0],
                   "kind": kind, "expectsReply": kind in ("result", "question"),
                   "paraphrase": "off"}
        if over:
            deliver["overCap"] = "summarize"
            deliver["maxWords"] = cap
        if body.get("ifIdle"):
            deliver["ifIdle"] = True
        req = urllib.request.Request(STATE["v2v"] + "/api/voice/deliver", data=json.dumps(deliver).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + STATE["v2v_bearer"]})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())
                st = resp.status
        except urllib.error.HTTPError as e:
            out = json.loads(e.read() or b"{}")
            st = e.code
        except Exception as e:  # noqa: BLE001
            out, st = {"delivered": False, "error": repr(e)}, 502
        with LOCK:
            line["spoken"] = bool(out.get("delivered"))
            line["vid"] = None
            if kind in ("result", "question") and not out.get("delivered"):
                ch["waiting"] = True
        return self._json(200, {"lineId": line["id"], "spoken": bool(out.get("delivered")),
                                "deliver": out, "deliverStatus": st, "overCap": over})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--bearer", default="")
    ap.add_argument("--v2v", default="http://127.0.0.1:8221")
    ap.add_argument("--v2v-bearer", default="")
    a = ap.parse_args()
    STATE.update({"bearer": a.bearer, "v2v": a.v2v, "v2v_bearer": a.v2v_bearer})
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[fake-engine] listening on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
