"""HTTP smoke for the room-voice V1 glue on a RUNNING v2v (no Dot audio is played).

  .venv\\Scripts\\python.exe tools\\room_voice_http_smoke.py [--server http://127.0.0.1:8221] [--token NAME]

Uses a bearer from voice_auth.json (default token name: local-test). Checks: device state
GET/POST, the simulate hook outside a room, and the deliver validation order (500-char 413,
2000-char summarize cap, kind/ifIdle fields echoed) against an UNKNOWN device so nothing is
synthesized or spoken.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FAILS = []


def call(server, tok, method, path, body=None):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}", flush=True)
    if not cond:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8221")
    ap.add_argument("--token", default="local-test")
    ap.add_argument("--device", default="echo-dot-biscuit")
    a = ap.parse_args()
    with open(os.path.join(ROOT, "voice_auth.json"), encoding="utf-8") as f:
        tok = (json.load(f).get("tokens") or {}).get(a.token)
    if not tok:
        print(f"no token named {a.token!r} in voice_auth.json")
        sys.exit(2)
    S, D = a.server, a.device

    st, b = call(S, tok, "GET", f"/api/voice/devices/{D}/state")
    check("state GET", st == 200 and "connected" in b, f"{st} {json.dumps(b)[:160]}")
    was_connected = bool(b.get("connected"))

    st, b = call(S, tok, "GET", "/api/voice/devices/nope/state")
    check("state GET unknown device -> 404", st == 404, st)

    st, b = call(S, tok, "POST", f"/api/voice/devices/{D}/simulate", {"text": "is anyone working right now"})
    if was_connected:
        check("simulate in room answers/relays", st == 200 and b.get("outcome") in ("answered", "sent"), f"{st} {b.get('outcome')} {b.get('say')!r}")
    else:
        check("simulate outside a room -> not_connected", st == 200 and b.get("outcome") == "not_connected", f"{st} {b.get('outcome')}")

    long500 = "word " * 120          # 600 chars
    st, b = call(S, tok, "POST", "/api/voice/deliver", {"device": "nope", "text": long500})
    check("deliver 600 chars without overCap -> 413", st == 413 and b.get("error") == "too_long", f"{st} {b.get('error')} max={b.get('max')}")

    st, b = call(S, tok, "POST", "/api/voice/deliver", {"device": "nope", "text": long500, "overCap": "summarize", "maxWords": 40})
    check("deliver 600 chars with overCap:summarize passes the cap (then device_unknown)", st == 404 and b.get("error") == "device_unknown", f"{st} {b.get('error')}")

    st, b = call(S, tok, "POST", "/api/voice/deliver", {"device": "nope", "text": "x" * 2100, "overCap": "summarize"})
    check("deliver 2100 chars with summarize -> 413 at 2000", st == 413 and b.get("max") == 2000, f"{st} max={b.get('max')}")

    # The clear is exercised on the FIXTURE device only. On 2026-09-29 this smoke cleared the
    # live Dot's room connection (the engine still had it open) right after a restart.
    st, b = call(S, tok, "POST", "/api/voice/devices/fixture-dot/state", {"channelId": "", "connected": False})
    check("state POST clear on fixture-dot", st == 200 and b.get("connected") is False, f"{st} {json.dumps(b)[:120]}")
    if was_connected:
        print(f"  note: {D} is connected to a room; left untouched.")

    print("\n" + ("FAILURES: " + str(FAILS) if FAILS else "ALL PASS"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
