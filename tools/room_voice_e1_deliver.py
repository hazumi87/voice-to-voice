"""Engine -> REAL v2v delivery leg against the scratch fixture run with --v2v (room-voice §7).

Prereqs (all local, nothing audible):
  - tools/fake_bridge.py on :8294 as device "fixture-dot" (voice_devices.json has the entry)
  - the engine fixture: node scripts/room-voice-fixture.mjs --serve --port 8291 ... \
        --v2v http://127.0.0.1:8221 --v2v-token <file with the 'fixture' caller key>
  - the fixture's stdout saved to a log; this script reads the engine token, voice token,
    pane id and channelId from it (tokens never printed)

  .venv\\Scripts\\python.exe tools\\room_voice_e1_deliver.py --log <fixture log> [--engine http://127.0.0.1:8291]

Sequence: connect fixture-dot (my client) -> seat ack (spoken, no expectsReply) -> short result
(spoken, expectsReply, replyId = vid) -> long result (overCap summarize -> my summary, "details
are in the chat") -> device busy 12 s then result (my 8 s busy wait -> 409 -> engine chat-only
device_busy) -> mute -> result (engine: spoken:false muted, no deliver) -> unmute -> disconnect.
Evidence: the engine's reply responses + the fake bridge's /__log + v2v's server log lines.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import room_agent  # noqa: E402
import room_voice  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + str(detail)[:220]) if detail else ''}", flush=True)
    if not cond:
        FAILS.append(name)


def http(method, url, body=None, bearer=None, timeout=90):
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--engine", default="http://127.0.0.1:8291")
    ap.add_argument("--bridge", default="http://127.0.0.1:8294")
    ap.add_argument("--device", default="fixture-dot")
    ap.add_argument("--sid", default="ef01f7ee2166")
    a = ap.parse_args()
    log = open(a.log, encoding="utf-8").read()
    engine_tok = re.search(r"^\s*engine token\s+(\S{20,})", log, re.M).group(1)
    voice_tok = re.search(r"^\s*voice token\s+(\S{20,})", log, re.M).group(1)
    pane = re.search(r"pane (recipe:\S+)", log).group(1)
    channel = re.search(r"channelId (room:\S+)", log).group(1)
    rid = channel[5:]
    print(f"  room {channel} pane {pane[:24]}…")

    tmp = tempfile.mkdtemp(prefix="rv-e1d-")
    modes = {}
    cfg = {"base_url": a.engine, "timeout_s": 10}
    eng = room_voice.EngineClient(lambda: cfg, lambda: voice_tok)
    state = room_voice.RoomVoiceState(os.path.join(tmp, "state.json"))
    memory = room_agent.RoomMemory(os.path.join(tmp, "memory.json"))
    rv = room_voice.RoomVoice(eng, state, memory, lambda d, m: modes.__setitem__(d, m))
    dev, sid = a.device, a.sid

    def bridge_log():
        try:
            with urllib.request.urlopen(a.bridge + "/__log", timeout=5) as r:
                return json.loads(r.read()).get("seen", [])
        except Exception:  # noqa: BLE001
            return []

    def reply(text, kind, re_=None):
        body = {"from": {"paneid": pane}, "text": text, "kind": kind}
        if re_:
            body["re"] = re_
        return http("POST", a.engine + "/api/voice/reply", body, bearer=engine_tok, timeout=120)

    n0 = len(bridge_log())
    print("connect")
    r = rv.connect_turn(dev, "voice fixture", "Eric", sid, 0.9)
    check("connected", r.outcome == "connected", f"{r.outcome} {r.say!r}")
    r = rv.room_turn(dev, "tell boss the button should say help", "Eric", sid, 0.9, "full", "d-u1")
    check("relay sent", r.outcome == "sent", f"{r.outcome} {r.extra.get('vid')}")
    vid = r.extra.get("vid")

    print("ack -> spoken, no expectsReply")
    st, b = reply("Received, three tasks ahead of it.", "ack", [vid])
    check("reply ack 200 spoken", st == 200 and b.get("spoken") is True, f"{st} {json.dumps(b)[:200]}")
    seen = bridge_log()[n0:]
    check("fake bridge played it with start_conversation=0", seen and seen[-1]["q"].get("start_conversation") == "0", seen[-1]["q"] if seen else None)

    print("short result -> spoken, expectsReply, replyId = vid")
    st, b = reply("The button now says help.", "result", [vid])
    check("reply result 200 spoken", st == 200 and b.get("spoken") is True, f"{st} {json.dumps(b)[:200]}")
    seen = bridge_log()[n0:]
    last = seen[-1] if seen else {}
    check("start_conversation=1 and reply_id carried to the bridge", last.get("q", {}).get("start_conversation") == "1" and bool(last.get("q", {}).get("reply_id")), last.get("q"))
    check("busy_wait=8 on a reply", last.get("q", {}).get("busy_wait") == "8", last.get("q"))

    print("long result -> overCap summarize")
    long_text = ("The lookup button is now blue across the desktop and phone layouts, the badge spacing "
                 "was adjusted to eight pixels, the export regained its timestamps, and the settings page "
                 "no longer crashes on Edge; the Windows run is green and the mockup review passed at both "
                 "widths, so I merged it to main and restarted the engine.")
    st, b = reply(long_text, "result", [vid])
    check("engine flags overCap", st == 200 and (b.get("overCap") in (True, "summarize")), f"{st} {json.dumps(b)[:240]}")
    seen = bridge_log()[n0:]
    check("summary was played (a WAV shorter than the full text would be)", seen and seen[-1].get("result") == "delivered", seen[-1] if seen else None)

    print("device busy -> 8 s wait -> chat-only device_busy")
    http("POST", a.bridge + "/__mode", {"busy": 14})
    t0 = time.time()
    st, b = reply("Are you still there?", "question", [vid])
    dt = time.time() - t0
    check("engine reports spoken:false device_busy", st == 200 and b.get("spoken") is False and "busy" in str(b.get("reason", "")), f"{st} {json.dumps(b)[:200]} after {dt:.1f}s")
    check("waited about 8 s, not 14", 7.0 <= dt <= 13.0, f"{dt:.1f}s")
    time.sleep(max(0.0, 14 - dt + 0.5))

    print("muted -> engine does not deliver")
    r = rv.room_turn(dev, "mute", "Eric", sid, 0.9, "full", "d-u2")
    check("mute 200", r.outcome == "mute" and r.extra.get("engineStatus") == 200)
    n1 = len(bridge_log())
    st, b = reply("Muted test line.", "result", [vid])
    check("reply while muted -> spoken:false muted", st == 200 and b.get("spoken") is False and "muted" in str(b.get("reason", "")), f"{st} {json.dumps(b)[:200]}")
    check("nothing reached the bridge", len(bridge_log()) == n1)
    r = rv.room_turn(dev, "unmute", "Eric", sid, 0.9, "full", "d-u3")
    check("unmute 200", r.outcome == "unmute")

    print("disconnect")
    r = rv.room_turn(dev, "disconnect", "Eric", sid, 0.9, "full", "d-u4")
    check("disconnected", r.outcome == "disconnected")
    st, b = reply("After disconnect.", "info")
    check("reply after disconnect -> spoken:false not-connected", st == 200 and b.get("spoken") is False, f"{st} {json.dumps(b)[:160]}")

    print()
    print("FAILURES:" if FAILS else "ALL PASS", FAILS if FAILS else "")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
