"""V1 round trip against the ENGINE's scratch fixture (briefing-table scripts/room-voice-fixture.mjs
--serve), driving room_voice.RoomVoice exactly as server.py does. Nothing plays on the Dot: the
fixture's stub v2v receives the deliveries and exposes them at GET /__log.

  .venv\\Scripts\\python.exe tools\\room_voice_e1_trip.py --engine http://127.0.0.1:8291 \\
        --token-file <path printed by the fixture or copied from its stdout> --stub http://127.0.0.1:8292

The token is read from a file or from the ROOM_VOICE_E1_TOKEN env var; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import room_agent  # noqa: E402
import room_voice  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + str(detail)[:200]) if detail else ''}", flush=True)
    if not cond:
        FAILS.append(name)


def stub_log(stub: str) -> list:
    try:
        with urllib.request.urlopen(stub + "/__log", timeout=5) as r:
            return json.loads(r.read()).get("seen", [])
    except Exception as e:  # noqa: BLE001
        print(f"  (stub /__log unreachable: {e!r})")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--token-file", default="")
    ap.add_argument("--stub", default="")
    ap.add_argument("--device", default="fixture-dot")
    ap.add_argument("--sid", default="ef01f7ee2166")
    a = ap.parse_args()
    tok = os.environ.get("ROOM_VOICE_E1_TOKEN", "")
    if a.token_file:
        with open(a.token_file, encoding="utf-8") as f:
            tok = f.read().strip()
    if not tok:
        print("no token (use --token-file or ROOM_VOICE_E1_TOKEN)")
        sys.exit(2)
    tmp = tempfile.mkdtemp(prefix="rv-e1-")
    modes = {}
    cfg = {"base_url": a.engine, "timeout_s": 10}
    engine = room_voice.EngineClient(lambda: cfg, lambda: tok)
    state = room_voice.RoomVoiceState(os.path.join(tmp, "state.json"))
    memory = room_agent.RoomMemory(os.path.join(tmp, "memory.json"))
    rv = room_voice.RoomVoice(engine, state, memory, lambda d, m: modes.__setitem__(d, m))
    dev, sid = a.device, a.sid
    seen0 = len(stub_log(a.stub)) if a.stub else 0

    print("connections")
    st, body = engine.connections()
    check("GET /api/voice/connections 200", st == 200, f"{st} {json.dumps(body)[:200]}")
    conns = body.get("connections") or []
    check("at least one room listed", bool(conns), [c.get("name") for c in conns])
    if not conns:
        print("FAILURES:", FAILS)
        sys.exit(1)
    room = conns[0]
    name = room.get("name") or ""
    room_id = room["channelId"][5:]
    print(f"  room: {name!r} channel {room['channelId']} lead {room.get('lead')} aliases {room.get('aliases')}")

    print("connect by spoken name")
    r = rv.connect_turn(dev, name, "Eric", sid, 0.9)
    check("connect resolves + opens", r.outcome == "connected", f"{r.outcome} {r.say!r} {r.extra}")
    check("mode -> development", modes.get(dev) == "development")
    check("settings.voice delivered on open", isinstance(state.get(dev).get("settings"), dict), state.get(dev).get("settings"))

    print("room question (answer + /said)")
    r = rv.room_turn(dev, "is anyone working right now", "Eric", sid, 0.9, "full", "e1-u1")
    check("answered", r.outcome == "answered" and bool(r.say), f"{r.outcome} {r.say!r}")

    print("explicit relay to the lead's handle")
    lead = room.get("lead") or ""
    r = rv.room_turn(dev, f"tell {lead.replace('-', ' ')} the button should say help", "Eric", sid, 0.9, "full", "e1-u2")
    check("relay accepted 202", r.outcome == "sent", f"{r.outcome} {r.say!r} {r.extra.get('engine_body')}")
    vid = r.extra.get("vid")
    check("vid returned", bool(vid), vid)

    print("implicit request -> lead")
    r = rv.room_turn(dev, "I'd like to add a hat to the character", "Eric", sid, 0.9, "full", "e1-u3")
    check("lead", r.outcome == "sent" and r.extra.get("to") == lead, f"{r.outcome} to={r.extra.get('to')} resolved={r.extra.get('resolved')}")

    print("fallback name")
    r = rv.room_turn(dev, "tell zorblax the badge is too small", "Eric", sid, 0.9, "full", "e1-u4")
    check("fallback-lead", r.outcome == "sent" and r.extra.get("resolved") == "fallback-lead", f"{r.outcome} resolved={r.extra.get('resolved')} say={r.say!r}")

    print("open-mic non-sequitur with followupTo")
    r = rv.room_turn(dev, "ok thanks", "Eric", sid, 0.9, "full", "e1-u5", is_followup=True, followup_to=vid)
    check("ignored", r.outcome == "ignored")

    print("context reflects the open messages")
    st, ctx = engine.context(room_id)
    check("context 200", st == 200, f"{st} keys={list(ctx)[:12]}")
    check("open list has our vid", any(o.get("vid") == vid for o in (ctx.get("open") or [])), [o.get("vid") for o in (ctx.get("open") or [])])

    print("what's waiting / say again / clear-waiting route")
    r = rv.room_turn(dev, "what's waiting", "Eric", sid, 0.9, "full", "e1-u6")
    check("what's waiting spoken", r.outcome in ("whats_waiting", "whats_waiting_local") and bool(r.say), f"{r.outcome} {r.say!r}")
    st, wb = engine.clear_waiting(room_id)
    check("POST rooms/:id/waiting 200", st == 200, f"{st} {json.dumps(wb)[:120]}")

    print("mute / unmute (pinned verb)")
    r = rv.room_turn(dev, "mute", "Eric", sid, 0.9, "full", "e1-u7")
    check("mute 200", r.outcome == "mute" and r.extra.get("engineStatus") == 200, f"{r.outcome} status={r.extra.get('engineStatus')}")
    r = rv.room_turn(dev, "unmute", "Eric", sid, 0.9, "full", "e1-u8")
    check("unmute 200", r.outcome == "unmute" and r.extra.get("engineStatus") == 200, f"status={r.extra.get('engineStatus')}")

    print("line search")
    st, lb = engine.lines(room_id, "hat", 5)
    check("GET lines 200", st == 200, f"{st} {json.dumps(lb)[:160]}")

    print("disconnect / reconnect remembers / self-heal")
    r = rv.room_turn(dev, "disconnect", "Eric", sid, 0.9, "full", "e1-u9")
    check("disconnected", r.outcome == "disconnected" and modes.get(dev) == "chat", f"{r.outcome} {r.say!r}")
    r = rv.connect_turn(dev, name, "Eric", sid, 0.9)
    check("reconnect remembers", r.outcome == "connected" and "remember" in r.say.lower(), r.say)
    engine.close_channel("room:" + room_id, dev, "test-external-close")
    r = rv.room_turn(dev, "tell the lead hello", "Eric", sid, 0.9, "full", "e1-u10")
    check("409/404 after external close -> back to chat", r.outcome == "channel_gone" and modes.get(dev) == "chat", f"{r.outcome} status={r.extra.get('status')}")

    if a.stub:
        seen = stub_log(a.stub)[seen0:]
        paths = [f"{s.get('method')} {s.get('path')}" for s in seen]
        print("stub v2v received since start:", json.dumps(paths))
        states = [s for s in seen if "/state" in (s.get("path") or "")]
        check("engine pushed device state to (stub) v2v", bool(states), [s.get("body") for s in states][:3])

    print()
    print("FAILURES:" if FAILS else "ALL PASS", FAILS if FAILS else "")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
