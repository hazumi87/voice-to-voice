"""Room-voice V1 round trip against the fake engine, in-process (no GPU, no Dot).

Drives room_voice.RoomVoice exactly as server.py does, with tools/fake_engine.py as the
engine, and asserts the §4-§7 behaviours: connect (resolve + open + mode flip), a room
question (answer + /said), an explicit relay (inbound route + resolved handle), an
implicit request (lead), a forbidden request (lead), a fallback name, the open-mic
non-sequitur (ignored, /said), what's waiting, say again, mute/unmute, hop (one
connection per device), disconnect, reconnect-remembers, and stale-mode self-heal.

  .venv\\Scripts\\python.exe tools\\room_voice_roundtrip.py [--model llama3.2:3b]

The routing steps call Ollama for real (the model must be reachable); everything else is
deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import room_agent  # noqa: E402
import room_voice  # noqa: E402

PORT = 8299
BEARER = "rt-token"
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + str(detail)) if detail else ''}", flush=True)
    if not cond:
        FAILS.append(name)


def calls():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_calls", timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=room_agent.ROUTER_MODEL)
    a = ap.parse_args()
    room_agent.ROUTER_MODEL = a.model

    eng = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_engine.py"), "--port", str(PORT),
                            "--bearer", BEARER], stdout=subprocess.DEVNULL)
    time.sleep(0.8)
    tmp = tempfile.mkdtemp(prefix="rv-")
    modes = {}
    try:
        cfg = {"base_url": f"http://127.0.0.1:{PORT}", "timeout_s": 8}
        engine = room_voice.EngineClient(lambda: cfg, lambda: BEARER)
        state = room_voice.RoomVoiceState(os.path.join(tmp, "state.json"))
        memory = room_agent.RoomMemory(os.path.join(tmp, "memory.json"))
        rv = room_voice.RoomVoice(engine, state, memory, lambda d, m: modes.__setitem__(d, m))
        dev, sid = "echo-dot-biscuit", "sim-eric"

        print("connect")
        r = rv.connect_turn(dev, "briefing table", "Eric", sid, 0.3)
        check("guest cannot connect", r.outcome == "refused_speaker", r.say)
        r = rv.connect_turn(dev, "nowhere land", "Eric", sid, 0.9)
        check("unknown room -> no_match", r.outcome == "no_match", r.say)
        r = rv.connect_turn(dev, "voice interaction", "Eric", sid, 0.9)
        check("connect resolves + opens", r.outcome == "connected" and r.extra.get("channelId") == "room:r1", r.say)
        check("device flipped to development", modes.get(dev) == "development")
        check("state holds the room", state.room_id(dev) == "r1" and state.get(dev).get("lead") == "briefing-table")

        print("room question")
        r = rv.room_turn(dev, "is anyone working right now", "Eric", sid, 0.9, "full", "u1")
        check("answer action", r.outcome == "answered" and bool(r.say), f"{r.say!r}")
        check("answer reopens mic", r.expects_reply is True)
        said = [c for c in calls()["calls"] if c["path"].endswith("/said")]
        check("/said recorded the answer", said and said[-1]["body"]["action"] == "answer")

        print("explicit relay")
        r = rv.room_turn(dev, "tell aurora the button should say help", "Eric", sid, 0.9, "full", "u2")
        check("relay sent", r.outcome == "sent", f"{r.say!r} to={r.extra.get('to')}")
        check("engine resolved aurora -> aurora-design", r.extra.get("to") == "aurora-design" and r.extra.get("resolved") == "alias", r.extra.get("resolved"))
        inb = [c for c in calls()["calls"] if c["path"] == "/api/voice/inbound"][-1]["body"]
        check("inbound carries route.to + confidence", inb["route"].get("to") and "confidence" in inb["route"], inb["route"])
        check("inbound body is the raw transcript", inb["text"] == "tell aurora the button should say help")

        print("implicit request -> lead")
        r = rv.room_turn(dev, "I'd like to add a hat to the character", "Eric", sid, 0.9, "full", "u3")
        check("lead", r.outcome == "sent" and r.extra.get("to") == "briefing-table", f"{r.say!r}")
        check("spoken line names the lead", "lead" in r.say.lower())

        print("forbidden -> lead")
        r = rv.room_turn(dev, "add the blender tool to the room", "Eric", sid, 0.9, "full", "u4")
        check("forbidden goes to the lead", r.outcome == "sent" and r.extra.get("to") == "briefing-table", f"{r.say!r}")

        print("fallback name")
        r = rv.room_turn(dev, "tell zorblax the badge is too small", "Eric", sid, 0.9, "full", "u5")
        check("fallback-lead resolved", r.outcome == "sent" and r.extra.get("resolved") == "fallback-lead", f"{r.say!r}")
        check("spoken line explains the fallback", "lead" in r.say.lower())

        print("open-mic non-sequitur")
        r = rv.room_turn(dev, "ok thanks", "Eric", sid, 0.9, "full", "u6", is_followup=True, followup_to="fake-1")
        check("ignored, nothing said", r.outcome == "ignored" and r.say == "")
        said = [c for c in calls()["calls"] if c["path"].endswith("/said")][-1]["body"]
        check("/said logged the ignore with followupTo", said["action"] == "ignore" and said.get("followupTo") == "fake-1")
        r = rv.room_turn(dev, "ok", "Eric", sid, 0.9, "full", "u7", is_followup=False)
        check("'ok' outside the window is NOT ignored", r.outcome != "ignored", r.outcome)

        print("what's waiting / say again")
        rv.note_spoken(dev, "briefing-table", "Message received, working on the hat.", "fake-9", vid="v2be")
        r = rv.room_turn(dev, "say that again", "Eric", sid, 0.9, "full", "u8")
        check("say again replays lastSpoken", r.outcome == "say_again" and "working on the hat" in r.say, r.say)
        r = rv.room_turn(dev, "what's waiting", "Eric", sid, 0.9, "full", "u9")
        check("what's waiting answers from context", r.outcome == "whats_waiting", r.say)

        print("mute / unmute")
        r = rv.room_turn(dev, "mute", "Eric", sid, 0.9, "full", "u10")
        check("muted", r.outcome == "mute" and state.get(dev).get("muted") is True and r.extra.get("engineStatus") == 200, r.say)
        r = rv.room_turn(dev, "unmute", "Eric", sid, 0.9, "full", "u11")
        check("unmuted", r.outcome == "unmute" and state.get(dev).get("muted") is False)

        print("hop -> one connection per device")
        r = rv.room_turn(dev, "connect to workrooms", "Eric", sid, 0.9, "full", "u12")
        check("hopped to r2", r.outcome == "connected" and state.room_id(dev) == "r2", r.say)
        ch = calls()["channels"]
        check("r1 closed on hop", "room:r1" not in ch and "room:r2" in ch, list(ch))

        print("disconnect / reconnect remembers")
        r = rv.room_turn(dev, "disconnect", "Eric", sid, 0.9, "full", "u13")
        check("disconnected", r.outcome == "disconnected" and modes.get(dev) == "chat" and not state.get(dev))
        r = rv.connect_turn(dev, "voice interaction", "Eric", sid, 0.9)
        check("reconnect remembers r1", r.outcome == "connected" and "remember" in r.say.lower(), r.say)

        print("stale mode self-heal")
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/_reset", data=b"{}", method="POST"), timeout=5)
        r = rv.room_turn(dev, "tell aurora hello", "Eric", sid, 0.9, "full", "u14")
        check("409 -> back to chat", r.outcome == "channel_gone" and modes.get(dev) == "chat" and not state.get(dev), r.say)
    finally:
        eng.terminate()
    print()
    print("FAILURES:" if FAILS else "ALL PASS", FAILS if FAILS else "")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
