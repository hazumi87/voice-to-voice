"""Offline tests for gpu_cop_client against a fake cop + fake Ollama (no GPU, no harbor).

    python tools/gpu_cop_client_test.py
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FAKE = {"state": None, "ps": {"models": []}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = FAKE["ps"] if self.path.startswith("/api/ps") else FAKE["state"]
        if body is None:
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
os.environ["GPU_COP_URL"] = f"http://127.0.0.1:{port}/gpu/state"
os.environ["OLLAMA_PS_URL"] = f"http://127.0.0.1:{port}/api/ps"
import gpu_cop_client as g  # noqa: E402

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def state(free, reserved, reservations=(), services=(), unmanaged=(), enabled=True, ok=True):
    return {"schema": "gpu-cop/1", "enabled": enabled,
            "gpu": {"ok": ok, "free_mib": free, "reserved_mib": reserved, "total_mib": 16376},
            "reservations": list(reservations), "services": list(services),
            "unmanaged": list(unmanaged)}


def fresh_ps(models):
    FAKE["ps"] = {"models": models}
    g._ps_cache["at"] = 0.0


# cop down -> allowed (today's behaviour)
FAKE["state"] = None
check("cop down: tts allowed", g.tts_gate(10752) is None)
fresh_ps([])
try:
    g.router_gate("llama3.2:3b")
    check("cop down: router allowed", True)
except g.RouterGpuWait:
    check("cop down: router allowed", False)

# master off / gpu unreadable -> allowed
FAKE["state"] = state(1000, 0, enabled=False)
check("master off: allowed", g.tts_gate(10752) is None)
FAKE["state"] = state(1000, 0, ok=False)
check("gpu not ok: allowed", g.tts_gate(10752) is None)

# v2v restarting, own enforced reservation: TTS ignores its own, router respects it
own = [{"service": "voice-to-voice", "floor_mib": 10752, "enforced": True}]
svc = [{"name": "voice-to-voice", "held_mib": 0}]
FAKE["state"] = state(12000, 10752, own, svc)
check("own reservation: tts proceeds", g.tts_gate(10752) is None)
fresh_ps([])
try:
    g.router_gate("llama3.2:3b")
    check("own reservation: router blocked", False)
except g.RouterGpuWait as e:
    check("own reservation: router blocked", "voice-to-voice is restarting" in str(e))
FAKE["state"] = state(14900, 10752, own, svc)  # 14900-10752 = 4148 >= router floor 3000
try:
    g.router_gate("llama3.2:3b")
    check("own reservation, room to spare: router allowed", True)
except g.RouterGpuWait:
    check("own reservation, room to spare: router allowed", False)

# router already resident -> never gated
fresh_ps([{"name": "llama3.2:3b", "size_vram": 2550 * 1024 * 1024}])
try:
    g.router_gate("llama3.2:3b")
    check("resident router: no gate", True)
except g.RouterGpuWait:
    check("resident router: no gate", False)
check("router_resident_mib", g.router_resident_mib(["llama3.2:3b"]) == 2550)

# Krea holding the card: tts blocked, reason names Krea
FAKE["state"] = state(6100, 0, [], [{"name": "asset-platform-krea", "held_mib": 3311}] + svc)
w = g.tts_gate(10752)
check("krea holds: tts blocked with name", w is not None and "asset-platform-krea" in w)

# plenty of room, someone else reserving -> subtract theirs
other = [{"service": "asset-platform-krea", "floor_mib": 3500, "enforced": True}]
FAKE["state"] = state(14500, 3500, other, svc)
check("other reservation fits: tts proceeds", g.tts_gate(10752) is None)
FAKE["state"] = state(13000, 3500, other, svc)
w = g.tts_gate(10752)
check("other reservation too big: blocked, names restart",
      w is not None and "asset-platform-krea is restarting" in w)

# observe-mode reservation (enforced false) is not in reserved_mib and not subtracted
FAKE["state"] = state(14900, 0, [dict(own[0], enforced=False)], svc)
check("observe reservation: tts proceeds", g.tts_gate(10752) is None)

srv.shutdown()
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
