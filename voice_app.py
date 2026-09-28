"""voice_app.py -- V2 room voice panel (vk-1826).

A small read-only web panel the briefing-table engine embeds in a workroom,
inside a sandboxed iframe, loaded THROUGH the engine's panel proxy (GET/HEAD
only; the proxy strips upstream CSP, so this module sends its own -- see
`_VIEW_CSP`). v2v serves it itself; nothing here talks to the engine, and
nothing here writes to room state -- Mute/Disconnect only postMessage to the
host page, which does the actual POST.

Mounted by server.py with exactly:

    import voice_app
    voice_app.mount(app, lambda: _room_state, _voice_devices)

`state_getter()` returns the live `room_voice.RoomVoiceState` instance
(`.get(device) -> dict`, `.room_id(device) -> str | None`).
`devices_getter()` returns the live device table (`{device_id: {...}}`).
Both are called fresh on every request -- this module holds no cached state
of its own besides the static asset bytes.

No auth (tailnet-only host, per the guide's contract for this shape of app).
Every response gets `Access-Control-Allow-Origin: *`: the frame's origin is
"null" (sandboxed, no allow-same-origin), so even same-server fetch()/@font-
face requests are cross-origin from the browser's point of view and need
this to not be silently blocked.

Reference: KB doc 08968e7e59226fd2 ("Guide -- building a room app for
briefing-table workrooms"), plus the phase's own binding visual contract
(blueprint sec9, carried in the vk-1826 task body) which is more specific
than the guide for this app's particular postMessage vocabulary and status-
ownership rules -- that contract wins where the two differ.
"""
from __future__ import annotations

import html
import json
import mimetypes
import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "voice_app_assets")

ROOM_PREFIX = "room:"
EXCHANGES_KEEP = 3
FIXTURE_ROOM_ID = "fixture"

# Sent on every view response. The proxy's own CSP is the enforced boundary
# (panel-proxy.ts) but every shipped room app also sends its own, per the
# guide sec5 "defense in depth" -- this is that second layer, plus it is
# what a direct/manual open (not through the proxy) gets.
_VIEW_CSP = (
    "default-src 'none'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; "
    "script-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}


def _cors(resp: Response) -> Response:
    """Every response from this module is CORS '*' -- see module docstring."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def _find_device(devices: dict, state, room_id: str):
    """Scan the device table for the one whose room-voice state row's
    channelId is this room. Returns (device_id, row) or (None, None)."""
    want = ROOM_PREFIX + room_id
    for device_id in devices.keys():
        try:
            row = state.get(device_id)
        except Exception:
            continue
        if (row or {}).get("channelId") == want:
            return device_id, row
    return None, None


def _real_state_payload(devices: dict, state, room_id: str) -> dict:
    device_id, row = _find_device(devices, state, room_id)
    if not device_id:
        return {
            "connected": False,
            "device": None,
            "name": None,
            "lead": None,
            "since": None,
            "muted": False,
            "waiting": False,
            "speaking": False,
            "exchanges": [],
            "serverTime": time.time(),
        }
    row = row or {}
    exchanges = list(row.get("exchanges") or [])[-EXCHANGES_KEEP:]
    speaking_until = row.get("speakingUntil")
    speaking = speaking_until is not None and time.time() < speaking_until
    return {
        "connected": True,
        "device": device_id,
        "name": row.get("name"),
        "lead": row.get("lead"),
        "since": row.get("since"),
        "muted": bool(row.get("muted")),
        "waiting": bool(row.get("waiting")),
        "speaking": speaking,
        "exchanges": exchanges,
        "serverTime": time.time(),
    }


# ---------------------------------------------------------------------------
# Fixture: deterministic sample data, reachable without the engine or a real
# device. /rooms/fixture/view?state=connected|disconnected&muted=0|1&
# waiting=0|1&exchanges=0|1|3&long=1&fallback=0|1&speaking=0|1
# ---------------------------------------------------------------------------
_FIXTURE_IN = [
    "What's the status on the north wing survey.",
    "Can someone read back the last decision on the beam spec.",
    "Any update on the delivery.",
]
_FIXTURE_IN_LONG = [
    "What's the status on the north wing survey, and did the two flagged "
    "sections from last week ever get resurveyed or are we still working "
    "off the old numbers.",
    "Can someone read back the last decision on the beam spec, I want to "
    "make sure I'm quoting the right revision to the client this afternoon.",
    "Is there any update from the fabrication team about the delay on the "
    "steel delivery, because the schedule board hasn't moved in two days "
    "and I want to know before I tell the client anything.",
]
_FIXTURE_OUT = [
    "The survey is 80 percent done, results by Friday.",
    "Last decision: go with the reinforced beam spec, seat kade signed off.",
    "Delivery is delayed a week.",
]
_FIXTURE_OUT_LONG = [
    "The survey is about 80 percent done. The two flagged sections were "
    "redone Tuesday and came back clean, so Friday's number should be final.",
    "Last decision, from the design review: go with the reinforced beam "
    "spec, revision C. Seat kade signed off on it Monday.",
    "Fabrication says the steel is delayed a week due to a supplier issue; "
    "kade is drafting a note to the client this afternoon.",
]


def _bool_q(qp, key: str, default: bool = False) -> bool:
    v = qp.get(key)
    if v is None:
        return default
    return str(v).strip() == "1"


def _fixture_state_payload(qp) -> dict:
    state = (qp.get("state") or "connected").strip().lower()
    connected = state != "disconnected"
    muted = _bool_q(qp, "muted", False)
    waiting = _bool_q(qp, "waiting", False)
    long_ = _bool_q(qp, "long", False)
    fallback = _bool_q(qp, "fallback", False)
    speaking = _bool_q(qp, "speaking", False)
    try:
        n = int(qp.get("exchanges", "3"))
    except (TypeError, ValueError):
        n = 3
    n = max(0, min(3, n))

    now = time.time()
    ins = _FIXTURE_IN_LONG if long_ else _FIXTURE_IN
    outs = _FIXTURE_OUT_LONG if long_ else _FIXTURE_OUT
    exchanges = []
    if connected:
        for i in range(n):
            t = now - (3 - i) * 40
            in_item = {"t": t, "kind": "in", "who": "user",
                       "text": ins[i], "lineId": "fx-in-%d" % i}
            if fallback and i == n - 1:
                # Aurora's binding (room-voice-v1): the fallback-lead note
                # rides an "in" exchange, plain --dim2 text, never amber/
                # coral. The most recent question is the natural one to mark.
                in_item["resolved"] = "fallback-lead"
            exchanges.append(in_item)
            exchanges.append({"t": t + 5, "kind": "out", "who": "voice",
                               "text": outs[i], "vid": "V%d" % (100 + i),
                               "lineId": "fx-out-%d" % i})
        exchanges = exchanges[-EXCHANGES_KEEP * 2:]

    return {
        "connected": connected,
        "device": "echo-dot-biscuit" if connected else None,
        "name": "North Wing Standup" if connected else None,
        "lead": "kade" if connected else None,
        "since": (now - 20 * 60) if connected else None,
        "muted": muted,
        "waiting": waiting,
        "speaking": speaking,
        "exchanges": exchanges,
        "serverTime": now,
    }


# ---------------------------------------------------------------------------
# View page
# ---------------------------------------------------------------------------
_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice</title>
<link rel="stylesheet" href="../../assets/tokens.css">
<link rel="stylesheet" href="../../assets/app.css">
</head>
<body>
<div class="fr" id="frame">
  <div class="fr-top">
    <span class="dev" id="dev"></span>
    <span class="spk" id="speakingRow"><i></i><i></i><i></i>Speaking</span>
    <button id="muteBtn" class="mute" type="button" aria-pressed="false"></button>
    <button id="disconnectBtn" class="disc" type="button">Disconnect</button>
  </div>
  <ul class="exs" id="exchanges"></ul>
</div>
<noscript>Enable scripts to use the voice panel.</noscript>
<script id="initial-state" type="application/json">%(seed)s</script>
<script src="../../assets/app.js"></script>
</body>
</html>
"""


def _render_view(payload: dict) -> str:
    # json.dumps output embedded in a <script type="application/json"> is
    # not HTML-parsed as markup, but escape "</script" defensively anyway --
    # none of these values are attacker-authored today, but exchange text
    # (transcribed speech) will be free text in the real (non-fixture) path.
    seed = json.dumps(payload).replace("</script", "<\\/script")
    return _PAGE_TEMPLATE % {"seed": seed}


def _head_safe(request: Request, resp: Response) -> Response:
    """The engine's panel proxy is GET/HEAD only (panel-proxy.ts). FastAPI/
    Starlette does not auto-derive a HEAD response from a GET route, so
    every route here is registered for both methods and this strips the
    body (keeping headers) when the request was a HEAD."""
    if request.method == "HEAD":
        resp.body = b""
        resp.headers["content-length"] = "0"
    return resp


def mount(app, state_getter, devices_getter) -> None:
    router = APIRouter()

    @router.api_route("/apps/voice/rooms/{room_id}/view", methods=["GET", "HEAD"])
    def view(room_id: str, request: Request):
        if room_id == FIXTURE_ROOM_ID:
            payload = _fixture_state_payload(request.query_params)
        else:
            payload = _real_state_payload(devices_getter(), state_getter(), room_id)
        resp = Response(content=_render_view(payload), media_type="text/html; charset=utf-8")
        resp.headers["Content-Security-Policy"] = _VIEW_CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return _head_safe(request, _cors(resp))

    @router.api_route("/apps/voice/rooms/{room_id}/state", methods=["GET", "HEAD"])
    def state(room_id: str, request: Request):
        if room_id == FIXTURE_ROOM_ID:
            payload = _fixture_state_payload(request.query_params)
        else:
            payload = _real_state_payload(devices_getter(), state_getter(), room_id)
        return _head_safe(request, _cors(JSONResponse(payload)))

    @router.api_route("/apps/voice/rooms/{room_id}/tool.json", methods=["GET", "HEAD"])
    def tool_json(room_id: str, request: Request):
        base = str(request.base_url).rstrip("/")
        payload = {
            "contract": "tool/draft-0",
            "name": "Voice Room Panel",
            "slug": "voice-room-panel",
            "version": "0.1.0",
            "description": (
                "Read-only panel: the Echo Dot's connection to this room, "
                "the last 3 voice exchanges, and Mute/Disconnect controls."
            ),
            "authority": "own",
            "operator": "none",
            "access": "shared",
            "needs": {"host": "any", "gpu": False, "runtime": "embedded-in-v2v"},
            "panel": {
                "title": "Voice",
                "url": "%s/apps/voice/rooms/%s/view" % (base, room_id),
                "kind": "app",
                "icon": "mic",
                "scripts": True,
                "uploads": False,
            },
            "note": (
                "Served directly by the always-on voice-to-voice FastAPI "
                "process (server.py via voice_app.py), not a standalone "
                "harbor-managed tool folder -- no lifecycle scripts (entry/"
                "pulse/stop) apply here. Registering this panel with the "
                "briefing-table engine (POST /api/rooms/:id/panels, plus "
                "adding vrpc-3.tail567253.ts.net to the engine's panelHosts "
                "allow-list) is the engine-side step this phase leaves open."
            ),
        }
        return _head_safe(request, _cors(JSONResponse(payload)))

    @router.api_route("/apps/voice/assets/{asset_path:path}", methods=["GET", "HEAD"])
    def assets(asset_path: str, request: Request):
        # No dotdot, no absolute escape -- resolve under ASSETS_DIR and verify.
        norm = os.path.normpath(asset_path).replace("\\", "/")
        if norm.startswith("..") or norm.startswith("/"):
            return _cors(PlainTextResponse("not found", status_code=404))
        full = os.path.join(ASSETS_DIR, norm)
        full = os.path.abspath(full)
        if not full.startswith(os.path.abspath(ASSETS_DIR) + os.sep):
            return _cors(PlainTextResponse("not found", status_code=404))
        if not os.path.isfile(full):
            return _cors(PlainTextResponse("not found", status_code=404))
        ext = os.path.splitext(full)[1].lower()
        ctype = _CONTENT_TYPES.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        resp = Response(content=body, media_type=ctype)
        return _head_safe(request, _cors(resp))

    app.include_router(router)
