"""Room voice — orchestration tier for the Dot in a workroom (room-voice §3-§7, vk-1825).

server.py owns HTTP, STT, speaker ID and synthesis. This module owns:
  - the engine client (v2v -> engine bearer)            EngineClient
  - the per-device room connection state, persisted     RoomVoiceState
  - the turn: intents -> non-sequitur -> room agent      room_turn(), connect_turn()
  - resolve-in-v2v of a spoken room name                 resolve_connection()

Everything here returns a `TurnResult` (what to SAY + what happened); it never
synthesizes and never touches FastAPI, so it is testable against tools/fake_engine.py.

Engine verbs used (docs/voice-channel.md + room-voice.md; E1 owns the engine side):
  GET  /api/voice/connections                       -> {provider, connections:[...]}
  POST /api/voice/channels/open  {channelId, device} -> 200 {channelId, name, lead, settings.voice}
  POST /api/voice/channels/close {channelId, device, reason}
  POST /api/voice/inbound        {channelId, text, ..., route:{to, confidence, note?}, followupTo?}
  POST /api/voice/rooms/:id/said {utteranceId, text, device, sid, action, answer?, confidence, followupTo?}
  GET  /api/voice/rooms/:id/context
  GET  /api/voice/rooms/:id/lines?q=&limit=
  POST /api/voice/rooms/:id/mute {muted}              -> 200 {voice}  (agent mute/unmute, §15)
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import room_agent

CONNECT_MIN_CONF = 0.50          # §4: a guest cannot connect the Dot
ROOM_PREFIX = "room:"


# ---------------------------------------------------------------------------------
# Engine client
# ---------------------------------------------------------------------------------
class EngineError(Exception):
    def __init__(self, status, body):
        super().__init__(f"engine {status}: {body}")
        self.status = status
        self.body = body or {}


class EngineClient:
    """Thin bearer client. cfg_getter() returns the voice_engine dict (hot-reloaded);
    token_getter() returns the bearer v2v presents."""

    def __init__(self, cfg_getter, token_getter):
        self._cfg = cfg_getter
        self._tok = token_getter

    def base(self) -> str:
        cfg = self._cfg() or {}
        b = cfg.get("base_url")
        if b:
            return b.rstrip("/")
        inbound = cfg.get("inbound_url") or ""
        return inbound.split("/api/voice/", 1)[0].rstrip("/")

    def timeout(self) -> float:
        try:
            return float((self._cfg() or {}).get("timeout_s", 8))
        except (TypeError, ValueError):
            return 8.0

    def _call(self, method: str, path: str, body: dict | None = None, params: dict | None = None):
        base = self.base()
        if not base:
            raise EngineError(None, {"error": "engine_unconfigured"})
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Content-Type": "application/json"}
        tok = self._tok()
        if tok:
            headers["Authorization"] = "Bearer " + tok
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout()) as r:
                out = json.loads(r.read() or b"{}")
                return r.status, (out if isinstance(out, dict) else {"result": out})
        except urllib.error.HTTPError as e:
            try:
                out = json.loads(e.read() or b"{}")
            except Exception:  # noqa: BLE001
                out = {}
            return e.code, (out if isinstance(out, dict) else {})
        except Exception as e:  # noqa: BLE001 — refused, DNS, timeout
            return None, {"error": "unreachable", "detail": repr(e)}

    def connections(self):
        return self._call("GET", "/api/voice/connections")

    def open_channel(self, channel_id: str, device: str):
        return self._call("POST", "/api/voice/channels/open", {"channelId": channel_id, "device": device})

    def close_channel(self, channel_id: str, device: str, reason: str):
        return self._call("POST", "/api/voice/channels/close",
                          {"channelId": channel_id, "device": device, "reason": reason})

    def inbound(self, body: dict):
        return self._call("POST", "/api/voice/inbound", body)

    def said(self, room_id: str, body: dict):
        return self._call("POST", f"/api/voice/rooms/{room_id}/said", body)

    def context(self, room_id: str):
        return self._call("GET", f"/api/voice/rooms/{room_id}/context")

    def lines(self, room_id: str, q: str, limit: int = 5):
        return self._call("GET", f"/api/voice/rooms/{room_id}/lines", params={"q": q, "limit": limit})

    def clear_waiting(self, room_id: str):
        # E1c: POST /api/voice/rooms/:id/waiting {waiting:false} -> 200 {voice}. Called after
        # "what's waiting" / "say that again" have been answered (followupTo clears it too).
        return self._call("POST", f"/api/voice/rooms/{room_id}/waiting", {"waiting": False})

    def set_muted(self, channel_id: str, device: str, muted: bool):
        # Pinned by the lead (room-voice §15): POST /api/voice/rooms/:id/mute {muted} -> {voice}.
        # voice_engine.mute_path may override with a "{room}" placeholder.
        room_id = channel_id[len(ROOM_PREFIX):] if channel_id.startswith(ROOM_PREFIX) else channel_id
        path = (self._cfg() or {}).get("mute_path") or "/api/voice/rooms/{room}/mute"
        return self._call("POST", path.replace("{room}", room_id),
                          {"muted": bool(muted), "channelId": channel_id, "device": device})


# ---------------------------------------------------------------------------------
# Per-device room connection state (what the engine told us + what we did), persisted
# so a v2v restart keeps "this Dot is in room X, muted, waiting".
# ---------------------------------------------------------------------------------
class RoomVoiceState:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._d: dict = {}
        try:
            with open(path, encoding="utf-8") as f:
                self._d = json.load(f) or {}
        except (OSError, ValueError):
            self._d = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._d, f, indent=1)
        os.replace(tmp, self.path)

    def get(self, device: str) -> dict:
        with self._lock:
            return dict(self._d.get(device) or {})

    def update(self, device: str, **fields) -> dict:
        with self._lock:
            row = self._d.setdefault(device, {})
            row.update(fields)
            row["updatedAt"] = time.time()
            self._save()
            return dict(row)

    def clear(self, device: str) -> None:
        with self._lock:
            self._d.pop(device, None)
            self._save()

    def room_id(self, device: str) -> str | None:
        ch = self.get(device).get("channelId") or ""
        return ch[len(ROOM_PREFIX):] if ch.startswith(ROOM_PREFIX) else None


# ---------------------------------------------------------------------------------
# Spoken lines (deterministic control flow; server may restyle them in-character later)
# ---------------------------------------------------------------------------------
LINES = {
    "connected": "Connected to {name}.",
    "reconnected": "Connected to {name}. I remember where we left off.",
    "disconnected": "Disconnected from {name}.",
    "not_connected": "You're not connected to a room.",
    "no_rooms": "I can't find any rooms to connect to.",
    "no_match": "I can't find a room called {target}.",
    "ambiguous": "Did you mean {options}?",
    "engine_down": "The table isn't answering.",
    "refused_speaker": "I'm not sure that's you.",
    "channel_gone": "The room connection is gone. Back to chat.",
    "relayed_lead": "Relayed to the lead.",
    "relayed": "Relayed to {name}.",
    "relayed_fallback": "I couldn't find {heard}, so I passed it to the lead.",
    "relay_failed": "I couldn't reach the room. Try again.",
    "muted": "Muted. I'll keep listening and write replies to the chat.",
    "unmuted": "Unmuted.",
    "nothing_waiting": "Nothing is waiting for you.",
    "waiting_one": "One message is waiting: {body}",
    "waiting_many": "{n} messages are waiting. The latest, from {author}: {body}",
    "say_again": "{author} said: {body}",
    "nothing_to_repeat": "Nothing has been said yet.",
    "open_messages": "{n} of your messages are still open, the oldest {ago}.",
}


def _fmt(key: str, **kw) -> str:
    return LINES[key].format(**kw)


def _spoken_handle(handle: str, lead: str | None = None) -> str:
    if not handle:
        return "the room"
    if lead and handle == lead:
        return "the lead"
    return handle.replace("-", " ").replace("_", " ")


def _ago(s) -> str:
    try:
        s = int(s)
    except (TypeError, ValueError):
        return "a while ago"
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{s // 60} minutes ago"
    h = s // 3600
    return "an hour ago" if h == 1 else f"{h} hours ago"


# ---------------------------------------------------------------------------------
# Resolve a spoken room name (§3: resolve in v2v, open in the engine)
# ---------------------------------------------------------------------------------
_STOP = {"the", "a", "room", "workroom", "workrooms", "rooms", "channel", "to", "please"}


def _tokens(s: str) -> list:
    raw = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split()
    kept = [t for t in raw if t not in _STOP]
    return kept or raw        # "workrooms" alone must still match a room called Workrooms


def _score(target: str, candidate: str) -> float:
    tt, ct = _tokens(target), _tokens(candidate)
    if not tt or not ct:
        return 0.0
    ratio = difflib.SequenceMatcher(None, " ".join(tt), " ".join(ct)).ratio()
    overlap = len(set(tt) & set(ct)) / max(1, len(set(tt) | set(ct)))
    # token-level fuzz for STT slips ("breifing" ~ "briefing")
    fuzzy = 0.0
    for a in tt:
        best = max((difflib.SequenceMatcher(None, a, b).ratio() for b in ct), default=0.0)
        fuzzy += best
    fuzzy /= len(tt)
    return max(ratio, 0.5 * overlap + 0.5 * fuzzy)


def resolve_connection(target: str, connections: list) -> dict:
    """Returns {"match": conn} | {"ambiguous": [conns]} | {"none": True}."""
    scored = []
    for c in connections or []:
        names = [c.get("name") or ""] + list(c.get("aliases") or [])
        s = max((_score(target, n) for n in names if n), default=0.0)
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < 0.55:
        return {"none": True}
    top = scored[0][0]
    close = [c for s, c in scored if s >= max(0.55, top - 0.12)]
    if len(close) == 1:
        return {"match": close[0]}
    return {"ambiguous": close[:3]}


# ---------------------------------------------------------------------------------
# Turn results
# ---------------------------------------------------------------------------------
class TurnResult:
    def __init__(self, say: str, outcome: str, expects_reply: bool = False, **extra):
        self.say = say
        self.outcome = outcome
        self.expects_reply = expects_reply
        self.extra = extra

    def as_dict(self) -> dict:
        return {"say": self.say, "outcome": self.outcome, "expectsReply": self.expects_reply,
                **self.extra}


# ---------------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------------
class RoomVoice:
    def __init__(self, engine: EngineClient, state: RoomVoiceState, memory: room_agent.RoomMemory,
                 set_device_mode, log=print):
        self.engine = engine
        self.state = state
        self.memory = memory
        self.set_device_mode = set_device_mode      # (device, "chat"|"development") -> None
        self.log = log

    # -- connect -----------------------------------------------------------------
    def connect_turn(self, device: str, target: str, speaker: str, sid: str, conf: float) -> TurnResult:
        if not sid or conf < CONNECT_MIN_CONF:
            return TurnResult(_fmt("refused_speaker"), "refused_speaker")
        status, body = self.engine.connections()
        if status != 200:
            return TurnResult(_fmt("engine_down"), "engine_down", status=status)
        conns = body.get("connections") or []
        if not conns:
            return TurnResult(_fmt("no_rooms"), "no_rooms")
        res = resolve_connection(target, conns)
        if res.get("none"):
            return TurnResult(_fmt("no_match", target=target), "no_match")
        if res.get("ambiguous"):
            names = [c.get("name") for c in res["ambiguous"]]
            opts = " or ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + ", or " + names[-1]
            return TurnResult(_fmt("ambiguous", options=opts), "ambiguous", options=names)
        conn = res["match"]
        return self._open(device, conn["channelId"], conn.get("name") or "the room", sid)

    def _open(self, device: str, channel_id: str, name: str, sid: str) -> TurnResult:
        prev = self.state.get(device)
        if prev.get("channelId") and prev["channelId"] != channel_id:
            # One connection per device (§3): hopping closes the old one first.
            self.engine.close_channel(prev["channelId"], device, "hop")
        status, body = self.engine.open_channel(channel_id, device)
        if status != 200:
            return TurnResult(_fmt("engine_down"), "open_failed", status=status, body=body)
        settings = body.get("settings", {}).get("voice") if isinstance(body.get("settings"), dict) else None
        self.state.update(device, channelId=body.get("channelId") or channel_id,
                          name=body.get("name") or name, lead=body.get("lead"),
                          settings=settings or body.get("settings") or {},
                          muted=False, waiting=False, since=time.time(),
                          lastSpoken=None, lastReplyId=None, speaker=sid)
        self.set_device_mode(device, "development")
        room_id = (body.get("channelId") or channel_id)[len(ROOM_PREFIX):]
        remembered = bool(self.memory.get(room_id, sid).get("turns"))
        line = _fmt("reconnected" if remembered else "connected", name=body.get("name") or name)
        return TurnResult(line, "connected", channelId=body.get("channelId") or channel_id,
                          name=body.get("name") or name)

    # -- disconnect --------------------------------------------------------------
    def disconnect(self, device: str, reason: str = "spoken", speak: bool = True) -> TurnResult:
        st = self.state.get(device)
        ch = st.get("channelId")
        if not ch:
            return TurnResult(_fmt("not_connected") if speak else "", "not_connected")
        self.engine.close_channel(ch, device, reason)
        self.state.clear(device)
        self.set_device_mode(device, "chat")
        return TurnResult(_fmt("disconnected", name=st.get("name") or "the room") if speak else "",
                          "disconnected", channelId=ch)

    def self_heal(self, device: str, status) -> TurnResult:
        """409 no_channel / 404 channel_gone from inbound: the engine no longer has us."""
        self.state.clear(device)
        self.set_device_mode(device, "chat")
        return TurnResult(_fmt("channel_gone"), "channel_gone", status=status)

    # -- engine-pushed state (§9.3) ------------------------------------------------
    def apply_engine_state(self, device: str, payload: dict) -> dict:
        fields = {}
        for k in ("channelId", "name", "lead", "muted", "waiting"):
            if k in payload:
                fields[k] = payload[k]
        if "settings" in payload:
            s = payload["settings"]
            fields["settings"] = s.get("voice") if isinstance(s, dict) and "voice" in s else s
        if payload.get("connected") is False or payload.get("channelId") in ("", None) and "channelId" in payload:
            self.state.clear(device)
            self.set_device_mode(device, "chat")
            return {"device": device, "connected": False}
        row = self.state.update(device, **fields)
        if row.get("channelId", "").startswith(ROOM_PREFIX):
            self.set_device_mode(device, "development")
        return {"device": device, "connected": True, **row}

    # -- the room turn (§5) --------------------------------------------------------
    def room_turn(self, device: str, transcript: str, speaker: str, sid: str, conf: float,
                  match: str, utterance_id: str, is_followup: bool = False,
                  followup_to: str | None = None) -> TurnResult:
        st = self.state.get(device)
        ch = st.get("channelId") or ""
        room_id = ch[len(ROOM_PREFIX):] if ch.startswith(ROOM_PREFIX) else None
        if not room_id:
            return TurnResult(_fmt("not_connected"), "not_connected")
        if not sid or conf < CONNECT_MIN_CONF:
            return TurnResult(_fmt("refused_speaker"), "refused_speaker")

        # 1. deterministic intents
        it = room_agent.parse_intent(transcript)
        if it:
            return self._intent(device, st, room_id, it, transcript, speaker, sid, conf, match,
                                utterance_id, followup_to)

        # 2. open-mic non-sequitur -> ignore (logged, never a room line)
        if is_followup and room_agent.is_non_sequitur(transcript):
            self.engine.said(room_id, {"utteranceId": utterance_id, "text": transcript,
                                       "device": device, "sid": sid, "action": "ignore",
                                       "confidence": 1.0, "followupTo": followup_to})
            return TurnResult("", "ignored")

        # 3. the room agent
        status, ctx = self.engine.context(room_id)
        if status in (404, 409):
            return self.self_heal(device, status)
        if status != 200:
            ctx = {"room": {"name": st.get("name"), "lead": st.get("lead")},
                   "connection": {"device": device, "muted": st.get("muted"), "waiting": st.get("waiting")}}
        dec = room_agent.route(transcript, ctx, history=self.memory.as_history(room_id, sid))
        action = dec["action"]
        if action == "answer":
            answer = dec["answer"] or ""
            self.engine.said(room_id, {"utteranceId": utterance_id, "text": transcript, "device": device,
                                       "sid": sid, "action": "answer", "answer": answer,
                                       "confidence": dec["confidence"], "followupTo": followup_to})
            self.memory.append(room_id, sid, transcript, answer, "answer")
            self._push_exchange(device, {"kind": "in", "who": "user", "text": transcript, "vid": None})
            self._push_exchange(device, {"kind": "out", "who": "voice", "text": answer, "vid": None})
            # §4: the open-mic window follows a voice-agent answer too.
            return TurnResult(answer, "answered", expects_reply=True, confidence=dec["confidence"])

        route = {"to": dec["to"] if action == "relay" else "lead", "confidence": dec["confidence"]}
        if dec.get("note"):
            route["note"] = dec["note"]
        body = {"channelId": ch, "text": transcript, "device": device, "speaker": speaker or "",
                "sid": sid, "utteranceId": utterance_id, "confidence": round(float(conf), 3),
                "match": match, "route": route}
        if followup_to:
            body["followupTo"] = followup_to
        status, resp = self.engine.inbound(body)
        if status == 202:
            resolved = resp.get("resolved") or ""
            to = resp.get("to") or ""
            lead = st.get("lead") or (ctx.get("room") or {}).get("lead")
            if resolved == "fallback-lead" and action == "relay":
                line = _fmt("relayed_fallback", heard=dec["to"] or "them")
            elif action == "lead" or to == lead or resolved == "lead":
                line = _fmt("relayed_lead")
            else:
                line = _fmt("relayed", name=_spoken_handle(to, lead))
            self.memory.append(room_id, sid, transcript, None, action)
            self._push_exchange(device, {"kind": "in", "who": "user", "text": transcript,
                                         "vid": resp.get("vid"), "lineId": resp.get("lineId"),
                                         "to": to, "resolved": resolved})
            return TurnResult(line, "sent", vid=resp.get("vid"), to=to, resolved=resolved,
                              route=route, engine_body=resp)
        if status in (404, 409):
            return self.self_heal(device, status)
        if status == 403:
            return TurnResult(_fmt("refused_speaker"), "refused_engine", status=status)
        return TurnResult(_fmt("engine_down") if status is None else _fmt("relay_failed"),
                          "engine_down", status=status, engine_body=resp)

    def _intent(self, device, st, room_id, it, transcript, speaker, sid, conf, match,
                utterance_id, followup_to) -> TurnResult:
        kind = it["intent"]
        ch = st.get("channelId")
        if kind == "connect":
            return self.connect_turn(device, it.get("target") or "", speaker, sid, conf)
        if kind == "disconnect":
            return self.disconnect(device, "spoken")
        if kind in ("mute", "unmute"):
            muted = kind == "mute"
            status, _ = self.engine.set_muted(ch, device, muted)
            self.state.update(device, muted=muted)
            return TurnResult(_fmt("muted" if muted else "unmuted"), kind, engineStatus=status)
        if kind == "say_again":
            ls = st.get("lastSpoken") or {}
            if not ls.get("body"):
                return TurnResult(_fmt("nothing_to_repeat"), "say_again")
            self.state.update(device, waiting=False)
            self.engine.clear_waiting(room_id)
            return TurnResult(_fmt("say_again", author=_spoken_handle(ls.get("author"), st.get("lead")),
                                   body=ls["body"]), "say_again", expects_reply=True)
        if kind == "whats_waiting":
            status, ctx = self.engine.context(room_id)
            if status in (404, 409):
                return self.self_heal(device, status)
            self.state.update(device, waiting=False)
            self.engine.clear_waiting(room_id)
            if status != 200:
                ls = st.get("lastSpoken") or {}
                if ls.get("body"):
                    return TurnResult(_fmt("waiting_one", body=ls["body"]), "whats_waiting_local")
                return TurnResult(_fmt("nothing_waiting"), "whats_waiting_local")
            open_ = ctx.get("open") or []
            ls = ctx.get("lastSpoken") or st.get("lastSpoken") or {}
            waiting = bool((ctx.get("connection") or {}).get("waiting") or st.get("waiting"))
            if waiting and ls.get("body"):
                line = _fmt("waiting_one", body=f"{_spoken_handle(ls.get('author'), st.get('lead'))} said, {ls['body']}")
            elif open_:
                oldest = max((o.get("agoS") or 0) for o in open_)
                line = _fmt("open_messages", n=len(open_), ago=_ago(oldest))
            else:
                line = _fmt("nothing_waiting")
            return TurnResult(line, "whats_waiting", expects_reply=waiting)
        return TurnResult(_fmt("relay_failed"), "unknown_intent")

    # -- delivery bookkeeping (called by the deliver route) ---------------------------
    def note_spoken(self, device: str, author: str | None, body: str, reply_id: str | None,
                    vid: str | None = None) -> None:
        if not self.state.get(device).get("channelId"):
            return
        self.state.update(device, lastSpoken={"author": author or "voice", "body": body,
                                              "vid": vid, "at": time.time()},
                          lastReplyId=reply_id)
        self._push_exchange(device, {"kind": "out", "who": author or "voice", "text": body,
                                     "vid": reply_id or vid, "inReplyTo": vid})

    def mark_speaking(self, device: str, seconds: float) -> None:
        """§9.2: the one fact the engine lacks. Playback is synchronous on our side, so
        'speaking' = now < speakingUntil; the panel's state route reads it."""
        if not self.state.get(device).get("channelId"):
            return
        self.state.update(device, speakingUntil=time.time() + max(0.0, float(seconds)))

    EXCHANGES_KEEP = 10

    def _push_exchange(self, device: str, entry: dict) -> None:
        """The panel (§9.5) shows the last exchanges; keep a short ring in the state row.
        Shape: {t, kind:"in"|"out", who:"user"|"voice"|<seat handle>, text, vid?, lineId?}."""
        st = self.state.get(device)
        if not st.get("channelId"):
            return
        ring = list(st.get("exchanges") or [])
        ring.append({"t": time.time(), **entry})
        self.state.update(device, exchanges=ring[-self.EXCHANGES_KEEP:])
