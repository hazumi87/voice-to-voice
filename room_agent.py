"""Room voice agent — logic tier (room-voice blueprint §5-§6, vk-1825 / harness vk-1819).

Pure module: no FastAPI, no globals from server.py. server.py wires it to the Dot turn;
tools/room_agent_harness.py scores it. Three layers, in the order the turn runs them:

  1. parse_intent(text)        deterministic: connect / disconnect / mute / unmute /
                               whats_waiting / say_again. Never touches the model.
  2. is_non_sequitur(text)     deterministic, open-mic window ONLY (the caller decides):
                               "ok", "thanks", "got it"... -> ignore. The model never sees them.
  3. route(text, context, ...) the Ollama JSON decision {action, to, answer, confidence}.
                               action is schema-enum'd to answer|relay|lead so a 3B model
                               cannot invent "add_tool". Any parse failure -> lead.

The agent has NO authority: it relays, answers about the room from the context snapshot
the engine serves (§6.2), and passes everything else to the lead (§6.3).
"""
from __future__ import annotations

import json
import os
import difflib
import re
import threading
import time
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
ROUTER_MODEL = os.environ.get("ROOM_ROUTER_MODEL", "llama3.2:3b")
ROUTER_KEEP_ALIVE = os.environ.get("ROOM_ROUTER_KEEP_ALIVE", "30m")   # §6.4: never cold-load a turn
ROUTER_NUM_CTX = 4096
ROUTER_TIMEOUT_S = 20.0

ACTIONS = ("answer", "relay", "lead")

# Ollama structured output (0.5+): the enum is the real guard against label drift.
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "to": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["action", "to", "answer", "confidence"],
}

# ---------------------------------------------------------------------------------
# 1. Deterministic intents (§4, §5). These run BEFORE the model, on the raw transcript.
# ---------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']")


def _norm(text: str) -> str:
    t = (text or "").lower().strip()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


_CONNECT = re.compile(
    r"^(?:please\s+)?(?:connect(?:\s+me)?|join|hook\s+me\s+up|link(?:\s+me)?)\s+(?:to|with|into)\s+"
    r"(?:the\s+)?(?P<target>.+?)(?:\s+(?:room|workroom|workrooms))?$")
_DISCONNECT = re.compile(
    r"^(?:please\s+)?(?:disconnect(?:\s+me)?(?:\s+from\s+(?:the\s+)?.+)?|hang\s+up|leave\s+(?:the\s+)?room|"
    r"log\s+me\s+off|end\s+(?:the\s+)?(?:voice\s+)?(?:connection|session))$")
_MUTE = re.compile(
    r"^(?:please\s+)?(?:mute(?:\s+yourself|\s+the\s+agent|\s+voice)?|be\s+quiet|go\s+quiet|hush|shut\s+up)$")
_UNMUTE = re.compile(
    r"^(?:please\s+)?(?:unmute(?:\s+yourself|\s+the\s+agent|\s+voice)?|you\s+can\s+talk(?:\s+again)?|speak\s+again)$")
_WAITING = re.compile(
    r"^(?:please\s+)?(?:what'?s\s+waiting|what\s+is\s+waiting|anything\s+waiting(?:\s+for\s+me)?|"
    r"any\s+messages(?:\s+waiting)?(?:\s+for\s+me)?|what\s+did\s+i\s+miss|what'?s\s+new|any\s+news)(?:\s+for\s+me)?$")
_SAY_AGAIN = re.compile(
    r"^(?:please\s+)?(?:say\s+(?:that|it)\s+again|repeat\s+(?:that|it)|come\s+again|what\s+was\s+that|"
    r"one\s+more\s+time|read\s+(?:that|it)\s+(?:back|again))$")


def parse_intent(text: str) -> dict | None:
    """Return {"intent": ..., "target"?: ...} for a deterministic intent, else None."""
    t = _norm(text)
    for lead in ("hey jarvis", "jarvis", "hey"):
        if t.startswith(lead + " "):
            t = t[len(lead) + 1:]
    m = _CONNECT.match(t)
    if m:
        return {"intent": "connect", "target": m.group("target").strip()}
    if _DISCONNECT.match(t):
        return {"intent": "disconnect"}
    if _MUTE.match(t):
        return {"intent": "mute"}
    if _UNMUTE.match(t):
        return {"intent": "unmute"}
    if _WAITING.match(t):
        return {"intent": "whats_waiting"}
    if _SAY_AGAIN.match(t):
        return {"intent": "say_again"}
    m = _VOICE_ID.match(t)
    if m:
        return {"intent": "voice_id_on" if _voice_id_on(m) else "voice_id_off"}
    return None


# §10.1 live voice-ID toggle by voice. TIGHTEN-ONLY on the engine: "turn on voice id" is
# honoured; "turn off" is answered with where to do it (anyone in earshot could say it).
_VOICE_ID = re.compile(
    r"^(?:please\s+)?(?:(?:turn|switch)\s+(?P<on>on|off)\s+(?:the\s+)?(?:voice|speaker)\s+(?:id|identification|verification|check)|"
    r"(?:turn|switch)\s+(?:the\s+)?(?:voice|speaker)\s+(?:id|identification|verification|check)\s+(?P<on2>on|off)|"
    r"(?P<on3>enable|disable|activate|deactivate)\s+(?:the\s+)?(?:voice|speaker)\s+(?:id|identification|verification|check))$")


def _voice_id_on(m) -> bool:
    v = (m.group("on") or m.group("on2") or m.group("on3") or "").lower()
    return v in ("on", "enable", "activate")


# ---------------------------------------------------------------------------------
# 2. Non-sequitur filter (§5, §6.5). Open-mic window only. Whole-utterance match:
#    "ok" is ignored, "ok tell aurora the mockup is late" is not.
# ---------------------------------------------------------------------------------
NON_SEQUITURS = {
    "ok", "okay", "k", "kay", "thanks", "thank you", "thanks a lot", "great thanks",
    "ok thanks", "okay thanks", "got it", "cool", "cool thanks", "alright", "all right",
    "sounds good", "i'll be waiting", "i will be waiting", "great", "perfect", "yep",
    "yup", "yes", "no", "nope", "sure", "fine", "roger", "roger that", "copy", "copy that",
    "noted", "will do", "good", "nice", "cheers", "never mind", "nevermind", "that's all",
    "that is all", "bye", "goodbye", "later", "talk later", "no thanks", "nothing",
    "nothing else", "that's it", "that is it", "understood", "right", "mhm", "uh huh",
    "okay great", "ok great", "ok cool", "okay cool", "good to know", "i see", "ah ok",
    "oh ok", "oh okay", "alright thanks", "thanks jarvis", "thank you jarvis", "ok jarvis",
}
_NON_SEQ_MAX_WORDS = 4


def is_non_sequitur(text: str) -> bool:
    t = _norm(text)
    if not t:
        return True
    if len(t.split()) > _NON_SEQ_MAX_WORDS:
        return False
    return t in NON_SEQUITURS


# ---------------------------------------------------------------------------------
# 3. The JSON routing decision (§6.4).
# ---------------------------------------------------------------------------------
def _ago(s) -> str:
    try:
        s = int(s)
    except (TypeError, ValueError):
        return "unknown"
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    return f"{s // 3600} h ago"


def render_context(ctx: dict) -> str:
    """Compact, line-oriented rendering of the §6.2 snapshot for a small model."""
    ctx = ctx or {}
    user = (ctx.get("user") or {}).get("name") or "the user"
    room = ctx.get("room") or {}
    conn = ctx.get("connection") or {}
    lines = [f"User: {user}",
             f"Room: {room.get('name', '?')} (lead: {room.get('lead', '?')})",
             f"Connection: device {conn.get('device', '?')}, muted={bool(conn.get('muted'))}, "
             f"waiting={bool(conn.get('waiting'))}"]
    seats = ctx.get("seats") or []
    lines.append(f"Seats ({len(seats)}):")
    for s in seats:
        spoken = f" (spoken name: {s['spokenName']})" if s.get("spokenName") else ""
        lead = " [LEAD]" if s.get("lead") else ""
        # Plain words, measured (G 2026-09-28): with "parked, streaming" the 3B model called a
        # working seat idle; with "working now" it answered correctly three times out of three.
        act = (s.get("activity") or "").lower()
        status = {"streaming": "working now", "working": "working now", "busy": "working now",
                  "waiting": "waiting for you", "idle": "idle"}.get(act, act or "idle")
        lines.append(f"  - {s.get('handle')}{spoken}{lead}: {status}, "
                     f"last line {_ago(s.get('lastLineAgoS'))}")
    tools = ctx.get("tools") or []
    lines.append("Tools: " + (", ".join(f"{t.get('title')} ({t.get('kind')})" for t in tools) or "none"))
    open_ = ctx.get("open") or []
    if open_:
        lines.append(f"Open voice messages ({len(open_)}):")
        for o in open_:
            ack = f", acked {_ago(o['ackAgoS'])}" if o.get("ackAgoS") is not None else ", no ack yet"
            res = f", answered {_ago(o['resultAgoS'])}" if o.get("resultAgoS") is not None else ""
            lines.append(f"  - {o.get('vid')} to {o.get('to')}, sent {_ago(o.get('agoS'))}{ack}{res}")
    else:
        lines.append("Open voice messages: none")
    ls = ctx.get("lastSpoken")
    if ls:
        lines.append(f"Last thing spoken to the user: {ls.get('author')} said \"{ls.get('body')}\" "
                     f"({_ago(ls.get('agoS'))})")
    recent = ctx.get("recent") or []
    if recent:
        lines.append("Recent room lines (newest last):")
        for r in recent[-8:]:
            tgt = f" @{r['target']}" if r.get("target") else ""
            lines.append(f"  [{_ago(r.get('agoS'))}] {r.get('author')}{tgt}: {str(r.get('body', ''))[:160]}")
    return "\n".join(lines)


SYSTEM_RULES = """You are the voice assistant for a workroom. A person speaks to you through a smart speaker. Decide what to do with ONE utterance. You have NO authority in the room: you cannot add or remove seats, tools or panels, invite anyone, dispatch agents, change settings, or run commands.

Pick exactly one action:
- "answer": the utterance is a QUESTION ABOUT THE ROOM (who is here, who is working, whether someone replied, what was said, what tools exist, whether anything is waiting, how long ago). Answer ONLY from the room snapshot below, in one short spoken sentence (max 30 words). If the snapshot does not contain the answer, say so briefly. Set "to" to null.
- "relay": the person explicitly addresses a message to a NAMED seat or agent ("tell X ...", "ask X ...", "message X ...", "let X know ..."). Set "to" to the name exactly as they said it. Do not rephrase the message. Set "answer" to null.
- "lead": EVERYTHING ELSE goes to the lead: any request, wish, feedback or instruction about the project ("I'd like ...", "make the button blue", "can we make the intro shorter"); any report of a problem or a bug ("the export is missing the timestamps", "the font is too small"); any message addressed to the lead; and every request you are not allowed to do yourself (adding or removing tools, seats, people; dissolving, renaming or archiving the room; changing settings; running commands; dispatching). Set "to" to null and "answer" to null. A question is only an "answer" when it asks about the ROOM ITSELF (people, seats, tools, messages, timing); "can we ..." / "could you ..." requests about the work are for the lead.

"confidence" is your confidence in the action, 0 to 1. Reply with JSON only.

Examples of the ACTION only (the answer text below is a placeholder; a real answer is one sentence built from the ROOM SNAPSHOT, never copied from here):
"I'd like to add a hat to the character" -> {"action":"lead","to":null,"answer":null,"confidence":0.9}
"tell mira the button should say help" -> {"action":"relay","to":"mira","answer":null,"confidence":0.95}
"message the lead that the logo is too big" -> {"action":"lead","to":null,"answer":null,"confidence":0.95}
"is anyone working right now" -> {"action":"answer","to":null,"answer":"<one sentence from the snapshot>","confidence":0.9}
"add the blender tool to the room" -> {"action":"lead","to":null,"answer":null,"confidence":0.9}
"can we make the intro shorter" -> {"action":"lead","to":null,"answer":null,"confidence":0.9}
"the export is missing the timestamps" -> {"action":"lead","to":null,"answer":null,"confidence":0.9}
"what tools are in the room" -> {"action":"answer","to":null,"answer":"<one sentence from the snapshot>","confidence":0.9}
"remove mira from the room" -> {"action":"lead","to":null,"answer":null,"confidence":0.9}
"did the lead answer me yet" -> {"action":"answer","to":null,"answer":"<one sentence from the snapshot>","confidence":0.85}
"""

# Words that only occur in the prompt's examples. An answer that contains one was copied
# from the examples instead of read from the snapshot (measured live at G: the 3B model
# returned the example sentence verbatim for "is anyone working right now").
_EXAMPLE_ONLY = ("kade", "mira", "one sentence from the snapshot", "<", ">")
_ANSWER_FALLBACK = "I can't tell that from the room right now."


# ---------------------------------------------------------------------------------
# Deterministic answers for the common room questions (G finding, 2026-09-28). The
# snapshot has the facts; a 3B model misreads "parked" and copies examples. So the questions
# Eric actually asks are answered from the snapshot in code, and the model only gets the
# rest. Returns None when the question is not one of these.
# ---------------------------------------------------------------------------------
_Q_WHO_HERE = re.compile(r"\b(who(?:'s| is| are)?\s+(?:all\s+)?(?:in|on)\s+(?:the\s+)?(?:room|here|call)|who(?:'s| is)\s+here|who\s+do\s+we\s+have)\b")
_Q_WORKING = re.compile(r"\b((?:is|are)\s+(?:anyone|anybody|someone|any\s+of\s+them|they|people)\s+(?:still\s+)?(?:working|busy|active|on\s+it)|who(?:'s| is)\s+(?:working|busy|active)|anyone\s+working|anybody\s+working)\b")
_Q_LEAD = re.compile(r"\b(who(?:'s| is)\s+(?:the\s+)?(?:lead|leader|in\s+charge|running\s+(?:this|the)\s+room))\b")
_Q_TOOLS = re.compile(r"\b((?:what|which)\s+(?:tools|apps|panels)|(?:any|are\s+there)\s+(?:tools|apps|panels))\b")
_Q_ROOM = re.compile(r"\b((?:what|which)\s+room\s+(?:am\s+i|are\s+we|is\s+this)|where\s+am\s+i\s+connected|what\s+am\s+i\s+connected\s+to)\b")
_Q_SEATS = re.compile(r"\bhow\s+many\s+(?:seats|people|agents|members)\b")
_Q_ANSWERED = re.compile(r"\b(?:did|has|have)\s+(?P<who>.+?)\s+(?:answer|answered|reply|replied|respond|responded|acknowledge|acknowledged|ack|acked|get\s+back|pick(?:ed)?\s+(?:that|it)\s+up)\b")
_Q_LAST_SAID = re.compile(r"\b(?:what(?:'s| is| was| did)\s+(?:the\s+)?(?:last\s+thing\s+)?(?P<who>.+?)\s+(?:say|said)(?:\s+last)?|last\s+thing\s+(?P<who2>.+?)\s+said)\b")
_Q_HOW_LONG = re.compile(r"\bhow\s+long\s+(?:ago\s+)?(?:did|since|has)\s+(?P<who>.+?)\s+(?:post|posted|spoke|said|wrote|write|reply|replied)\b")
_Q_MUTED = re.compile(r"\b(?:am\s+i|are\s+you|is\s+(?:the\s+)?(?:room|voice|agent))\s+muted\b")


def _spoken(handle: str, lead: str | None = None) -> str:
    if lead and handle == lead:
        return "the lead"
    return (handle or "").replace("-", " ").replace("_", " ")


def _match_seat(who: str, ctx: dict):
    """Fuzzy-map spoken words to a seat handle; 'the lead' -> the lead."""
    who = _norm(who)
    who = re.sub(r"^(?:the|our|my)\s+", "", who)
    lead = (ctx.get("room") or {}).get("lead")
    if who in ("lead", "leader", "team lead", "room lead", "boss") and lead:
        return lead
    best, score = None, 0.0
    for s in ctx.get("seats") or []:
        for cand in (s.get("handle") or "", s.get("spokenName") or ""):
            if not cand:
                continue
            c = _norm(cand.replace("-", " "))
            r = difflib.SequenceMatcher(None, who, c).ratio()
            if who and (who in c or c in who):
                r = max(r, 0.85)
            if r > score:
                best, score = s.get("handle"), r
    return best if score >= 0.6 else None


def _join(items: list) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def answer_room_question(text: str, ctx: dict) -> str | None:
    t = _norm(text)
    ctx = ctx or {}
    room = ctx.get("room") or {}
    lead = room.get("lead")
    seats = ctx.get("seats") or []
    # Working/status words win over the roster ("who in the room is working right now and
    # what's their status" is a working question, G 2026-09-28).
    asks_status = bool(re.search(r"\b(working|busy|active|status|doing|up\s+to)\b", t))
    if _Q_WHO_HERE.search(t) and not asks_status:
        if not seats:
            return "No one is seated in the room right now."
        names = [_spoken(s.get("handle"), lead) for s in seats]
        return f"{len(seats)} seat{'s' if len(seats) != 1 else ''}: {_join(names)}."
    if _Q_WORKING.search(t) or (asks_status and re.search(r"\b(who|everyone|everybody|anyone|anybody|they|people|the\s+room|seats?)\b", t)):
        working = [s for s in seats if (s.get("activity") or "").lower() in ("streaming", "working", "busy")]
        idle = [s for s in seats if s not in working]
        if working:
            line = f"Yes. {_join([_spoken(s.get('handle'), lead) for s in working])} {'is' if len(working) == 1 else 'are'} working"
            if idle:
                line += f"; {_join([_spoken(s.get('handle'), lead) for s in idle])} {'is' if len(idle) == 1 else 'are'} idle"
            return line + "."
        return "No one is working right now." + (f" {_join([_spoken(s.get('handle'), lead) for s in idle])} {'is' if len(idle) == 1 else 'are'} idle." if idle else "")
    if _Q_LEAD.search(t):
        return f"The lead is {(lead or 'not set').replace('-', ' ')}." if lead else "This room has no lead set."
    if _Q_TOOLS.search(t):
        tools = [tl.get("title") for tl in (ctx.get("tools") or []) if tl.get("title")]
        return f"{len(tools)} tool{'s' if len(tools) != 1 else ''}: {_join(tools)}." if tools else "There are no tools in the room."
    if _Q_ROOM.search(t):
        return f"You're connected to {room.get('name') or 'the room'}."
    if _Q_SEATS.search(t):
        return f"{len(seats)} seat{'s' if len(seats) != 1 else ''}."
    if _Q_MUTED.search(t):
        return "Yes, I'm muted; replies go to the chat." if (ctx.get("connection") or {}).get("muted") else "No, I'm not muted."
    m = _Q_HOW_LONG.search(t)
    if m:
        h = _match_seat(m.group("who"), ctx)
        s = next((x for x in seats if x.get("handle") == h), None)
        if s and s.get("lastLineAgoS") is not None:
            return f"{_spoken(h, lead).capitalize()} last posted {_ago(s['lastLineAgoS'])}."
        return "I don't see a post from them in the room."
    m = _Q_ANSWERED.search(t)
    if m:
        h = _match_seat(m.group("who"), ctx) or lead
        opens = [o for o in (ctx.get("open") or []) if o.get("to") == h]
        ls = ctx.get("lastSpoken") or {}
        if opens:
            o = opens[-1]
            if o.get("resultAgoS") is not None:
                return f"Yes. {_spoken(h, lead).capitalize()} answered {_ago(o['resultAgoS'])}."
            if o.get("ackAgoS") is not None:
                return f"Not yet. {_spoken(h, lead).capitalize()} acknowledged it {_ago(o['ackAgoS'])} but hasn't answered."
            return f"Not yet. Your message from {_ago(o.get('agoS'))} hasn't been picked up."
        if ls.get("author") == h and ls.get("body"):
            return f"The last thing from {_spoken(h, lead)} was {_ago(ls.get('agoS'))}: {ls['body']}"
        return f"There's nothing open for {_spoken(h, lead)} right now."
    m = _Q_LAST_SAID.search(t)
    if m:
        who = m.group("who") or m.group("who2") or ""
        h = _match_seat(who, ctx)
        if h:
            lines = [r for r in (ctx.get("recent") or []) if r.get("author") == h]
            if lines:
                r = lines[-1]
                return f"{_ago(r.get('agoS')).capitalize()}, {_spoken(h, lead)} said: {str(r.get('body', ''))[:240]}"
            return f"I don't see a recent line from {_spoken(h, lead)}."
    return None

_LEAD_WORDS = {"lead", "the lead", "leader", "the leader", "team lead", "the team lead",
               "our lead", "my lead", "room lead", "the room lead"}

# Explicit relay = the utterance STARTS with a relay verb and names someone. The addressee
# is taken from the words, never from the model: a 3B model will happily swap an unknown
# name for a seat it knows (measured), and the engine owns resolution anyway (§5).
_RELAY_VERB = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:tell|ask|message|text|ping|inform|notify|remind|"
    r"send|relay\s+to|let|say\s+to|pass\s+(?:on\s+)?to|forward\s+to)\s+(?P<rest>.+)$")
_ADDRESSEE_END = re.compile(
    r"^(?P<name>.+?)(?:\s+(?:that|to|if|whether|about|know|a\s+note|the|we|i|i'd|i'm|it|he|she|they|"
    r"please|when|what|why|how|where|is|are|for|on|this|there)\b.*|\s*$)")


def extract_addressee(text: str) -> str | None:
    """'tell aurora the button …' -> 'aurora'; 'let the lead know …' -> 'the lead';
    None when the utterance is not an explicit relay."""
    t = _norm(text)
    for lead in ("hey jarvis", "jarvis", "hey"):
        if t.startswith(lead + " "):
            t = t[len(lead) + 1:]
    m = _RELAY_VERB.match(t)
    if not m:
        return None
    rest = m.group("rest").strip()
    m2 = _ADDRESSEE_END.match(rest)
    name = (m2.group("name") if m2 else rest).strip()
    name = re.sub(r"^(?:the|our|my)\s+", "", name)
    # drop a trailing "agent"/"seat" the user appended to the name
    name = re.sub(r"\s+(?:agent|seat|bot)$", "", name)
    if not name or len(name.split()) > 4:
        return None
    return name


def _ollama_chat(messages: list, model: str, fmt, timeout: float, keep_alive: str) -> dict:
    import gpu_cop_client  # GPU cop: don't load the router into space held for a restart
    gpu_cop_client.router_gate(model)  # raises -> route() falls back to `lead` (never drops)
    payload = json.dumps({
        "model": model, "messages": messages, "stream": False, "format": fmt,
        "keep_alive": keep_alive,
        "options": {"temperature": 0, "num_ctx": ROUTER_NUM_CTX, "num_predict": 160},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def route(text: str, context: dict, model: str = None, timeout: float = ROUTER_TIMEOUT_S,
          history: list | None = None) -> dict:
    """One routing decision. Never raises for model/parse trouble: that is a `lead` with
    parse_ok=False and a note, because a dropped utterance is the one unacceptable outcome."""
    model = model or ROUTER_MODEL
    system = SYSTEM_RULES + "\n\nROOM SNAPSHOT:\n" + render_context(context)
    messages = [{"role": "system", "content": system}]
    for h in (history or [])[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": text})
    t0 = time.time()
    out = {"action": "lead", "to": None, "answer": None, "confidence": 0.0,
           "parse_ok": False, "note": None, "raw": None, "latency_s": None, "model": model}
    try:
        data = _ollama_chat(messages, model, ROUTE_SCHEMA, timeout, ROUTER_KEEP_ALIVE)
        raw = (data.get("message") or {}).get("content") or ""
        out["raw"] = raw
        out["eval_count"] = data.get("eval_count")
        out["prompt_eval_count"] = data.get("prompt_eval_count")
        dec = json.loads(raw)
        action = str(dec.get("action") or "").strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        to = dec.get("to")
        to = str(to).strip() if to not in (None, "", "null") else None
        answer = dec.get("answer")
        answer = str(answer).strip() if answer not in (None, "", "null") else None
        conf = dec.get("confidence")
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5
        # Deterministic clean-up of the label the model picked.
        addressee = extract_addressee(text)
        if addressee is not None:
            # The words name someone ("ask aurora if…", "tell the lead…"): relay to THAT name
            # (or the lead), whatever the model said, even when it called the utterance a question.
            if addressee.lower() in _LEAD_WORDS:
                action, to = "lead", None
            else:
                action, to = "relay", addressee
        elif action == "relay" and addressee is None:
            # No relay verb in the words: an implicit request, so it is the lead's.
            action, to = "lead", None
        if action == "relay":
            if not to or to.lower() in _LEAD_WORDS:
                action, to = "lead", None
        if action == "answer" and answer and any(w in answer.lower() for w in _EXAMPLE_ONLY):
            out["note"] = "answer copied from the examples; replaced"
            answer, conf = _ANSWER_FALLBACK, min(conf, 0.3)
        if action == "answer" and not answer:
            action = "lead"
        if action == "lead":
            to = None
        out.update({"action": action, "to": to, "answer": answer if action == "answer" else None,
                    "confidence": conf, "parse_ok": True})
    except Exception as e:  # noqa: BLE001 — model down, timeout, bad JSON: all -> lead
        out["note"] = f"routing failed: {type(e).__name__}: {e}"
    out["latency_s"] = round(time.time() - t0, 3)
    return out


# ---------------------------------------------------------------------------------
# Memory across disconnects (§6.4): per (room, speaker), last 12 turns + a one-paragraph
# summary, persisted, 7-day TTL. Room FACTS are never remembered (always refetched).
# ---------------------------------------------------------------------------------
MEMORY_TURNS = 12
MEMORY_TTL_S = 7 * 24 * 3600


class RoomMemory:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (OSError, ValueError):
            self._data = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _key(room_id: str, speaker: str) -> str:
        return f"{room_id}|{speaker or 'unknown'}"

    def get(self, room_id: str, speaker: str) -> dict:
        with self._lock:
            row = self._data.get(self._key(room_id, speaker))
            if not row:
                return {"turns": [], "summary": "", "updated": 0}
            if time.time() - float(row.get("updated") or 0) > MEMORY_TTL_S:
                self._data.pop(self._key(room_id, speaker), None)
                self._save()
                return {"turns": [], "summary": "", "updated": 0}
            return dict(row)

    def append(self, room_id: str, speaker: str, user_text: str, agent_text: str | None,
               action: str) -> None:
        with self._lock:
            k = self._key(room_id, speaker)
            row = self._data.setdefault(k, {"turns": [], "summary": "", "updated": 0})
            row["turns"].append({"ts": int(time.time()), "user": user_text,
                                 "agent": agent_text, "action": action})
            row["turns"] = row["turns"][-MEMORY_TURNS:]
            row["updated"] = time.time()
            self._save()

    def set_summary(self, room_id: str, speaker: str, summary: str) -> None:
        with self._lock:
            k = self._key(room_id, speaker)
            row = self._data.setdefault(k, {"turns": [], "summary": "", "updated": 0})
            row["summary"] = (summary or "")[:1200]
            row["updated"] = time.time()
            self._save()

    def as_history(self, room_id: str, speaker: str) -> list:
        """The last exchanges as ONE system note. Raw user/assistant turns made the 3B
        model treat the new utterance as a continuation of the previous one (it kept the
        previous addressee), so memory is rendered as background, not as dialogue."""
        row = self.get(room_id, speaker)
        parts = []
        if row.get("summary"):
            parts.append("Earlier in this room: " + row["summary"])
        turns = row.get("turns", [])[-3:]
        if turns:
            lines = []
            for t in turns:
                what = t.get("action") or "?"
                said = f' -> you answered "{t["agent"]}"' if t.get("agent") else f" -> {what}"
                lines.append(f'- user said "{t["user"]}"{said}')
            parts.append("Previous exchanges (background only; decide the NEW utterance on its own):\n"
                         + "\n".join(lines))
        if not parts:
            return []
        return [{"role": "system", "content": "\n\n".join(parts)}]
