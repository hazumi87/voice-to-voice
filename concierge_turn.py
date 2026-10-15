"""concierge_turn.py — the Concierge's judgement on one dictated utterance that carries
its name. Logic tier, no FastAPI: server.py exposes it as POST /api/voice/concierge-turn.

    decide(draft, utterance, name, handle) -> {
        addressed: bool,                 # spoken TO the Concierge (not merely mentioned)
        kind: "edit" | "question" | "content",
        draft: str,                      # the draft after any edit (unchanged otherwise)
        reply: str | None,               # short spoken line; None when there is nothing to say
        changes: [{from, to}],           # what was applied, as found in the draft
        text: str,                       # the utterance with the display name written as @handle
        proofread: bool,                 # kind "edit" produced by a proofread request
        via: "rule" | "model",           # whether a model call was needed
        model_ms: int | None, model_error: str | None,
    }

Rules first, model second (spec: briefing-table docs/specs/concierge-name-stt.md, roadmap
"the name is a CUE, and the concierge decides", Eric #8338):

  * addressed?  grammar decides the clear cases: the name at the start ("Olivia, …",
    "hey Olivia …", "ok Olivia …") or at the end ("…, Olivia") is a vocative; the name
    after an object marker ("tell platform to get Olivia the data", "ask Olivia to …",
    "send it to Olivia") is a mention. Only the rest goes to the model, and when the model
    cannot answer the line is treated as content, because losing dictation is the one
    unacceptable outcome.
  * kind:  "change X to Y", "replace X with Y", "remove X", "remove the last sentence",
    "can you hear me", "how many words", "read it back" are handled without a model.
    Free-form edits and proofreads ask the model for a CHANGE LIST; every change must be
    located in the draft (exact, case-insensitive, then fuzzy for STT slips) before it is
    applied. The model never writes the draft itself.
  * proofread:  STT slips only (homophones, dropped/doubled small words, misheard names);
    each change is short and must resemble what it replaces, so a rewrite is dropped.

Owner: voice-to-voice. Tests: tools/concierge_turn_test.py (fake model, no Ollama).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("CONCIERGE_TURN_MODEL", os.environ.get("ROOM_ROUTER_MODEL", "llama3.2:3b"))
KEEP_ALIVE = os.environ.get("ROOM_ROUTER_KEEP_ALIVE", "30m")
TIMEOUT_S = 20.0
NUM_CTX = 4096

FILLERS = {"hey", "hi", "hello", "ok", "okay", "so", "um", "uh", "oh", "yo", "alright",
           "right", "now", "please", "and", "well", "then", "also", "yeah", "yes", "no"}
# the word right before the name when the name is an OBJECT, not a vocative
OBJECT_MARKERS = {"to", "for", "with", "at", "about", "from", "get", "tell", "ask", "give",
                  "let", "send", "show", "of", "on", "by", "via", "and", "or", "than", "like",
                  "called", "named", "is", "was", "that", "remind", "ping", "cc", "copy",
                  "email", "message", "text", "call", "thank", "thanks", "dear"}
# the word right after the name when the name is a vocative mid-sentence ("… Olivia can you …")
ADDRESS_FOLLOWERS = {"can", "could", "would", "will", "please", "let's", "lets", "change",
                     "replace", "remove", "delete", "make", "add", "fix", "do", "did", "are",
                     "what", "where", "how", "why", "who", "when", "which", "stop", "wait", "hold"}

_POLITE = r"(?:(?:hey|ok|okay|so|um|please|can you|could you|would you|will you|i want you to|i need you to|i'd like you to|let's|lets|i want to|i need to|we need to|we should|you should|just|go ahead and|now)\s+)*"
_CHANGE_RE = re.compile(
    r"^" + _POLITE + r"(?:change|replace|swap|switch|turn|update|correct|fix)\s+"
    r"(?:(?:the|that|this)\s+(?:word|words|phrase|name|text|part|bit)\s+)?(?P<body>.+?)\s*$")
_CHANGE_SPLIT = re.compile(r"\s+(?:to|with|into|for)\s+")
_DELETE_RE = re.compile(
    r"^" + _POLITE + r"(?:delete|remove|drop|take out|cut|strike|get rid of|erase|scrap)\s+"
    r"(?:(?:the|that|this)\s+(?:word|words|phrase|part|bit)\s+)?(?P<body>.+?)\s*$")
_LAST_FIRST_RE = re.compile(r"^(?:the\s+)?(?P<which>last|first)\s+(?P<unit>sentence|line|word|part|bit|thing)$")
_INSERT_RE = re.compile(
    r"^" + _POLITE + r"(?:add|insert|put|append|include)\s+"
    r"(?:(?:the|a)\s+(?:word|words|phrase)\s+)?(?P<what>.+?)\s+"
    r"(?:(?P<where>after|before|following|behind|in front of)\s+(?:the\s+(?:word|words|phrase)\s+)?(?P<anchor>.+?)"
    r"|(?P<end>at the end|to the end|at the back|on the end)|(?P<start>at the start|at the beginning|at the front|up front))\s*$")
# doubled small words are a classic STT slip; "had had" / "that that" are real English, so only these
_DOUBLE_WORDS = {"the", "a", "an", "to", "of", "in", "on", "and", "is", "it", "i", "we", "you", "for",
                 "with", "at", "my", "our", "your", "this", "or", "but", "so"}
_DOUBLE_RE = re.compile(r"\b(" + "|".join(sorted(_DOUBLE_WORDS)) + r")\s+\1\b", re.IGNORECASE)
_PROOF_RE = re.compile(
    r"\b(?:proof\s?read|proofread|spell\s?check|check\s+(?:the\s+|my\s+|this\s+)?(?:spelling|draft|text|typos|transcript)"
    r"|fix\s+(?:the\s+|my\s+|any\s+)?(?:typos|spelling|transcription|mistakes)"
    r"|clean\s+(?:this|it|that)\s+up|tidy\s+(?:this|it|that)\s+up"
    r"|correct\s+(?:the\s+|any\s+)?(?:typos|spelling|mistakes|transcript))\b")
_EDIT_CUE_RE = re.compile(
    r"\b(?:change|replace|swap|switch|delete|remove|drop|take out|cut|strike|add|insert|put|append|prepend"
    r"|capitali[sz]e|lower\s?case|upper\s?case|rename|rephrase|reword|rewrite|shorten|fix|correct|spell"
    r"|move|instead of|should\s+(?:be|say|read)|make\s+(?:it|that|this|the)|edit|update)\b")
_QUESTION_START_RE = re.compile(
    r"^(?:can|could|do|does|did|are|is|what|what's|where|how|why|who|when|will|would|which|have|has|am)\b")
_HEAR_RE = re.compile(r"\b(?:hear me|are you there|you there|still there|are you listening|you listening|awake)\b")
_COUNT_RE = re.compile(r"\b(?:how many words|word count|how long is (?:it|that|this|the draft)|how much (?:do i|have i|is there))\b")
_LOOK_RE = re.compile(
    r"\b(?:take a look|have a look|look at (?:this|that|it|the draft)|check (?:this|that|it) out|check (?:this|that|it)"
    r"|review (?:this|that|it|the draft)|see (?:this|that|it)|what do you think|thoughts|look (?:this|that|it) over|go over (?:this|that|it))\b")
_READBACK_RE = re.compile(r"\b(?:read (?:it|that|this|the draft|the text|the message) back|read back|what did i say|what have i said|what's in the draft|what do you have|what have you got|repeat (?:it|that|the draft))\b")


# ---------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w' ]+", re.UNICODE)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", str(text or "").lower()).replace("_", " ")).strip()


def words(text: str) -> list[str]:
    n = norm(text)
    return n.split(" ") if n else []


def _name_words(name: str) -> list[str]:
    return words(name)


def _name_hits(utter_words: list[str], name: str, handle: str) -> list[tuple[int, int]]:
    """(start, length) of every whole-word match of the display name, '@handle'
    (normalised to 'handle' after punctuation stripping) or 'at handle'."""
    hits: list[tuple[int, int]] = []
    forms: list[list[str]] = []
    nw = _name_words(name)
    if nw:
        forms.append(nw)
    hw = words(handle)
    if hw:
        forms.append(hw)
        forms.append(["at"] + hw)
    for i in range(len(utter_words)):
        for f in forms:
            if utter_words[i:i + len(f)] == f:
                hits.append((i, len(f)))
                break
    return hits


def substitute(text: str, name: str, handle: str) -> str:
    """Spec rules 1/2/4/5: every whole-word display name (and 'at handle') -> '@handle'.
    'Olivia's' and longer words are untouched; without a handle the text is left as spoken."""
    h = str(handle or "").strip().lstrip("@").lower()
    if not h:
        return text
    out = text
    pat_at = re.compile(r"(?<![\w@])at\s+" + re.escape(h) + r"(?!['\w])", re.IGNORECASE)
    out = pat_at.sub("@" + h, out)              # "at olivia" first, or the name pass eats it
    n = str(name or "").strip()
    if n:
        pat = re.compile(r"(?<![\w@])" + re.escape(n) + r"(?!['\w])", re.IGNORECASE)
        out = pat.sub("@" + h, out)
    pat_h = re.compile(r"(?<![\w@])" + re.escape(h) + r"(?!['\w])", re.IGNORECASE)
    out = pat_h.sub("@" + h, out)
    return out


def _strip_name(utter_words: list[str], hits: list[tuple[int, int]]) -> list[str]:
    drop = set()
    for s, n in hits:
        drop.update(range(s, s + n))
    return [w for i, w in enumerate(utter_words) if i not in drop]


def _strip_fillers(ws: list[str]) -> list[str]:
    i = 0
    while i < len(ws) and ws[i] in FILLERS:
        i += 1
    return ws[i:]


# ---------------------------------------------------------------------------------
# locating text in the draft
# ---------------------------------------------------------------------------------
def _spans(draft: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", draft)]


def locate(draft: str, frag: str, fuzzy: bool = True, cutoff: float = 0.78) -> tuple[int, int] | None:
    """(start, end) of `frag` in `draft`: exact (case-insensitive, word-bounded) first, then
    the best fuzzy word n-gram of similar length (STT slips: 'tabel' for 'table')."""
    frag = str(frag or "").strip()
    if not frag or not draft:
        return None
    pat = r"(?<!\w)" + re.escape(frag) + r"(?!\w)" if re.match(r"\w", frag) and re.search(r"\w$", frag) else re.escape(frag)
    m = re.search(pat, draft, re.IGNORECASE)
    if m:
        return (m.start(), m.end())
    if not fuzzy:
        return None
    target = norm(frag)
    n = len(target.split(" "))
    spans = _spans(draft)
    best, best_r = None, 0.0
    for size in sorted({n, max(1, n - 1), n + 1}):
        for i in range(0, len(spans) - size + 1):
            cand = norm(" ".join(s[2] for s in spans[i:i + size]))
            r = difflib.SequenceMatcher(None, target, cand).ratio()
            if r > best_r:
                best_r, best = r, (spans[i][0], spans[i + size - 1][1])
    if best and best_r >= cutoff:
        return best
    return None


def _match_case(src: str, repl: str) -> str:
    if not repl or not src:
        return repl
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper() and not repl[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def apply_changes(draft: str, changes: list[dict], all_occurrences: bool = False,
                  fuzzy: bool = True) -> tuple[str, list[dict], list[dict]]:
    """Apply [{from,to}] to the draft. Returns (new_draft, applied, dropped). `applied`
    carries the text as actually found in the draft."""
    out = draft
    applied: list[dict] = []
    dropped: list[dict] = []
    for ch in changes or []:
        frm = str((ch or {}).get("from") or "").strip()
        to = str((ch or {}).get("to") or "").strip()
        if not frm or norm(frm) == norm(to):
            dropped.append({"from": frm, "to": to, "why": "empty or no-op"})
            continue
        loc = locate(out, frm, fuzzy=fuzzy)
        if loc is None:
            dropped.append({"from": frm, "to": to, "why": "not in draft"})
            continue
        found = out[loc[0]:loc[1]]
        repl = _match_case(found, to)
        if all_occurrences:
            pat = re.compile((r"(?<!\w)" if re.match(r"\w", found) else "") + re.escape(found)
                             + (r"(?!\w)" if re.search(r"\w$", found) else ""), re.IGNORECASE)
            out = pat.sub(lambda m: _match_case(m.group(0), to), out)
        else:
            out = out[:loc[0]] + repl + out[loc[1]:]
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out).strip()
        if not to:  # a deletion can leave "table. ." behind
            out = re.sub(r"([,.;:!?])\s*(?:[,.;:!?]\s*)+", r"\1", out).strip()
        applied.append({"from": found, "to": repl})
    return out, applied, dropped


def _drop_unit(draft: str, which: str, unit: str) -> tuple[str, list[dict]]:
    """'remove the last sentence' / 'the first word' without a model."""
    d = draft.strip()
    if not d:
        return draft, []
    if unit == "word":
        ws = _spans(d)
        s = ws[-1] if which == "last" else ws[0]
        return (d[:s[0]] + d[s[1]:]).strip(), [{"from": s[2], "to": ""}]
    parts = re.split(r"(?<=[.!?])\s+|\n+", d)
    parts = [p for p in parts if p.strip()]
    if len(parts) < 2:
        return "", [{"from": d, "to": ""}]
    gone = parts[-1] if which == "last" else parts[0]
    rest = parts[:-1] if which == "last" else parts[1:]
    return " ".join(rest).strip(), [{"from": gone, "to": ""}]


# ---------------------------------------------------------------------------------
# the model (Ollama, schema-constrained JSON, temperature 0)
# ---------------------------------------------------------------------------------
_ADDRESSED_SCHEMA = {"type": "object", "properties": {"addressed": {"type": "boolean"}},
                     "required": ["addressed"]}
_CHANGES_SCHEMA = {"type": "object",
                   "properties": {"changes": {"type": "array", "items": {
                       "type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                       "required": ["from", "to"]}},
                       "reply": {"type": "string"}},
                   "required": ["changes", "reply"]}
_REPLY_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}


def ollama_chat(messages: list, fmt, num_predict: int = 300) -> dict:
    """One schema-constrained call. Raises on any trouble; decide() turns that into a
    model_error and a safe fallback. Gated by the GPU cop like the room router."""
    try:
        import gpu_cop_client
        gpu_cop_client.router_gate(MODEL)
    except ImportError:
        pass
    payload = json.dumps({
        "model": MODEL, "messages": messages, "stream": False, "format": fmt,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    raw = (data.get("message") or {}).get("content") or ""
    return json.loads(raw)


def _judge_addressed(chat, name: str, utterance: str) -> bool:
    system = (
        f"A person dictates a message. An assistant named {name} is listening. Decide whether "
        f"the utterance is spoken TO {name} (an instruction, request or question for {name} to act on "
        f"right now) or merely MENTIONS {name} as part of the message content (something for someone "
        f"else to do that involves {name}, or {name} as the topic).\n"
        f"Examples: 'tell platform to get {name} the data' -> addressed false (a task for platform). "
        f"'we need to change the word ice to nice, {name}' -> addressed true (an edit for {name}). "
        f"'ask {name} to check the colours' -> addressed false. "
        f"'{name} can you hear me' -> addressed true.\n"
        'Answer JSON only: {"addressed": true|false}.')
    out = chat([{"role": "system", "content": system}, {"role": "user", "content": utterance}],
               _ADDRESSED_SCHEMA, 20)
    return bool(out.get("addressed"))


def _model_edit(chat, name: str, draft: str, instruction: str) -> dict:
    system = (
        f"You are {name}. You edit a dictated draft exactly as instructed and nothing more. "
        "Answer JSON only: {\"changes\": [{\"from\": \"exact text copied from the draft\", \"to\": \"replacement\"}], "
        "\"reply\": \"one short spoken confirmation, at most eight words\"}.\n"
        "Rules: 'from' must be copied verbatim from the draft (a few words, the smallest span that "
        "covers the change). 'to' may be an empty string to delete. To insert text, use the "
        "neighbouring words as 'from' and the same words plus the insertion as 'to'. Never change "
        "anything you were not asked to; never restyle. If the instruction cannot be applied to this "
        "draft, answer changes [] and say why in the reply, briefly.")
    user = f"DRAFT:\n{draft}\n\nINSTRUCTION:\n{instruction}"
    return chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                _CHANGES_SCHEMA, 300)


def _model_proofread(chat, name: str, draft: str, names: list[str]) -> dict:
    known = ", ".join(n for n in names if n) or "none"
    system = (
        "You proofread a transcript produced by speech-to-text. Fix ONLY likely transcription slips: "
        "a wrong homophone or near-sounding word (ice -> nice, there -> their, won -> one), a dropped "
        "or doubled small word, or a misheard name from this list: " + known + ". "
        "Keep every other word exactly as it is. Never reword, reorder, shorten or improve the style; "
        "never fix grammar that the speaker could have meant. "
        "Answer JSON only: {\"changes\": [{\"from\": \"exact words from the transcript\", \"to\": \"fixed words\"}], "
        "\"reply\": \"\"}. Each change is a few words. If it reads fine, changes [].")
    return chat([{"role": "system", "content": system}, {"role": "user", "content": "TRANSCRIPT:\n" + draft}],
                _CHANGES_SCHEMA, 300)


def _model_reply(chat, name: str, draft: str, question: str) -> str:
    system = (
        f"You are {name}, a concierge helping a person dictate a message. They asked you something "
        "while composing. Answer in ONE short spoken sentence (at most fifteen words), plainly, no "
        "emoji, no lists. Only state what the draft itself says; you know nothing about the world "
        "outside it. If the question is open or unclear, ask what they want you to do, in one short "
        "sentence. Answer JSON only: {\"reply\": \"...\"}.")
    user = f"DRAFT SO FAR:\n{draft or '(nothing yet)'}\n\nTHEY SAID:\n{question}"
    out = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], _REPLY_SCHEMA, 80)
    reply = str(out.get("reply") or "").strip()
    first = re.split(r"(?<=[.!?])\s+", reply)[0] if reply else ""
    return first if 0 < len(words(first)) <= 20 else reply[:140]


def _proof_ok(frm: str, to: str, draft: str) -> bool:
    """A proofread change must be short, must resemble what it replaces, and must point at
    ONE place: a single word that occurs more than once in the draft is ambiguous (the 3B
    offered 'the' -> 'their' for a draft with six 'the's) and is dropped."""
    fw, tw = words(frm), words(to)
    if not fw or len(fw) > 6 or len(tw) > 6:
        return False
    if abs(len(fw) - len(tw)) > 2:
        return False
    if len(fw) == 1 and len(re.findall(r"(?<!\w)" + re.escape(fw[0]) + r"(?!\w)", norm(draft))) > 1:
        return False
    r = difflib.SequenceMatcher(None, " ".join(fw), " ".join(tw)).ratio()
    return r >= 0.4 or (len(fw) <= 2 and len(tw) <= 2)


def _fix_doubles(draft: str) -> tuple[str, list[dict]]:
    """'the the' -> 'the' for the small-word list, before the model sees the draft."""
    applied: list[dict] = []

    def one(m):
        applied.append({"from": m.group(0), "to": m.group(1)})
        return m.group(1)

    return _DOUBLE_RE.sub(one, draft), applied


def _say_changes(applied: list[dict], prefix: str = "Done") -> str:
    if not applied:
        return "Nothing to change."
    bits = []
    for ch in applied[:3]:
        if ch["to"]:
            bits.append(f"{ch['from']} to {ch['to']}")
        else:
            bits.append(f"removed {ch['from']}")
    more = f", and {len(applied) - 3} more" if len(applied) > 3 else ""
    return f"{prefix}: " + "; ".join(bits) + more + "."


# ---------------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------------
def decide(draft: str, utterance: str, name: str, handle: str = "", chat=ollama_chat) -> dict:
    draft = str(draft or "")
    utterance = str(utterance or "")
    name = str(name or "").strip()
    handle = str(handle or "").strip().lstrip("@").lower()
    out = {"addressed": False, "kind": "content", "draft": draft, "reply": None, "changes": [],
           "text": substitute(utterance, name, handle), "proofread": False, "via": "rule",
           "model_ms": None, "model_error": None}
    out["keep"] = ""
    if not (name or handle):
        return out
    # Sentence by sentence: "I'm wondering what else to do. Olivia, can you look at this?"
    # is content plus a question; the content half comes back as `keep`.
    sents = [s for s in re.split(r"(?<=[.!?])\s+", utterance.strip()) if s.strip()]
    per = [(s, words(s)) for s in sents]
    hit_idx = [(k, _name_hits(ws, name, handle)) for k, (s, ws) in enumerate(per)]
    hit_idx = [(k, h) for k, h in hit_idx if h]
    if not hit_idx:
        return out

    t_model = [0.0]

    def timed_chat(messages, fmt, num_predict=300):
        t0 = time.time()
        try:
            return chat(messages, fmt, num_predict)
        finally:
            t_model[0] += time.time() - t0
            out["via"] = "model"
            out["model_ms"] = int(t_model[0] * 1000)

    # -- addressed or mentioned? (grammar per sentence; the model only for the unclear ones)
    addressed_k = None
    unclear: list[int] = []
    for k, hits in hit_idx:
        uw = per[k][1]
        verdict = None
        for (i, n) in hits:
            before = uw[:i]
            prev = uw[i - 1] if i > 0 else None
            nxt = uw[i + n] if i + n < len(uw) else None
            if all(w in FILLERS for w in before):
                verdict = True
            elif prev in OBJECT_MARKERS:
                verdict = verdict or False
            elif nxt is None:
                verdict = True
            elif nxt in ADDRESS_FOLLOWERS:
                verdict = True
            if verdict:
                break
        if verdict:
            addressed_k = k
            break
        if verdict is None:
            unclear.append(k)
    if addressed_k is None and unclear:
        k = unclear[0]
        try:
            if _judge_addressed(timed_chat, name or handle, per[k][0]):
                addressed_k = k
        except Exception as e:  # noqa: BLE001 — model down / GPU busy: keep the line as content
            out["model_error"] = f"{type(e).__name__}: {e}"[:200]
    out["addressed"] = addressed_k is not None
    if addressed_k is None:
        return out

    # -- what kind of line is it? ----------------------------------------------------
    uw = per[addressed_k][1]
    hits = dict(hit_idx)[addressed_k]
    rest = _strip_fillers(_strip_name(uw, hits))
    instr = " ".join(rest)
    out["instruction"] = instr
    others = " ".join(s for k, (s, _w) in enumerate(per) if k != addressed_k).strip()
    out["keep"] = substitute(others, name, handle)
    if not instr:
        out["kind"] = "question"
        out["reply"] = "Yes?"
        return out

    # proofread
    if _PROOF_RE.search(instr):
        out["kind"] = "edit"
        out["proofread"] = True
        if not draft.strip():
            out["reply"] = "There is nothing in the draft yet."
            return out
        pre, applied0 = _fix_doubles(draft)
        try:
            res = _model_proofread(timed_chat, name or "the concierge", pre, [name, handle])
        except Exception as e:  # noqa: BLE001
            out["model_error"] = f"{type(e).__name__}: {e}"[:200]
            out["draft"], out["changes"] = pre, applied0
            out["reply"] = "I can't proofread right now." if not applied0 else _say_changes(applied0, "Fixed")
            return out
        wanted = [c for c in (res.get("changes") or [])
                  if _proof_ok(str(c.get("from", "")), str(c.get("to", "")), pre)]
        new, applied, _dropped = apply_changes(pre, wanted, all_occurrences=False, fuzzy=False)
        applied = applied0 + applied
        out["draft"] = new
        out["changes"] = applied
        out["reply"] = _say_changes(applied, "Fixed") if applied else "It reads clean to me."
        return out

    # deterministic edits: add X after Y / at the end; change X to Y; remove X; remove the last sentence
    miss = None   # a rule understood the instruction but could not place it: the model gets a go
    m = _INSERT_RE.match(instr)
    if m:
        out["kind"] = "edit"
        what = m.group("what").strip().strip("\"'")
        d = draft.rstrip()
        if m.group("end"):
            sep = "" if not d else (" " if d[-1] in ".!?,;:" else " ")
            new = (d + sep + what).strip()
            out["draft"], out["changes"] = new, [{"from": "", "to": what}]
            out["reply"] = f"Added {what} at the end."
            return out
        if m.group("start"):
            new = (what + " " + d.lstrip()).strip()
            out["draft"], out["changes"] = new, [{"from": "", "to": what}]
            out["reply"] = f"Added {what} at the start."
            return out
        anchor = m.group("anchor").strip().strip("\"'")
        loc = locate(draft, anchor)
        if loc is None:
            out["reply"] = f"I can't find {anchor} in the draft."
            return out
        found = draft[loc[0]:loc[1]]
        after = m.group("where") in ("after", "following", "behind")
        repl = (found + " " + what) if after else (what + " " + found)
        new = draft[:loc[0]] + repl + draft[loc[1]:]
        out["draft"], out["changes"] = new, [{"from": found, "to": repl}]
        out["reply"] = f"Added {what} {'after' if after else 'before'} {found}."
        return out
    m = _DELETE_RE.match(instr)
    if m:
        body = m.group("body").strip()
        lf = _LAST_FIRST_RE.match(body)
        out["kind"] = "edit"
        if lf:
            new, applied = _drop_unit(draft, lf.group("which"), lf.group("unit"))
            out["draft"], out["changes"] = new, applied
            out["reply"] = _say_changes(applied, "Done") if applied else "There is nothing to remove."
            return out
        new, applied, _d = apply_changes(draft, [{"from": body, "to": ""}], all_occurrences=True)
        if applied:
            out["draft"], out["changes"] = new, applied
            out["reply"] = f"Removed {applied[0]['from']}."
            return out
        miss = f"I can't find {body} in the draft."   # "remove the greeting": let the model try
    m = _CHANGE_RE.match(instr)
    if m and not miss:
        body = m.group("body").strip()
        splits = list(_CHANGE_SPLIT.finditer(body))
        if splits:
            out["kind"] = "edit"
            # several "to"s ("change go to the store to walk to the shop"): take the LONGEST
            # left side that is actually in the draft, exact before fuzzy, else the first split
            cands = [(body[:s.start()].strip(), body[s.end():].strip()) for s in splits]
            longest = sorted(cands, key=lambda c: -len(c[0]))
            pick = next((c for c in longest if locate(draft, c[0], fuzzy=False)), None) \
                or next((c for c in longest if locate(draft, c[0], fuzzy=True)), None) \
                or cands[0]
            frm, to = pick
            frm = re.sub(r"^(?:the\s+(?:word|words|phrase)\s+)?[\"']?(.*?)[\"']?$", r"\1", frm).strip()
            to = re.sub(r"^(?:the\s+(?:word|words|phrase)\s+)?[\"']?(.*?)[\"']?[.!?]*$", r"\1", to).strip()
            new, applied, _d = apply_changes(draft, [{"from": frm, "to": to}], all_occurrences=True)
            if applied:
                out["draft"], out["changes"] = new, applied
                out["reply"] = f"Done, {applied[0]['to']}." if applied[0]["to"] else f"Removed {applied[0]['from']}."
                return out
            miss = f"I can't find {frm} in the draft."  # "change the greeting to hi": model's turn

    # deterministic questions
    if _HEAR_RE.search(instr):
        out["kind"] = "question"
        out["reply"] = "Loud and clear."
        return out
    if _COUNT_RE.search(instr):
        out["kind"] = "question"
        n = len(words(draft))
        out["reply"] = f"{n} words so far." if n else "Nothing yet."
        return out
    if _READBACK_RE.search(instr):
        out["kind"] = "question"
        out["reply"] = draft.strip() or "Nothing yet."
        return out
    if _LOOK_RE.search(instr) and not _EDIT_CUE_RE.search(instr):
        # an open "take a look": a read-back and a question, never a free-form claim
        out["kind"] = "question"
        dw = len(words(draft))
        if not dw:
            out["reply"] = "There is nothing in the draft yet. What should I look for?"
        elif dw <= 16:
            out["reply"] = f"I have: {draft.strip()} What should I look for?"
        else:
            out["reply"] = f"I have the draft, {dw} words. What should I look for?"
        return out

    # free-form edit via the model, verified against the draft
    if _EDIT_CUE_RE.search(instr):
        out["kind"] = "edit"
        if not draft.strip():
            out["reply"] = "There is nothing in the draft yet."
            return out
        try:
            res = _model_edit(timed_chat, name or "the concierge", draft, instr)
        except Exception as e:  # noqa: BLE001
            out["model_error"] = f"{type(e).__name__}: {e}"[:200]
            out["reply"] = miss or "I can't make edits right now."
            return out
        new, applied, _dropped = apply_changes(draft, res.get("changes") or [], all_occurrences=False)
        if applied:
            out["draft"], out["changes"] = new, applied
            reply = str(res.get("reply") or "").strip()
            out["reply"] = reply if 0 < len(words(reply)) <= 12 else _say_changes(applied, "Done")
        else:
            reply = str(res.get("reply") or "").strip()
            out["reply"] = miss or (reply if 0 < len(words(reply)) <= 20 else "I couldn't place that change in the draft.")
        return out

    # a question for the concierge
    if _QUESTION_START_RE.match(instr) or utterance.rstrip().endswith("?"):
        out["kind"] = "question"
        try:
            out["reply"] = _model_reply(timed_chat, name or "the concierge", draft, instr) or "Sorry, say that again?"
        except Exception as e:  # noqa: BLE001
            out["model_error"] = f"{type(e).__name__}: {e}"[:200]
            out["reply"] = "I can't answer right now."
        return out

    # addressed, but neither an instruction nor a question: dictation that names the concierge
    out["kind"] = "content"
    out["keep"] = ""   # `text` carries the whole line
    return out


# =================================================================================
# assess(): after every pause, what the Concierge THINKS the person wants, name or not.
# Display only (Eric #8475): intent delegate | act | dictate, with the cue she read, what
# she would do, and the target for a delegate. Rules where the line is obvious; the 3B for
# the rest. A line that names the Concierge goes through decide() and maps edit -> act,
# question -> delegate (to the Concierge), content -> dictate.
# =================================================================================
SEND_PHRASES = ["send that", "send it", "send this", "go ahead and send", "go ahead and send it",
                "that's it send it", "that's it send", "send the message", "submit", "ship it", "send"]
_LOOKUP_RE = re.compile(
    r"^" + _POLITE + r"(?:look up|look into|find out|find me|find|search for|search|pull up|fetch|get me"
    r"|check (?:on|with|whether|if|the|how|what|when)|see if|see whether|figure out|dig up|grab|bring up)\b\s*(?P<rest>.*)$")
_ACT_RE = re.compile(
    r"^" + _POLITE + r"(?:undo|redo|change|replace|swap|switch|delete|remove|drop|take out|cut|strike|scrap|erase"
    r"|insert|add|put|append|prepend|capitali[sz]e|lower\s?case|upper\s?case|rename|rephrase|reword|rewrite|shorten"
    r"|fix|correct|move|go back|start over|clear (?:the draft|that|it|everything)|make (?:it|that|this|the)"
    r"|instead of|never mind|nevermind|scratch that|forget that)\b")
_DICTATE_LEAD_RE = re.compile(
    r"^(?:the|a|an|i|i'm|i'd|i'll|i've|we|we're|we'll|we've|it|it's|this|that|these|those|there|there's|here|he|she|they"
    r"|my|our|your|so|and|but|also|then|when|if|because|for|in|on|at|with|as|to|of|by|first|second|next|finally|one|two|three"
    r"|no|not|yes|ok|okay|um|uh|well|basically|actually|maybe|probably|something|some|all|every|each|most|more|less"
    r"|you|you're|what|which|how|why|where|who|is|are|was|were|should|would|could|will|can|do|does|did|has|have|had)\b")
_UNDO_RE = re.compile(r"^" + _POLITE + r"(?:undo|go back|scratch that|never ?mind|forget that|start over|redo)\b")

_ASSESS_SCHEMA = {"type": "object",
                  "properties": {"intent": {"type": "string", "enum": ["delegate", "act", "dictate"]},
                                 "why": {"type": "string"}, "would": {"type": "string"},
                                 "target": {"type": ["string", "null"]}},
                  "required": ["intent", "why", "would", "target"]}


def detect_send(text: str, phrases=SEND_PHRASES):
    """Python twin of voice-test-turns.mjs detectSend: the line IS or ENDS WITH a send phrase."""
    t = norm(text)
    if not t:
        return None
    for p in sorted((norm(x) for x in phrases), key=len, reverse=True):
        if t == p:
            return {"phrase": p, "rest": ""}
        if t.endswith(" " + p):
            return {"phrase": p, "rest": t[:-len(p)].strip()}
    return None


def _clip(text: str, n: int) -> str:
    ws = str(text or "").split()
    return " ".join(ws[:n]) + ("…" if len(ws) > n else "")


def _would_from_changes(d: dict) -> str:
    ch = d.get("changes") or []
    if not ch:
        return _clip(d.get("reply") or "edit the draft", 15)
    c = ch[0]
    if not c["to"]:
        w = f"remove '{_clip(c['from'], 6)}'"
    elif not c["from"]:
        w = f"add '{_clip(c['to'], 6)}'"
    else:
        w = f"change '{_clip(c['from'], 5)}' to '{_clip(c['to'], 5)}'"
    if len(ch) > 1:
        w += f" and {len(ch) - 1} more"
    return w


def _model_assess(chat, name: str, draft: str, utterance: str) -> dict:
    system = (
        f"You are {name or 'the concierge'}, listening while a person dictates a message in a workroom. "
        "After each pause you say what you think they want. Classify the NEWEST line:\n"
        "- delegate: they want data looked up, or a message delivered to a person or agent.\n"
        "- act: they want words changed in the draft or in something already said (edit, undo, remove).\n"
        "- dictate: no action; they are composing the message itself (specs, descriptions, opinions).\n"
        "Answer JSON only: {\"intent\": \"delegate\"|\"act\"|\"dictate\", \"why\": \"the cue you read, at most 12 words\", "
        "\"would\": \"what you would do, at most 15 words, e.g. change 'blue' to 'green' / ask platform for the data / keep it in the draft\", "
        "\"target\": \"who to reach for a delegate, else null\"}. When unsure, dictate.")
    user = f"DRAFT SO FAR:\n{draft or '(nothing yet)'}\n\nNEWEST LINE:\n{utterance}"
    return chat([{"role": "system", "content": system}, {"role": "user", "content": user}], _ASSESS_SCHEMA, 120)


def assess(draft: str, utterance: str, name: str = "", handle: str = "", chat=ollama_chat) -> dict:
    draft = str(draft or "")
    utterance = str(utterance or "")
    name = str(name or "").strip()
    out = {"intent": "dictate", "why": "", "would": "keep it in the draft", "target": None,
           "via": "rule", "model_ms": None, "model_error": None}
    t = norm(utterance)
    if not t:
        return out
    t_model = [0.0]

    def timed_chat(messages, fmt, num_predict=120):
        t0 = time.time()
        try:
            return chat(messages, fmt, num_predict)
        finally:
            t_model[0] += time.time() - t0
            out["via"] = "model"
            out["model_ms"] = int(t_model[0] * 1000)

    # 1. the line names the Concierge: concierge-turn's classifier is the cheap path
    if name or handle:
        d = decide(draft, utterance, name, handle, chat=timed_chat)
        out["model_error"] = d.get("model_error")
        if d["addressed"]:
            out["concierge"] = d["kind"]
            if d["kind"] == "edit":
                out["intent"] = "act"
                out["why"] = f"{name or 'the concierge'} was asked to " + ("proofread" if d.get("proofread") else "edit")
                out["would"] = _would_from_changes(d)
                return out
            if d["kind"] == "question":
                out["intent"] = "delegate"
                out["target"] = name or handle
                out["why"] = f"a question for {name or handle}"
                out["would"] = _clip(d.get("reply") or "answer", 15)
                return out
            out["why"] = f"{name or handle} named, no instruction"
            return out

    # 2. send phrase -> deliver the draft
    s = detect_send(utterance)
    if s:
        out["intent"] = "delegate"
        out["why"] = f"send phrase '{s['phrase']}'"
        try:
            import room_agent
            who = room_agent.extract_addressee(draft) or (room_agent.extract_addressee(s["rest"]) if s["rest"] else None)
        except Exception:  # noqa: BLE001
            who = None
        m = re.search(r"@([a-z][a-z0-9_-]{1,31})", (draft + " " + utterance).lower())
        out["target"] = who or (m.group(1) if m else None)
        out["would"] = f"send the draft to {out['target']}" if out["target"] else "send the draft"
        return out

    # 3. "tell X / ask X / let X know" -> deliver a message
    try:
        import room_agent
        who = room_agent.extract_addressee(utterance)
    except Exception:  # noqa: BLE001
        who = None
    if who:
        out["intent"] = "delegate"
        out["target"] = who
        out["why"] = f"message for {who}"
        out["would"] = f"deliver it to {who}"
        return out

    # 4. "look up / find / check whether" -> a lookup
    m = _LOOKUP_RE.match(t)
    if m:
        out["intent"] = "delegate"
        out["why"] = "a lookup request"
        out["would"] = f"look up {_clip(m.group('rest'), 8)}" if m.group("rest") else "look it up"
        return out

    # 5. edit verbs up front -> act (decide() with the name prefixed gives the concrete change)
    if _ACT_RE.match(t):
        out["intent"] = "act"
        if _UNDO_RE.match(t):
            out["why"] = "an undo"
            out["would"] = "undo the last change"
            return out
        nm = name or "Concierge"
        d = decide(draft, nm + ", " + utterance, nm, handle, chat=timed_chat)
        out["model_error"] = d.get("model_error")
        out["why"] = "an edit instruction"
        out["would"] = _would_from_changes(d)
        return out

    # 6. plain prose -> dictate, no model
    if _DICTATE_LEAD_RE.match(t) or len(t.split()) >= 14:
        out["why"] = "plain dictation"
        return out

    # 7. the rest: the model decides, dictate when it can't
    try:
        res = _model_assess(timed_chat, name, draft, utterance)
        intent = str(res.get("intent") or "dictate")
        out["intent"] = intent if intent in ("delegate", "act", "dictate") else "dictate"
        out["why"] = _clip(res.get("why") or "", 12)
        out["would"] = _clip(res.get("would") or "", 15) or ("keep it in the draft" if out["intent"] == "dictate" else "")
        tgt = res.get("target")
        if tgt and out["intent"] == "delegate" and str(tgt).strip().lower() not in ("null", "none", ""):
            out["target"] = str(tgt).strip()
    except Exception as e:  # noqa: BLE001
        out["model_error"] = f"{type(e).__name__}: {e}"[:200]
        out["why"] = "unclear; model unavailable"
    return out
