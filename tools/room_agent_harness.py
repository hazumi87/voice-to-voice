"""Room-agent accuracy harness (room-voice §6.5, vk-1819).

Scores room_agent.py against tools/room_agent_cases.json and writes
working/harness/latest.json (+ previous.json) plus a markdown summary for the room.

  .venv\\Scripts\\python.exe tools\\room_agent_harness.py --model llama3.2:3b
  .venv\\Scripts\\python.exe tools\\room_agent_harness.py --model llama3.1:8b --no-stt

STT path: every case flagged stt:true is spoken by the running v2v server (POST /synthesize,
OmniVoice) and transcribed by the same server's faster-whisper (POST /api/stt), so the router
sees what the STT actually produces for names like "aurora" and "vrpc". This is the Crawl
proxy for "real Dot recordings": same STT model, same text, no room acoustics. Real Dot
captures replace these at the live gate (G).

Gates (§6.5): >= 90% correct action, 100% of forbidden -> lead, p90 latency <= 1.5 s,
parse-failure rate reported. The non-sequitur filter and the deterministic intents are
tested separately and must be 100%.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import room_agent  # noqa: E402

CASES_PATH = os.path.join(HERE, "room_agent_cases.json")
OUT_DIR = os.path.join(ROOT, "working", "harness")


def _post_json(url: str, body: dict, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _post_wav(url: str, wav: bytes, timeout: float = 120.0) -> dict:
    boundary = "----v2vharness"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"u.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def tts_stt(server: str, text: str, voice: str, cache_dir: str) -> tuple[str, str]:
    """Speak `text` with OmniVoice and transcribe it with faster-whisper on the server.
    Returns (transcript, wav_path). WAVs are cached by text so re-runs cost nothing."""
    os.makedirs(cache_dir, exist_ok=True)
    key = "".join(c if c.isalnum() else "_" for c in text.lower())[:80]
    wav_path = os.path.join(cache_dir, key + ".wav")
    if not os.path.exists(wav_path):
        wav = _post_json(server + "/synthesize", {"text": text, "voice": voice, "speed": 1.0})
        with open(wav_path, "wb") as f:
            f.write(wav)
    with open(wav_path, "rb") as f:
        wav = f.read()
    out = _post_wav(server + "/api/stt", wav)
    return (out.get("text") or "").strip(), wav_path


def check_case(case: dict, dec: dict) -> tuple[bool, str]:
    exp = case["expect"]
    allowed = exp.get("actions") or [exp["action"]]
    if dec["action"] not in allowed:
        return False, f"action {dec['action']} not in {allowed}"
    if dec["action"] == "relay" and exp.get("to_contains"):
        if not dec.get("to") or exp["to_contains"].lower() not in dec["to"].lower():
            return False, f"to={dec.get('to')!r} lacks {exp['to_contains']!r}"
    if dec["action"] == "relay" and not dec.get("to"):
        return False, "relay without a target"
    return True, ""


def run(args) -> dict:
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    ctx = spec["context"]
    contexts = spec.get("contexts") or {"default": ctx}
    results = []
    # -- deterministic layers first: these must be perfect and cost nothing --------------
    ns_fail = []
    for t in spec["non_sequiturs"]["ignore"]:
        if not room_agent.is_non_sequitur(t):
            ns_fail.append(("should ignore", t))
    for t in spec["non_sequiturs"].get("ignore_unverified", []):
        if not room_agent.is_non_sequitur(t):
            ns_fail.append(("should ignore (unverified)", t))
    for t in spec["non_sequiturs"]["keep"]:
        if room_agent.is_non_sequitur(t):
            ns_fail.append(("should keep", t))
    intent_fail = []
    for it in spec["intents"]:
        got = room_agent.parse_intent(it["text"])
        exp = it["expect"]
        if exp["intent"] is None:
            if got is not None:
                intent_fail.append((it["text"], got))
        else:
            if not got or got.get("intent") != exp["intent"]:
                intent_fail.append((it["text"], got))
            elif exp.get("target") and exp["target"] not in (got.get("target") or ""):
                intent_fail.append((it["text"], got))

    det_fail = []
    for it in spec.get("deterministic", []):
        ans = room_agent.answer_room_question(it["text"], ctx) or ""
        low = ans.lower()
        if any(w.lower() not in low for w in it.get("must", [])) or any(w.lower() in low for w in it.get("must_not", [])):
            det_fail.append((it["text"], ans))

    # -- the model ---------------------------------------------------------------------
    cases = spec["cases"]
    if args.only:
        cases = [c for c in cases if c["class"] in args.only.split(",")]
    stt_done = 0
    for case in cases:
        text = case["text"]
        heard = None
        wav = None
        if case.get("stt") and not args.no_stt:
            try:
                heard, wav = tts_stt(args.server, text, args.voice,
                                     os.path.join(OUT_DIR, "stt_cache"))
                stt_done += 1
            except Exception as e:  # noqa: BLE001
                heard = None
                print(f"  [stt] {case['id']}: failed {e!r}; routing the typed text", flush=True)
        routed_text = heard if heard else text
        case_ctx = contexts.get(case.get("context") or "default", ctx)
        dec = room_agent.route(routed_text, case_ctx, model=args.model)
        ok, why = check_case(case, dec)
        if ok and case.get("class") == "unverified":
            # No identity talk: the speaker gate is off for this room by the room's choice.
            blob = " ".join(str(v) for v in (dec.get("answer"), dec.get("raw")) if v).lower()
            if any(w in blob for w in ("who are you", "who is this", "identify", "not sure that", "recogni", "verify", "unverified")):
                ok, why = False, "identity talk in an unverified-room answer"
            det = room_agent.answer_room_question(routed_text, case_ctx)
            if det and any(w in det.lower() for w in ("who are you", "identify", "verif", "recogni")):
                ok, why = False, "identity talk in the deterministic answer"
        row = {"id": case["id"], "class": case["class"], "text": text, "heard": heard,
               "routed_text": routed_text, "expect": case["expect"],
               "action": dec["action"], "to": dec.get("to"), "answer": dec.get("answer"),
               "confidence": dec.get("confidence"), "parse_ok": dec.get("parse_ok"),
               "note": dec.get("note"), "latency_s": dec.get("latency_s"),
               "eval_count": dec.get("eval_count"), "ok": ok, "why": why}
        results.append(row)
        mark = "PASS" if ok else "FAIL"
        h = f"  heard: {heard!r}" if heard and heard.lower() != text.lower() else ""
        print(f"[{mark}] {case['id']} {case['class']:15} {dec['latency_s']:5.2f}s -> {dec['action']}"
              f"{(' to=' + repr(dec.get('to'))) if dec.get('to') else ''}"
              f"{'' if ok else '  (' + why + ')'}{h}", flush=True)

    # -- metrics -----------------------------------------------------------------------
    n = len(results)
    correct = sum(1 for r in results if r["ok"])
    by_class = {}
    for r in results:
        c = by_class.setdefault(r["class"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += 1 if r["ok"] else 0
    heldout = [r for r in results if r["class"] == "heldout"]
    heldout_ok = sum(1 for r in heldout if r["ok"])
    tuned = [r for r in results if r["class"] != "heldout"]
    tuned_ok = sum(1 for r in tuned if r["ok"])
    forbidden = [r for r in results if r["class"] == "forbidden"]
    forbidden_to_lead = sum(1 for r in forbidden if r["action"] == "lead")
    lats = sorted(r["latency_s"] for r in results if r["latency_s"] is not None)
    p50 = statistics.median(lats) if lats else None
    p90 = lats[int(0.9 * (len(lats) - 1))] if lats else None
    parse_fail = sum(1 for r in results if not r["parse_ok"])
    summary = {
        "model": args.model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None,
        "by_class": {k: {**v, "accuracy": round(v["ok"] / v["n"], 3)} for k, v in by_class.items()},
        "tuned_n": len(tuned), "tuned_correct": tuned_ok,
        "heldout_n": len(heldout), "heldout_correct": heldout_ok,
        "heldout_accuracy": round(heldout_ok / len(heldout), 4) if heldout else None,
        "forbidden_n": len(forbidden), "forbidden_to_lead": forbidden_to_lead,
        "forbidden_rate": round(forbidden_to_lead / len(forbidden), 4) if forbidden else None,
        "stt_cases": stt_done, "parse_failures": parse_fail,
        "latency_p50_s": p50, "latency_p90_s": p90,
        "non_sequitur_failures": ns_fail, "intent_failures": intent_fail,
        "deterministic_failures": det_fail,
        "gates": {
            "accuracy_ge_90": (correct / n >= 0.90) if n else False,
            "forbidden_100": (forbidden_to_lead == len(forbidden)) if forbidden else False,
            "p90_le_1_5s": (p90 is not None and p90 <= 1.5),
            "non_sequitur_100": not ns_fail,
            "intents_100": not intent_fail,
            "deterministic_100": not det_fail,
        },
        "gpu_note": args.note,
    }
    return {"summary": summary, "results": results}


def write_out(report: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    latest = os.path.join(OUT_DIR, "latest.json")
    prev = os.path.join(OUT_DIR, "previous.json")
    if os.path.exists(latest):
        os.replace(latest, prev)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return latest


def markdown(report: dict) -> str:
    s = report["summary"]
    g = s["gates"]
    lines = [f"Harness {s['model']} @ {s['ts']}: {s['correct']}/{s['n']} = {s['accuracy']*100:.1f}% "
             f"(gate >=90%: {'PASS' if g['accuracy_ge_90'] else 'FAIL'})",
             f"tuned set {s['tuned_correct']}/{s['tuned_n']}; HELD-OUT (never tuned on) "
             f"{s['heldout_correct']}/{s['heldout_n']}",
             f"forbidden -> lead: {s['forbidden_to_lead']}/{s['forbidden_n']} "
             f"({'PASS' if g['forbidden_100'] else 'FAIL'})",
             f"latency p50 {s['latency_p50_s']:.2f}s, p90 {s['latency_p90_s']:.2f}s "
             f"(gate <=1.5s: {'PASS' if g['p90_le_1_5s'] else 'FAIL'}); "
             f"parse failures {s['parse_failures']}; STT cases {s['stt_cases']}",
             f"non-sequitur filter: {'PASS' if g['non_sequitur_100'] else 'FAIL ' + str(s['non_sequitur_failures'])}; "
             f"intents: {'PASS' if g['intents_100'] else 'FAIL ' + str(s['intent_failures'])}; "
             f"deterministic answers: {'PASS' if g['deterministic_100'] else 'FAIL ' + str(s['deterministic_failures'])}",
             "per class: " + ", ".join(f"{k} {v['ok']}/{v['n']}" for k, v in s["by_class"].items())]
    fails = [r for r in report["results"] if not r["ok"]]
    if fails:
        lines.append("failures: " + "; ".join(
            f"{r['id']} '{r['routed_text']}' -> {r['action']}{(' to=' + str(r['to'])) if r['to'] else ''}"
            for r in fails))
    if s.get("gpu_note"):
        lines.append("note: " + s["gpu_note"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=room_agent.ROUTER_MODEL)
    ap.add_argument("--server", default="http://127.0.0.1:8221")
    ap.add_argument("--voice", default="f_us", help="OmniVoice voice id for the STT path")
    ap.add_argument("--no-stt", action="store_true", help="route the typed text only")
    ap.add_argument("--only", default="", help="comma list of classes to run")
    ap.add_argument("--note", default="", help="free text recorded with the run (GPU state etc.)")
    args = ap.parse_args()
    report = run(args)
    path = write_out(report)
    print()
    print(markdown(report))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
