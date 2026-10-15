"""Scenarios for concierge_turn.assess with a FAKE model (no Ollama). Run:
    python tools/concierge_assess_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concierge_turn as ct  # noqa: E402

fails = 0
calls = []


def check(name, cond, detail=""):
    global fails
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'  ' + str(detail) if detail else ''}")
    if not cond:
        fails += 1


def fake_chat(messages, fmt, num_predict=120):
    system = messages[0]["content"]
    user = messages[-1]["content"]
    calls.append(system[:30])
    if "After each pause" in system:
        if "igloo" in user:
            return {"intent": "delegate", "why": "asks for results", "would": "ask concierge-ops for igloo results", "target": "concierge-ops"}
        return {"intent": "dictate", "why": "sounds like spec text", "would": "keep it in the draft", "target": None}
    if "edit a dictated draft" in system:
        return {"changes": [{"from": "quick", "to": "fast"}], "reply": "Changed quick to fast."}
    if '"addressed"' in system:
        return {"addressed": False}
    return {"reply": "ok"}


def boom(messages, fmt, num_predict=120):
    raise RuntimeError("ollama down")


D = "I want three bananas on the table. Make it quick."


def a(utt, draft=D, chat=fake_chat, name="Olivia", handle="olivia"):
    return ct.assess(draft, utt, name, handle, chat=chat)


print("rules: delegate")
r = a("send that")
check("send phrase, no target", r["intent"] == "delegate" and r["target"] is None and r["would"] == "send the draft" and r["via"] == "rule", r)
r = a("ok send it", draft="tell aurora the button is wrong")
check("send phrase, target from the draft", r["intent"] == "delegate" and r["target"] == "aurora" and r["would"] == "send the draft to aurora", r)
r = a("ship it", draft="@platform the export is missing timestamps")
check("send phrase, @handle in the draft", r["target"] == "platform", r)
r = a("tell the lead the build is green")
check("tell X", r["intent"] == "delegate" and r["target"] == "lead" and r["would"] == "deliver it to lead", r)
r = a("can you ask aurora whether the mockup passed")
check("ask X", r["intent"] == "delegate" and r["target"] == "aurora", r)
r = a("look up the igloo results from yesterday")
check("look up", r["intent"] == "delegate" and r["target"] is None and r["would"].startswith("look up the igloo results"), r)
r = a("find out when the export runs")
check("find out", r["intent"] == "delegate")

print("rules: act")
r = a("change bananas to teddy bears")
check("change X to Y without the name", r["intent"] == "act" and r["would"] == "change 'bananas' to 'teddy bears'" and r["via"] == "rule", r)
r = a("delete the last sentence")
check("delete the last sentence", r["intent"] == "act" and r["would"] == "remove 'Make it quick.'", r)
r = a("undo that")
check("undo", r["intent"] == "act" and r["would"] == "undo the last change" and r["via"] == "rule", r)
r = a("scratch that")
check("scratch that", r["intent"] == "act" and r["would"] == "undo the last change")
r = a("make the second sentence say make it fast")
check("free-form edit -> model change", r["intent"] == "act" and r["would"] == "change 'quick' to 'fast'" and r["via"] == "model", r)
r = a("add please at the end")
check("add at the end", r["intent"] == "act" and r["would"] == "add 'please'", r)
r = a("change unicorns to horses", chat=boom)
check("act with no placeable change and model down", r["intent"] == "act" and "can't find unicorns" in r["would"], r)

print("rules: dictate")
r = a("the button should be green and sit under the logo")
check("plain prose", r["intent"] == "dictate" and r["why"] == "plain dictation" and r["via"] == "rule", r)
r = a("we need the export to carry timestamps and the font is too small")
check("prose starting with we", r["intent"] == "dictate" and r["via"] == "rule")
r = a("please note the colours are approximate but the layout is final so keep the spacing exactly as drawn")
check("long line is dictation whatever it starts with", r["intent"] == "dictate" and r["via"] == "rule")

print("named lines go through concierge-turn")
r = a("Olivia, change bananas to apples")
check("edit -> act", r["intent"] == "act" and r["concierge"] == "edit" and r["would"] == "change 'bananas' to 'apples'", r)
r = a("Olivia can you hear me")
check("question -> delegate to Olivia", r["intent"] == "delegate" and r["target"] == "Olivia" and r["would"] == "Loud and clear.", r)
r = a("ok Olivia this is the spec for the button")
check("addressed content -> dictate", r["intent"] == "dictate" and r["concierge"] == "content")
r = a("tell platform to get Olivia the data")
check("mentioned -> tell X rule", r["intent"] == "delegate" and r["target"] == "platform", r)
r = a("can you proofread this Olivia", draft="it would be ice to have the the bananas")
check("proofread -> act with the fixes", r["intent"] == "act" and r["why"].endswith("proofread") and r["would"].startswith("change 'the the' to 'the'"), r)

print("model for the rest")
calls.clear()
r = a("grab the igloo results for me")
check("lookup rule beats the model", r["intent"] == "delegate" and r["via"] == "rule" and not calls)
r = a("igloo results please")
check("unclear -> model delegate with target", r["intent"] == "delegate" and r["target"] == "concierge-ops" and r["via"] == "model" and r["would"] == "ask concierge-ops for igloo results", r)
r = a("mostly green with a touch of teal")
check("unclear short line -> model dictate", r["intent"] == "dictate" and r["via"] == "model" and r["why"] == "sounds like spec text", r)
r = a("mostly green with a touch of teal", chat=boom)
check("model down -> dictate with error", r["intent"] == "dictate" and "ollama down" in r["model_error"] and r["would"] == "keep it in the draft", r)
r = a("")
check("empty -> dictate, nothing", r["intent"] == "dictate" and r["why"] == "")

print("FAILURES: %d" % fails if fails else "ALL PASS")
sys.exit(1 if fails else 0)
