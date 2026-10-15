"""Scenarios for concierge_turn.decide with a FAKE model (no Ollama). Run:
    python tools/concierge_turn_test.py
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


def fake_chat(messages, fmt, num_predict=300):
    """Answers by the system prompt's job: judge / edit / proofread / reply."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    calls.append(system[:40])
    if '"addressed"' in system:
        return {"addressed": "we should wait" in user}
    if "proofread a transcript" in system:
        return {"changes": [{"from": "ice", "to": "nice"}, {"from": "the", "to": "their"},
                            {"from": "I want three bananas on the tabel", "to": "Three bananas, please, on the table"},
                            {"from": "not here", "to": "nope"}], "reply": ""}
    if "edit a dictated draft" in system:
        if "no such" in user:
            return {"changes": [], "reply": "I don't see that in the draft."}
        if "unicorns" in user:
            return {"changes": [], "reply": "There are no unicorns in the draft."}
        if "drop the second" in user:
            return {"changes": [{"from": "Make it quick", "to": ""}], "reply": "Dropped it."}
        return {"changes": [{"from": "quick", "to": "fast"}], "reply": "Changed quick to fast."}
    if "concierge helping" in system:
        return {"reply": "It is about bananas on a table."}
    return {}


def boom(messages, fmt, num_predict=300):
    raise RuntimeError("ollama down")


D = "I want three bananas on the table. Make it quick."
N, H = "Olivia", "olivia"


def dec(utt, draft=D, chat=fake_chat, name=N, handle=H):
    return ct.decide(draft, utt, name, handle, chat=chat)


print("substitution (spec rules 1/2/4/5)")
check("name -> @handle anywhere", ct.substitute("Olivia, ask Olivia. olivia!", N, H) == "@olivia, ask @olivia. @olivia!")
check("possessive and longer words untouched", ct.substitute("Olivia's desk and Olivian", N, H) == "Olivia's desk and Olivian")
check("'at olivia' -> @olivia", ct.substitute("at olivia change it", N, H) == "@olivia change it")
check("no handle = as spoken", ct.substitute("Olivia hi", N, "") == "Olivia hi")
check("already @olivia stays single @", ct.substitute("@olivia hi", N, H) == "@olivia hi")

print("addressed: grammar")
r = dec("Olivia, can you change bananas to teddy bears?")
check("start vocative + change X to Y", r["addressed"] and r["kind"] == "edit" and r["draft"] == "I want three teddy bears on the table. Make it quick.", r)
check("change list as found", r["changes"] == [{"from": "bananas", "to": "teddy bears"}], r["changes"])
check("reply", r["reply"] == "Done, teddy bears.", r["reply"])
check("no model call", r["via"] == "rule" and r["model_ms"] is None)
r = dec("we need to change ice to nice, Olivia", draft="it would be ice to have that")
check("end vocative + change", r["addressed"] and r["draft"] == "it would be nice to have that", r)
r = dec("tell platform to get Olivia the data")
check("object marker -> mentioned", not r["addressed"] and r["kind"] == "content" and r["text"] == "tell platform to get @olivia the data", r)
r = dec("ask Olivia to check the colors")
check("'ask Olivia' -> mentioned", not r["addressed"] and r["kind"] == "content")
r = dec("send the data to Olivia")
check("'to Olivia' at the end -> mentioned", not r["addressed"])
r = dec("hey Olivia can you hear me")
check("'hey' filler then name -> addressed question", r["addressed"] and r["kind"] == "question" and r["reply"] == "Loud and clear.", r)
r = dec("ok Olivia this is the spec for the button")
check("addressed statement -> content, draft kept", r["addressed"] and r["kind"] == "content" and r["draft"] == D and r["reply"] is None, r)
r = dec("@olivia change the word quick to fast")
check("@handle form addressed", r["addressed"] and r["draft"].endswith("Make it fast."), r["draft"])
r = dec("at olivia replace table with desk")
check("'at olivia' form + replace/with", r["addressed"] and "on the desk" in r["draft"], r["draft"])
r = dec("no name in this line at all")
check("no name -> content", not r["addressed"] and r["kind"] == "content" and r["text"] == "no name in this line at all")
r = dec("Olivia's version is better")
check("possessive is not a hit", not r["addressed"])

print("addressed: model for the ambiguous middle")
calls.clear()
r = dec("I think Olivia we should wait on this")
check("model says addressed -> content kept", r["addressed"] and r["kind"] == "content" and r["via"] == "model" and len(calls) == 1, r)
r = dec("I think Olivia likes this one")
check("model says mentioned", not r["addressed"] and r["via"] == "model")
r = dec("I think Olivia we should wait on this", chat=boom)
check("model down -> content, error noted", not r["addressed"] and r["kind"] == "content" and "ollama down" in r["model_error"], r)

print("deterministic edits")
r = dec("Olivia, remove quick")
check("remove X", r["kind"] == "edit" and r["draft"] == "I want three bananas on the table. Make it." and r["changes"] == [{"from": "quick", "to": ""}], r)
r = dec("Olivia delete the last sentence")
check("remove the last sentence", r["draft"] == "I want three bananas on the table." and r["changes"][0]["from"] == "Make it quick.", r)
r = dec("Olivia remove the first word")
check("remove the first word", r["draft"].startswith("want three"), r["draft"])
r = dec("Olivia, change the word tabel to desk", draft="put it on the tabel now")
check("exact match on a slip", r["draft"] == "put it on the desk now")
r = dec("Olivia change table to desk", draft="put it on the tabel now")
check("fuzzy match finds the slip", r["draft"] == "put it on the desk now" and r["changes"] == [{"from": "tabel", "to": "desk"}], r)
r = dec("Olivia change Bananas to apples")
check("case kept from the draft", r["draft"].startswith("I want three apples"), r["draft"])
r = dec("Olivia change bananas to apples", draft="Bananas are great. I like bananas.")
check("all occurrences, case-matched", r["draft"] == "Apples are great. I like apples.", r["draft"])
r = dec("Olivia change go to the store to walk to the shop", draft="we go to the store daily")
check("several 'to's: split chosen by the draft", r["draft"] == "we walk to the shop daily", r["draft"])
r = dec("Olivia change unicorns to horses")
check("not found -> no change, says so", r["kind"] == "edit" and r["changes"] == [] and r["draft"] == D and "can't find unicorns" in r["reply"], r["reply"])
r = dec("Olivia, could you please switch bananas for pears?")
check("polite prefix + switch/for", r["draft"].startswith("I want three pears"), r["draft"])

print("deterministic inserts")
r = dec("Olivia, add the word please after table")
check("add X after Y", r["kind"] == "edit" and r["draft"] == "I want three bananas on the table please. Make it quick." and r["changes"] == [{"from": "table", "to": "table please"}] and r["via"] == "rule", r)
r = dec("Olivia insert fresh before bananas")
check("insert X before Y", r["draft"].startswith("I want three fresh bananas"), r["draft"])
r = dec("Olivia add thanks at the end")
check("add X at the end", r["draft"] == D + " thanks" and r["changes"] == [{"from": "", "to": "thanks"}], r["draft"])
r = dec("Olivia put hi team at the start")
check("add X at the start", r["draft"].startswith("hi team I want"), r["draft"])
r = dec("Olivia add please after unicorns")
check("anchor missing -> says so", r["changes"] == [] and "can't find unicorns" in r["reply"])

print("deterministic questions")
r = dec("Olivia how many words is that")
check("word count", r["kind"] == "question" and r["reply"] == "10 words so far.", r["reply"])
r = dec("Olivia read it back")
check("read back = the draft", r["reply"] == D)
r = dec("Olivia?")
check("name alone", r["kind"] == "question" and r["reply"] == "Yes?")

print("model edit, verified against the draft")
r = dec("Olivia, make the second sentence say make it fast")
check("model change applied", r["kind"] == "edit" and r["via"] == "model" and r["draft"] == "I want three bananas on the table. Make it fast." and r["reply"] == "Changed quick to fast.", r)
r = dec("Olivia, rephrase the no such thing part")
check("model declines -> its reason", r["kind"] == "edit" and r["changes"] == [] and r["reply"] == "I don't see that in the draft.", r)
r = dec("Olivia, drop the second sentence please")
check("model deletion leaves clean punctuation", r["draft"] == "I want three bananas on the table.", r["draft"])
r = dec("Olivia, rephrase the second sentence", chat=boom)
check("model down -> edit with no change", r["kind"] == "edit" and r["draft"] == D and r["reply"] == "I can't make edits right now.")
r = dec("Olivia, rephrase the second sentence", draft="")
check("empty draft", r["reply"] == "There is nothing in the draft yet.")

print("proofread: STT slips only")
r = dec("can you proof read this, Olivia?", draft="it would be ice to have the the bananas. I want three bananas on the tabel")
check("kind edit + proofread flag", r["kind"] == "edit" and r["proofread"] is True)
check("doubled word fixed first; slip applied; ambiguous 'the', rewrite and unknown dropped",
      r["draft"] == "it would be nice to have the bananas. I want three bananas on the tabel" and
      r["changes"] == [{"from": "the the", "to": "the"}, {"from": "ice", "to": "nice"}], r)
check("reply lists the fixes", r["reply"].startswith("Fixed: the the to the; ice to nice"), r["reply"])
r = dec("Olivia proofread it", draft="all good here")
check("nothing to fix -> clean", r["changes"] == [] and r["reply"] == "It reads clean to me.")

print("mixed lines -> keep")
r = dec("I'm wondering if there's anything else I can do there. Olivia, can you take a look at this?")
check("content sentence kept, question answered", r["kind"] == "question" and r["keep"] == "I'm wondering if there's anything else I can do there." and r["draft"] == D, r)
r = dec("Olivia, change bananas to teddy bears. Also ask Olivia for the cake order.")
check("edit applied, other sentence kept with @handle", r["kind"] == "edit" and r["draft"].startswith("I want three teddy bears") and r["keep"] == "Also ask @olivia for the cake order.", r)
r = dec("tell platform to get Olivia the data")
check("not addressed: keep empty, text carries it", r["keep"] == "" and r["text"].startswith("tell platform to get @olivia"))
r = dec("ok Olivia this is the spec for the button")
check("addressed content: keep empty", r["kind"] == "content" and r["keep"] == "")
r = dec("Olivia, remove unicorns")
check("rule miss -> model -> still not found -> rule's reply", r["kind"] == "edit" and r["changes"] == [] and r["reply"] == "I can't find unicorns in the draft." and r["via"] == "model", r)

print("open 'take a look' -> read-back + question, no model")
r = dec("I'm wondering if there's anything else I can do there. Olivia, can you take a look at this?", draft="I want three bananas on the table.")
check("short draft read back", r["kind"] == "question" and r["via"] == "rule" and r["reply"] == "I have: I want three bananas on the table. What should I look for?", r["reply"])
r = dec("Olivia what do you think", draft=" ".join(["word"] * 30))
check("long draft -> count", r["reply"] == "I have the draft, 30 words. What should I look for?", r["reply"])
r = dec("Olivia have a look", draft="")
check("empty draft", r["reply"].startswith("There is nothing in the draft yet."))
r = dec("Olivia take a look and change bananas to apples")
check("look + edit cue stays an edit", r["kind"] == "edit", r)

print("model question")
r = dec("Olivia what is the draft about")
check("short model reply", r["kind"] == "question" and r["reply"] == "It is about bananas on a table." and r["draft"] == D)

print("FAILURES: %d" % fails if fails else "ALL PASS")
sys.exit(1 if fails else 0)
