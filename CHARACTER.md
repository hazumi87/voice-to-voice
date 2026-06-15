# Verbal Character Model

This document is the canonical spec for the **character** asset type in voice-to-voice.
It covers the JSON schema, bio/style split, how consumers assemble prompts, and token budgets.

---

## 1. Asset-Library Progression

```
audio  →  voice  →  character
```

| Level     | Contains                                                          | Usable in         |
|-----------|-------------------------------------------------------------------|-------------------|
| audio     | Raw sample recording                                              | —                 |
| voice     | Cleaned audio + STT bundle                                        | OmniVoice TTS     |
| character | voice + personality (bio + style) + tuning + referenceable name   | reword, chat      |

**audio** is just a recording. **voice** attaches a title and makes the sample usable in
OmniVoice as a TTS voice reference; it carries no personality. **character** is the full
verbal persona: it wraps a voice reference with identity text, tuning parameters, and the
name consumers use when addressing or imitating the person.

### Visual stream (out of scope here)

A parallel stream handles phonetics, lip-sync, and avatar rendering. Discoveries from that
work live in `avatar/DISCOVERIES.md`. The visual stream eventually adds an image or webcam
feed, and later a 3D avatar, layered on top of the verbal character. This document covers
the verbal layer only.

---

## 2. Character Schema

Characters are stored as JSON files. Current schema version: **2**.

```json
{
  "schema_version": 2,
  "id": "seth1",
  "title": "Seth1",
  "name": "Seth Meyers",
  "voice": "seth_meyers_ref",
  "bio": "You are Seth Meyers. ...",
  "style": "Dry wit, quick pivot from absurd to sincere. ...",
  "strength": "light",
  "tuning": {
    "speed": 1.0,
    "guidance": 3.0,
    "temperature": 0.85,
    "steps": 20
  }
}
```

### Field table

| Field            | Type    | Meaning                                                              | Default      |
|------------------|---------|----------------------------------------------------------------------|--------------|
| schema_version   | int     | Schema revision; increment on breaking changes                       | 2            |
| id               | string  | Slug of title; used as filename and lookup key                       | —            |
| title            | string  | Variant handle shown in the library (e.g. `Seth1`, `Seth_final`)     | —            |
| name             | string  | Referenceable display name used in prompts (e.g. `Seth Meyers`)      | —            |
| voice            | string  | OmniVoice voice reference ID                                         | —            |
| bio              | string  | BIO block — who they are and facts (see §3)                          | ""           |
| style            | string  | STYLE block — tone and speech patterns (see §3)                      | ""           |
| strength         | string  | Default reword strength: `"none"`, `"light"`, or `"full"`            | `"light"`    |
| tuning.speed     | float   | TTS speed multiplier                                                 | 1.0          |
| tuning.guidance  | float   | Voice guidance scale                                                 | 3.0          |
| tuning.temperature | float | LLM sampling temperature                                            | 0.85         |
| tuning.steps     | int     | TTS diffusion steps                                                  | 20           |

**Migration note:** Fields added in future schema versions should be filled with defaults
on load when absent, so old character files remain usable without manual edits.

---

## 3. Bio vs Style

The personality is split into two discrete text blocks. Keep them separate — they are
injected at different points and in different consumer paths.

### BIO — who they are + facts

Factual identity. Who the person is, their background, their career, their relationships.
Written in the second person addressed to the model: *You are ...*

**Budget: ≤ 600 tokens (~450 words)**

> You are Seth Meyers. You were born December 28, 1973, in Evanston, Illinois.
> You host Late Night with Seth Meyers on NBC, a role you have held since 2014.
> You are married to Alexi Ashe and have three children.

### STYLE — tone + speech patterns

How they talk, not who they are. Cadence, word choices, recurring phrases, comic timing,
segment names, and the instruction that when asked about themselves they answer factually
in their own voice.

**Budget: ≤ 400 tokens (~300 words)**

> Dry wit with quick pivots from absurd to sincere. Often opens a riff with a flat
> declarative before the punchline. Signature segment: "A Closer Look." Catchphrase
> register is understated — the laugh comes from timing, not volume.

---

## 4. Consumer Assembly

The LLM (Ollama llama3.2:3b) receives a system prompt assembled by the consumer.
**Identity (bio + style) is shared and lives in the character. The intro sentence and
the strength block are owned by the consumer — they are never stored in the character.**

### 4a. reword (transform path)

The model is asked to restyle a line, not to roleplay. BIO is optional (off by default —
see §6). The strength block controls how aggressively the model rewrites.

```
[REWORD_INTRO]    "Re-voice this line as {name}."

[STYLE]           character.style

[BIO]             character.bio          ← optional, toggled by consumer

[STRENGTH]        one of:
                    none  → "Return the line unchanged."
                    light → "Preserve meaning; adjust phrasing to match {name}'s cadence."
                    full  → "Fully rewrite in {name}'s voice; meaning may shift."

[VOICE_STYLE]     shared block: pacing, TTS-friendly prose rules, no markdown

--- user message ---
{the line to restyle}
```

### 4b. chat (generation path)

The model IS the character. No strength block. History is included.

```
[CHAT_INTRO]      "You are {name}."

[BIO]             character.bio

[STYLE]           character.style

[VOICE_STYLE]     shared block: pacing, TTS-friendly prose rules, no markdown

--- messages ---
[history, capped ~12 turns]
[user message]
```

**Key difference:** reword injects the line as the user message; chat injects the
conversation history. BIO always appears in chat; it is optional in reword.

---

## 5. Budgets

| Block       | Char limit  | Token estimate (chars ÷ 4) |
|-------------|-------------|---------------------------|
| BIO         | ~1 800 chars | ≤ 600 tokens              |
| STYLE       | ~1 200 chars | ≤ 400 tokens              |
| Combined    | ~3 000 chars | ~1 000 tokens             |

**Why these numbers fit:**
Ollama llama3.2:3b defaults to a 2 048-token context. The combined persona (~1 000 tokens)
leaves ~1 000 tokens for the intro, strength block, VOICE_STYLE, and the speech line on
the reword path — enough for most inputs.

**Chat path adjustment:** The chat consumer bumps `num_ctx` to **8 192** and caps message
history at **~12 turns** to keep the full conversation within context while preserving the
persona blocks.

**UI enforcement:** The talk-back character builder shows a live `(x of y)` word counter
for each field and disables the Save button when either field is over budget.

---

## 6. Open Question Being Tested

**Does BIO help or pollute the reword path?**

Including BIO in reword gives the model factual grounding (so it doesn't invent details
when the line references the character). Excluding it keeps the prompt lighter and may
produce cleaner style-only rewrites without the model reaching for facts.

The talk-back UI exposes an `include_bio` toggle on the reword path for exactly this test.
Results from that testing will inform whether BIO becomes on-by-default, off-by-default,
or character-level configurable.
