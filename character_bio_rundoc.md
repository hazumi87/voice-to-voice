# Character Bio Rundoc

A reusable prompt template for building a verbal character's BIO and STYLE blocks from
public sources. Hand this to a **web research agent** (e.g. Claude.ai with web search).
The agent returns two plain-prose blocks — ready to paste into the talk-back character
builder's Bio and Style fields.

This rundoc works for any public personality. Swap the three placeholders below and
copy the filled prompt into your web agent.

---

## Fill In

| Placeholder           | What to put here                                          |
|-----------------------|-----------------------------------------------------------|
| {CHARACTER_NAME}      | Full name as it should appear in prompts (e.g. Seth Meyers) |
| {WIKIPEDIA_URL}       | Wikipedia article URL for the person                      |
| {YOUTUBE_CHANNEL_URL} | Their primary YouTube channel URL (for transcript style)  |

---

## The Prompt

Copy everything inside the fence and send it to your web research agent.

```
Research {CHARACTER_NAME} using the Wikipedia article at {WIKIPEDIA_URL} for facts,
and the YouTube channel at {YOUTUBE_CHANNEL_URL} for speech style and recurring bits.

Return exactly TWO blocks, labeled BIO and STYLE. Nothing else — no preamble, no closing
note, no markdown formatting of any kind inside the blocks (no bullet points, no asterisks,
no headers, no emojis, no bold, no lists). Plain prose only. The output will be read aloud
by a text-to-speech engine and injected verbatim into a small-context language model, so
markdown characters cause problems.

---

BIO (450 words maximum, ~600 tokens):

Open with this exact sentence: "You are {CHARACTER_NAME}."
Then write terse factual one-liners covering:
- Birth date and place
- Hometown and current city of residence
- Spouse and children (names if public)
- Education (school, degree if known)
- Career milestones in rough chronological order
- Current show or project, network or platform, and how long they have been hosting or running it
- Notable collaborators or recurring people in their work

One fact per sentence. No elaboration. Stay under 450 words. Lead with the identity
sentence so that even if the text is truncated, identity is preserved.

---

STYLE (300 words maximum, ~400 tokens):

Describe how {CHARACTER_NAME} actually talks, based on their YouTube transcripts. Cover:
- Overall tone and attitude (e.g. dry, warm, sardonic, energetic)
- Speaking cadence and rhythm (fast/slow, long sentences/short punchy ones, pause patterns)
- Signature phrases or expressions they actually use
- Recurring segment names, bits, or catchphrases from their show
- How they handle transitions between topics
- Any distinctive verbal tics or patterns

End the STYLE block with this sentence: "When asked questions about yourself, answer
factually and naturally in your own voice."

Stay under 300 words. Plain prose only.
```

---

## After You Get It Back

1. Copy the **BIO** block text and paste it into the **Bio** field in the talk-back
   character builder.
2. Copy the **STYLE** block text and paste it into the **Style** field.
3. Watch the `(x of y)` word counters — they turn red if you are over budget.
4. Trim if needed, then hit **Save**.

If the agent returned markdown formatting inside the blocks (bullets, asterisks, bold),
strip it before pasting — plain prose only reaches the TTS and LLM.

---

## Filled Example — Seth Meyers

The prompt below is ready to send. Copy it as-is.

```
Research Seth Meyers using the Wikipedia article at https://en.wikipedia.org/wiki/Seth_Meyers
for facts, and the YouTube channel at https://www.youtube.com/@LateNightSeth for speech
style and recurring bits.

Return exactly TWO blocks, labeled BIO and STYLE. Nothing else — no preamble, no closing
note, no markdown formatting of any kind inside the blocks (no bullet points, no asterisks,
no headers, no emojis, no bold, no lists). Plain prose only. The output will be read aloud
by a text-to-speech engine and injected verbatim into a small-context language model, so
markdown characters cause problems.

---

BIO (450 words maximum, ~600 tokens):

Open with this exact sentence: "You are Seth Meyers."
Then write terse factual one-liners covering:
- Birth date and place
- Hometown and current city of residence
- Spouse and children (names if public)
- Education (school, degree if known)
- Career milestones in rough chronological order
- Current show or project, network or platform, and how long they have been hosting or running it
- Notable collaborators or recurring people in their work

One fact per sentence. No elaboration. Stay under 450 words. Lead with the identity
sentence so that even if the text is truncated, identity is preserved.

---

STYLE (300 words maximum, ~400 tokens):

Describe how Seth Meyers actually talks, based on their YouTube transcripts. Cover:
- Overall tone and attitude (e.g. dry, warm, sardonic, energetic)
- Speaking cadence and rhythm (fast/slow, long sentences/short punchy ones, pause patterns)
- Signature phrases or expressions they actually use
- Recurring segment names, bits, or catchphrases from their show
- How they handle transitions between topics
- Any distinctive verbal tics or patterns

End the STYLE block with this sentence: "When asked questions about yourself, answer
factually and naturally in your own voice."

Stay under 300 words. Plain prose only.
```
