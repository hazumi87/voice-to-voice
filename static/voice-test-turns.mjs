// voice-test-turns.mjs — the conversation turn logic for the speech test page
// (briefing-table web/voice-test.html) and, later, the Dot. Pure logic: no DOM, no
// audio, no timers of its own. The page feeds it what the mic/STT saw and plays what
// the events ask for. Owner: voice-to-voice. Shape agreed with vrpc-briefing-table
// (Voice Interaction room, line 7977, 2026-10-14).
//
//   import { createTurns, DEFAULTS, detectSend, detectYesNo } from './voice-test-turns.mjs';
//   const t = createTurns({ ackSec: 1.5, promptSec: 8, ... });
//   t.speechStart()                               -> Event[]
//   t.segment({ text, words, speechMs, pauseMs }) -> Event[]
//   t.tick(silenceMs)                             -> Event[]   (~every 100 ms while silent)
//   t.reset()
//
//   Event = {type:'ack'}
//         | {type:'prompt', text}
//         | {type:'send', block, via:'phrase'|'yes'}
//         | {type:'no-send', block}
//         | {type:'pending', block}
//
// Behaviour (Eric, line 7967): talk until a break; a short break gets ONE soft "ok"
// (cooldown so it is never chatty) while the text keeps accumulating; nothing is sent
// until a send phrase ("send that") or, after a long pause, ONE prompt ("want me to
// send that?") answered yes. "No" keeps the block and keeps listening. A send phrase at
// the END of a segment sends the words before it. Everything is deterministic.

export const DEFAULTS = Object.freeze({
  ackSec: 1.5,            // silence before the soft "ok"
  ackCooldownSec: 20,     // at most one "ok" per this many seconds
  promptSec: 8,           // silence before "want me to send that?"
  promptOn: true,
  promptText: 'Want me to send that?',
  sendPhrases: ['send that', 'send it', 'send this', 'go ahead and send', 'go ahead and send it',
    "that's it send it", "that's it send", 'send the message', 'submit', 'ship it', 'send'],
  yesWords: ['yes', 'yeah', 'yep', 'yup', 'sure', 'go ahead', 'send it', 'do it', 'please', 'correct', 'affirmative', 'ok send'],
  noWords: ['no', 'nope', 'not yet', 'hold on', 'wait', 'cancel', "don't", 'do not', 'keep going', 'not now', 'negative', 'scrap that'],
  now: () => Date.now(),  // injectable clock for tests
});

const PUNCT = /[^\p{L}\p{N}' ]+/gu;

export function normalize(text) {
  return String(text || '').toLowerCase().replace(PUNCT, ' ').replace(/\s+/g, ' ').trim();
}

// Returns {hit:true, phrase, rest} when `text` is, or ends with, a send phrase; `rest` is the
// content before the phrase (may be ''). Longest phrase wins.
export function detectSend(text, phrases = DEFAULTS.sendPhrases) {
  const t = normalize(text);
  if (!t) return { hit: false, rest: t };
  const sorted = [...phrases].map(normalize).filter(Boolean).sort((a, b) => b.length - a.length);
  for (const p of sorted) {
    if (t === p) return { hit: true, phrase: p, rest: '' };
    if (t.endsWith(' ' + p)) return { hit: true, phrase: p, rest: t.slice(0, -p.length).trim() };
  }
  return { hit: false, rest: t };
}

// 'yes' | 'no' | null, judged on the first few words so "yes, and also …" still counts.
export function detectYesNo(text, yesWords = DEFAULTS.yesWords, noWords = DEFAULTS.noWords) {
  const t = normalize(text);
  if (!t) return null;
  const head = t.split(' ').slice(0, 4).join(' ');
  const has = (list) => list.map(normalize).some((w) => w && (head === w || head.startsWith(w + ' ') || (' ' + head + ' ').includes(' ' + w + ' ')));
  if (has(noWords)) return 'no';     // "no" beats "yes" when both appear ("no, don't send it")
  if (has(yesWords)) return 'yes';
  return null;
}

export function createTurns(settings = {}) {
  const cfg = { ...DEFAULTS, ...settings };
  let block = [];            // composed segments (strings)
  let ackedThisSilence = false;
  let promptedThisSilence = false;
  let pendingPrompt = false; // a prompt was spoken; the next segment is the answer
  let lastAckAt = -Infinity;

  const text = () => block.join(' ').replace(/\s+/g, ' ').trim();

  function reset() {
    block = [];
    ackedThisSilence = false;
    promptedThisSilence = false;
    pendingPrompt = false;
    lastAckAt = -Infinity;
  }

  function speechStart() {
    // New speech cancels the per-silence flags; a pending prompt stays pending until the
    // segment arrives, because the answer is in it.
    ackedThisSilence = false;
    promptedThisSilence = false;
    return [];
  }

  function segment(seg) {
    const events = [];
    const raw = (seg && seg.text) || '';
    const t = normalize(raw);
    if (!t) return events;

    if (pendingPrompt) {
      const yn = detectYesNo(t, cfg.yesWords, cfg.noWords);
      pendingPrompt = false;
      if (yn === 'yes') {
        const b = text();
        reset();
        events.push({ type: 'send', block: b, via: 'yes' });
        return events;
      }
      if (yn === 'no') {
        events.push({ type: 'no-send', block: text() });
        return events;
      }
      // Neither: the user kept composing; fall through and append it.
    }

    const s = detectSend(t, cfg.sendPhrases);
    if (s.hit) {
      if (s.rest) block.push(s.rest);
      const b = text();
      reset();
      if (b) events.push({ type: 'send', block: b, via: 'phrase' });
      return events;
    }

    block.push(t);
    events.push({ type: 'pending', block: text() });
    return events;
  }

  function tick(silenceMs) {
    const events = [];
    if (!block.length) return events;
    const ms = Number(silenceMs) || 0;
    const now = cfg.now();
    if (!ackedThisSilence && !pendingPrompt && ms >= cfg.ackSec * 1000
        && now - lastAckAt >= cfg.ackCooldownSec * 1000) {
      ackedThisSilence = true;
      lastAckAt = now;
      events.push({ type: 'ack' });
    } else if (!ackedThisSilence && ms >= cfg.ackSec * 1000) {
      ackedThisSilence = true;  // inside the cooldown: skip the ack for this silence, once
    }
    if (cfg.promptOn && !promptedThisSilence && !pendingPrompt && ms >= cfg.promptSec * 1000) {
      promptedThisSilence = true;
      pendingPrompt = true;
      events.push({ type: 'prompt', text: cfg.promptText });
    }
    return events;
  }

  return {
    speechStart, segment, tick, reset,
    get block() { return text(); },
    get pendingPrompt() { return pendingPrompt; },
    settings: cfg,
  };
}
