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
// send that?") answered yes. "No" keeps the block and keeps listening: no second prompt
// until new content is added. A send phrase at the END of a segment sends the words
// before it. A weak yes ("please", "sure") only counts as a short answer; a long utterance
// after a prompt is content. Everything is deterministic.

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

// Yes-words that count only when the whole answer is short. "Please change the button to
// blue" after an unheard prompt is content, not consent; "please" or "sure, go" alone is.
export const WEAK_YES = Object.freeze(['please', 'sure', 'correct', 'affirmative', 'ok', 'okay', 'fine']);
const SHORT_ANSWER_WORDS = 4;

// 'yes' | 'no' | null. A "no" is judged on the first few words (a no is safe: it keeps the
// block). A "yes" sends, so it needs either a short answer or a strong yes-word up front;
// a long utterance that merely contains "sure"/"please" is composing, not consent.
export function detectYesNo(text, yesWords = DEFAULTS.yesWords, noWords = DEFAULTS.noWords) {
  const t = normalize(text);
  if (!t) return null;
  const words = t.split(' ');
  const head = words.slice(0, SHORT_ANSWER_WORDS).join(' ');
  const inHead = (w) => head === w || head.startsWith(w + ' ') || (' ' + head + ' ').includes(' ' + w + ' ');
  const starts = (w) => t === w || t.startsWith(w + ' ');
  if (noWords.map(normalize).some((w) => w && inHead(w))) return 'no';  // "no" beats "yes" ("no, don't send it")
  const short = words.length <= SHORT_ANSWER_WORDS;
  for (const w of yesWords.map(normalize)) {
    if (!w) continue;
    if (WEAK_YES.includes(w)) {
      if (short && inHead(w)) return 'yes';
    } else if (starts(w) || (short && inHead(w))) {
      return 'yes';
    }
  }
  return null;
}

export function createTurns(settings = {}) {
  const cfg = { ...DEFAULTS, ...settings };
  let block = [];            // composed segments (strings)
  let ackedThisSilence = false;
  let promptedThisSilence = false;
  let pendingPrompt = false; // a prompt was spoken; the next segment is the answer
  let promptArmed = false;   // new content since the last "no": a prompt may fire again
  let lastAckAt = -Infinity;

  const text = () => block.join(' ').replace(/\s+/g, ' ').trim();

  function reset() {
    block = [];
    ackedThisSilence = false;
    promptedThisSilence = false;
    pendingPrompt = false;
    promptArmed = false;
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
        // "No" keeps the block and keeps listening: no further prompt until new content
        // arrives, however long the silence runs.
        promptArmed = false;
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
    promptArmed = true;
    events.push({ type: 'pending', block: text() });
    return events;
  }

  function tick(silenceMs) {
    const events = [];
    if (!block.length) return events;
    const ms = Number(silenceMs) || 0;
    const now = cfg.now();
    // While a prompt is pending nothing here touches the ack flag: the answer's STT can
    // land later than ackSec, and the silence after the answer still deserves its "ok".
    if (!ackedThisSilence && !pendingPrompt && ms >= cfg.ackSec * 1000) {
      ackedThisSilence = true;  // one decision per silence, ack or not
      if (now - lastAckAt >= cfg.ackCooldownSec * 1000) {
        lastAckAt = now;
        events.push({ type: 'ack' });
      }
    }
    if (cfg.promptOn && promptArmed && !promptedThisSilence && !pendingPrompt && ms >= cfg.promptSec * 1000) {
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
