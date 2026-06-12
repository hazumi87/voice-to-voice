#!/usr/bin/env node
// speak-shim — a per-project voice shim for the agent-speech-relay `speak` tool.
//
// WHY THIS EXISTS
// ---------------
// The real speak MCP (agent-speech-relay) is shared and stateless: it has no idea
// WHICH project an agent is working on, so it can't pick a project-appropriate voice
// on its own. And a shared agent (e.g. /opt/dev/domain-experts/architect) launches
// from a home folder and cd's all over the place, so project-CLAUDE.md can't carry
// the voice either.
//
// This shim closes that gap WITHOUT making the agent reason about voices. It is a
// tiny local stdio-MCP server that exposes a single `speak` tool. When Claude Code
// spawns it, the shim reads ONE env var set at launch — SPEAK_VOICE — and stamps
// that voice onto every speak() call before forwarding to the real relay. The agent
// just calls speak("...") and the correct project voice is GUARANTEED by how the
// process was launched, regardless of cwd, regardless of where the agent wanders.
//
// HOW IT TALKS
// ------------
//   - Upstream (to the agent): stdio MCP — newline-delimited JSON-RPC 2.0 on
//     stdin/stdout. Implemented by hand (no SDK dep) so it's a single portable file.
//   - Downstream (to the relay): HTTP POST /say on the agent-speech-relay, which is
//     the same doSpeak() core the relay's own MCP speak tool uses. Accepts
//     {text, voice, intent, engine} and returns the delivery result.
//
// ENV (all read once at launch)
//   SPEAK_VOICE   the voice id to stamp, e.g. "cust_sammuel_1781140440". REQUIRED to
//                 be useful; if unset, the shim forwards with no voice (relay default).
//   SPEAK_ENGINE  engine to use. Default "omnivoice".
//   SPEAK_RELAY_URL  base URL of the relay. Default "http://127.0.0.1:8217"
//                    (the relay binds loopback on the NUC). On the VRPC, point this
//                    at the NUC's tailscale addr, e.g. http://100.110.14.59:8217.
//   SPEAK_LABEL   optional human label for logs (e.g. "asset-platform"). Cosmetic.
//
// Run: node speak-shim.mjs   (Claude Code launches it; you don't run it by hand.)

import process from 'node:process';

// Default voice when SPEAK_VOICE is unset. Alle is the house default; per-project
// launches override it via the SPEAK_VOICE env var (that's the whole point of the shim).
const DEFAULT_VOICE = 'cust_alle_1781140440';
const VOICE = process.env.SPEAK_VOICE || DEFAULT_VOICE;
const ENGINE = process.env.SPEAK_ENGINE || 'omnivoice';
const RELAY = (process.env.SPEAK_RELAY_URL || 'http://127.0.0.1:8217').replace(/\/+$/, '');
const LABEL = process.env.SPEAK_LABEL || '';
const SAY_TIMEOUT_MS = 60000;

const SERVER_NAME = 'speak-shim';
const VERSION = '0.1.0';

// stderr is for diagnostics — stdout is reserved for the JSON-RPC stream ONLY.
function diag(...a) {
  process.stderr.write(`[speak-shim] ${a.join(' ')}\n`);
}
diag(`up voice=${VOICE || '(none)'} engine=${ENGINE} relay=${RELAY}${LABEL ? ` label=${LABEL}` : ''}`);

// ── downstream: forward a speak to the relay's /say endpoint ──────────────────
async function forward({ text, intent }) {
  // The shim's whole job: stamp the launch-env voice/engine onto the call.
  const body = { text, voice: VOICE || undefined, engine: ENGINE, intent: intent || 'reply' };
  const r = await fetch(`${RELAY}/say`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(SAY_TIMEOUT_MS),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.ok === false) {
    throw new Error(j.error || `relay /say HTTP ${r.status}`);
  }
  // Echo back the voice we stamped so the agent (and logs) can confirm routing.
  return { ...j, stamped_voice: VOICE || '(relay default)', via: RELAY };
}

// ── upstream: minimal stdio JSON-RPC (MCP) ────────────────────────────────────
const TOOLS = [
  {
    name: 'speak',
    description:
      'Speak a short message aloud to the user in THIS project\'s voice. The voice is ' +
      'fixed by how this tool was launched — you do not choose it. Use for brief, ' +
      'user-directed lines (a reply, a status note, a heads-up); one or two sentences. ' +
      'Long-form output stays in the chat. Returns delivery info.',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', minLength: 1, maxLength: 2000, description: 'What to say. One or two sentences.' },
        intent: { type: 'string', enum: ['reply', 'status', 'question', 'blocker'], description: 'Why you are speaking. Default reply.' },
      },
      required: ['text'],
    },
  },
];

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}

function reply(id, result) {
  send({ jsonrpc: '2.0', id, result });
}

function replyError(id, code, message) {
  send({ jsonrpc: '2.0', id, error: { code, message } });
}

async function handle(req) {
  const { id, method, params } = req;

  // Notifications (no id) — acknowledge by doing nothing; never reply.
  if (id === undefined || id === null) {
    return;
  }

  switch (method) {
    case 'initialize':
      reply(id, {
        protocolVersion: params?.protocolVersion || '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: SERVER_NAME, version: VERSION },
      });
      return;

    case 'tools/list':
      reply(id, { tools: TOOLS });
      return;

    case 'tools/call': {
      const name = params?.name;
      const args = params?.arguments || {};
      if (name !== 'speak') {
        replyError(id, -32601, `unknown tool: ${name}`);
        return;
      }
      const text = (args.text || '').toString();
      if (!text.trim()) {
        replyError(id, -32602, 'text is required');
        return;
      }
      try {
        const result = await forward({ text: text.slice(0, 2000), intent: args.intent });
        reply(id, { content: [{ type: 'text', text: JSON.stringify(result) }] });
        diag(`spoke voice=${VOICE || '(default)'} listeners=${result.listeners ?? '?'} engine=${result.engine ?? ENGINE}`);
      } catch (err) {
        // Surface as a tool error, not a transport error — the agent can read it.
        reply(id, { isError: true, content: [{ type: 'text', text: `speak failed: ${err.message}` }] });
        diag(`speak failed: ${err.message}`);
      }
      return;
    }

    case 'ping':
      reply(id, {});
      return;

    default:
      replyError(id, -32601, `method not found: ${method}`);
  }
}

// ── stdin line reader: newline-delimited JSON-RPC ─────────────────────────────
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let req;
    try {
      req = JSON.parse(line);
    } catch (e) {
      diag(`bad json line: ${e.message}`);
      continue;
    }
    // Fire-and-forget; handle() awaits its own async work and replies in order
    // per-message (Claude Code tolerates out-of-order responses by id anyway).
    handle(req).catch((e) => diag(`handler error: ${e.message}`));
  }
});

process.stdin.on('end', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
