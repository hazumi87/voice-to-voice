#!/usr/bin/env node
// test-shim — drives speak-shim.mjs over stdio exactly the way Claude Code would,
// so we can prove the MCP handshake + voice stamping + relay forward without a CLI.
//
// Usage:
//   node test-shim.mjs                 # uses default voice (Alle)
//   SPEAK_VOICE=cust_sammuel_... node test-shim.mjs   # override voice
//
// Point at the relay via SPEAK_RELAY_URL (NUC). Pass-through env to the child.

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SHIM = join(__dirname, 'speak-shim.mjs');

const child = spawn(process.execPath, [SHIM], {
  stdio: ['pipe', 'pipe', 'inherit'], // stderr (diag) streams straight through
  env: { ...process.env },
});

let outBuf = '';
const pending = new Map();
child.stdout.setEncoding('utf8');
child.stdout.on('data', (chunk) => {
  outBuf += chunk;
  let nl;
  while ((nl = outBuf.indexOf('\n')) >= 0) {
    const line = outBuf.slice(0, nl).trim();
    outBuf = outBuf.slice(nl + 1);
    if (!line) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { console.log('<< (non-json)', line); continue; }
    console.log('<<', JSON.stringify(msg));
    if (msg.id && pending.has(msg.id)) {
      const { resolve } = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(msg);
    }
  }
});

let nextId = 1;
function call(method, params) {
  const id = nextId++;
  const req = { jsonrpc: '2.0', id, method, params };
  console.log('>>', JSON.stringify(req));
  child.stdin.write(JSON.stringify(req) + '\n');
  return new Promise((resolve) => pending.set(id, { resolve }));
}

function notify(method, params) {
  const req = { jsonrpc: '2.0', method, params };
  console.log('>> (notify)', JSON.stringify(req));
  child.stdin.write(JSON.stringify(req) + '\n');
}

async function main() {
  await call('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'test-shim', version: '0' } });
  notify('notifications/initialized', {});
  await call('tools/list', {});
  const r = await call('tools/call', {
    name: 'speak',
    arguments: { text: process.env.TEST_TEXT || 'Shim self test. If you can hear me, the stdio MCP and voice stamping both work.', intent: 'status' },
  });
  // Pull the inner result for a clean readout.
  try {
    const inner = JSON.parse(r.result.content[0].text);
    console.log('\n=== RESULT ===');
    console.log('stamped_voice:', inner.stamped_voice);
    console.log('engine       :', inner.engine);
    console.log('device       :', inner.device);
    console.log('listeners    :', inner.listeners);
    console.log('delivered    :', inner.delivered);
    console.log('via          :', inner.via);
  } catch (e) {
    console.log('result parse failed:', e.message, r.result);
  }
  child.stdin.end();
  setTimeout(() => process.exit(0), 300);
}

main().catch((e) => { console.error('test error', e); process.exit(1); });
