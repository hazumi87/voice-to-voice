// node tools/voice_test_turns.test.mjs — scenarios for static/voice-test-turns.mjs
import { createTurns, detectSend, detectYesNo } from '../static/voice-test-turns.mjs';

let fails = 0;
function check(name, cond, detail = '') {
  console.log(`  [${cond ? 'ok' : 'FAIL'}] ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) fails++;
}
const types = (evs) => evs.map((e) => e.type).join(',');

// a controllable clock
let clock = 0;
const now = () => clock;
const mk = (over = {}) => createTurns({ ackSec: 1.5, ackCooldownSec: 20, promptSec: 8, now, ...over });

console.log('detectSend / detectYesNo');
check('"send that" alone', detectSend('Send that.').hit && detectSend('Send that.').rest === '');
check('trailing phrase keeps the content', detectSend('make the button blue, send it').rest === 'make the button blue');
check('"send" inside a sentence is not a send', !detectSend('send the invoice to Tina tomorrow').hit);
check('yes', detectYesNo('Yes please.') === 'yes');
check('no', detectYesNo('No, not yet.') === 'no');
check('"no don\'t send it" is no', detectYesNo("no, don't send it") === 'no');
check('neither', detectYesNo('and also the logo') === null);

console.log('compose -> short break -> ack once -> continue -> send phrase');
let t = mk();
let ev = t.segment({ text: 'tell the lead the build is green', speechMs: 2000, pauseMs: 0 });
check('pending after first segment', types(ev) === 'pending' && ev[0].block === 'tell the lead the build is green');
check('no ack before ackSec', types(t.tick(1000)) === '');
clock = 5000;
check('ack at ackSec', types(t.tick(1600)) === 'ack');
check('ack only once per silence', types(t.tick(2000)) === '');
t.speechStart();
ev = t.segment({ text: 'and the mockup passed, send that', speechMs: 2500, pauseMs: 2100 });
check('send via phrase with both parts', types(ev) === 'send' && ev[0].via === 'phrase' && ev[0].block === 'tell the lead the build is green and the mockup passed', ev[0] && ev[0].block);
check('block cleared after send', t.block === '');

console.log('ack cooldown');
t = mk();
clock = 0;
t.segment({ text: 'first thought', speechMs: 1000, pauseMs: 0 });
check('first ack', types(t.tick(1600)) === 'ack');
t.speechStart();
t.segment({ text: 'second thought', speechMs: 1000, pauseMs: 2000 });
clock = 10000; // 10 s later: inside the 20 s cooldown
check('no second ack inside cooldown', types(t.tick(1600)) === '');
t.speechStart();
t.segment({ text: 'third thought', speechMs: 1000, pauseMs: 2000 });
clock = 30000; // past the cooldown
check('ack again after cooldown', types(t.tick(1600)) === 'ack');

console.log('long pause -> one prompt -> yes sends');
t = mk();
clock = 0;
t.segment({ text: 'please add a hat to the character', speechMs: 2000, pauseMs: 0 });
check('ack first', types(t.tick(1600)) === 'ack');
ev = t.tick(8100);
check('prompt at promptSec', types(ev) === 'prompt' && ev[0].text === 'Want me to send that?');
check('prompt only once', types(t.tick(9000)) === '');
t.speechStart();
ev = t.segment({ text: 'Yes.', speechMs: 400, pauseMs: 9000 });
check('yes sends the block', types(ev) === 'send' && ev[0].via === 'yes' && ev[0].block === 'please add a hat to the character');

console.log('long pause -> prompt -> no keeps composing');
t = mk();
clock = 0;
t.segment({ text: 'the export is missing timestamps', speechMs: 2000, pauseMs: 0 });
t.tick(8100);
t.speechStart();
ev = t.segment({ text: 'no, hold on', speechMs: 600, pauseMs: 9000 });
check('no -> no-send with the block kept', types(ev) === 'no-send' && t.block === 'the export is missing timestamps');
t.speechStart();
ev = t.segment({ text: 'and the font is too small, send it', speechMs: 2000, pauseMs: 1000 });
check('then a phrase sends everything', types(ev) === 'send' && ev[0].block === 'the export is missing timestamps and the font is too small');

console.log('prompt answered with more content (neither yes nor no)');
t = mk();
clock = 0;
t.segment({ text: 'first part', speechMs: 1000, pauseMs: 0 });
t.tick(8100);
t.speechStart();
ev = t.segment({ text: 'second part', speechMs: 1000, pauseMs: 9000 });
check('content after a prompt is appended', types(ev) === 'pending' && ev[0].block === 'first part second part');
check('prompt can fire again on the next long silence', types(t.tick(8100)) === 'prompt');

console.log('prompt off');
t = mk({ promptOn: false });
clock = 0;
t.segment({ text: 'hello there', speechMs: 1000, pauseMs: 0 });
t.tick(1600);
check('no prompt when off', types(t.tick(60000)) === '');

console.log('empty block: silence does nothing');
t = mk();
check('no events with nothing composed', types(t.tick(60000)) === '');

console.log(fails ? `FAILURES: ${fails}` : 'ALL PASS');
process.exit(fails ? 1 : 0);
