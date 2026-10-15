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

console.log('lead #1: prompt pending, STT of the next segment lands after ackSec -> the ack is not spent');
t = mk({ ackCooldownSec: 0 });
clock = 0;
t.segment({ text: 'first part', speechMs: 1000, pauseMs: 0 });
t.tick(1600);                 // ack
check('prompt at 8 s', types(t.tick(8100)) === 'prompt');
t.speechStart();              // the person keeps composing instead of answering
check('ticks while STT is in flight do nothing', types(t.tick(1900)) === '');
ev = t.segment({ text: 'second part', speechMs: 1500, pauseMs: 9000 });
check('segment appends', types(ev) === 'pending' && ev[0].block === 'first part second part');
check('the silence after it still gets its ok', types(t.tick(1900)) === 'ack');
check('and only once', types(t.tick(2500)) === '');

console.log('lead #2: after "no" there is no second prompt until new content');
t = mk();
clock = 0;
t.segment({ text: 'draft of the note', speechMs: 1000, pauseMs: 0 });
t.tick(1600);                 // ack
check('first prompt', types(t.tick(8100)) === 'prompt');
t.speechStart();
ev = t.segment({ text: "No, don't send it.", speechMs: 800, pauseMs: 9000 });
check('no-send keeps the block', types(ev) === 'no-send' && t.block === 'draft of the note');
check('no re-prompt after another 8 s', types(t.tick(8100)) === '');
check('nor after a minute', types(t.tick(60000)) === '');
t.speechStart();
t.segment({ text: 'and the deadline', speechMs: 800, pauseMs: 60000 });
check('new content re-arms the prompt', types(t.tick(8100)) === 'prompt');
t.speechStart();
ev = t.segment({ text: 'yes', speechMs: 300, pauseMs: 9000 });
check('yes sends everything', types(ev) === 'send' && ev[0].block === 'draft of the note and the deadline');

console.log('lead #2b: weak yes-words do not send long utterances');
check('"Please change the button to blue" is not a yes', detectYesNo('Please change the button to blue') === null);
check('"Sure, and make the logo bigger too" is not a yes', detectYesNo('Sure, and make the logo bigger too') === null);
check('"please" alone is a yes', detectYesNo('Please.') === 'yes');
check('"sure, go" is a yes', detectYesNo('Sure, go.') === 'yes');
check('"yes, and also the logo" still counts (strong yes up front)', detectYesNo('yes, and also the logo') === 'yes');
check('"go ahead" is a yes', detectYesNo('Go ahead.') === 'yes');
check('"the answer is yes I think we should wait" is not a yes', detectYesNo('the answer is yes I think we should wait') === null);
t = mk();
clock = 0;
t.segment({ text: 'first thought', speechMs: 1000, pauseMs: 0 });
t.tick(8100);
t.speechStart();
ev = t.segment({ text: 'Please change the title to draft', speechMs: 2000, pauseMs: 9000 });
check('content after an unheard prompt is appended, not sent', types(ev) === 'pending' && ev[0].block === 'first thought please change the title to draft');

console.log(fails ? `FAILURES: ${fails}` : 'ALL PASS');
process.exit(fails ? 1 : 0);
