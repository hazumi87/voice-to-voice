// visemeTimeline.smoke.js
// Tiny smoke check for the pure timeline-lookup logic. No test framework — just
// assertions. Browser-safe (no node builtins, no top-level await) so it can be
// imported by the app and run against already-fetched data; the assertions then
// show up in the browser console.
//
// For a standalone CLI run see visemeTimeline.smoke.node.mjs (kept separate so
// node-only imports never reach the Vite bundle).
//
// The contract being proven (per task spec):
//   - rest (10) BEFORE the clip starts
//   - the correct viseme MID-clip (derived from the timeline, not hardcoded, so
//     this never goes stale when the timeline is regenerated)
//   - rest (10) AFTER the clip ends

import { visemeAt, REST_VISEME } from './visemeTimeline.js';

function assertEq(label, actual, expected) {
  const ok = actual === expected;
  const line = `[smoke] ${label}: got "${actual}", expected "${expected}" -> ${ok ? 'PASS' : 'FAIL'}`;
  if (ok) {
    console.log(line);
  } else {
    console.error(line);
  }
  return ok;
}

/**
 * Run the three-point smoke check against a timeline.
 * @param {Array} timeline
 * @returns {boolean} true if all assertions pass
 */
export function runSmokeCheck(timeline) {
  // Pick a mid-clip sample point from a real entry so the expectation is
  // self-derived and stays valid across timeline regenerations.
  const sample = timeline[Math.floor(timeline.length / 2)];
  const midT = (sample.t_start + sample.t_end) / 2;
  const midExpected = sample.viseme;

  const before = visemeAt(-1.0, timeline);
  const mid = visemeAt(midT, timeline);
  const after = visemeAt(9999.0, timeline);

  let pass = true;
  pass = assertEq('before clip (t=-1.0)', before, REST_VISEME) && pass;
  pass = assertEq(`mid clip (t=${midT.toFixed(3)})`, mid, midExpected) && pass;
  pass = assertEq('after clip (t=9999)', after, REST_VISEME) && pass;

  console.log(`[smoke] overall: ${pass ? 'PASS' : 'FAIL'}`);
  return pass;
}
