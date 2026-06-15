// visemeTimeline.js
// PURE LOGIC — no React, no Pixi, no DOM. This module is the single place that
// answers "given a time t (seconds), which viseme is active?" against a sorted
// timeline. Kept dependency-free so it is unit-testable and reusable.
//
// Timeline shape (from full_stella_timeline.json):
//   { duration, timeline: [ { t_start, t_end, viseme, src }, ... ] }
// Entries are sorted by t_start and contiguous. "viseme" is a NUMBERED id
// (1..10, scheme "blair-10-numbered"). 10 is the rest / closed mouth and is the
// fallback whenever no entry is active (before the clip, in gaps, or after end).
// (Numbers, not letters, so a pose id can never collide with a phoneme letter.)

export const REST_VISEME = 10;

/**
 * Find the index of the active timeline entry for time t using binary search.
 * Returns -1 if t is before the first entry, past the last entry, or otherwise
 * falls in no entry's [t_start, t_end) half-open interval.
 *
 * @param {number} t - time in seconds
 * @param {Array<{t_start:number,t_end:number,viseme:string}>} timeline - sorted by t_start
 * @returns {number} index into timeline, or -1
 */
export function findActiveIndex(t, timeline) {
  if (!timeline || timeline.length === 0) {
    return -1;
  }
  if (t < timeline[0].t_start) {
    return -1;
  }

  let lo = 0;
  let hi = timeline.length - 1;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const entry = timeline[mid];

    if (t < entry.t_start) {
      hi = mid - 1;
    } else if (t >= entry.t_end) {
      lo = mid + 1;
    } else {
      // t_start <= t < t_end : active entry
      return mid;
    }
  }

  return -1;
}

/**
 * Given a time t and a sorted timeline, return the active viseme id.
 * Falls back to REST_VISEME (10) whenever nothing is active.
 *
 * @param {number} t - time in seconds
 * @param {Array} timeline - sorted by t_start
 * @returns {number} viseme id (1..10)
 */
export function visemeAt(t, timeline) {
  const idx = findActiveIndex(t, timeline);
  if (idx === -1) {
    return REST_VISEME;
  }
  return timeline[idx].viseme;
}
