// visemeTimeline.smoke.node.mjs
// Standalone CLI runner for the timeline smoke check. Kept separate from the
// browser-safe visemeTimeline.smoke.js so node builtins (fs/url/path) and
// top-level await never reach the Vite bundle.
//
// Run:  node src/lib/visemeTimeline.smoke.node.mjs

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { runSmokeCheck } from './visemeTimeline.smoke.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataPath = join(__dirname, '..', '..', 'public', 'data', 'full_stella_timeline.json');
const data = JSON.parse(readFileSync(dataPath, 'utf8'));

const pass = runSmokeCheck(data.timeline);
process.exit(pass ? 0 : 1);
