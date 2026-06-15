// MouthStage.jsx
// RENDERING tier. Lives INSIDE <Application> so it can use the Pixi ticker via
// useTick. It owns no clock logic and no timeline math beyond calling the pure
// visemeAt() lookup. Every frame it:
//   1) asks the clock master for the current time (readTime())
//   2) advances the virtual clock if we are in fallback mode (advance())
//   3) resolves the active viseme via the pure lookup
//   4) swaps the mouth sprite's texture (instant, because all 9 are preloaded)
//
// @pixi/react v8 does NOT export <Sprite>/<Container> as components — we extend()
// the pixi.js classes and use them as lowercase JSX (<container>, <sprite>).

import { extend, useTick } from '@pixi/react';
import { Container, Sprite, Texture } from 'pixi.js';
import { useRef } from 'react';
import { visemeAt } from '../lib/visemeTimeline.js';

extend({ Container, Sprite, Texture });

const DEBUG_VISEME_LOG = true; // log viseme changes (project standard: observable behavior)
// Whole-frame-swap mode: each viseme sprite is a FULL-FACE pose, so the sprite
// fills the stage. (If sprites later become isolated mouth tiles over a static
// face, switch this back to a centered fixed size.)

/**
 * @param {object} props
 * @param {number} props.stageWidth
 * @param {number} props.stageHeight
 * @param {Object<string, import('pixi.js').Texture>} props.textures - viseme id -> Texture (all 9 preloaded)
 * @param {Array} props.timeline - sorted timeline entries
 * @param {() => number} props.readTime - clock master: returns current clip time in seconds
 * @param {(deltaSeconds:number) => void} props.advance - virtual-clock integrator (no-op in real-audio mode)
 * @param {(state:{ time:number, viseme:string, frameIndex:number }) => void} props.onFrame - readout sink
 */
export default function MouthStage({
  stageWidth,
  stageHeight,
  textures,
  timeline,
  readTime,
  advance,
  onFrame,
}) {
  const spriteRef = useRef(null);
  const lastVisemeRef = useRef(null);

  useTick((ticker) => {
    // Pixi v8: ticker.deltaMS is the elapsed wall time for this frame in ms.
    const deltaSeconds = ticker.deltaMS / 1000;

    // In virtual-clock mode this integrates time; in real-audio mode it is a
    // no-op (the <audio> element advances its own currentTime).
    advance(deltaSeconds);

    const t = readTime();
    const viseme = visemeAt(t, timeline);

    if (viseme !== lastVisemeRef.current) {
      lastVisemeRef.current = viseme;
      const tex = textures[viseme];
      if (spriteRef.current && tex) {
        spriteRef.current.texture = tex;
      }
      if (DEBUG_VISEME_LOG) {
        console.log(`[viseme] t=${t.toFixed(3)}s -> ${viseme}`);
      }
    }

    // Push readout state up. frameIndex is the active timeline row (-1 if none).
    if (onFrame) {
      let frameIndex = -1;
      for (let i = 0; i < timeline.length; i += 1) {
        if (t >= timeline[i].t_start && t < timeline[i].t_end) {
          frameIndex = i;
          break;
        }
      }
      onFrame({ time: t, viseme, frameIndex });
    }
  });

  // Full-face sprite fills the stage. anchor 0.5 + centered position keeps it
  // put regardless of which pose texture is swapped in.
  return (
    <container>
      <sprite
        ref={spriteRef}
        texture={textures['10']}
        anchor={0.5}
        x={stageWidth / 2}
        y={stageHeight / 2}
        width={stageWidth}
        height={stageHeight}
      />
    </container>
  );
}
