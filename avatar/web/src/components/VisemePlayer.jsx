// VisemePlayer.jsx
// Orchestrator for the lip-sync player engine. Responsibilities:
//   - load timeline + viseme contract (from /data/*.json)
//   - PRELOAD all 9 mouth textures up front so swaps are instant (no flicker)
//   - mount the <audio> clock master (real file if present, else virtual fallback)
//   - host transport controls + a live readout (plain HTML, NOT in Pixi)
//   - wrap the Pixi <Application> with the MouthStage renderer
//
// Decoupling: clock LOGIC lives in useAudioClock; timeline LOGIC lives in
// visemeTimeline; this file is wiring + presentation only.

import { Application } from '@pixi/react';
import { Assets } from 'pixi.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useAudioClock } from '../lib/useAudioClock.js';
import { runSmokeCheck } from '../lib/visemeTimeline.smoke.js';
import MouthStage from './MouthStage.jsx';

const STAGE_W = 640;
const STAGE_H = 480;

// Where data + (optional) audio live. public/ is served under Vite's base
// (BASE_URL = '/avatar/' in the built app), so all asset paths MUST be prefixed
// with it — absolute '/data/...' would 404 under the /avatar/ mount.
const BASE = import.meta.env.BASE_URL;
const TIMELINE_URL = `${BASE}data/full_stella_timeline.json`;
const VISEME_MAP_URL = `${BASE}data/viseme_map.json`;
// Optional audio. If this 404s, the clock falls back to a virtual timer.
const AUDIO_URL = `${BASE}audio/full_stella.wav`;

// Numbered scheme "blair-10-numbered". rendered_set = 9 distinct sprites
// (id 7 W/Q aliases id 6 and is not rendered). 10 = rest / closed mouth.
const VISEME_IDS = ['1', '2', '3', '4', '5', '6', '8', '9', '10'];
const REST_ID = '10';

export default function VisemePlayer() {
  const [clip, setClip] = useState(null);          // { duration, text, timeline, ... }
  const [visemeMap, setVisemeMap] = useState(null); // viseme_map.json
  const [textures, setTextures] = useState(null);   // { A: Texture, ... }
  const [loadError, setLoadError] = useState(null);

  const audioRef = useRef(null);

  // Readout state (driven by MouthStage every frame).
  const [readout, setReadout] = useState({ time: 0, viseme: 10, frameIndex: -1 });

  const duration = clip?.duration ?? 0;
  const timeline = clip?.timeline ?? [];

  const clock = useAudioClock({ audioRef, duration });

  // --- Load data + preload textures up front --------------------------------
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [clipRes, mapRes] = await Promise.all([
          fetch(TIMELINE_URL),
          fetch(VISEME_MAP_URL),
        ]);
        const clipJson = await clipRes.json();
        const mapJson = await mapRes.json();

        // Preload all 9 textures BEFORE first render of the mouth so swapping is
        // instant. We resolve each viseme id's sprite filename from the contract.
        const entries = await Promise.all(
          VISEME_IDS.map(async (id) => {
            const sprite = mapJson.visemes[id]?.sprite;
            const url = `${BASE}sprites/${sprite}`;
            const tex = await Assets.load(url);
            return [id, tex];
          }),
        );

        if (cancelled) {
          return;
        }

        const texMap = Object.fromEntries(entries);
        setClip(clipJson);
        setVisemeMap(mapJson);
        setTextures(texMap);

        // In-app smoke check — proves the pure lookup against the real timeline.
        // Observable in the browser console (and gated to dev usefulness).
        runSmokeCheck(clipJson.timeline);

        console.log(`[player] loaded: ${clipJson.timeline.length} timeline entries, duration=${clipJson.duration}s, ${entries.length} textures preloaded`);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err?.message || String(err));
          console.error('[player] load failed:', err);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const visemeLabel = useMemo(() => {
    if (!visemeMap) {
      return '';
    }
    return visemeMap.visemes[readout.viseme]?.label ?? '';
  }, [visemeMap, readout.viseme]);

  if (loadError) {
    return (
      <p style={{ color: '#ff6b6b', fontFamily: 'sans-serif' }}>
        Load error: {loadError}
      </p>
    );
  }

  const ready = clip && textures;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
        padding: '16px',
      }}
    >
      {/* Clock master. No <source> tag is rendered if the file may be absent;
          we set src directly so a 404 triggers the element's 'error' event and
          the virtual-clock fallback kicks in. Drop a real file at AUDIO_URL and
          it becomes the master with no code change. */}
      <audio ref={audioRef} src={AUDIO_URL} preload="auto" />

      <div style={{ position: 'relative', width: STAGE_W, height: STAGE_H }}>
        <Application width={STAGE_W} height={STAGE_H} background={0x16213e} antialias={true}>
          {ready ? (
            <MouthStage
              stageWidth={STAGE_W}
              stageHeight={STAGE_H}
              textures={textures}
              timeline={timeline}
              readTime={clock.readTime}
              advance={clock.advance}
              onFrame={setReadout}
            />
          ) : null}
        </Application>
      </div>

      {/* --- Transport controls (plain HTML, below the canvas) --- */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button onClick={clock.play} disabled={!ready} style={btnStyle}>
          Play
        </button>
        <button onClick={clock.pause} disabled={!ready} style={btnStyle}>
          Pause
        </button>
        <button
          onClick={() => {
            clock.restart();
            setReadout((r) => ({ ...r, time: 0 }));
          }}
          disabled={!ready}
          style={btnStyle}
        >
          Restart
        </button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={readout.time}
          disabled={!ready}
          onChange={(e) => {
            const t = parseFloat(e.target.value);
            clock.seek(t);
            // Update readout time immediately so the slider tracks even while paused.
            setReadout((r) => ({ ...r, time: t }));
          }}
          style={{ width: '320px' }}
        />
      </div>

      {/* --- Live readout --- */}
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: '13px',
          color: '#cfd8dc',
          background: '#0f1626',
          padding: '8px 12px',
          borderRadius: '6px',
          minWidth: '420px',
          textAlign: 'left',
        }}
      >
        <div>
          time: {readout.time.toFixed(3)}s / {duration.toFixed(2)}s
        </div>
        <div>
          viseme: {readout.viseme} — {visemeLabel}
        </div>
        <div>frame index: {readout.frameIndex}</div>
        <div style={{ color: '#7a8aa0' }}>
          clock master: {clock.hasAudio ? 'audio file' : 'virtual (no audio file)'} ·{' '}
          {clock.isPlaying ? 'playing' : 'paused'}
        </div>
      </div>
    </div>
  );
}

const btnStyle = {
  fontFamily: 'sans-serif',
  fontSize: '13px',
  padding: '6px 14px',
  background: '#2ec4b6',
  color: '#06202a',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
};
