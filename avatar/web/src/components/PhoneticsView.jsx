// PhoneticsView.jsx
// PoC2 alignment prototype, rendered as a visual strip. Shows HOW speech becomes
// a viseme timeline: faster-whisper gives word [start,end] timings; each word is
// spread into viseme ids via the grapheme map; gaps become rest. This view makes
// that timeline legible at a glance (each block = one viseme frame, width = its
// duration, colour = its id), with the source words tracked underneath.
//
// Pure HTML/CSS — no Pixi, no canvas. Reads the same /data/*.json the player uses.

import { useEffect, useMemo, useState } from 'react';

// public/ is served under Vite's base (BASE_URL = '/avatar/' in the built app);
// absolute '/data/...' would 404 under the /avatar/ mount.
const BASE = import.meta.env.BASE_URL;
const TIMELINE_URL = `${BASE}data/full_stella_timeline.json`;
const VISEME_MAP_URL = `${BASE}data/viseme_map.json`;

// Distinct colour per viseme id (10 = rest is muted grey).
const VISEME_COLOR = {
  1: '#6c5ce7', 2: '#0984e3', 3: '#00b894', 4: '#e17055',
  5: '#fdcb6e', 6: '#e84393', 7: '#e84393', 8: '#00cec9',
  9: '#a29bfe', 10: '#2d3540',
};

export default function PhoneticsView() {
  const [clip, setClip] = useState(null);
  const [vmap, setVmap] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [c, m] = await Promise.all([
          fetch(TIMELINE_URL).then((r) => r.json()),
          fetch(VISEME_MAP_URL).then((r) => r.json()),
        ]);
        if (!cancelled) {
          setClip(c);
          setVmap(m);
        }
      } catch (e) {
        if (!cancelled) setErr(e?.message || String(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const stats = useMemo(() => {
    if (!clip) return null;
    const tl = clip.timeline;
    const rest = 10;
    const speaking = tl
      .filter((f) => f.viseme !== rest)
      .reduce((s, f) => s + (f.t_end - f.t_start), 0);
    return {
      frames: tl.length,
      duration: clip.duration,
      speakingPct: Math.round((speaking / clip.duration) * 100),
    };
  }, [clip]);

  if (err) {
    return <p style={{ color: '#ff6b6b', fontFamily: 'sans-serif' }}>Load error: {err}</p>;
  }
  if (!clip || !vmap) {
    return <p style={{ color: '#7a8aa0', fontFamily: 'sans-serif' }}>Loading alignment…</p>;
  }

  const dur = clip.duration || 1;
  const W = 880; // strip width in px

  return (
    <div style={{ width: W, fontFamily: 'sans-serif', color: '#cfd8dc' }}>
      <p style={{ fontSize: '13px', lineHeight: 1.5, color: '#9fb0c0' }}>
        “{clip.text}”
      </p>

      {/* viseme strip */}
      <div
        style={{
          position: 'relative',
          height: 56,
          width: W,
          background: '#0f1626',
          borderRadius: 6,
          overflow: 'hidden',
          border: '1px solid #1f2937',
        }}
      >
        {clip.timeline.map((f, i) => {
          const left = (f.t_start / dur) * W;
          const w = Math.max(1, ((f.t_end - f.t_start) / dur) * W);
          return (
            <div
              key={i}
              title={`${f.t_start.toFixed(2)}–${f.t_end.toFixed(2)}s · id ${f.viseme} · ${f.src}`}
              style={{
                position: 'absolute',
                left,
                width: w,
                top: 0,
                bottom: 0,
                background: VISEME_COLOR[f.viseme] || '#555',
                borderRight: '1px solid rgba(0,0,0,0.35)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                color: f.viseme === 10 ? '#5a6b7a' : 'rgba(0,0,0,0.7)',
                fontWeight: 700,
              }}
            >
              {w > 11 ? f.viseme : ''}
            </div>
          );
        })}
      </div>

      {/* time axis */}
      <div style={{ position: 'relative', height: 18, width: W, fontSize: 10, color: '#6b7a8a' }}>
        {Array.from({ length: Math.floor(dur) + 1 }).map((_, s) => (
          <span key={s} style={{ position: 'absolute', left: (s / dur) * W }}>
            {s}s
          </span>
        ))}
      </div>

      {/* legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
        {Object.entries(vmap.visemes)
          .filter(([, v]) => !('alias_of' in v))
          .map(([id, v]) => (
            <span key={id} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
              <span
                style={{
                  width: 12, height: 12, borderRadius: 3,
                  background: VISEME_COLOR[id] || '#555', display: 'inline-block',
                }}
              />
              {id} · {v.label}
            </span>
          ))}
      </div>

      {/* stats */}
      {stats && (
        <div
          style={{
            marginTop: 12, fontFamily: 'monospace', fontSize: 12,
            color: '#7a8aa0', background: '#0f1626', padding: '8px 12px', borderRadius: 6,
          }}
        >
          scheme: {clip.scheme} · frames: {stats.frames} · duration: {stats.duration.toFixed(2)}s ·
          {' '}mouth-moving: {stats.speakingPct}%
        </div>
      )}
    </div>
  );
}
