// Avatar3D.jsx
// PoC: 3D Ready Player Me head driven by the Stella phonetics timeline.
//
// Key proof point: SMOOTH weight interpolation (lerp each frame) vs the 2D
// sprite-swap which snaps hard between poses. The smoothing factor k is
// frame-rate-aware using delta time.
//
// Controls:
//   - Play/Pause toggle
//   - Restart
//   - Scrub slider
//   - Playback speed  0.1x – 1.0x (virtual clock only, no audio pitch-shift)
//   - Live readout: time / duration, active viseme id + target name, speed
//
// Architecture:
//   - three.js scene set up in a useEffect, torn down on unmount
//   - RAF loop runs independently of React render cycle
//   - State flows out via setState only for the readout (low freq)

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BASE = import.meta.env.BASE_URL;
const MODEL_URL = `${BASE}models/rpm_head.glb`;
const TIMELINE_URL = `${BASE}data/full_stella_timeline.json`;

// Blair-10-numbered viseme id -> Oculus morph target name on Wolf3D_Head / Wolf3D_Teeth
// id 7 aliases 6 (W/Q), id 10 = sil/rest
const VISEME_MAP = {
  1:  'viseme_PP',   // MBP bilabial closed
  2:  'viseme_DD',   // general consonant
  3:  'viseme_E',    // eh
  4:  'viseme_aa',   // ah / wide open
  5:  'viseme_O',    // O
  6:  'viseme_U',    // U / pucker
  7:  'viseme_U',    // W/Q alias
  8:  'viseme_FF',   // F / V
  9:  'viseme_nn',   // L / N
  10: 'viseme_sil',  // rest / neutral
};

// All Oculus viseme target names we care about (we drive these to 0 or 1).
const OCULUS_TARGETS = [
  'viseme_sil', 'viseme_PP', 'viseme_FF', 'viseme_TH', 'viseme_DD',
  'viseme_kk',  'viseme_CH', 'viseme_SS', 'viseme_nn', 'viseme_RR',
  'viseme_aa',  'viseme_E',  'viseme_I',  'viseme_O',  'viseme_U',
];

// Easing constant: weight += (goal - weight) * k_per_sec * deltaTime
// At 60fps with k_per_sec = 12, each frame k_frame ≈ 0.18 → ~80ms to 90% of target.
// Frame-rate-aware because we multiply by delta.
const EASE_K_PER_SEC = 12;

// Canvas size
const W = 640;
const H = 480;

export default function Avatar3D() {
  const mountRef = useRef(null);      // div that receives the renderer's canvas
  const sceneRef = useRef(null);       // { scene, camera, renderer, heads[] }
  const clockRef = useRef({            // virtual clock (no audio in this view)
    time: 0,
    playing: false,
    speed: 0.25,
    lastRAF: null,
  });
  const weightsRef = useRef({});       // { targetName: currentWeight }
  const rafIdRef = useRef(null);

  // React state — only for UI display, NOT the hot path
  const [status, setStatus] = useState('loading model…');
  const [readout, setReadout] = useState({
    time: 0,
    duration: 0,
    visemeId: 10,
    targetName: 'viseme_sil',
    speed: 0.25,
  });
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const timelineRef = useRef([]);

  // ---------------------------------------------------------------------------
  // Helpers exposed via callbacks (must read/write refs, not stale closure state)
  // ---------------------------------------------------------------------------
  const togglePlay = useCallback(() => {
    const ck = clockRef.current;
    ck.playing = !ck.playing;
    setPlaying(ck.playing);
  }, []);

  const restart = useCallback(() => {
    clockRef.current.time = 0;
    clockRef.current.playing = false;
    setPlaying(false);
    setReadout((r) => ({ ...r, time: 0, visemeId: 10, targetName: 'viseme_sil' }));
  }, []);

  const seek = useCallback((t) => {
    clockRef.current.time = t;
  }, []);

  const setSpeed = useCallback((s) => {
    clockRef.current.speed = s;
    setReadout((r) => ({ ...r, speed: s }));
  }, []);

  // ---------------------------------------------------------------------------
  // Three.js setup + teardown
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    // --- Scene, camera, lights, renderer ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x16213e);

    const camera = new THREE.PerspectiveCamera(35, W / H, 0.01, 100);
    camera.position.set(0, 0, 2); // will be repositioned after model loads

    const ambient = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(W, H);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    if (mountRef.current) {
      mountRef.current.appendChild(renderer.domElement);
    }

    // Store on ref so the RAF loop can access them
    sceneRef.current = { scene, camera, renderer, heads: [] };

    // Initialize morph weights to 0
    OCULUS_TARGETS.forEach((name) => {
      weightsRef.current[name] = 0;
    });

    // --- Load timeline ---
    fetch(TIMELINE_URL)
      .then((r) => r.json())
      .then((json) => {
        if (cancelled) { return; }
        timelineRef.current = json.timeline;
        const dur = json.duration;
        setDuration(dur);
        clockRef.current.time = 0;
        setReadout((r) => ({ ...r, duration: dur }));
        console.log(`[Avatar3D] timeline loaded: ${json.timeline.length} entries, duration=${dur}s`);
      })
      .catch((err) => {
        console.error('[Avatar3D] timeline load failed:', err);
        if (!cancelled) { setStatus(`Timeline load error: ${err.message}`); }
      });

    // --- Load GLB ---
    const loader = new GLTFLoader();
    loader.load(
      MODEL_URL,
      (gltf) => {
        if (cancelled) { return; }

        // Collect the two head meshes (Wolf3D_Head, Wolf3D_Teeth)
        const heads = [];
        gltf.scene.traverse((obj) => {
          if (
            obj.isMesh &&
            (obj.name === 'Wolf3D_Head' || obj.name === 'Wolf3D_Teeth') &&
            obj.morphTargetDictionary
          ) {
            // Enable morph targets
            obj.morphTargetInfluences = obj.morphTargetInfluences || [];
            // Zero all influences to start
            const count = Object.keys(obj.morphTargetDictionary).length;
            for (let i = 0; i < count; i++) {
              obj.morphTargetInfluences[i] = 0;
            }
            heads.push(obj);
            console.log(
              `[Avatar3D] found mesh: ${obj.name}, targets:`,
              Object.keys(obj.morphTargetDictionary).filter((k) => !k.endsWith('.001'))
            );
          }
        });

        if (heads.length === 0) {
          console.warn('[Avatar3D] no head meshes with morph targets found — check mesh names');
          setStatus('Warning: morph target meshes not found');
        }

        sceneRef.current.heads = heads;

        // --- Auto-frame: position camera to show just the head ---
        // Find Wolf3D_Head for bounding box reference
        const headMesh = heads.find((h) => h.name === 'Wolf3D_Head') || heads[0];
        if (headMesh) {
          headMesh.geometry.computeBoundingBox();
          const box = new THREE.Box3().setFromObject(headMesh);
          const center = new THREE.Vector3();
          box.getCenter(center);
          const size = new THREE.Vector3();
          box.getSize(size);

          // Camera: look at face center, step back enough to frame the head
          // The head is ~0.25m tall typically in RPM; FOV=35° → ~0.4m stand-off
          const headHeight = size.y;
          const vFovRad = (camera.fov * Math.PI) / 180;
          const dist = (headHeight * 1.1) / (2 * Math.tan(vFovRad / 2));

          // Tilt slightly down to show more of the face (eyes/mouth)
          camera.position.set(center.x, center.y + headHeight * 0.05, center.z + dist);
          camera.lookAt(center.x, center.y - headHeight * 0.05, center.z);

          console.log(
            `[Avatar3D] head center: ${center.toArray().map((v) => v.toFixed(3))}`,
            `size: ${size.toArray().map((v) => v.toFixed(3))}`,
            `camera dist: ${dist.toFixed(3)}`
          );
        }

        scene.add(gltf.scene);
        setStatus('ready');
        console.log('[Avatar3D] model loaded and added to scene');
      },
      (progress) => {
        if (!cancelled && progress.total > 0) {
          const pct = Math.round((progress.loaded / progress.total) * 100);
          setStatus(`loading model… ${pct}%`);
        }
      },
      (err) => {
        if (!cancelled) {
          console.error('[Avatar3D] GLB load error:', err);
          setStatus(`Model load error: ${err.message || String(err)}`);
        }
      }
    );

    // --- RAF animation loop ---
    function animate(now) {
      rafIdRef.current = requestAnimationFrame(animate);

      const ck = clockRef.current;
      const sc = sceneRef.current;
      if (!sc) { return; }

      // Delta time (seconds), clamped to avoid spiral-of-death on tab refocus
      const dt = ck.lastRAF != null ? Math.min((now - ck.lastRAF) / 1000, 0.1) : 0;
      ck.lastRAF = now;

      // Advance virtual clock
      if (ck.playing && ck.time < ck.duration) {
        ck.time = Math.min(ck.time + dt * ck.speed, ck.duration ?? Infinity);
        if (ck.time >= (ck.duration ?? Infinity)) {
          ck.playing = false;
          setPlaying(false);
        }
      }

      // Find active timeline entry
      const tl = timelineRef.current;
      const t = ck.time;
      let activeEntry = null;
      for (let i = 0; i < tl.length; i++) {
        const e = tl[i];
        if (t >= e.t_start && t < e.t_end) {
          activeEntry = e;
          break;
        }
      }
      const activeVisemeId = activeEntry ? activeEntry.viseme : 10;
      const activeTarget = VISEME_MAP[activeVisemeId] ?? 'viseme_sil';

      // Smooth lerp of all morph weights
      const k = EASE_K_PER_SEC * dt;
      const w = weightsRef.current;
      OCULUS_TARGETS.forEach((name) => {
        const goal = name === activeTarget ? 1.0 : 0.0;
        w[name] = w[name] + (goal - w[name]) * Math.min(k, 1.0);
      });

      // Apply to both Wolf3D_Head and Wolf3D_Teeth
      sc.heads.forEach((mesh) => {
        const dict = mesh.morphTargetDictionary;
        OCULUS_TARGETS.forEach((name) => {
          // Skip .001 duplicates
          if (dict[name] !== undefined) {
            mesh.morphTargetInfluences[dict[name]] = w[name];
          }
        });
      });

      renderer.render(scene, camera);

      // Update React readout at ~10fps to avoid thrashing
      if (!ck._lastReadout || now - ck._lastReadout > 100) {
        ck._lastReadout = now;
        setReadout({
          time: t,
          duration: ck.duration ?? 0,
          visemeId: activeVisemeId,
          targetName: activeTarget,
          speed: ck.speed,
        });
      }
    }

    rafIdRef.current = requestAnimationFrame(animate);

    // --- Cleanup ---
    return () => {
      cancelled = true;
      if (rafIdRef.current != null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      renderer.dispose();
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
    };
  }, []); // run once on mount

  // Keep duration in the clock ref when it updates from the fetch
  useEffect(() => {
    clockRef.current.duration = duration;
  }, [duration]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const isReady = status === 'ready';

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
      {/* Three.js canvas mount point */}
      <div
        ref={mountRef}
        style={{
          width: W,
          height: H,
          background: '#16213e',
          borderRadius: '4px',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {/* Status overlay while loading */}
        {!isReady && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#7a8aa0',
              fontFamily: 'monospace',
              fontSize: '14px',
              pointerEvents: 'none',
            }}
          >
            {status}
          </div>
        )}
      </div>

      {/* Transport controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <button
          onClick={togglePlay}
          disabled={!isReady}
          style={btnStyle}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={restart} disabled={!isReady} style={btnStyle}>
          Restart
        </button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.05}
          value={readout.time}
          disabled={!isReady}
          onChange={(e) => {
            const t = parseFloat(e.target.value);
            seek(t);
            setReadout((r) => ({ ...r, time: t }));
          }}
          style={{ width: '300px' }}
        />
      </div>

      {/* Speed control */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: 'monospace', fontSize: '13px', color: '#cfd8dc' }}>
        <span>Speed:</span>
        {[0.1, 0.25, 0.5, 0.75, 1.0].map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            style={{
              ...btnStyle,
              background: Math.abs(readout.speed - s) < 0.001 ? '#1f6feb' : '#2a313b',
              color: '#fff',
              padding: '4px 10px',
              fontSize: '12px',
            }}
          >
            {s}x
          </button>
        ))}
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={readout.speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          style={{ width: '120px' }}
          title="Playback speed (0.1x – 1.0x)"
        />
        <span style={{ minWidth: '36px' }}>{readout.speed.toFixed(2)}x</span>
      </div>

      {/* Live readout */}
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
          time: {readout.time.toFixed(3)}s / {readout.duration.toFixed(2)}s
        </div>
        <div>
          viseme id: {readout.visemeId} — target: <strong>{readout.targetName}</strong>
        </div>
        <div style={{ color: '#7a8aa0' }}>
          speed: {readout.speed.toFixed(2)}x · clock: virtual (RAF) ·{' '}
          {playing ? 'playing' : 'paused'}
        </div>
        <div style={{ color: '#4a5a6a', fontSize: '11px', marginTop: '4px' }}>
          easing: lerp k={EASE_K_PER_SEC}/s × Δt per frame (~80ms transition at 60fps)
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
