// useAudioClock.js
// CLOCK MASTER hook. The single source of truth for "where are we in the clip"
// is an HTMLAudioElement's currentTime — NOT Date.now(), NOT the Pixi ticker's
// accumulated time. Audio and mouth must never drift, so the audio element owns
// the clock.
//
// Two modes, chosen automatically by whether the <audio> element has a real,
// playable source:
//
//   1) REAL AUDIO MODE (a file is present, e.g. /audio/full_stella.wav):
//      We just play/pause the <audio> element. currentTime advances naturally as
//      audio plays. readTime() returns audio.currentTime. The audio IS the clock.
//
//   2) VIRTUAL CLOCK FALLBACK (no audio source present yet):
//      There is nothing to play, so we advance a virtual time variable ourselves
//      by the ticker delta each frame (see advance()). readTime() returns that
//      virtual time. This keeps the animation runnable for visual testing.
//
// When a real audio file is dropped into /audio later, mode (1) activates with no
// rework: the same readTime() contract is used by the renderer either way.

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * @param {object} opts
 * @param {React.RefObject<HTMLAudioElement>} opts.audioRef
 * @param {number} opts.duration - clip duration in seconds (used by virtual clock to clamp/loop-stop)
 */
export function useAudioClock({ audioRef, duration }) {
  // Whether the <audio> element actually has a usable source. Decided after the
  // element mounts and (maybe) fires metadata/error events.
  const [hasAudio, setHasAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // Virtual clock state lives in a ref so the Pixi ticker can mutate it every
  // frame without forcing React re-renders.
  const virtualTimeRef = useRef(0);
  const playingRef = useRef(false);

  // Detect whether the audio element has a real source once it mounts.
  useEffect(() => {
    const el = audioRef.current;
    if (!el) {
      return;
    }

    // If no <source>/src resolves, the browser fires 'error' on the element (or
    // never reaches readyState>0). We treat "has duration metadata" as the
    // signal for a real, playable file.
    function onLoadedMetadata() {
      if (Number.isFinite(el.duration) && el.duration > 0) {
        setHasAudio(true);
        console.log(`[clock] real audio detected (duration=${el.duration.toFixed(2)}s) — audio is clock master`);
      }
    }
    function onError() {
      setHasAudio(false);
      console.log('[clock] no playable audio source — using virtual clock fallback');
    }

    el.addEventListener('loadedmetadata', onLoadedMetadata);
    el.addEventListener('error', onError);

    // If the element already has metadata by the time we attach, use it.
    if (Number.isFinite(el.duration) && el.duration > 0) {
      onLoadedMetadata();
    }

    return () => {
      el.removeEventListener('loadedmetadata', onLoadedMetadata);
      el.removeEventListener('error', onError);
    };
  }, [audioRef]);

  // --- Time source --------------------------------------------------------
  // readTime() is the contract the renderer calls every frame. It returns the
  // master time regardless of which mode we are in.
  const readTime = useCallback(() => {
    if (hasAudio && audioRef.current) {
      return audioRef.current.currentTime;
    }
    return virtualTimeRef.current;
  }, [hasAudio, audioRef]);

  // advance() is called by the Pixi ticker each frame. In real-audio mode it is
  // a no-op (the <audio> element advances its own currentTime). In virtual mode
  // it integrates the frame delta into the virtual clock.
  const advance = useCallback((deltaSeconds) => {
    if (hasAudio) {
      return;
    }
    if (!playingRef.current) {
      return;
    }
    let next = virtualTimeRef.current + deltaSeconds;
    if (next >= duration) {
      next = duration;
      playingRef.current = false;
      setIsPlaying(false);
      console.log('[clock] virtual clock reached end of clip');
    }
    virtualTimeRef.current = next;
  }, [hasAudio, duration]);

  // --- Transport ----------------------------------------------------------
  const play = useCallback(() => {
    if (hasAudio && audioRef.current) {
      audioRef.current.play().catch((err) => {
        console.log('[clock] audio.play() rejected:', err?.message || err);
      });
    } else {
      // Restart from 0 if we were parked at the end.
      if (virtualTimeRef.current >= duration) {
        virtualTimeRef.current = 0;
      }
      playingRef.current = true;
    }
    setIsPlaying(true);
    console.log('[clock] play');
  }, [hasAudio, audioRef, duration]);

  const pause = useCallback(() => {
    if (hasAudio && audioRef.current) {
      audioRef.current.pause();
    } else {
      playingRef.current = false;
    }
    setIsPlaying(false);
    console.log('[clock] pause');
  }, [hasAudio, audioRef]);

  const restart = useCallback(() => {
    if (hasAudio && audioRef.current) {
      audioRef.current.currentTime = 0;
    } else {
      virtualTimeRef.current = 0;
    }
    console.log('[clock] restart');
  }, [hasAudio, audioRef]);

  // seek() jumps the master clock to an absolute time (seconds). Used by the
  // scrub slider; the mouth must update immediately, which it does because the
  // renderer reads readTime() every frame.
  const seek = useCallback((t) => {
    const clamped = Math.max(0, Math.min(t, duration));
    if (hasAudio && audioRef.current) {
      audioRef.current.currentTime = clamped;
    } else {
      virtualTimeRef.current = clamped;
    }
  }, [hasAudio, audioRef, duration]);

  // Keep isPlaying in sync if a real audio element ends/pauses on its own.
  useEffect(() => {
    const el = audioRef.current;
    if (!el || !hasAudio) {
      return;
    }
    function onEnded() {
      setIsPlaying(false);
      console.log('[clock] audio ended');
    }
    function onPause() {
      setIsPlaying(false);
    }
    function onPlay() {
      setIsPlaying(true);
    }
    el.addEventListener('ended', onEnded);
    el.addEventListener('pause', onPause);
    el.addEventListener('play', onPlay);
    return () => {
      el.removeEventListener('ended', onEnded);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('play', onPlay);
    };
  }, [audioRef, hasAudio]);

  return { hasAudio, isPlaying, readTime, advance, play, pause, restart, seek };
}
