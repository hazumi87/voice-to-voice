#!/usr/bin/env bash
# Overnight stability watch for voice-to-voice (:8221).
# Runs from the NUC (reliable path to the VRPC). Every INTERVAL it:
#   1. GET /health        — liveness + gpu.busy + in-flight op
#   2. POST /synthesize    — exercises the GPU tts.generate path (the wedge path),
#                            rotating through the 3 custom voices. WAV bytes = proof.
#   3. records VRAM-free is N/A from NUC, so we just log health's gpu field.
# On ANY failure it flips to rapid re-probe so we capture the moment it dies, and
# notes it loudly in the log. No speakers / browser / listener needed.
#
# Stop condition: runs until STOP_EPOCH (passed in), then exits.

VRPC="${VRPC:-100.108.144.20}"
BASE="http://$VRPC:8221"
LOG="${LOG:-/tmp/v2v-overnight.log}"
INTERVAL="${INTERVAL:-600}"        # 10 min between healthy probes
STOP_EPOCH="${STOP_EPOCH:-0}"      # unix epoch to stop at (0 = ~9h from start)
VOICES=(cust_alle cust_sammuel cust_unclelo)

now() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(now)] $*" | tee -a "$LOG"; }

if [ "$STOP_EPOCH" = "0" ]; then STOP_EPOCH=$(( $(date +%s) + 32400 )); fi

say "=== overnight watch start. base=$BASE interval=${INTERVAL}s stop=$(date -d @$STOP_EPOCH '+%H:%M') ==="
i=0
fails=0
while [ "$(date +%s)" -lt "$STOP_EPOCH" ]; do
  i=$((i+1))
  voice="${VOICES[$(( i % 3 ))]}"

  # 1. health
  hcode=$(curl -s -o /tmp/v2v_h.json -m 8 -w "%{http_code}" "$BASE/health")
  if [ "$hcode" != "200" ]; then
    fails=$((fails+1))
    say "!!! PROBE $i HEALTH FAIL http=$hcode (fail #$fails) — capturing crash state"
    say "    (service not answering /health — likely wedged or restarting)"
    # rapid re-probe for 60s to see if it self-recovers or stays dead
    for r in 1 2 3 4 5 6; do
      sleep 10
      rc=$(curl -s -o /dev/null -m 8 -w "%{http_code}" "$BASE/health")
      say "    re-probe $r: http=$rc"
      if [ "$rc" = "200" ]; then say "    recovered after ~$((r*10))s"; break; fi
    done
    sleep "$INTERVAL"
    continue
  fi
  gpu=$(python3 -c "import json;d=json.load(open('/tmp/v2v_h.json'));print(d.get('gpu'))" 2>/dev/null)

  # 2. synth (the GPU path) — time it, capture failure
  t0=$(date +%s.%N)
  scode=$(curl -s -o /tmp/v2v_out.wav -m 45 -w "%{http_code}" -X POST "$BASE/synthesize" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"Overnight probe number $i. Voice $voice.\",\"voice\":\"$voice\"}")
  t1=$(date +%s.%N)
  dur=$(python3 -c "print(f'{$t1-$t0:.1f}')" 2>/dev/null)
  bytes=$(stat -c %s /tmp/v2v_out.wav 2>/dev/null || echo 0)
  magic=$(head -c 4 /tmp/v2v_out.wav 2>/dev/null)

  if [ "$scode" = "200" ] && [ "$bytes" -gt 1000 ] && [ "$magic" = "RIFF" ]; then
    say "probe $i OK  voice=$voice synth=${dur}s bytes=$bytes gpu=$gpu"
  else
    fails=$((fails+1))
    say "!!! PROBE $i SYNTH FAIL http=$scode bytes=$bytes magic='$magic' dur=${dur}s gpu=$gpu (fail #$fails)"
    if [ "$scode" = "503" ]; then
      say "    503 = gpu_busy (timeout insurance fired — an op held the GPU past the limit). GOOD: service survived."
    fi
  fi
  sleep "$INTERVAL"
done
say "=== overnight watch done. probes=$i failures=$fails ==="
