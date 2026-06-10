#!/usr/bin/env bash
# play-clip.sh — render a line in an ad-hoc voice via OmniVoice and serve it for browser playback.
#
# Prototype player primitive. OmniVoice just RETURNS wav bytes (see contract below); this helper
# is a CONSUMER that routes them to the browser by dropping the result in the service's static dir
# and printing the Tailscale URL. Host-speaker playback (target=host) and Workbench audio-bridge
# routing (duck music + play) are FUTURE consumer-side modes — NOT engine changes. The engine never
# plays audio or knows a "target"; the consumer decides what to do with the bytes.
#
# Usage:
#   ./play-clip.sh <ref_wav> "<text to speak>" [options]
# Options:
#   --strength none|light|full   reword strength (default: none = verbatim)
#   --personality <id>           personality id for reword (default: friendly; ignored if none)
#   --voice <id>                 use a registered preset/custom voice INSTEAD of a ref wav
#                                (pass "-" as <ref_wav> to skip the upload and use --voice)
#   --ref-text "<transcript>"    transcript for the ref wav (skips STT if given)
#   --speed <f> --guidance <f> --temperature <f> --steps <n>   tuning (defaults 1.0/2.0/0.0/32)
#   --host <h>                   service host:port (default 127.0.0.1:8221)
#   --base-url <url>             public base for the printed link (default Tailscale front)
#
# Examples:
#   ./play-clip.sh kim-huat.wav "Where we landed, the seam is done."
#   ./play-clip.sh kim-huat.wav "Where we landed, the seam is done." --strength full --personality genx
#   ./play-clip.sh - "Hello there." --voice f_us            # registered preset, no ref wav

set -euo pipefail

HOST="127.0.0.1:8221"
BASE_URL="https://vrpc-3.tail567253.ts.net"
STRENGTH="none"
PERSONALITY="friendly"
VOICE="f_us"
REF_TEXT=""
SPEED="1.0"; GUIDANCE="2.0"; TEMPERATURE="0.0"; STEPS="32"

REF_WAV="${1:-}"; shift || true
TEXT="${1:-}"; shift || true

if [[ -z "$REF_WAV" || -z "$TEXT" ]]; then
  echo "usage: play-clip.sh <ref_wav|-> \"<text>\" [--strength none|light|full] [--personality id] [--voice id] [--ref-text ...] [tuning]" >&2
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strength)    STRENGTH="$2"; shift 2;;
    --personality) PERSONALITY="$2"; shift 2;;
    --voice)       VOICE="$2"; shift 2;;
    --ref-text)    REF_TEXT="$2"; shift 2;;
    --speed)       SPEED="$2"; shift 2;;
    --guidance)    GUIDANCE="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --steps)       STEPS="$2"; shift 2;;
    --host)        HOST="$2"; shift 2;;
    --base-url)    BASE_URL="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

# Deterministic output name from inputs so repeat renders overwrite (clean static dir).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (working/.. )
STATIC_DIR="$HERE/static"
SLUG="$(echo "${REF_WAV##*/}-$STRENGTH-$PERSONALITY-$VOICE" | tr -c 'a-zA-Z0-9._-' '_' )"
OUT_NAME="_clip_${SLUG}.wav"
OUT_PATH="$STATIC_DIR/$OUT_NAME"

echo "[render] host=$HOST strength=$STRENGTH personality=$PERSONALITY voice=${REF_WAV/=-/$VOICE}" >&2

# Build the curl form. Ad-hoc wav unless REF_WAV is "-" (then use the registered --voice).
ARGS=( -s -m 180 -X POST "http://$HOST/api/prototype/speak"
       -F "text=$TEXT" -F "strength=$STRENGTH" -F "personality=$PERSONALITY"
       -F "speed=$SPEED" -F "guidance=$GUIDANCE" -F "temperature=$TEMPERATURE" -F "steps=$STEPS" )
if [[ "$REF_WAV" == "-" ]]; then
  ARGS+=( -F "voice=$VOICE" )
else
  [[ -f "$REF_WAV" ]] || { echo "ref wav not found: $REF_WAV" >&2; exit 1; }
  ARGS+=( -F "wav=@$REF_WAV;type=audio/wav" )
  [[ -n "$REF_TEXT" ]] && ARGS+=( -F "ref_text=$REF_TEXT" )
fi

HDRS="$(mktemp)"
CODE="$(curl "${ARGS[@]}" -D "$HDRS" -o "$OUT_PATH" -w '%{http_code}')"
if [[ "$CODE" != "200" ]]; then
  echo "[error] HTTP $CODE" >&2; cat "$OUT_PATH" >&2; echo >&2; rm -f "$HDRS"; exit 1
fi

# Surface the actually-spoken (reworded) text from the header.
SPOKEN="$(grep -i '^x-spoken-text:' "$HDRS" | sed 's/^[^:]*: *//' | tr -d '\r' | python3 -c 'import sys,urllib.parse;print(urllib.parse.unquote(sys.stdin.read().strip()))' 2>/dev/null || true)"
rm -f "$HDRS"

BYTES="$(wc -c < "$OUT_PATH")"
echo "[ok] ${BYTES} bytes -> $OUT_NAME" >&2
[[ -n "$SPOKEN" ]] && echo "[spoken] $SPOKEN" >&2
echo "$BASE_URL/$OUT_NAME"
