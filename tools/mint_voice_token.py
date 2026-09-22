"""Mint (or rotate) a bearer token for a voice-channel caller.

Usage:  .venv\\Scripts\\python.exe tools\\mint_voice_token.py <caller-name>

Writes/updates voice_auth.json next to server.py ({"tokens": {name: token}}), which
server.py hot-reloads on mtime change — no restart. Prints the token ONCE; hand it to
the caller sealed (harbor deliver_sealed), never over a relay file or chat.
The file is git-ignored. Re-running for the same name ROTATES that caller's token.
"""
import json
import os
import secrets
import sys


def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not name:
        print("usage: mint_voice_token.py <caller-name>", file=sys.stderr)
        sys.exit(2)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "voice_auth.json")
    data = {"tokens": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    tokens = data.setdefault("tokens", {})
    rotated = name in tokens
    tokens[name] = secrets.token_hex(32)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    print(f"{'rotated' if rotated else 'minted'} token for '{name}' in {path}", file=sys.stderr)
    print(tokens[name])


if __name__ == "__main__":
    main()
