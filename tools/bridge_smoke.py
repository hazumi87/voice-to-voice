"""Pre-restart smoke for the Dot bridge: import the module AND build a VoiceBridge for
every enabled device with a fake API client, using the interpreter the service runs on.

  python tools\\bridge_smoke.py        (the system python has webrtcvad/aioesphomeapi;
                                        the server's .venv does not)

Exit 1 on any failure. Added after three restart crashes on 2026-09-28 that py_compile
could not see (an attribute read before __init__ set it, a module name used before its
definition, and the same again): compile checks syntax, this checks construction.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

BRIDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "working", "atom_echo", "esphome")


class FakeClient:
    def __getattr__(self, name):
        return lambda *a, **k: None


def main() -> int:
    sys.path.insert(0, BRIDGE_DIR)
    os.chdir(BRIDGE_DIR)
    try:
        m = importlib.import_module("va_bridge")
    except Exception as e:  # noqa: BLE001
        print(f"IMPORT FAILED: {e!r}")
        return 1
    with open(os.path.join(BRIDGE_DIR, "va_bridge_devices.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    bad = 0
    for dev in cfg.get("devices", []):
        if not dev.get("enabled", True):
            continue
        try:
            b = m.VoiceBridge(FakeClient(), dev, cfg)
            print(f"OK {dev['id']}: hang {b.silence_hang}s room {b.silence_hang_room}s "
                  f"no_speech {b.no_speech_timeout}s fillers {len(b._fillers)}")
        except Exception as e:  # noqa: BLE001
            print(f"CONSTRUCT FAILED {dev.get('id')}: {e!r}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
