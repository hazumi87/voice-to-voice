"""GPU traffic cop client (harbor gpu-cop/1, harbor docs/gpu-cop-schema.md).

Logic tier only: no FastAPI, no torch. server.py builds /gpu/status from these helpers and
calls the load gates before a model load.

The shared load rule (schema §4): before loading a model, read the cop's /gpu/state with a
2 s timeout and load only if free_mib - reserved_mib >= my floor_mib. If the cop is down,
slow, disabled or can't read the GPU, return "allowed" and let the caller's own local floor
guard decide. The cop being down never blocks voice.

Two gates:
- tts_gate(): before OmniVoice loads. It SUBTRACTS v2v's own restart reservation from
  reserved_mib, because a service must not wait on the space held for itself.
- router_gate(): before an Ollama call that would LOAD the router (the model isn't in
  api/ps yet). It does NOT subtract v2v's own reservation: the router loading into the gap
  while OmniVoice restarts is exactly the race the reservation exists to stop (four times
  on 2026-09-28).
"""
import json
import os
import threading
import time
import urllib.request

SERVICE = "voice-to-voice"
COP_STATE_URL = os.environ.get("GPU_COP_URL", "http://127.0.0.1:8210/gpu/state")
COP_TIMEOUT_S = 2.0
OLLAMA_PS_URL = os.environ.get("OLLAMA_PS_URL", "http://127.0.0.1:11434/api/ps")
# Router START FLOOR: llama3.2:3b is 2,550 MiB resident (measured, api/ps); the rest is
# headroom for the load. Override per model via env when the 8B is trialled.
ROUTER_FLOOR_MIB = int(os.environ.get("ROUTER_FLOOR_MIB", "3000"))
MIB = 1024 * 1024

_ps_cache = {"at": 0.0, "models": None}
_ps_lock = threading.Lock()


class RouterGpuWait(Exception):
    """The router can't load right now. str(e) is a speakable reason."""


def _get_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def cop_state():
    """The cop's /gpu/state, or None when it's unreachable, slow, disabled or can't read
    the GPU (all of those mean 'today's behaviour')."""
    try:
        st = _get_json(COP_STATE_URL, COP_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - cop down never blocks an app
        return None
    if not isinstance(st, dict) or st.get("enabled") is False:
        return None
    if not (st.get("gpu") or {}).get("ok"):
        return None
    return st


def _own_reserved_mib(st: dict) -> int:
    """This service's contribution to gpu.reserved_mib: max(0, floor - held) over its
    ENFORCED reservations (the cop counts only enforced ones)."""
    mine = [r for r in (st.get("reservations") or [])
            if r.get("service") == SERVICE and r.get("enforced")]
    if not mine:
        return 0
    row = next((s for s in (st.get("services") or []) if s.get("name") == SERVICE), {})
    held = int(row.get("held_mib") or 0)
    return sum(max(0, int(r.get("floor_mib") or 0) - held) for r in mine)


def _blocker_text(st: dict, exclude_self: bool) -> str:
    """Who is holding the space, in words a person can act on."""
    for r in st.get("reservations") or []:
        if exclude_self and r.get("service") == SERVICE:
            continue
        if r.get("enforced"):
            return f"{r.get('service')} is restarting"
    # The biggest holder across harbor services AND unmanaged processes (an unmanaged
    # Ollama runner can outweigh every service). An unmanaged row attributed to a service
    # is named after that service.
    holders = [(s.get("held_mib") or 0, s.get("name")) for s in (st.get("services") or [])
               if s.get("name") != SERVICE]
    holders += [(u.get("mib") or 0, (f"{u['for_service']} ({u.get('name')})" if u.get("for_service")
                                     else u.get("name")))
                for u in (st.get("unmanaged") or []) if u.get("for_service") != SERVICE]
    holders = [h for h in holders if h[0] > 0]
    if holders:
        mib, name = max(holders)
        return f"{name} is holding {mib / 1024:.1f} GB"
    return "another app is using it"


def check_load(floor_mib: int, exclude_self: bool):
    """Schema §4. Returns (allowed, reason, headroom_mib). allowed=True with reason None
    means the cop is unavailable, so the caller's local guard decides."""
    st = cop_state()
    if st is None:
        return True, None, None
    gpu = st["gpu"]
    reserved = int(gpu.get("reserved_mib") or 0)
    if exclude_self:
        reserved = max(0, reserved - _own_reserved_mib(st))
    headroom = int(gpu.get("free_mib") or 0) - reserved
    if headroom >= floor_mib:
        return True, None, headroom
    return False, _blocker_text(st, exclude_self), headroom


def tts_gate(floor_mib: int, own_cache_mib: int = 0):
    """Before OmniVoice loads. Returns None to proceed, else a load_error string.
    own_cache_mib: memory this process already holds in torch's cache but isn't using. The
    load reuses it, so it counts as free for us (it's invisible in nvidia-smi's free)."""
    ok, why, headroom = check_load(max(0, floor_mib - own_cache_mib), exclude_self=True)
    if headroom is not None:
        headroom += own_cache_mib
    if ok:
        return None
    return (f"waiting for GPU: {why} (need {floor_mib / 1024:.1f} GB free, "
            f"{max(0, headroom) / 1024:.1f} GB available)")


def ollama_models():
    """{model_name: size_vram_mib} from Ollama api/ps (cached 3 s). None if Ollama is down."""
    with _ps_lock:
        if time.time() - _ps_cache["at"] < 3.0:
            return _ps_cache["models"]
    try:
        d = _get_json(OLLAMA_PS_URL, 2.0)
        models = {m.get("name") or m.get("model"): int((m.get("size_vram") or 0) / MIB)
                  for m in d.get("models") or []}
    except Exception:  # noqa: BLE001
        models = None
    with _ps_lock:
        _ps_cache.update(at=time.time(), models=models)
    return models


def _norm(name: str) -> str:
    return name if ":" in name else name + ":latest"


def router_resident_mib(model_names) -> int | None:
    """VRAM Ollama holds for v2v's router models (the schema's router_resident_mib)."""
    models = ollama_models()
    if models is None:
        return None
    want = {_norm(m) for m in model_names if m}
    return sum(v for k, v in models.items() if k and _norm(k) in want)


def router_gate(model: str) -> None:
    """Before an Ollama call. Resident model -> no load -> no check. Otherwise apply §4
    with the router floor; raise RouterGpuWait with a speakable reason when it can't fit."""
    models = ollama_models()
    if models is not None and (models.get(_norm(model)) or 0) > 0:
        return
    ok, why, headroom = check_load(ROUTER_FLOOR_MIB, exclude_self=False)
    if ok:
        return
    print(f"[gpu] router gate: not loading {model}: {why} "
          f"(headroom {headroom} MiB < {ROUTER_FLOOR_MIB})", flush=True)
    raise RouterGpuWait(f"The GPU is busy right now: {why}. Try me again in a minute.")
