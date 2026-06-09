@echo off
REM Voice-to-voice prototype launcher (VRPC).
REM Starts the FastAPI server; tailscale serve already fronts it over HTTPS.
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
".venv\Scripts\python.exe" server.py
