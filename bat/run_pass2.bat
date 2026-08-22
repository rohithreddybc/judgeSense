@echo off
REM Launches 4 pass-2 models x 4 tasks = 16 sweep processes for the 4 *new* judges,
REM plus 4 windows for the deepseek-r1 re-run at max_tokens=1024 = 20 windows total.
REM Each window logs to bat\logs\<model>_<task>.log.
REM Resumability: each process skips already-completed (pair_id, run) tuples.
REM
REM Provider rate limits are shared across windows that share the same key.
REM Novita carries 8 of these (deepseek + deepseek-v4-flash across 4 tasks each),
REM so expect throttling there. The per-call 0.5s sleep in evaluate.py mitigates this.

setlocal
cd /d "%~dp0"

echo Launching 20 sweep windows...

REM ── deepseek (re-run at max_tokens=1024) ─────────────────────────
start "" cmd /c "deepseek_factuality.bat"
start "" cmd /c "deepseek_coherence.bat"
start "" cmd /c "deepseek_relevance.bat"
start "" cmd /c "deepseek_preference.bat"

REM ── gpt-5.5 ─────────────────────────────────────────────────────
start "" cmd /c "gpt-5.5_factuality.bat"
start "" cmd /c "gpt-5.5_coherence.bat"
start "" cmd /c "gpt-5.5_relevance.bat"
start "" cmd /c "gpt-5.5_preference.bat"

REM ── claude-opus-4-7 ─────────────────────────────────────────────
start "" cmd /c "claude-opus-4-7_factuality.bat"
start "" cmd /c "claude-opus-4-7_coherence.bat"
start "" cmd /c "claude-opus-4-7_relevance.bat"
start "" cmd /c "claude-opus-4-7_preference.bat"

REM ── qwen-3.6-flash (DashScope) ──────────────────────────────────
start "" cmd /c "qwen-3.6-flash_factuality.bat"
start "" cmd /c "qwen-3.6-flash_coherence.bat"
start "" cmd /c "qwen-3.6-flash_relevance.bat"
start "" cmd /c "qwen-3.6-flash_preference.bat"

REM ── deepseek-v4-flash ───────────────────────────────────────────
start "" cmd /c "deepseek-v4-flash_factuality.bat"
start "" cmd /c "deepseek-v4-flash_coherence.bat"
start "" cmd /c "deepseek-v4-flash_relevance.bat"
start "" cmd /c "deepseek-v4-flash_preference.bat"

echo.
echo All 20 windows launched. Tail bat\logs\*.log to monitor progress.
echo Each window auto-closes 10s after success or pauses on failure.
