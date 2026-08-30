@echo off
REM Daily Groq quota run, launched by the Windows scheduled task
REM "JudgeSense-GroqDaily". Kept as a .cmd because schtasks quoting around a
REM path with spaces plus arguments is fragile; this wrapper takes none.
REM
REM Register with:
REM   schtasks /Create /TN JudgeSense-GroqDaily /SC DAILY /ST 09:07 /F ^
REM            /TR "\"<repo>\scripts\groq_daily.cmd\""
REM Remove with:
REM   schtasks /Delete /TN JudgeSense-GroqDaily /F

setlocal
set REPO=%~dp0..
set PYTHONIOENCODING=utf-8
cd /d "%REPO%"
"C:\Users\rohit\anaconda3\python.exe" -u "%REPO%\scripts\groq_daily.py" --max-minutes 330 >> "%REPO%\logs\groq_daily.out" 2>&1
endlocal
