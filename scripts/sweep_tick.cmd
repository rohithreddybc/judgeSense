@echo off
REM One sweep tick, launched by the Windows scheduled task "JudgeSense-Sweep".
REM Kept as a .cmd because schtasks quoting around a path with spaces plus
REM arguments is fragile; this wrapper takes none.
REM
REM Register with:
REM   schtasks /Create /TN JudgeSense-Sweep /SC MINUTE /MO 30 /F ^
REM            /TR "\"<repo>\scripts\sweep_tick.cmd\""
REM Remove with:
REM   schtasks /Delete /TN JudgeSense-Sweep /F

setlocal
set REPO=%~dp0..
set PYTHONIOENCODING=utf-8
cd /d "%REPO%"
"C:\Users\rohit\anaconda3\python.exe" -u "%REPO%\scripts\sweep_tick.py" --max-minutes 27 >> "%REPO%\logs\sweep_tick.out" 2>&1
endlocal
