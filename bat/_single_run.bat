@echo off
REM Helper invoked by per-model-per-task .bat files.
REM Args:  %1 = model name, %2 = task name
REM Runs from the judgeSense project root, writes its own log under bat\logs\.

setlocal
cd /d "%~dp0\.."
if not exist "bat\logs" mkdir "bat\logs"
set LOGFILE=bat\logs\%~1_%~2.log

title judgesense %~1 / %~2
echo ============================================================
echo  judgesense  %~1 / %~2  --runs 3
echo ============================================================
echo.
python src/evaluate.py --model %~1 --task %~2 --runs 3
set RC=%ERRORLEVEL%
echo.

if %RC% NEQ 0 (
    echo.
    echo ============================================================
    echo  FAILED  -  %~1 / %~2  (exit %RC%) - see %LOGFILE%
    echo ============================================================
    pause
) else (
    echo.
    echo ============================================================
    echo  DONE  -  %~1 / %~2  (press any key to close)
    echo ============================================================
    pause
)
