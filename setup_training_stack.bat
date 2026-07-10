@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\echo_bootstrap.ps1" -Mode setup -Model qwen2.5:3b -IncludeTraining
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo Training stack completed.
) else (
    echo Training stack failed. Exit code: %EXIT_CODE%
)
pause
exit /b %EXIT_CODE%
