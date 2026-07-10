@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\echo_bootstrap.ps1" -Mode launch -Model qwen2.5:3b
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Echo launch failed. Exit code: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
