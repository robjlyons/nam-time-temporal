@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%run_temporal_windows_cuda.ps1"

if not exist "%PS_SCRIPT%" (
  echo [error] Could not find PowerShell script:
  echo         %PS_SCRIPT%
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [error] Training script failed with exit code %EXIT_CODE%.
  exit /b %EXIT_CODE%
)

exit /b 0

