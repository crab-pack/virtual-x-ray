@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo Python is required: https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment and installing packages...
  py -3.11 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)

".venv\Scripts\python.exe" main.py %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" pause
exit /b %ERR%

:fail
echo Setup failed.
pause
exit /b 1
