@echo off
cd /d "%~dp0"
title Squad
python -c "import pygame,numpy" >nul 2>&1
if errorlevel 1 (
  echo Preparing Squad...
  python -m pip install -q -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo Setup failed.
    pause
    exit /b 1
  )
)
python game.py
if errorlevel 1 (
  echo.
  pause
)
