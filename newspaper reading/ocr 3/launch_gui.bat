@echo off
title AI Newspaper Reader - OCR 3 Architecture
cd /d "%~dp0"
echo ==============================================================
echo    AI NEWSPAPER READER - OCR 3 ARCHITECTURE (GUI LAUNCHER)
echo ==============================================================
echo.
echo Launching Interactive Reader with Multi-Column & Fine-Tuned OCR...
python gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application closed with code %ERRORLEVEL%.
    pause
)

