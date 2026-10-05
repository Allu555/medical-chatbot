@echo off
title AI Medical Chatbot
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo =======================================================
echo          AI Medical Chatbot - Launching UI
echo =======================================================
echo.

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0venv
    echo Please create it and install requirements:
    echo   python -m venv venv
    echo   venv\Scripts\pip.exe install -r requirements.txt
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo Starting server on http://localhost:7860 ...
echo Press Ctrl+C in this terminal to stop the server.
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0app.py"
pause
