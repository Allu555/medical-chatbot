# =============================================================================
# AI Medical Chatbot - PowerShell Launcher
# =============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "        AI Medical Chatbot - Gradio UI                 " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $AppDir "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Python virtual environment not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Please set up the environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\pip.exe install -r requirements.txt"
    exit 1
}

Write-Host "`n[INFO] Virtual environment verified." -ForegroundColor Green
Write-Host "[INFO] Server address: http://localhost:7860" -ForegroundColor Cyan

$PortActive = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
if ($PortActive) {
    Write-Host "[INFO] Server is already running on port 7860." -ForegroundColor Yellow
    Write-Host "[INFO] Opening in default web browser..." -ForegroundColor Green
    Start-Process "http://localhost:7860"
    exit 0
}

Write-Host "[INFO] Starting Gradio application..." -ForegroundColor Green
$AppScript = Join-Path $AppDir "app.py"
& $VenvPython $AppScript
