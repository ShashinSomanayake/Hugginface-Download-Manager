@echo off
REM ============================================================
REM  HuggingFace Download Manager - Windows Installer
REM  Run this once to set up the environment.
REM ============================================================
title HF Download Manager - Setup

echo.
echo  ================================================
echo   HuggingFace Download Manager - Setup Script
echo  ================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo.
    echo  Please install Python 3.9 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found:
python --version
echo.

REM Create virtual environment
echo  [SETUP] Creating virtual environment...
if exist venv (
    echo  [INFO] Virtual environment already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

REM Activate virtual environment
echo  [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip
echo  [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo  [SETUP] Installing Python dependencies (this may take a minute)...
echo         Installing: PySide6, huggingface_hub, requests, tqdm
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [WARN] PySide6 failed, trying PyQt6 as fallback...
    pip install PyQt6 huggingface_hub requests tqdm --quiet
    if errorlevel 1 (
        echo  [ERROR] Failed to install Qt. Please check your internet connection.
        pause
        exit /b 1
    )
)
echo  [OK] Python dependencies installed.

REM Download aria2c
echo.
echo  [SETUP] Checking aria2c...
if exist aria2\aria2c.exe (
    echo  [OK] aria2c already present.
) else (
    echo  [INFO] aria2c not found. The app will download it automatically on first run.
    echo         Or you can download it manually from:
    echo           https://github.com/aria2/aria2/releases
    echo         And place aria2c.exe in the 'aria2' subfolder.
)

REM Create run.bat shortcut
echo.
echo  [SETUP] Creating run.bat launcher...
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo python main.py >> run.bat
echo  [OK] run.bat created.

echo.
echo  ================================================
echo   Setup Complete!
echo  ================================================
echo.
echo  To start the app, run:  run.bat
echo  Or double-click run.bat
echo.
echo  Starting app now...
echo.

timeout /t 3 >nul
call run.bat
