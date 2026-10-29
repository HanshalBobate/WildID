@echo off
setlocal enabledelayedexpansion

:: Navigate to script directory
cd /d "%~dp0"

title Tiger Semantic Segmentation Generator

echo =====================================================================
echo   Tiger Semantic Segmentation Dataset Generator
echo =====================================================================
echo.

:: 1. Try finding Python first
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Python detected in PATH.
    echo [INFO] Starting segmentation collector...
    echo.
    python run_segmentation_collector.py %*
    goto :end
)

:: 2. If Python is not in PATH, look for Blender directly
set "BLENDER_EXE="
where blender >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%i in ('where blender') do (
        set "BLENDER_EXE=%%i"
        goto :found_blender
    )
)

for %%p in (
    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
    "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
    "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
    "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
) do (
    if exist %%p (
        set "BLENDER_EXE=%%~p"
        goto :found_blender
    )
)

:found_blender
if defined BLENDER_EXE (
    echo [INFO] Blender detected at: "!BLENDER_EXE!"
    echo [INFO] Starting segmentation collector directly with Blender...
    echo.
    "!BLENDER_EXE!" -b target.blend --python run_segmentation_collector.py -- %*
    goto :end
)

echo [ERROR] Neither Python nor Blender executable could be found!
echo Please ensure Python 3.10+ or Blender 4.0+ is installed.
echo.

:end
echo.
echo =====================================================================
echo Process finished.
echo =====================================================================
pause
