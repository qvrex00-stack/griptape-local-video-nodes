@echo off
chcp 65001 >nul
title Griptape Video Generation Server (Port 8090)
echo ============================================================
echo  Griptape Video Generation Server
echo  Port: 8090
echo ============================================================
echo.

set "SERVER_DIR=C:\Foundry\Griptape\video_gen_server"
set "VENV_DIR=C:\Foundry\Griptape\video_gen_server\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd /d "%SERVER_DIR%"

if not exist "%PYTHON%" (
    echo [ERROR] Python venv not found!
    echo Expected: %PYTHON%
    echo.
    pause
    exit /b 1
)

echo Starting server on http://127.0.0.1:8090
echo Press Ctrl+C to stop
echo.

"%PYTHON%" "%SERVER_DIR%\server.py"
goto end

:error
echo.
echo ERROR occurred. Please check the output above.
echo.

:end
pause
