@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "BACKEND_ENV=%PROJECT_ROOT%myenv\Scripts\activate.bat"

call :launch_backend
call :launch_frontend

echo [Promply-V2] Backend and frontend launch commands dispatched.
exit /b 0

:launch_backend
    echo [Promply-V2] Opening backend window...
    start "Promply Backend" cmd /k "call "%BACKEND_ENV%" && cd /d "%PROJECT_ROOT%" && python app.py"
    goto :eof

:launch_frontend
    echo [Promply-V2] Opening frontend window...
    start "Promply Interface" cmd /k "cd /d "%PROJECT_ROOT%interface" && npm run dev"
    goto :eof
