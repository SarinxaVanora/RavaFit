@echo off
setlocal
where py >nul 2>nul || (echo Python launcher 'py' was not found. Install Python 3.13 first.& exit /b 1)
py -3.13 -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo B14 runtime installed.
pause
