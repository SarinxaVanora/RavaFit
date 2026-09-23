@echo off
setlocal
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe scripts\build_B14_fresh.py
) else (
  py -3.13 scripts\build_B14_fresh.py
)
if errorlevel 1 exit /b 1
echo.
echo Output: candidates\B14_final_garment.glb
echo Output: candidates\B14_with_Selected_Body.glb
pause
