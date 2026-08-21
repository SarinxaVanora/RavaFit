@echo off
setlocal
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe scripts\verify_B14_reference.py
) else (
  py -3.13 scripts\verify_B14_reference.py
)
pause
