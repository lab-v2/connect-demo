@echo off
setlocal

REM Build a Windows EXE for the Streamlit app.
REM Run this in a Windows terminal from the repository root.

python -m pip install --upgrade pip
python -m pip install pyinstaller streamlit

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist run_narrate.spec del /q run_narrate.spec

pyinstaller --noconfirm --clean --onedir --name connect-demo ^
  --add-data "streamlit_app.py;." ^
  --add-data "data;data" ^
  --add-data "src;src" ^
  run_narrate.py

echo.
echo Build complete. EXE path:
echo dist\connect-demo\connect-demo.exe
echo.
pause
