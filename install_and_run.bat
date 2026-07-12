@echo off
echo Verifying requirements for Ultimate Xenoverse Extractor...
echo.
setlocal
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1
echo.
echo All requirements met, starting tool! Give it some time to boot up please.
py ultimate_xenoverse_extractor.py
if errorlevel 1 pause