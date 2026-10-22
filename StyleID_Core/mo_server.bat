@echo off
chcp 65001 > nul
cd /d "%~dp0"
python 04_demo\server.py
pause
