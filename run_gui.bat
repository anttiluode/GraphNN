@echo off
cd /d "%~dp0"
python janus_cabbage_graph.py %*
if errorlevel 1 pause
