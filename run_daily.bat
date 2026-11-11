@echo off
rem Daily refresh on Windows. Schedule it with Task Scheduler (see README).
cd /d "%~dp0"
if not exist data mkdir data
python pipeline\run_all.py --daily >> data\daily.log 2>&1
