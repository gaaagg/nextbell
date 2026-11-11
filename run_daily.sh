#!/usr/bin/env bash
# Daily refresh on macOS / Linux. Schedule it with cron (see README).
cd "$(dirname "$0")" || exit 1
mkdir -p data
python3 pipeline/run_all.py --daily >> data/daily.log 2>&1
