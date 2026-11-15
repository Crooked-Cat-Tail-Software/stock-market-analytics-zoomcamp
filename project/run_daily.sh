#!/usr/bin/env bash
# Daily job for cron: fetch the new days, rebuild features, score with the saved models,
# print/append today's trades to reports/latest_trades.csv. Logs go to logs/daily_YYYY-MM-DD.log
#
# crontab -e   (weekdays 22:30 local time = after the US close and after yfinance updates)
#   30 22 * * 1-5 /full/path/to/project/run_daily.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
.venv/bin/python main.py run --mode fetch >> "logs/daily_$(date +%F).log" 2>&1
