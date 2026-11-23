#!/bin/sh
set -eu
cd "$(dirname "$0")"
if systemctl --user is-active --quiet gestureflow-dashboard.service; then
    echo "GestureFlow is already running at http://127.0.0.1:5000"
    echo "To stop that instance: systemctl --user stop gestureflow-dashboard"
    exit 0
fi
if ss -H -ltn 'sport = :5000' | grep -q .; then
    echo "Port 5000 is already in use. Check http://127.0.0.1:5000 before starting another copy." >&2
    exit 1
fi
exec systemd-run --user --pty --wait --collect --setenv=GESTUREFLOW_DESKTOP_SESSION=1 --unit=gestureflow-dashboard --working-directory="$PWD" "$PWD/.venv/bin/python" -u "$PWD/app.py"
