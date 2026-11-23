#!/usr/bin/env bash
set -e
echo "=========================================================="
echo "  FreeCloud AI Studio — 16GB RAM Cloud PC Initialization  "
echo "=========================================================="

echo "=== System Memory and CPU Specifications ==="
free -h
lscpu | grep "Model name\|CPU(s):" || true
df -h /

echo "=== Installing Dependencies ==="
python -m pip install --upgrade pip
pip install -r requirements.txt || true
pip install opencv-python-headless numpy imageio-ffmpeg || true

echo "=== 16GB RAM Cloud Environment Ready ==="
