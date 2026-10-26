#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ISAAC=/c/isaacsim/python.bat
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
idle
$ISAAC src/course/run_head2head.py --mode skill --trials 6 --seed 900 --frames artifacts/h2h_skill --report results/h2h_skill.json > logs/h2h_skill.out 2>&1
echo "[$(date +%H:%M)] skill done" >> logs/h2h.log
idle
$ISAAC src/course/run_head2head.py --mode direct --trials 6 --seed 900 --frames artifacts/h2h_direct --report results/h2h_direct.json > logs/h2h_direct.out 2>&1
echo "[$(date +%H:%M)] direct done" >> logs/h2h.log
echo "H2H DONE" >> logs/h2h.log
