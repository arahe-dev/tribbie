#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ISAAC=/c/isaacsim/python.bat; LOG=logs/record_v2.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
summ() { grep -oE '"deliveries_ok": [0-9]+, "deliveries": [0-9]+' "$1" | awk -F'[:,]' '{ok+=$2; tot+=$4} END {print ok "/" tot}'; }
idle
$ISAAC src/course/run_skills.py --mode student --goto-planner --checkpoint models/skills_s2.pt --checkpoint-pick models/skills_p3.pt --episodes 4 --seed 820 --deliveries 2 --start-bin-prob 0 --frames artifacts/demo2v3_frames --report results/demo2v3.json > logs/demo2v3.out 2>&1
note "demo2v3: $(summ logs/demo2v3.out)"
idle
$ISAAC src/course/run_opus_skills.py --goto-planner --checkpoint models/skills_s2.pt --checkpoint-pick models/skills_p3.pt --max-seconds 220 --frames artifacts/demo3v3_frames --report results/demo3v3_run.json > logs/demo3v3.out 2>&1
note "demo3v3: $(grep -oE 'DEMO3_JSON .*' logs/demo3v3.out | head -c 700) $(grep -c Traceback logs/demo3v3.out)"
note "RECORD V3 DONE"
