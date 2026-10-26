#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ISAAC=/c/isaacsim/python.bat; LOG=logs/record_all.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
idle
$ISAAC src/relay/run.py --mode expert --episodes 1 --seconds 60 --seed 12 --frames artifacts/demo1_teacher_frames --report results/demo1_teacher.json > logs/demo1_teacher.out 2>&1
note "demo1 teacher: $(grep -oE '"transfers": [0-9]+, "drops": [0-9]+' logs/demo1_teacher.out)"
idle
$ISAAC src/course/run.py --mode student --no-teacher --fixed --checkpoint models/course_fixed_s3.pt --episodes 2 --seconds 45 --seed 995 --frames artifacts/demo2_frames_s3 --report results/demo2_s3_student.json > logs/demo2_s3.out 2>&1
note "demo2 s3: $(grep -oE '"success": [a-z]+|"frames": \[[0-9, ]+\]|"policy_ms_mean": [0-9.]+' logs/demo2_s3.out | paste - - - | tr '\n' ' ')"
idle
$ISAAC src/course/run_planner.py --checkpoint models/course_fixed_s3.pt --fixed --instruction "Put the blue ball in the teal bin and the red cube in the magenta bin." --max-seconds 150 --exec-timeout 50 --frames artifacts/demo3_frames --report results/demo3_run.json > logs/demo3.out 2>&1
note "demo3: $(grep -oE '"deliveries": \[[^]]*\]|Traceback' logs/demo3.out | head -c 900)"
note "RECORD DONE"
