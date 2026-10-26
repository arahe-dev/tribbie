#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; ISAAC=/c/isaacsim/python.bat; LOG=logs/fast_course.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until [ "$(grep -c EPISODE_JSON logs/course_fixed_expert.out)" -ge 45 ]; do sleep 10; done
idle
note "teacher: $(grep -oE '"success": [a-z]+' logs/course_fixed_expert.out | sort | uniq -c | tr '\n' ' ')"
$PY src/lm/train_bc.py --task course --data data/course/fixed_expert --init models/course_student_h3.pt --lr 8e-4 --out models/course_fixed_s1.pt --epochs 15 --batch 256 > logs/train_fixed_s1.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_fixed_s1.out)"
$ISAAC src/course/run.py --mode dagger --fixed --beta 0.3 --noise 0.2 --checkpoint models/course_fixed_s1.pt --episodes 20 --seconds 50 --seed 950 --start-bin-prob 0.3 --prefill-prob 0.3 --save-data data/course/fixed_dagger1 --report results/course_fixed_dagger1.json > logs/course_fixed_dagger1.out 2>&1
note "dagger1: $(grep -oE '"success": [a-z]+' logs/course_fixed_dagger1.out | sort | uniq -c | tr '\n' ' ')"
idle
$PY src/lm/train_bc.py --task course --data data/course/fixed_expert data/course/fixed_dagger1 --init models/course_fixed_s1.pt --lr 6e-4 --out models/course_fixed_s2.pt --epochs 15 --batch 256 > logs/train_fixed_s2.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_fixed_s2.out)"
$ISAAC src/course/run.py --mode student --no-teacher --fixed --checkpoint models/course_fixed_s2.pt --episodes 6 --seconds 45 --seed 980 --start-bin-prob 0.0 --frames artifacts/demo2_frames_fixed --report results/demo2_fixed_student.json > logs/demo2_fixed.out 2>&1
note "eval: $(grep -oE '"success": [a-z]+|"frames": \[[0-9, ]+\]' logs/demo2_fixed.out | paste - - | tr '\n' ' ')"
note "FAST COURSE DONE"
