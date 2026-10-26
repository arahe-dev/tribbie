#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; ISAAC=/c/isaacsim/python.bat; LOG=logs/fast_course.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
idle
DATA="data/course/fixed_expert data/course/fixed_dagger1"
PREV=models/course_fixed_s2.pt
R=2
for BETA in 0.12 0.04; do
  $ISAAC src/course/run.py --mode dagger --fixed --beta $BETA --noise 0.15 --checkpoint $PREV --episodes 24 --seconds 45 --seed $((960 + R)) --start-bin-prob 0.3 --prefill-prob 0.3 --save-data data/course/fixed_dagger$R --report results/course_fixed_dagger$R.json > logs/course_fixed_dagger$R.out 2>&1
  note "dagger$R beta=$BETA: $(grep -oE '"success": [a-z]+' logs/course_fixed_dagger$R.out | sort | uniq -c | tr '\n' ' ')"
  DATA="$DATA data/course/fixed_dagger$R"
  idle
  $PY src/lm/train_bc.py --task course --data $DATA --init $PREV --lr 6e-4 --out models/course_fixed_s$((R+1)).pt --epochs 15 --batch 256 > logs/train_fixed_s$((R+1)).out 2>&1
  note "$(grep -E 'SAVED|Traceback' logs/train_fixed_s$((R+1)).out)"
  PREV=models/course_fixed_s$((R+1)).pt
  R=$((R+1))
done
$ISAAC src/course/run.py --mode student --no-teacher --fixed --checkpoint $PREV --episodes 6 --seconds 45 --seed 990 --frames artifacts/demo2_frames_final --report results/demo2_final_student.json > logs/demo2_final.out 2>&1
note "FINAL eval $PREV: $(grep -oE '"success": [a-z]+|"frames": \[[0-9, ]+\]' logs/demo2_final.out | paste - - | tr '\n' ' ')"
idle
$ISAAC src/relay/run.py --mode expert --episodes 1 --seconds 60 --seed 12 --frames artifacts/demo1_teacher_frames --report results/demo1_teacher.json > logs/demo1_teacher.out 2>&1
note "relay teacher footage: $(grep -oE '"transfers": [0-9]+, "drops": [0-9]+' logs/demo1_teacher.out)"
note "FAST2 DONE"
