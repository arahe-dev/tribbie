#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; ISAAC=/c/isaacsim/python.bat; LOG=logs/goto_pipeline.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
summ() { grep -oE '"deliveries_ok": [0-9]+, "deliveries": [0-9]+' "$1" | awk -F'[:,]' '{ok+=$2; tot+=$4} END {print ok "/" tot}'; }
idle
$ISAAC src/course/run_skills.py --mode dagger --only-goto --beta 0.2 --noise 0.15 --checkpoint models/skills_s2.pt --episodes 36 --seed 700 --save-data data/skills/goto_dagger1 > logs/goto_dagger1.out 2>&1
note "goto dagger beta=0.2: $(summ logs/goto_dagger1.out)"
$PY src/lm/train_bc.py --task skills --data data/skills/expert data/skills/dagger1 data/skills/dagger2 data/skills/goto_dagger1 --init models/skills_s2.pt --lr 5e-4 --out models/skills_g1.pt --epochs 12 --batch 256 > logs/train_skills_g1.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_skills_g1.out)"
idle
$ISAAC src/course/run_skills.py --mode student --only-goto --checkpoint models/skills_g1.pt --episodes 10 --seed 720 > logs/goto_eval.out 2>&1
note "GOTO STUDENT EVAL: $(summ logs/goto_eval.out)"
note "GOTO DONE"
