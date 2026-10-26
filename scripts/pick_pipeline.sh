#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; ISAAC=/c/isaacsim/python.bat; LOG=logs/pick_pipeline.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
summ() { grep -oE '"deliveries_ok": [0-9]+, "deliveries": [0-9]+' "$1" | awk -F'[:,]' '{ok+=$2; tot+=$4} END {print ok "/" tot}'; }
idle
$ISAAC src/course/run_skills.py --mode expert --only-pick --noise 0.4 --episodes 120 --seed 400 --save-data data/skills/pick_expert > logs/pick_expert.out 2>&1
note "pick teacher: $(summ logs/pick_expert.out)"
DATA="data/skills/expert data/skills/dagger1 data/skills/dagger2 data/skills/pick_expert"
$PY src/lm/train_bc.py --task skills --data $DATA --init models/skills_s2.pt --lr 6e-4 --out models/skills_p1.pt --epochs 15 --batch 256 > logs/train_skills_p1.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_skills_p1.out)"
PREV=models/skills_p1.pt; R=1
for BETA in 0.25 0.1; do
  idle
  $ISAAC src/course/run_skills.py --mode dagger --only-pick --beta $BETA --noise 0.15 --checkpoint $PREV --episodes 80 --seed $((500 + R)) --save-data data/skills/pick_dagger$R > logs/pick_dagger$R.out 2>&1
  note "pick dagger$R beta=$BETA: $(summ logs/pick_dagger$R.out)"
  DATA="$DATA data/skills/pick_dagger$R"
  $PY src/lm/train_bc.py --task skills --data $DATA --init $PREV --lr 5e-4 --out models/skills_p$((R+1)).pt --epochs 12 --batch 256 > logs/train_skills_p$((R+1)).out 2>&1
  note "$(grep -E 'SAVED|Traceback' logs/train_skills_p$((R+1)).out)"
  PREV=models/skills_p$((R+1)).pt; R=$((R+1))
done
idle
$ISAAC src/course/run_skills.py --mode student --only-pick --checkpoint $PREV --episodes 20 --seed 600 > logs/pick_eval.out 2>&1
note "PICK STUDENT EVAL $PREV: $(summ logs/pick_eval.out)"
idle
$ISAAC src/course/run_skills.py --mode student --checkpoint $PREV --episodes 5 --seed 610 --deliveries 2 --report results/skills_eval_full.json > logs/skills_eval_full.out 2>&1
note "FULL STUDENT EVAL $PREV: $(summ logs/skills_eval_full.out)"
note "PICK DONE"
