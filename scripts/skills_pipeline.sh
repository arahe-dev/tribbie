#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; ISAAC=/c/isaacsim/python.bat; LOG=logs/skills_pipeline.log
idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
summ() { .venv/Scripts/python.exe -c "
import json,sys
ok=tot=0
for l in open(sys.argv[1], encoding='utf-8', errors='ignore'):
    i=l.find('EPISODE_JSON ')
    if i>=0:
        r=json.loads(l[i+13:]); ok+=r['deliveries_ok']; tot+=r['deliveries']
print(f'{ok}/{tot} deliveries')" "$1"; }
idle
$ISAAC src/course/run_skills.py --mode expert --noise 0.3 --episodes 40 --seed 100 --save-data data/skills/expert > logs/skills_expert.out 2>&1
note "teacher: $(summ logs/skills_expert.out)"
$PY src/lm/train_bc.py --task skills --data data/skills/expert --out models/skills_s1.pt --epochs 30 --batch 256 > logs/train_skills_s1.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_skills_s1.out) $(grep EPOCH logs/train_skills_s1.out | tail -1 | grep -oE '"val": \{[^}]*\}')"
DATA=data/skills/expert; PREV=models/skills_s1.pt; R=1
for BETA in 0.3 0.15 0.05; do
  idle
  $ISAAC src/course/run_skills.py --mode dagger --beta $BETA --noise 0.15 --checkpoint $PREV --episodes 20 --seed $((200 + R)) --save-data data/skills/dagger$R > logs/skills_dagger$R.out 2>&1
  note "dagger$R beta=$BETA: $(summ logs/skills_dagger$R.out)"
  DATA="$DATA data/skills/dagger$R"
  $PY src/lm/train_bc.py --task skills --data $DATA --init $PREV --lr 6e-4 --out models/skills_s$((R+1)).pt --epochs 15 --batch 256 > logs/train_skills_s$((R+1)).out 2>&1
  note "$(grep -E 'SAVED|Traceback' logs/train_skills_s$((R+1)).out)"
  PREV=models/skills_s$((R+1)).pt; R=$((R+1))
done
idle
$ISAAC src/course/run_skills.py --mode student --checkpoint $PREV --episodes 6 --seed 300 --report results/skills_eval.json > logs/skills_eval.out 2>&1
note "STUDENT EVAL $PREV: $(summ logs/skills_eval.out)"
note "SKILLS DONE"
