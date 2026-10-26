#!/usr/bin/env bash
# Sequential relay pipeline: clean teacher data, then DAgger rounds, then a pure eval.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
ISAAC=/c/isaacsim/python.bat
LOG=logs/pipeline_relay.log
wait_isaac_idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_isaac_idle
note "collect clean relay teacher data"
$ISAAC src/relay/run.py --mode expert --noise 0.5 --episodes 10 --seconds 90 --seed 9 --save-data data/relay/expert_v6 --report results/relay_expert_v6.json > logs/relay_collect6.out 2>&1
note "teacher: $(grep -oE '"transfers": [0-9]+' logs/relay_collect6.out | tr '\n' ' ')"
DATA="data/relay/expert_v6"
OUT=models/relay_student_r1.pt
$PY src/lm/train_bc.py --task relay --data $DATA --out $OUT --epochs 25 --batch 256 > logs/train_relay_r1.out 2>&1
note "$(grep -E 'SAVED|Traceback' logs/train_relay_r1.out)"
PREV=$OUT
ROUND=1
for BETA in 0.4 0.2 0.1; do
  wait_isaac_idle
  D=data/relay/rdagger${ROUND}
  note "dagger beta=$BETA with $PREV -> $D"
  $ISAAC src/relay/run.py --mode dagger --beta $BETA --noise 0.3 --checkpoint "$PREV" --episodes 4 --seconds 90 --seed $((60 + ROUND)) \
      --save-data "$D" --report results/relay_rdagger${ROUND}.json > logs/relay_rdagger${ROUND}.out 2>&1
  note "dagger result: $(grep -oE '"transfers": [0-9]+, "drops": [0-9]+' logs/relay_rdagger${ROUND}.out | tr '\n' ' ')"
  DATA="$DATA $D"
  ROUND=$((ROUND + 1))
  OUT=models/relay_student_r${ROUND}.pt
  $PY src/lm/train_bc.py --task relay --data $DATA --init "$PREV" --lr 6e-4 --out "$OUT" --epochs 20 --batch 256 > logs/train_relay_r${ROUND}.out 2>&1
  note "$(grep -E 'SAVED|Traceback' logs/train_relay_r${ROUND}.out)"
  PREV=$OUT
done
wait_isaac_idle
note "pure eval $PREV"
$ISAAC src/relay/run.py --mode student --no-teacher --checkpoint "$PREV" --episodes 2 --seconds 90 --seed 80 --report results/relay_eval_r${ROUND}.json > logs/relay_eval_r${ROUND}.out 2>&1
note "eval: $(grep -oE '"transfers": [0-9]+, "drops": [0-9]+|"policy_ms_mean": [0-9.]+' logs/relay_eval_r${ROUND}.out | tr '\n' ' ')"
note "RELAY PIPELINE DONE"
