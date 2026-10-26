#!/usr/bin/env bash
# Sequential DAgger pipeline for the course student. Never runs two Isaac
# instances at once (concurrent RTX renderers corrupt camera frames).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
ISAAC=/c/isaacsim/python.bat
LOG=logs/pipeline_course.log

wait_isaac_idle() { while powershell -NoProfile -Command "if (Get-Process kit -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 3; done; }
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

DATA="data/course/expert_v5 data/course/hdagger1 data/course/hdagger2"
PREV=models/course_student_h2.pt
ROUND=3
for BETA in 0.2 0.1 0.05; do
  OUT=models/course_student_h${ROUND}.pt
  note "train $OUT on: $DATA"
  $PY src/lm/train_bc.py --task course --data $DATA --init "$PREV" --lr 6e-4 --out "$OUT" --epochs 20 --batch 256 > logs/train_course_h${ROUND}.out 2>&1
  grep -E "SAVED|Traceback" logs/train_course_h${ROUND}.out | tee -a "$LOG"
  wait_isaac_idle
  D=data/course/hdagger${ROUND}
  note "dagger beta=$BETA with $OUT -> $D"
  $ISAAC src/course/run.py --mode dagger --beta $BETA --noise 0.3 --checkpoint "$OUT" --episodes 30 --seconds 60 --seed $((40 + ROUND)) \
      --start-bin-prob 0.4 --prefill-prob 0.4 --save-data "$D" --report results/course_hdagger${ROUND}.json > logs/course_hdagger${ROUND}.out 2>&1
  note "dagger result: $(grep -E 'EPISODE_JSON' logs/course_hdagger${ROUND}.out | sed -E 's/.*"success": (true|false).*/\1/' | sort | uniq -c | tr '\n' ' ') flat: $(grep -oE '"head_flat_frac": [0-9.]+' logs/course_hdagger${ROUND}.out | awk -F': ' '{s+=$2} END {print s/NR}')"
  DATA="$DATA $D"
  PREV=$OUT
  ROUND=$((ROUND + 1))
done
OUT=models/course_student_h${ROUND}.pt
note "final train $OUT"
$PY src/lm/train_bc.py --task course --data $DATA --init "$PREV" --lr 5e-4 --out "$OUT" --epochs 25 --batch 256 > logs/train_course_h${ROUND}.out 2>&1
wait_isaac_idle
note "pure eval $OUT"
$ISAAC src/course/run.py --mode student --no-teacher --checkpoint "$OUT" --episodes 10 --seconds 60 --seed 520 --start-bin-prob 0.3 --prefill-prob 0.3 \
    --report results/course_eval_h${ROUND}.json > logs/course_eval_h${ROUND}.out 2>&1
note "eval: $(grep -E 'EPISODE_JSON' logs/course_eval_h${ROUND}.out | sed -E 's/.*"success": (true|false).*/\1/' | sort | uniq -c | tr '\n' ' ')"
note "COURSE PIPELINE DONE"
