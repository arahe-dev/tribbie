#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
D="data/skills/expert data/skills/pick_expert"
for SEED in 8 9; do
  for spec in "both16:--latent 16" "head16:--latent 16 --zero-hand" "wrist16:--latent 16 --zero-scene" "both4:--latent 4" "both64:--latent 64"; do
    name=${spec%%:*}; flags=${spec#*:}
    $PY src/lm/train_bc.py --task skills --data $D --epochs 10 --batch 256 --seed $SEED --out models/abl_${name}_s$SEED.pt --metrics-out results/ablations/${name}_s$SEED.json $flags > logs/abl_${name}_s$SEED.out 2>&1
    echo "[$(date +%H:%M:%S)] $name seed $SEED done" >> logs/ablations.log
  done
done
echo "SEEDS DONE" >> logs/ablations.log
