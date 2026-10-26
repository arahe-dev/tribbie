#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
D="data/skills/expert data/skills/pick_expert"
run() { name=$1; shift; $PY src/lm/train_bc.py --task skills --data $D --epochs 10 --batch 256 --out models/abl_$name.pt --metrics-out results/ablations/$name.json "$@" > logs/abl_$name.out 2>&1; echo "[$(date +%H:%M:%S)] $name $(cat results/ablations/$name.json | tr -d '\n ' | head -c 220)" >> logs/ablations.log; }
run both16 --latent 16
run head16 --latent 16 --zero-hand
run wrist16 --latent 16 --zero-scene
run both4 --latent 4
run both64 --latent 64
echo "ABLATIONS DONE" >> logs/ablations.log
