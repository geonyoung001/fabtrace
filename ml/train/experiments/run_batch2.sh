#!/bin/bash
set -e
cd /home/geonyoung_lee/fab_scope
source /home/geonyoung_lee/fab_scope_venv/bin/activate
export PYTHONPATH=/home/geonyoung_lee/fab_scope/ml/train

DATA=data/processed

echo "=== [1/2] no_alpha (baseline data, alpha=None, gamma=2.0) ==="
python ml/train/experiment.py --tag no_alpha \
  --train-x $DATA/X_train.npy --train-y $DATA/y_train.npy --seed 42 \
  --alpha-mode none --gamma 2.0

echo "=== [2/2] sqrt_gamma1 (baseline data, sqrt_balanced alpha, gamma=1.0) ==="
python ml/train/experiment.py --tag sqrt_gamma1 \
  --train-x $DATA/X_train.npy --train-y $DATA/y_train.npy --seed 42 \
  --alpha-mode sqrt_balanced --gamma 1.0

echo "=== BATCH2 DONE ==="
