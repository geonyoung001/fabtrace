#!/bin/bash
set -e
cd /home/geonyoung_lee/fab_scope
source /home/geonyoung_lee/fab_scope_venv/bin/activate
export PYTHONPATH=/home/geonyoung_lee/fab_scope/ml/train

DATA=data/processed
EXP=data/processed/experiments

echo "=== [1/4] baseline_rerun ==="
python ml/train/experiment.py --tag baseline_rerun \
  --train-x $DATA/X_train.npy --train-y $DATA/y_train.npy --seed 42

echo "=== [2/4] scratch_only_ae ==="
python ml/train/experiment.py --tag scratch_only_ae \
  --train-x $EXP/X_train_scratch_only_ae.npy --train-y $EXP/y_train_scratch_only_ae.npy --seed 42

echo "=== [3/4] oversample_naive ==="
python ml/train/experiment.py --tag oversample_naive \
  --train-x $EXP/X_train_oversample_naive.npy --train-y $EXP/y_train_oversample_naive.npy --seed 42

echo "=== [4/4] vae_3class ==="
python ml/train/experiment.py --tag vae_3class \
  --train-x $EXP/X_train_vae_3class.npy --train-y $EXP/y_train_vae_3class.npy --seed 42

echo "=== ALL DONE ==="
