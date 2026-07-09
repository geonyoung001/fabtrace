# ml/train/experiments/prep_scratch_only_ae.py
"""
실험 1: Scratch(가장 recall/precision 낮은 class)만 Denoising AE로 3배 증강.
Donut/Near-full은 건드리지 않음 (기존 3-class 동시 증강이 다른 class 성능을 깎아먹었기 때문에
가장 문제였던 Scratch 하나만 타겟팅해서 부작용을 격리).
"""
import sys
sys.path.append("/home/geonyoung_lee/fab_scope/ml/train")

import numpy as np
from augementaion_ae import augment_minority_classes

PROCESSED_DIR = "/home/geonyoung_lee/fab_scope/data/processed"
OUT_DIR = "/home/geonyoung_lee/fab_scope/data/processed/experiments"

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy", allow_pickle=True)
y_train_str = np.load(f"{PROCESSED_DIR}/y_train.npy", allow_pickle=True)
classes = np.load(f"{PROCESSED_DIR}/label_encoder_classes.npy", allow_pickle=True)

label_to_idx = {label: i for i, label in enumerate(classes)}
y_train = np.array([label_to_idx[l] for l in y_train_str])

scratch_idx = list(classes).index("Scratch")

X_aug, y_aug = augment_minority_classes(
    X_train, y_train, target_classes=[scratch_idx], multiplier=3
)
y_aug_str = classes[y_aug]

print(f"\n증강 전: {len(X_train)}장 -> 증강 후: {len(X_aug)}장")
np.save(f"{OUT_DIR}/X_train_scratch_only_ae.npy", X_aug)
np.save(f"{OUT_DIR}/y_train_scratch_only_ae.npy", y_aug_str)
