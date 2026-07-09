# ml/train/experiments/prep_vae_3class.py
"""
실험 3: 기존 AE 증강과 동일한 대상(Scratch/Near-full/Donut, multiplier=2)을
VAE로 교체해서 생성. 같은 조건에서 AE(0.830) 대비 VAE가 나은지 비교.
"""
import sys
sys.path.append("/home/geonyoung_lee/fab_scope/ml/train")

import numpy as np
from augmentation_vae import augment_minority_classes_vae

PROCESSED_DIR = "/home/geonyoung_lee/fab_scope/data/processed"
OUT_DIR = "/home/geonyoung_lee/fab_scope/data/processed/experiments"

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy", allow_pickle=True)
y_train_str = np.load(f"{PROCESSED_DIR}/y_train.npy", allow_pickle=True)
classes = np.load(f"{PROCESSED_DIR}/label_encoder_classes.npy", allow_pickle=True)

label_to_idx = {label: i for i, label in enumerate(classes)}
y_train = np.array([label_to_idx[l] for l in y_train_str])

target_classes = [list(classes).index(c) for c in ["Scratch", "Near-full", "Donut"]]

X_aug, y_aug = augment_minority_classes_vae(
    X_train, y_train, target_classes=target_classes, multiplier=2
)
y_aug_str = classes[y_aug]

print(f"\n증강 전: {len(X_train)}장 -> 증강 후: {len(X_aug)}장")
np.save(f"{OUT_DIR}/X_train_vae_3class.npy", X_aug)
np.save(f"{OUT_DIR}/y_train_vae_3class.npy", y_aug_str)
