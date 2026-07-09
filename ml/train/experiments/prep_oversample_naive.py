# ml/train/experiments/prep_oversample_naive.py
"""
실험 2 (대조군): AE 합성 없이 원본 wafer map을 그냥 복제해서 2배로 오버샘플링.
WM811KDataset이 augment=True일 때 매 epoch 랜덤 회전/flip을 적용하므로,
같은 원본이 여러 번 등장해도 완전히 동일한 텐서로 학습되진 않는다.
-> AE로 새로운 패턴을 "생성"하는 것과 "단순 반복 노출"만으로 얼마나 차이가 나는지 비교하기 위한 베이스라인.
"""
import numpy as np

PROCESSED_DIR = "/home/geonyoung_lee/fab_scope/data/processed"
OUT_DIR = "/home/geonyoung_lee/fab_scope/data/processed/experiments"

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy", allow_pickle=True)
y_train_str = np.load(f"{PROCESSED_DIR}/y_train.npy", allow_pickle=True)
classes = np.load(f"{PROCESSED_DIR}/label_encoder_classes.npy", allow_pickle=True)

label_to_idx = {label: i for i, label in enumerate(classes)}
y_train = np.array([label_to_idx[l] for l in y_train_str])

target_classes = [list(classes).index(c) for c in ["Scratch", "Near-full", "Donut"]]
multiplier = 2

X_list = [X_train]
y_list = [y_train_str]

for cls_idx in target_classes:
    mask = y_train == cls_idx
    maps_raw = X_train[mask]
    current_count = len(maps_raw)
    target_count = current_count * multiplier
    n_needed = target_count - current_count

    # 부족한 만큼 원본에서 랜덤 복원추출로 복제
    dup_idx = np.random.choice(current_count, size=n_needed, replace=True)
    duplicated = maps_raw[dup_idx]

    print(f"Class {classes[cls_idx]}: {current_count}장 -> {target_count}장 (복제 {n_needed}장)")

    X_list.append(duplicated)
    y_list.append(np.full(n_needed, classes[cls_idx]))

X_aug = np.concatenate(X_list, axis=0)
y_aug_str = np.concatenate(y_list, axis=0)

print(f"\n증강 전: {len(X_train)}장 -> 증강 후: {len(X_aug)}장")
np.save(f"{OUT_DIR}/X_train_oversample_naive.npy", X_aug)
np.save(f"{OUT_DIR}/y_train_oversample_naive.npy", y_aug_str)
