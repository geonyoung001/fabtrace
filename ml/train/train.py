import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from ml.config import (BEST_MODEL_PATH, CHECKPOINTS_DIR, LABEL_TO_IDX,
                       NUM_CLASSES, PROCESSED_DATA_DIR)
from ml.train.dataset import WM811KDataset
from ml.train.model import WaferClassifier, FocalLoss

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

DEVICE = torch.device("cuda")

# AE/VAE 합성 증강(X_train_augmented) 실험 결과 macro F1이 오히려 원본보다 낮았음(~0.82) → 원본 데이터 사용
train_maps = np.load(PROCESSED_DATA_DIR / "X_train.npy", allow_pickle=True)
train_labels = np.load(PROCESSED_DATA_DIR / "y_train.npy", allow_pickle=True)
val_maps = np.load(PROCESSED_DATA_DIR / "X_val.npy", allow_pickle=True)
val_labels = np.load(PROCESSED_DATA_DIR / "y_val.npy", allow_pickle=True)

# failureType 문자열 -> 정수 인덱스 (FAILURE_CLASSES 순서 = 라벨 인덱스, config.py 참고)
train_labels = np.array([LABEL_TO_IDX[l] for l in train_labels])
val_labels = np.array([LABEL_TO_IDX[l] for l in val_labels])


# 데이터
train_ds = WM811KDataset(train_maps, train_labels, augment=True)
val_ds = WM811KDataset(val_maps, val_labels, augment=False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

# class weight(alpha) 없이 FocalLoss(gamma=2.0)만 사용 — sqrt-balanced alpha를 주면
# 모델이 소수 class를 과하게 예측해 Edge-Loc/Loc 등 인접 class의 precision이 깎여 macro F1이 낮아짐
# (alpha 제거만으로 macro F1 0.82 → 0.90+, 실험 결과는 ml/EXPERIMENT_RESULTS.md 참고)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# 모델, 손실함수, optimizer, scheduler
model = WaferClassifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
criterion = FocalLoss(alpha=None, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 학습 루프
best_val_f1 = -float('inf')
patience_counter = 0

for epoch in range(50):
    # --- Train ---
    model.train()                          # 학습 모드 (dropout 켜짐, augment 작동)
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()              # 이전 기울기 초기화
        logits = model(X_batch)            # forward (예측)
        loss = criterion(logits, y_batch)  # 손실 계산
        loss.backward()                    # backward (기울기 계산)
        optimizer.step()                   # 가중치 업데이트
        train_loss += loss.item()

    # --- Validate ---
    model.eval()                           # 평가 모드 (dropout 꺼짐)
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():                  # 기울기 계산 안 함 (메모리 절약)
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            val_loss += criterion(logits, y_batch).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # --- 지표 계산 ---
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1} | train_loss={train_loss/len(train_loader):.4f} | "
          f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    scheduler.step()                       # lr 조정

    # --- Early stopping + best 저장 ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        patience_counter = 0
        print("  → Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"  → Early stopping at epoch {epoch+1}")
            break