# ml/train/experiment.py
"""
train.py + 02_model_evaluation.ipynb 로직을 하나로 합친 실험용 러너.
서로 다른 증강 데이터셋(X/y npy 경로)을 받아 학습 -> best checkpoint -> test set 평가까지 한 번에 수행하고,
결과(macro F1 등)를 JSON으로 저장한다. 여러 증강 기법을 같은 하이퍼파라미터로 공정하게 비교하기 위한 스크립트.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.config import (EXPERIMENTS_CHECKPOINT_DIR, FAILURE_CLASSES, LABEL_TO_IDX,
                       NUM_CLASSES, PROCESSED_DATA_DIR)
from ml.train.dataset import WM811KDataset
from ml.train.model import WaferClassifier, FocalLoss


def run(tag, train_x_path, train_y_path, epochs=50, patience=10, seed=None,
        alpha_mode="sqrt_balanced", gamma=2.0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda")
    EXPERIMENTS_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = EXPERIMENTS_CHECKPOINT_DIR / f"{tag}.pt"
    log_path = EXPERIMENTS_CHECKPOINT_DIR / f"{tag}.log"
    report_path = EXPERIMENTS_CHECKPOINT_DIR / f"{tag}_report.json"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    train_maps = np.load(train_x_path, allow_pickle=True)
    train_labels_str = np.load(train_y_path, allow_pickle=True)
    val_maps = np.load(PROCESSED_DATA_DIR / "X_val.npy", allow_pickle=True)
    val_labels_str = np.load(PROCESSED_DATA_DIR / "y_val.npy", allow_pickle=True)

    train_labels = np.array([LABEL_TO_IDX[l] for l in train_labels_str])
    val_labels = np.array([LABEL_TO_IDX[l] for l in val_labels_str])

    log(f"=== {tag} ===")
    log(f"train: {train_x_path} ({len(train_labels)} samples)")

    train_ds = WM811KDataset(train_maps, train_labels, augment=True)
    val_ds = WM811KDataset(val_maps, val_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    if alpha_mode == "none":
        alpha = None
    else:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        if alpha_mode == "sqrt_balanced":
            class_weights = np.sqrt(class_weights)
        elif alpha_mode == "balanced":
            pass  # full balanced weight, no dampening
        else:
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        alpha = torch.FloatTensor(class_weights).to(device)

    log(f"alpha_mode={alpha_mode} gamma={gamma} alpha={alpha.tolist() if alpha is not None else None}")

    model = WaferClassifier(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -float("inf")
    best_epoch = -1
    patience_counter = 0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        _, _, f1_per_cls, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=range(NUM_CLASSES), zero_division=0)
        val_f1 = f1_per_cls.mean()

        elapsed = time.time() - t_start
        log(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss/len(train_loader):.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
            f"elapsed={elapsed:.0f}s")

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            log("  -> best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"  -> early stopping at epoch {epoch+1}")
                break

    # --- test set 평가 (02_model_evaluation.ipynb 동일 로직) ---
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_maps = np.load(PROCESSED_DATA_DIR / "X_test.npy", allow_pickle=True)
    test_labels_str = np.load(PROCESSED_DATA_DIR / "y_test.npy", allow_pickle=True)
    test_labels = np.array([LABEL_TO_IDX[l] for l in test_labels_str])
    test_ds = WM811KDataset(test_maps, test_labels, augment=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1.mean()
    weighted_f1 = float(np.average(f1, weights=support))

    per_class = {
        FAILURE_CLASSES[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        } for i in range(NUM_CLASSES)
    }

    result = {
        "tag": tag,
        "train_x_path": train_x_path,
        "train_samples": int(len(train_labels)),
        "alpha_mode": alpha_mode,
        "gamma": gamma,
        "best_epoch": best_epoch,
        "best_val_f1": round(float(best_val_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "total_time_s": round(time.time() - t_start, 1),
        "checkpoint": str(ckpt_path),
    }

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log(f"\n=== TEST RESULT [{tag}] ===")
    log(f"test_macro_f1={macro_f1:.4f} test_weighted_f1={weighted_f1:.4f} test_acc={test_acc:.4f}")
    for cls, m in per_class.items():
        log(f"  {cls:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    log(f"report saved: {report_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-x", required=True)
    parser.add_argument("--train-y", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha-mode", choices=["sqrt_balanced", "balanced", "none"], default="sqrt_balanced")
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    run(args.tag, args.train_x, args.train_y, epochs=args.epochs,
        patience=args.patience, seed=args.seed,
        alpha_mode=args.alpha_mode, gamma=args.gamma)
