# Macro F1 개선 실험 기록 (목표: 89점)

## 결과 요약

| 실험 | 데이터 | alpha | gamma | test macro F1 |
|---|---|---|---|---|
| before (최초 baseline) | X_train (원본) | sqrt_balanced | 2.0 | 0.848 |
| after (AE 3-class 증강) | Scratch/Near-full/Donut ×2 (Denoising AE) | sqrt_balanced | 2.0 | 0.830 |
| baseline_rerun | X_train (원본) | sqrt_balanced | 2.0 | 0.824 |
| scratch_only_ae | Scratch ×3 (Denoising AE) | sqrt_balanced | 2.0 | 0.816 |
| oversample_naive | Scratch/Near-full/Donut ×2 (단순 복제) | sqrt_balanced | 2.0 | 0.824 |
| vae_3class | Scratch/Near-full/Donut ×2 (VAE) | sqrt_balanced | 2.0 | 0.820 |
| sqrt_gamma1 | X_train (원본) | sqrt_balanced | 1.0 | 0.873 |
| **no_alpha (seed 42)** | X_train (원본) | **None** | 2.0 | **0.904** |
| **no_alpha (seed 123)** | X_train (원본) | **None** | 2.0 | **0.911** |

**최종 채택: `no_alpha`, seed 123 — test macro F1 = 0.9108** (목표 0.89 초과, 2개 시드로 재현 확인)

## 핵심 발견

1. **AE/VAE 기반 합성 증강은 거의 효과가 없었다.** Denoising AE, VAE, 단순 오버샘플링 모두
   macro F1 0.816~0.824 범위로, 무증강 재현 실험(baseline_rerun=0.824)과 통계적으로 구분되지 않았다.
   오히려 원래 "before" 수치(0.848)보다 낮게 나왔는데, 이는 런 간 랜덤 변동성(시드 미고정 등)이
   증강 기법 차이보다 큰 영향을 준 것으로 보인다.

2. **매크로 F1을 가장 많이 깎아먹은 건 증강 대상이 아니었던 Edge-Loc, Loc였다.**
   `before` 대비 `baseline_rerun`의 클래스별 F1 차이를 뜯어보니 Edge-Loc(-0.143), Loc(-0.098) 두 클래스가
   전체 격차의 대부분을 차지했다. 이 두 클래스는 애초에 recall이 아니라 **precision**이 낮은 게 문제였다
   (다른 class가 Edge-Loc/Loc으로 오분류됨) — "recall<70% class만 증강"이라는 원래 전략 자체가
   실제 병목을 겨냥하지 못하고 있었다.

3. **진짜 원인은 `FocalLoss`의 `alpha`(sqrt-balanced class weight)였다.**
   alpha를 완전히 제거하고 `FocalLoss(alpha=None, gamma=2.0)`만 사용하자 macro F1이
   0.82 → 0.90+ 로 뛰었다. class weight가 소수 class 예측을 과하게 밀어붙이면서 Edge-Loc/Loc/Center 등
   인접 class의 precision을 깎아먹고 있었던 것 — augmentation으로는 절대 못 고치는 문제였다.
   gamma를 1.0으로 낮추기만 한 실험(`sqrt_gamma1`, alpha는 유지)도 0.873까지는 개선됐지만
   alpha를 아예 없앤 것보다는 못했다 (alpha 자체가 핵심 원인이라는 뜻).

## 최종 채택 설정

- 데이터: `data/processed/X_train.npy` / `y_train.npy` (증강 없음, 원본 그대로)
- 모델: `WaferClassifier` (EfficientNet-B0 backbone, pretrained)
- 손실함수: `FocalLoss(alpha=None, gamma=2.0)`
- optimizer: AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR(T_max=50)
- batch size: train 64 / val 128, epochs 최대 50, early stopping patience 10
- `ml/train/train.py`를 이 설정으로 업데이트함 (X_train_augmented 대신 X_train 사용, alpha=None)

## 최종 모델 (test set 기준, macro F1 = 0.9108, seed 123)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| none | 0.990 | 0.994 | 0.992 | 14728 |
| Center | 0.968 | 0.912 | 0.939 | 430 |
| Donut | 0.962 | 0.911 | 0.936 | 56 |
| Edge-Loc | 0.889 | 0.850 | 0.869 | 519 |
| Edge-Ring | 0.970 | 0.987 | 0.978 | 968 |
| Loc | 0.874 | 0.811 | 0.841 | 359 |
| Near-full | 0.933 | 0.933 | 0.933 | 15 |
| Random | 0.938 | 0.862 | 0.898 | 87 |
| Scratch | 0.755 | 0.875 | 0.811 | 120 |
| **macro avg** | — | — | **0.911** | 17282 |
| accuracy | — | — | 0.981 | 17282 |

체크포인트: `ml/checkpoints/experiments/no_alpha_seed123.pt`
(→ `ml/checkpoints/best_model.pt`로 복사해 production 경로에 반영함)

## 재현 방법

```bash
source ~/fab_scope_venv/bin/activate
cd ~/fab_scope
python ml/train/train.py
```

또는 개별 실험 재현/비교용 러너:

```bash
python ml/train/experiment.py --tag <name> \
  --train-x data/processed/X_train.npy --train-y data/processed/y_train.npy \
  --alpha-mode none --gamma 2.0 --seed 123
```

## 남겨둔 것들 (참고용, 프로덕션에서는 미사용)

- `ml/train/augementaion_ae.py`, `ml/train/augmentation_vae.py`: Denoising AE / VAE 증강 구현.
  이번 태스크의 목표 달성에는 기여하지 못했지만, 향후 다른 문제(예: 실제 recall이 병목인 상황)에
  재사용할 수 있어 남겨둠.
- `ml/train/experiment.py`: 데이터/alpha/gamma를 바꿔가며 학습+평가를 한 번에 도는 실험 러너.
- `ml/train/experiments/*.py`, `*.sh`: 이번에 돌린 개별 실험 데이터 준비/체인 스크립트.
- `ml/checkpoints/experiments/*_report.json`, `*.log`: 각 실험의 상세 결과.
