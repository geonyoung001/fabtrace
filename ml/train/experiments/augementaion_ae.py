# ml/train/augmentation_ae.py
import torch
import torch.nn as nn
import numpy as np

from dataset import WM811KDataset


# ═══════════════════════════════════════════════════════════
# Denoising Convolutional Autoencoder (Bao et al. 2024)
# ═══════════════════════════════════════════════════════════
class DenoisingConvAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder: (3, 64, 64) → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # 64→32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 32→16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )
        # Decoder: latent_dim → (3, 64, 64)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            # Softmax 제거: raw logits 그대로 출력 (CrossEntropyLoss가 내부에서 softmax 처리)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


# ═══════════════════════════════════════════════════════════
# 학습: Denoising 방식 (입력=noisy, 정답=원본)
# ═══════════════════════════════════════════════════════════
def train_denoising_ae(minority_maps, latent_dim=128,
                       epochs=200, noise_level=0.2, device="cuda"):
    """
    소수 class wafer map으로 Denoising AE 학습.
    핵심: 입력에 noise를 주되, 정답은 깨끗한 원본.
    """
    ae = DenoisingConvAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X = torch.FloatTensor(np.array(minority_maps)).to(device)  # (N, 3, 64, 64) one-hot
    y_target = X.argmax(dim=1).long()  # (N, 64, 64) class index {0,1,2} — CrossEntropyLoss 타깃

    ae.train()
    for epoch in range(epochs):
        # --- Denoising 핵심: 입력에 noise 추가 ---
        X_noisy = X + torch.randn_like(X) * noise_level
        X_noisy = torch.clamp(X_noisy, 0, 1)  # 0~1 범위 유지

        # 복원 (raw logits, (N, 3, 64, 64))
        X_recon = ae(X_noisy)

        # --- 정답은 noisy가 아니라 깨끗한 원본(class index) ---
        loss = criterion(X_recon, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} | recon loss: {loss.item():.4f} | lr: {lr:.6f}")

    return ae


# ═══════════════════════════════════════════════════════════
# 생성: latent에 noise 주입 → 새 wafer map (Bao 2024 핵심)
# ═══════════════════════════════════════════════════════════
def generate_synthetic(ae, minority_maps, n_per_sample=5,
                       latent_noise=0.1, device="cuda"):
    """
    각 원본을 latent로 인코딩 → 작은 noise 주입 → 재구성.
    latent_noise가 논문의 'small latent feature variation'.
    """
    ae.eval()
    X = torch.FloatTensor(np.array(minority_maps)).to(device)
    synthetic = []

    with torch.no_grad():
        for i in range(len(X)):
            z = ae.encode(X[i:i+1])  # latent vector

            for _ in range(n_per_sample):
                # --- latent space에 작은 noise 주입 ---
                z_noisy = z + torch.randn_like(z) * latent_noise
                new_img = ae.decode(z_noisy)  # (1, 3, 64, 64) raw logits

                # 3채널 logits → 카테고리 맵(값 0/1/2)으로 이산화 (argmax는 softmax 여부와 무관하게 동일)
                # X_train 원본과 동일한 raw 포맷(2D, {0,1,2})을 유지해야
                # WM811KDataset의 resize_and_onehot과 그대로 이어붙일 수 있음
                categorical = new_img.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)

                synthetic.append(categorical)  # (64, 64)

    # X_train과 동일하게 object dtype 1D 배열로 반환 (raw wafer map 리스트)
    synthetic_arr = np.empty(len(synthetic), dtype=object)
    for i, s in enumerate(synthetic):
        synthetic_arr[i] = s
    return synthetic_arr  # (N * n_per_sample,), 각 원소는 (64, 64)


# ═══════════════════════════════════════════════════════════
# 전체 파이프라인
# ═══════════════════════════════════════════════════════════
def augment_minority_classes(X_train, y_train, target_classes,
                             multiplier=2, device="cuda"):
    """
    recall 낮은 소수 class들을 (현재 개수 × multiplier)까지 증강.
    """
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for cls_idx in target_classes:
        # 해당 class 샘플만 추출 (원본 wafer map, 리사이즈/one-hot 미적용)
        mask = y_train == cls_idx
        minority_maps_raw = X_train[mask]
        current_count = len(minority_maps_raw)
        target_count = current_count * multiplier

        if current_count == 0 or current_count >= target_count:
            continue

        n_needed = target_count - current_count
        n_per_sample = max(1, n_needed // current_count + 1)

        print(f"\nClass {cls_idx}: {current_count}장 → {target_count}장 목표")

        # WM811KDataset의 전처리(리사이즈 64x64 + one-hot)를 그대로 재사용
        minority_ds = WM811KDataset(minority_maps_raw, np.zeros(current_count), augment=False)
        minority_maps = np.stack([minority_ds[i][0].numpy() for i in range(current_count)])

        # 1. Denoising AE 학습
        ae = train_denoising_ae(minority_maps, device=device)

        # 2. 합성 생성
        synthetic = generate_synthetic(ae, minority_maps,
                                       n_per_sample=n_per_sample, device=device)
        synthetic = synthetic[:n_needed]  # 필요한 만큼만

        # 3. 추가
        X_aug_list.append(synthetic)
        y_aug_list.append(np.full(len(synthetic), cls_idx))
        print(f"  → {len(synthetic)}장 합성 추가")

    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list, axis=0)

    return X_augmented, y_augmented


# ═══════════════════════════════════════════════════════════
# 실행 예시
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy", allow_pickle=True)
    y_train_str = np.load("data/processed/y_train.npy", allow_pickle=True)
    classes = np.load("data/processed/label_encoder_classes.npy", allow_pickle=True)

    # failureType 문자열 -> 정수 인덱스 (classes 순서 기준)
    label_to_idx = {label: i for i, label in enumerate(classes)}
    y_train = np.array([label_to_idx[l] for l in y_train_str])

    # 02_model_evaluation에서 recall<70%로 판단된 class
    scratch_idx = list(classes).index("Scratch")
    nearfull_idx = list(classes).index("Near-full")
    donut_idx = list(classes).index("Donut")
    target_classes = [scratch_idx, nearfull_idx, donut_idx]

    X_aug, y_aug = augment_minority_classes(
        X_train, y_train, target_classes, multiplier=2
    )

    # y_train.npy와 동일하게 문자열 라벨로 되돌려서 저장 (train.py가 포맷 구분 없이 사용 가능)
    y_aug_str = classes[y_aug]

    print(f"\n증강 전: {len(X_train)}장 → 증강 후: {len(X_aug)}장")
    np.save("data/processed/X_train_augmented.npy", X_aug)
    np.save("data/processed/y_train_augmented.npy", y_aug_str)
    # 이후 train.py를 X_train_augmented로 재학습