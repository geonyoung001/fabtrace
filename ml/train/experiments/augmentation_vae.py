# ml/train/augmentation_vae.py
import torch
import torch.nn as nn
import numpy as np

from dataset import WM811KDataset


# ═══════════════════════════════════════════════════════════
# Convolutional VAE
# augementaion_ae.py의 DenoisingConvAE와 달리, latent를 mu/logvar로 모델링하고
# reparameterization trick으로 샘플링 -> KL divergence로 latent space를 정규화.
# "노이즈를 임의로 주입"하는 대신 학습된 분포에서 직접 샘플링하므로 더 다양하고
# 그럴듯한 합성 샘플을 만들 수 있다는 게 AE 대비 기대되는 이점.
# ═══════════════════════════════════════════════════════════
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # 64→32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 32→16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            # raw logits 출력 (CrossEntropyLoss가 내부에서 softmax)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ═══════════════════════════════════════════════════════════
# 학습: recon(CrossEntropy) + KL divergence
# ═══════════════════════════════════════════════════════════
def train_vae(minority_maps, latent_dim=128, epochs=200,
              beta=0.01, device="cuda"):
    """
    beta: KL 항 가중치. 작을수록 recon 품질 우선(posterior collapse 방지),
    클수록 latent가 정규분포에 가까워져 샘플링 품질이 좋아짐.
    소수 class처럼 데이터가 적을 땐 beta를 작게 잡아 recon을 우선한다.
    """
    vae = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    recon_criterion = nn.CrossEntropyLoss()

    X = torch.FloatTensor(np.array(minority_maps)).to(device)  # (N, 3, 64, 64) one-hot
    y_target = X.argmax(dim=1).long()  # (N, 64, 64)

    vae.train()
    for epoch in range(epochs):
        X_recon, mu, logvar = vae(X)

        recon_loss = recon_criterion(X_recon, y_target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} | recon={recon_loss.item():.4f} | "
                  f"kl={kl_loss.item():.4f} | lr: {lr:.6f}")

    return vae


# ═══════════════════════════════════════════════════════════
# 생성: 각 원본의 posterior(mu, logvar)에서 직접 재샘플링
# ═══════════════════════════════════════════════════════════
def generate_synthetic(vae, minority_maps, n_per_sample=5, device="cuda"):
    vae.eval()
    X = torch.FloatTensor(np.array(minority_maps)).to(device)
    synthetic = []

    with torch.no_grad():
        for i in range(len(X)):
            mu, logvar = vae.encode(X[i:i+1])

            for _ in range(n_per_sample):
                z = vae.reparameterize(mu, logvar)  # posterior에서 새로 샘플링
                new_img = vae.decode(z)
                categorical = new_img.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)
                synthetic.append(categorical)

    synthetic_arr = np.empty(len(synthetic), dtype=object)
    for i, s in enumerate(synthetic):
        synthetic_arr[i] = s
    return synthetic_arr


# ═══════════════════════════════════════════════════════════
# 전체 파이프라인 (augementaion_ae.augment_minority_classes와 동일 구조)
# ═══════════════════════════════════════════════════════════
def augment_minority_classes_vae(X_train, y_train, target_classes,
                                  multiplier=2, device="cuda"):
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for cls_idx in target_classes:
        mask = y_train == cls_idx
        minority_maps_raw = X_train[mask]
        current_count = len(minority_maps_raw)
        target_count = current_count * multiplier

        if current_count == 0 or current_count >= target_count:
            continue

        n_needed = target_count - current_count
        n_per_sample = max(1, n_needed // current_count + 1)

        print(f"\nClass {cls_idx}: {current_count}장 → {target_count}장 목표 (VAE)")

        minority_ds = WM811KDataset(minority_maps_raw, np.zeros(current_count), augment=False)
        minority_maps = np.stack([minority_ds[i][0].numpy() for i in range(current_count)])

        vae = train_vae(minority_maps, device=device)
        synthetic = generate_synthetic(vae, minority_maps, n_per_sample=n_per_sample, device=device)
        synthetic = synthetic[:n_needed]

        X_aug_list.append(synthetic)
        y_aug_list.append(np.full(len(synthetic), cls_idx))
        print(f"  → {len(synthetic)}장 합성 추가")

    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list, axis=0)

    return X_augmented, y_augmented
