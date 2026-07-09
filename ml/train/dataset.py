import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize as sk_resize
import torchvision.transforms as T

class WM811KDataset(Dataset):
    def __init__(self, wafer_maps, labels, augment=False):
        # [1] 초기화 — 데이터를 받아 저장
        self.wafer_maps = wafer_maps    # 원본 wafer map 리스트 (0/1/2 정수)
        self.labels = labels            # 정답 라벨 (0~8 정수)
        self.augment = augment          # train이면 True, val/test면 False

        # augmentation 정의 (train일 때만 사용)
        self.transform = T.Compose([
            T.RandomRotation(180),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]) if augment else None

    def __len__(self):
        # [2] 전체 데이터 개수 반환
        return len(self.labels)

    def __getitem__(self, idx):
        # [3] idx번째 데이터 1장을 꺼내 변환
        wm = self.wafer_maps[idx]           # 원본 로드
        wm = self.resize_and_onehot(wm)     # resize + one-hot
        x = torch.FloatTensor(wm)           # 텐서 변환

        if self.transform:
            x = self.transform(x)           # train이면 augmentation

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def resize_and_onehot(self, wm, target_size=(64, 64)):
        # nearest-neighbour resize (이산값 0/1/2 보존)
        wm_resized = sk_resize(wm, target_size, order=0,
                            preserve_range=True, anti_aliasing=False).astype(np.int8)
        # 3-channel one-hot
        ch0 = (wm_resized == 0).astype(np.float32)  # wafer 밖
        ch1 = (wm_resized == 1).astype(np.float32)  # 정상 die
        ch2 = (wm_resized == 2).astype(np.float32)  # 불량 die
        return np.stack([ch0, ch1, ch2], axis=0)     # (3, 64, 64)