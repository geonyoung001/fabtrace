# ml/train/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ml.config import NUM_CLASSES

class WaferClassifier(nn.Module):      # Part 1: 모델
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):            # Part 2: 손실함수
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()