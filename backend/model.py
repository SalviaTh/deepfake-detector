import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.backbone._fc.in_features
        
        self.backbone._fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_maps(self, x):
        return self.backbone.extract_features(x)
