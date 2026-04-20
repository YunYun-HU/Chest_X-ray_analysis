import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet(nn.Module):
    def __init__(
        self,
        input_size=224,
        num_classes=8,
        dense_units=256,
        dropout_rate=0.4,
        pretrained=False
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # 載入 DenseNet-121
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)

        # -------- 改第一層：3 channel -> 1 channel --------
        old_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,   # 64
            kernel_size=old_conv.kernel_size,     # 7
            stride=old_conv.stride,               # 2
            padding=old_conv.padding,             # 3
            bias=False
        )

        # 如果用 pretrained，將 RGB 權重平均成灰階權重
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.backbone.features.conv0 = new_conv

        # -------- 改分類頭 --------
        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)