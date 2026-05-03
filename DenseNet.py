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


#早停
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # val_loss 有明顯下降
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True