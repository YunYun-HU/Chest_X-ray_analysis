import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


# -------------------------
# Channel Attention
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        hidden_channels = in_channels // reduction

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Max pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * attention


# -------------------------
# Spatial Attention
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿 channel 維度做平均與最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))

        return x * attention


# -------------------------
# CBAM
# -------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# -------------------------
# DenseNet + CBAM
# -------------------------
class DenseNet(nn.Module):
    def __init__(
        self,
        input_size=224,
        num_classes=8,
        dense_units=256,
        dropout_rate=0.4,
        pretrained=False,
        use_cbam=True,
        input_channels=1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.use_cbam = use_cbam

        # 載入 DenseNet-121
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)

        # -------- 改第一層：3 channel -> 1 channel --------
        old_conv = self.backbone.features.conv0

        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # 如果用 pretrained，將 RGB 權重平均成灰階權重
        if pretrained:
            with torch.no_grad():
                gray_weight = old_conv.weight.mean(dim=1, keepdim=True)

                if input_channels == 1:
                    new_conv.weight.copy_(gray_weight)

                elif input_channels == 2:
                    new_weight = gray_weight.repeat(1, 2, 1, 1) / 2.0
                    new_conv.weight.copy_(new_weight)

                else:
                    new_weight = gray_weight.repeat(1, input_channels, 1, 1) / input_channels
                    new_conv.weight.copy_(new_weight)

        self.backbone.features.conv0 = new_conv

        # DenseNet121 最後 feature channel 是 1024
        in_features = self.backbone.classifier.in_features

        # -------- 加 CBAM：放在 features 後面、pooling 前面 --------
        if use_cbam:
            self.cbam = CBAM(
                in_channels=in_features,
                reduction=16,
                kernel_size=7
            )
        else:
            self.cbam = nn.Identity()

        # -------- 改分類頭 --------
        self.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        # DenseNet feature extractor
        features = self.backbone.features(x)

        # CBAM attention
        features = self.cbam(features)

        # DenseNet 原本 forward 的後半段
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        logits = self.classifier(out)

        return logits


# -------------------------
# Early Stopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True