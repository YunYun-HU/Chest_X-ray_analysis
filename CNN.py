import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_size=224, num_classes=8,
                 base_filters=32, dense_units=256, dropout_rate=0.4):

        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        conv1 = base_filters
        conv2 = base_filters * 2
        conv3 = base_filters * 4
        conv4 = base_filters * 8
        conv5 = base_filters * 8

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, 3, padding=1),
            nn.BatchNorm2d(conv1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(conv1, conv2, 3, padding=1),
            nn.BatchNorm2d(conv2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(conv2, conv3, 3, padding=1),
            nn.BatchNorm2d(conv3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(conv3, conv4, 3, padding=1),
            nn.BatchNorm2d(conv4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(conv4, conv5, 3, padding=1),
            nn.BatchNorm2d(conv5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv5, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x