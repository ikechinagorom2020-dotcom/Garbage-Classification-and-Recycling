import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        return self.drop(self.pool(self.conv(x)))


class WasteNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, 0.10),
            ConvBlock(32, 64, 0.15),
            ConvBlock(64, 128, 0.20),
            ConvBlock(128, 256, 0.25),
            ConvBlock(256, 256, 0.30),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))
