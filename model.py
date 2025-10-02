import torch.nn as nn
import torch

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class CustomCIFARNet(nn.Module):
    """
    Custom CNN for CIFAR-10 with:
    - C1C2C3C4 architecture (No MaxPooling)
    - Dilated convolutions for downsampling
    - Depthwise Separable Convolution
    - GAP + FC
    - Params < 200k
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1 Block
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 20, 3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )

        # C2 Block
        self.c2 = nn.Sequential(
            nn.Conv2d(20, 28, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 36, 3, padding=1, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(),
        )

        # C3 Block
        self.c3 = nn.Sequential(
            DepthwiseSeparableConv(36, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 56, 3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(),
        )

        # C4 Block
        self.c4 = nn.Sequential(
            nn.Conv2d(56, 64, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 72, 3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 40, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(40, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_model(num_classes: int = 10) -> nn.Module:
    """Factory to get the CIFAR model used in this project."""
    return CustomCIFARNet(num_classes=num_classes)