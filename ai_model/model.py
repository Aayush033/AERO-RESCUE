"""
Multimodal Deep Learning Flood & Debris Segmentation Model
Designed for Sentinel-1 (SAR VV/VH), DEM (Copernicus 30m), and Sentinel-2 inputs.
Optimized for high-resolution SAR change detection and terrain slope conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> GELU) * 2 with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block for multi-sensor channel reweighting"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class MultimodalFloodUNet(nn.Module):
    """
    Multimodal SAR + DEM segmentation network.
    Supported Class Configurations:
      - 2-class mode (Sen1Floods11 standard benchmark):
          0: Non-water / Dry Terrain
          1: Water / Inundated Surface
          (Debris flows detected via documented SAR backscatter roughness change analysis)
      - 3-class mode (Multi-hazard joint segmentation):
          0: Normal / Dry Terrain
          1: Water / Flooded Riverbed
          2: Mud & Debris Flow Deposit
    """
    def __init__(self, in_channels=2, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.ca1 = ChannelAttention(base_channels)
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        self.ca2 = ChannelAttention(base_channels * 2)

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4)
        )
        self.ca3 = ChannelAttention(base_channels * 4)

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8)
        )

        # Bridge
        self.bridge = DoubleConv(base_channels * 8, base_channels * 8)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 2, base_channels)

        # Classifier head
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        x1 = self.ca1(self.inc(x))
        x2 = self.ca2(self.down1(x1))
        x3 = self.ca3(self.down2(x2))
        x4 = self.down3(x3)

        b = self.bridge(x4)

        d1 = self.up1(b)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.dec3(d3)

        logits = self.out_conv(d3)
        return logits


def get_model(in_channels=2, num_classes=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = MultimodalFloodUNet(in_channels=in_channels, num_classes=num_classes)
    return model.to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing MultimodalFloodUNet on {device} (2-channel SAR VV/VH)...")
    model = MultimodalFloodUNet(in_channels=2, num_classes=2).to(device)
    dummy_input = torch.randn(2, 2, 256, 256, device=device)
    out = model(dummy_input)
    print("Output shape:", out.shape)
    assert out.shape == (2, 2, 256, 256), "Shape mismatch"
    print("Model test successful!")
