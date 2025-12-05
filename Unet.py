import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) # Bias = False because Batch Normalization is used.
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False) # Bias = False because Batch Normalization is used.
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super().__init__()

        # --------------------------------------------------------------
        # Encoder (contracting path): extract features + reduce spatial size
        # Each DoubleConv increases feature richness while MaxPool downsamples.
        # --------------------------------------------------------------
        self.enc1 = DoubleConv(in_channels, 64)    # Level 1 features
        self.enc2 = DoubleConv(64, 128)            # Level 2 features
        self.enc3 = DoubleConv(128, 256)           # Level 3 features
        self.enc4 = DoubleConv(256, 512)           # Level 4 features

        # Downsampling layer (shared across encoder stages)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --------------------------------------------------------------
        # Bottleneck: deepest representation (lowest resolution, richest features)
        # --------------------------------------------------------------
        self.bottleneck = DoubleConv(512, 1024)

        # --------------------------------------------------------------
        # Decoder (expanding path): upsample + recover spatial details
        # ConvTranspose2d upsamples; skip-connections add encoder features.
        # --------------------------------------------------------------
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)   # concat(encoder_4, upsampled)

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # --------------------------------------------------------------
        # Final classifier: 1×1 convolution → one logit per class per pixel
        # --------------------------------------------------------------
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------------------------
        # Encoder: feature extraction
        # -------------------------------
        x1 = self.enc1(x)             # Preserve spatial details (for skip)
        x2 = self.enc2(self.pool(x1)) # Downsample + extract deeper features
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # -------------------------------
        # Bottleneck
        # -------------------------------
        x_bottleneck = self.bottleneck(self.pool(x4))

        # -------------------------------
        # Decoder: upsample and fuse with encoder features ("skip connections")
        # -------------------------------
        # Level 4
        x = self.up4(x_bottleneck)              # Upsample
        x = torch.cat([x4, x], dim=1)           # Skip-connection restores details
        x = self.dec4(x)                        # Refine features

        # Level 3
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        # Level 2
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        # Level 1
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        # -------------------------------
        # Final prediction (logits)
        # -------------------------------
        logits = self.classifier(x)  # (B, num_classes, H, W)
        return logits
