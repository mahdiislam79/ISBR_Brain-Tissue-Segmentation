import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1):
        super(Encoder, self).__init__()
        self.conv = self._block(in_channels, out_channels * factor)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _block(self, in_channels, out_channels):
        mid_channels = out_channels * 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, factor=1, use_concatenate=False):
        super(Decoder, self).__init__()
        self.use_concatenate = use_concatenate
        self.up = nn.ConvTranspose2d(in_channels * factor, out_channels * factor, kernel_size=2, stride=2)
        self.conv = self._block(out_channels * factor * (2 if use_concatenate else 1), out_channels * factor)

    def _block(self, in_channels, out_channels):
        mid_channels = out_channels * 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Align dimensions if mismatched
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)

        if self.use_concatenate:
            x = torch.cat([x, skip], dim=1)
        else:
            x = x + skip
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, factor=1, use_concatenate=True):
        super(UNet, self).__init__()

        # Encoder Path
        self.enc1 = Encoder(in_channels, 32, factor)
        self.enc2 = Encoder(32, 64, factor)
        self.enc3 = Encoder(64, 128, factor)
        self.enc4 = Encoder(128, 256, factor)

        # Decoder Path
        self.dec1 = Decoder(256, 128, factor, use_concatenate)
        self.dec2 = Decoder(128, 64, factor, use_concatenate)
        self.dec3 = Decoder(64, 32, factor, use_concatenate)

        # Output layer
        self.final_conv = nn.Conv2d(32 * factor, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1, pool1 = self.enc1(x)
        skip2, pool2 = self.enc2(pool1)
        skip3, pool3 = self.enc3(pool2)
        _, bottleneck = self.enc4(pool3)

        # Decoder
        up1 = self.dec1(bottleneck, skip3)
        up2 = self.dec2(up1, skip2)
        up3 = self.dec3(up2, skip1)

        # Output
        return torch.sigmoid(self.final_conv(up3))
