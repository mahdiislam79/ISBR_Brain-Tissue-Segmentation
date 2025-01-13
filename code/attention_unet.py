import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = x + skip  # Additive skip connection
        return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gating_channels):
        super().__init__()
        self.W_skip = nn.Conv2d(skip_channels, skip_channels // 2, kernel_size=1)
        self.W_gating = nn.Conv2d(gating_channels, skip_channels // 2, kernel_size=1)
        self.psi = nn.Conv2d(skip_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, gating):
        gating = F.interpolate(gating, size=skip.size()[2:], mode='bilinear', align_corners=False)
        attention = self.sigmoid(self.psi(F.relu(self.W_skip(skip) + self.W_gating(gating))))
        return skip * attention

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.att4 = AttentionGate(512, 1024)
        self.att3 = AttentionGate(256, 512)
        self.att2 = AttentionGate(128, 256)
        self.att1 = AttentionGate(64, 128)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1, pool1 = self.enc1(x)
        enc2, pool2 = self.enc2(pool1)
        enc3, pool3 = self.enc3(pool2)
        enc4, pool4 = self.enc4(pool3)

        bottleneck = self.bottleneck(pool4)

        dec4 = self.dec4(bottleneck, self.att4(enc4, bottleneck))
        dec3 = self.dec3(dec4, self.att3(enc3, dec4))
        dec2 = self.dec2(dec3, self.att2(enc2, dec3))
        dec1 = self.dec1(dec2, self.att1(enc1, dec2))

        return torch.sigmoid(self.final(dec1))
