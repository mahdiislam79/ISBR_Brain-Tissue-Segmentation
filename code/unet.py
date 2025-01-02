import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, factor=1):
        super(UNet, self).__init__()
        
        # Downsampling Path
        self.conv1 = self._block(in_channels, 32 * factor)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = self._block(32 * factor, 64 * factor)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self._block(64 * factor, 128 * factor)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = self._block(128 * factor, 256 * factor)

        # Upsampling Path
        self.up5 = nn.ConvTranspose2d(256 * factor, 128 * factor, kernel_size=2, stride=2)
        self.conv5 = self._block(256 * factor, 128 * factor)

        self.up6 = nn.ConvTranspose2d(128 * factor, 64 * factor, kernel_size=2, stride=2)
        self.conv6 = self._block(128 * factor, 64 * factor)

        self.up7 = nn.ConvTranspose2d(64 * factor, 32 * factor, kernel_size=2, stride=2)
        self.conv7 = self._block(64 * factor, 32 * factor)

        # Output layer
        self.final_conv = nn.Conv2d(32 * factor, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        """Helper function to create a conv-batchnorm-relu block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downsampling
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        # Upsampling
        up5 = self.up5(conv4)
        up5 = torch.cat([up5, conv3], dim=1)
        conv5 = self.conv5(up5)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv2], dim=1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv1], dim=1)
        conv7 = self.conv7(up7)

        return torch.sigmoid(self.final_conv(conv7))


