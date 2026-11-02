from typing import Any

import torch 
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channel, out_channels) :
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2 ,stride=2),
            DoubleConv(in_channel, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channels) :
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        self.inc = DoubleConv(in_channels,64)

        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)

        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self,x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)

        return self.outc(x)

class UpFlex(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class PretrainedUNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Encoder: pieces taken from ResNet34
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64, 128x128
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64,  64x64
        self.layer2 = resnet.layer2   # 128, 32x32
        self.layer3 = resnet.layer3   # 256, 16x16
        self.layer4 = resnet.layer4   # 512, 8x8

        # Decoder
        self.up1 = Up(512, 256)               # skip: layer3 (256)
        self.up2 = Up(256, 128)               # skip: layer2 (128)
        self.up3 = Up(128, 64)                # skip: layer1 (64)
        self.up4 = UpFlex(64, 64, 32)         # skip: stem (64), goes to 128x128
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # 256x256
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)                # 64,  128
        s1 = self.layer1(self.pool(s0))  # 64,  64
        s2 = self.layer2(s1)             # 128, 32
        s3 = self.layer3(s2)             # 256, 16
        b = self.layer4(s3)              # 512, 8

        x = self.up1(b, s3)              # 256, 16
        x = self.up2(x, s2)              # 128, 32
        x = self.up3(x, s1)              # 64,  64
        x = self.up4(x, s0)              # 32,  128
        x = self.final_up(x)             # 32,  256
        return self.outc(x)              # 3,   256



