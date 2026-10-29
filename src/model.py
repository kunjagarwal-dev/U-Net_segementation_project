from typing import Any

import torch 
import torch.nn as nn

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
