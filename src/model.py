import torch 
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channels) -> None:
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