# 实现ResNet中block模块。
import torch
from torch import nn

class ResBlock(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size,stride,padding)
        )

    def forward(self, x):
        return torch.relu(self.sequential(x) + x)

if __name__ == '__main__':
    x = torch.randn(10, 3, 224, 224)
    resBlock = ResBlock(3,16,3,1,1)
    y = resBlock(x)
    print(y.shape)