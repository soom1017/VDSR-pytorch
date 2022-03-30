from audioop import bias
import torch
import torch.nn as nn

class Conv_Relu_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class VDSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_layer = self.make_layer(Conv_Relu_Block, 18)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, num_block):
        layers = []
        for _ in range(num_block):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.relu(self.input(x))
        res = self.residual_layer(res)
        res = self.output(res)
        out = torch.add(x, res)
        return out