import torch
from torch import nn
from typing import Tuple

from LotteryLayers import LotteryConv2D, LotteryDense

models = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

valid = (11, 13, 16, 19)

class VGG(nn.Module):

    class ConvBN(nn.Module):
    
        def __init__(self, input_channels: int, 
                        num_filters: int, 
                        kernel_size: Tuple[int, int] = (3, 3), 
                        stride: Tuple[int, int] = (1, 1), 
                        padding: str = "same"): 
            
            super(VGG.ConvBN, self).__init__()

            self.register_module("conv", LotteryConv2D(input_channels, num_filters, 
                                                        kernel_size, stride, padding))
            
            self.register_module("norm", nn.BatchNorm2d(num_filters, track_running_stats = False))

            self.register_module("relu", nn.ReLU())

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            return self.get_submodule("relu")(x)
        
    class OutBlock(nn.Module):

        def __init__(self, in_features: int, dropout: float):

            super(VGG.OutBlock, self).__init__()

            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.drop = nn.Dropout(dropout)
            self.norm = nn.BatchNorm1d(in_features, track_running_stats = False)
            self.fc = nn.Linear(in_features, 10)
        
        def forward(self, x):
            x = self.gap(x)
            x = x.squeeze()
            x = self.drop(x)
            x = self.norm(x)
            x = self.fc(x)
            return x

    def __init__(self, depth: int = 19, input_channels: int = 3):
        super(VGG, self).__init__()

        if depth not in valid:
            raise ValueError("VGG architecture must be one of", [f"VGG{n}" for n in valid])
        
        curr_block = 1
        curr_num = 1
        self.layers = []

        for value in models[depth]:
            
            if value == "M":
                self.register_module(f"block{curr_block}p", nn.MaxPool2d((2, 2), (2, 2)))
                self.layers.append(f"block{curr_block}p")
                curr_block += 1
                curr_num = 1

            else:
                self.register_module(f"block{curr_block}{curr_num}", self.ConvBN(input_channels, value))
                self.layers.append(f"block{curr_block}{curr_num}")
                curr_num += 1
                input_channels = value

        self.out = self.OutBlock(512, 0.8)

    
    def reinit_bn(self):
        for name, module in self.named_modules:
            if "norm" in name:
                module.get_parameter("weight").random_()
        return 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = self.get_submodule(layer)(x)

        x = self.out(x)

        return x