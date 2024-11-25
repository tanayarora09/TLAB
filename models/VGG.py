import torch
from torch import nn
from typing import Tuple, Callable

from models.LotteryLayers import LotteryConv2D, LotteryDense
from models.base import BaseModel

modelcfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

valid = (11, 13, 16, 19)
    

class VGG(BaseModel):

    class ConvBN(nn.Module):
    
        def __init__(self, input_channels: int, 
                        num_filters: int, 
                        kernel_size: int = 3, 
                        stride: int = 1, 
                        padding: str = "same"): 
            
            super(VGG.ConvBN, self).__init__()

            self.register_module("conv", LotteryConv2D(input_channels, num_filters, 
                                                        kernel_size, stride, padding))
            
            self.register_module("norm", nn.BatchNorm2d(num_filters, track_running_stats = False))

            self.register_module("relu", nn.ReLU())

            #self.get_parameter("norm.weight").data = torch.rand(self.get_parameter("norm.weight").shape) # Reinit BN 

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            return self.get_submodule("relu")(x)
        
    class OutBlock(nn.Module):

        def __init__(self, in_features: int):

            super(VGG.OutBlock, self).__init__()

            self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))
            #self.register_module("drop", nn.Dropout(dropout))
            #self.register_module("norm", nn.BatchNorm1d(in_features))
            #self.register_module("relu", FakeHReLU())
            self.register_module("fc", nn.Linear(in_features, 10))

            #self.get_parameter("norm.weight").data = torch.rand(self.get_parameter("norm.weight").shape)
        
        def forward(self, x):
            x = self.get_submodule("gap")(x)
            x = x.squeeze((-1, -2))
            #x = self.get_submodule("drop")(x)
            #x = self.get_submodule("norm")(x)
            x = self.get_submodule("fc")(x)
            return x

    def __init__(self, rank: int, depth: int = 19, input_channels: int = 3):
        super(VGG, self).__init__()

        if depth not in valid:
            raise ValueError("VGG architecture must be one of", [f"VGG{n}" for n in valid])
        
        curr_block = 1
        curr_num = 1
        self.layers = []

        for value in modelcfgs[depth]:
            
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

        self.out = self.OutBlock(512)

        self.init_base(rank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = self.get_submodule(layer)(x)

        x = self.out(x)

        return x