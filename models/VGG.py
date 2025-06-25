import torch
from torch import nn
from typing import Tuple, Callable

from models.LotteryLayers import LotteryConv2D, LotteryDense
from models.base import BaseModel

import types

modelcfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

valid = (11, 13, 16, 19)


def bn_init(self) -> None:
    self.reset_running_stats()
    if self.affine:
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

def conv_fc_init(self) -> None:
    nn.init.kaiming_normal_(self.weight)

class VGG(BaseModel):

    class ConvBN(nn.Module):
    
        def __init__(self, input_channels: int, 
                        num_filters: int, 
                        kernel_size: int = 3, 
                        stride: int = 1, 
                        padding: str = "same",
                        custom_init = False): 
            
            super(VGG.ConvBN, self).__init__()

            conv_obj = LotteryConv2D(input_channels, num_filters, 
                                kernel_size, stride, padding)

            if custom_init: 
                conv_obj.reset_parameters = types.MethodType(conv_fc_init, conv_obj)
                conv_obj.reset_parameters()

            self.register_module("conv", conv_obj)
            
            norm_obj = nn.BatchNorm2d(num_filters, track_running_stats = False)

            if custom_init: 
                norm_obj.reset_parameters = types.MethodType(bn_init, norm_obj)
                norm_obj.reset_parameters()

            self.register_module("norm", norm_obj)

            self.register_module("relu", nn.ReLU())

            #self.get_parameter("norm.weight").data = torch.rand(self.get_parameter("norm.weight").shape) # Reinit BN 

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            return self.get_submodule("relu")(x)
        
    class OutBlock(nn.Module):

        def __init__(self, in_features: int, custom_init = False):

            super(VGG.OutBlock, self).__init__()

            self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))
            #self.register_module("drop", nn.Dropout(dropout))
            #self.register_module("norm", nn.BatchNorm1d(in_features))
            #self.register_module("relu", FakeHReLU())

            fc_obj = nn.Linear(in_features, 10)

            if custom_init: 
                fc_obj.reset_parameters = types.MethodType(conv_fc_init, fc_obj)
                fc_obj.reset_parameters()

            self.register_module("fc", fc_obj)

            #self.get_parameter("norm.weight").data = torch.rand(self.get_parameter("norm.weight").shape)
        
        def forward(self, x):
            x = self.get_submodule("gap")(x)
            x = x.squeeze((-1, -2))
            #x = self.get_submodule("drop")(x)
            #x = self.get_submodule("norm")(x)
            x = self.get_submodule("fc")(x)
            return x

    def __init__(self, rank: int, world_size: int, depth: int = 19, input_channels: int = 3, custom_init = True):
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
                self.register_module(f"block{curr_block}{curr_num}", self.ConvBN(input_channels, value, custom_init = custom_init))
                self.layers.append(f"block{curr_block}{curr_num}")
                curr_num += 1
                input_channels = value

        self.out = self.OutBlock(512, custom_init = custom_init)

        self.init_base(rank, world_size)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = self.get_submodule(layer)(x)

        x = self.out(x)

        return x