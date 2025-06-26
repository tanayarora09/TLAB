import torch
from torch import nn
from typing import Tuple, Callable

from models.LotteryLayers import LotteryConv2D, LotteryDense
from models.base import BaseModel

import types

def modelcfgs(depth: int):
    d = (depth - 2)//6
    w = 16
    return [(w, d), (2*w, d), (4*w, d)]

valid = tuple(2 + 6 * n for n in range(1, 10))


def bn_init(self) -> None:
    self.reset_running_stats()
    if self.affine:
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

def conv_fc_init(self) -> None:
    nn.init.kaiming_normal_(self.weight)

class ResNet(BaseModel):

    class ResBlock(nn.Module):
        
        def _make_conv_obj(self, input_channels, num_filters, kernel_size, stride, padding, custom_init):

            conv_obj = LotteryConv2D(input_channels, num_filters, kernel_size, stride, padding)

            if custom_init: 
                conv_obj.reset_parameters = types.MethodType(conv_fc_init, conv_obj)
                conv_obj.reset_parameters()
                
            return conv_obj
        
        def _make_bn_obj(self, num_filters, custom_init):

            norm_obj = nn.BatchNorm2d(num_filters, track_running_stats = False)

            if custom_init: 
                norm_obj.reset_parameters = types.MethodType(bn_init, norm_obj)
                norm_obj.reset_parameters()
            
            return norm_obj


        def __init__(self, input_channels: int,
                        num_filters:int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1,
                        custom_init = True,
                        downsample: bool = False):

            super(ResNet.ResBlock, self).__init__()        

            self.downsample = downsample

            if downsample and stride == 1: raise ValueError

            self.register_module("1conv", self._make_conv_obj(input_channels, num_filters,
                                                              kernel_size, stride, padding, 
                                                              custom_init))


            self.register_module("1norm", self._make_bn_obj(num_filters, custom_init))

            self.register_module("1relu", nn.ReLU())

            self.register_module("2conv", self._make_conv_obj(num_filters, num_filters,
                                                              kernel_size, 1, padding, 
                                                              custom_init))

            self.register_module("2norm", self._make_bn_obj(num_filters, custom_init))

            if downsample or input_channels != num_filters:
                
                self.register_module("3conv", self._make_conv_obj(input_channels, num_filters,
                                                                kernel_size, stride, padding, 
                                                                custom_init))              

                self.register_module("3norm", self._make_bn_obj(num_filters, custom_init))

            self.register_module("2relu", nn.ReLU())

        def forward(self, x):

            res = x

            x = self.get_submodule("1conv")(x)
            x = self.get_submodule("1norm")(x)
            x = self.get_submodule("1relu")(x)

            x = self.get_submodule("2conv")(x)
            x = self.get_submodule("2norm")(x)

            if self.downsample:
                res = self.get_submodule("3conv")(res)
                res = self.get_submodule("3norm")(res)

            x += res
            x = self.get_submodule("2relu")(x)

            return x

    class InBlock(nn.Module):

        def _make_conv_obj(self, input_channels, num_filters, kernel_size, stride, padding, custom_init):

            conv_obj = LotteryConv2D(input_channels, num_filters, kernel_size, stride, padding)

            if custom_init: 
                conv_obj.reset_parameters = types.MethodType(conv_fc_init, conv_obj)
                conv_obj.reset_parameters()
                
            return conv_obj
        
        def _make_bn_obj(self, num_filters, custom_init):

            norm_obj = nn.BatchNorm2d(num_filters, track_running_stats = False)

            if custom_init: 
                norm_obj.reset_parameters = types.MethodType(bn_init, norm_obj)
                norm_obj.reset_parameters()
            
            return norm_obj

        def __init__(self, input_channels: int,
                        num_filters:int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1,
                        custom_init = True):

            super(ResNet.InBlock, self).__init__()        

            self.register_module("conv", self._make_conv_obj(input_channels, num_filters,
                                                            kernel_size, stride, padding, 
                                                            custom_init))


            self.register_module("norm", self._make_bn_obj(num_filters, custom_init))

            self.register_module("relu", nn.ReLU())

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            x = self.get_submodule("relu")(x)
            return x

    class OutBlock(nn.Module):

        def __init__(self, in_features: int, custom_init = False):

            super(ResNet.OutBlock, self).__init__()

            self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))

            fc_obj = LotteryDense(in_features, 10)

            if custom_init: 
                fc_obj.reset_parameters = types.MethodType(conv_fc_init, fc_obj)
                fc_obj.reset_parameters()

            self.register_module("fc", fc_obj)

        def forward(self, x):
            x = self.get_submodule("gap")(x)
            x = x.squeeze((-1, -2))
            x = self.get_submodule("fc")(x)
            return x

    def __init__(self, rank: int, world_size: int, depth: int = 20, input_channels: int = 3, custom_init = True):
        super(ResNet, self).__init__()

        if depth not in valid:
            raise ValueError("ResNet architecture must have depth 2 + 6n")
        
        #curr_block = 1
        #curr_num = 1
        self.layers = []
        plan = modelcfgs(depth)
        
        current = plan[0][0]
        self.inblock = self.InBlock(3, current, custom_init = custom_init)


        for idx, (filters, num_blocks) in enumerate(plan):
            for block_idx in range(num_blocks):
                downsample = idx > 0 and block_idx == 0
                self.register_module(f"block{idx}{block_idx}", 
                                     self.ResBlock(current, filters, 
                                                stride = 2 if downsample else 1,
                                                downsample = downsample,
                                                custom_init = custom_init))
                self.layers.append(f"block{idx}{block_idx}")
                current = filters

        self.outblock = self.OutBlock(plan[-1][0], custom_init = custom_init)

        self.init_base(rank, world_size)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.inblock(x)

        for layer in self.layers:
            x = self.get_submodule(layer)(x)

        x = self.outblock(x)

        return x
    