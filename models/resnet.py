import torch
from torch import nn
from typing import Tuple, Callable

from models.LotteryLayers import LotteryConv2D, LotteryDense
from models.base import BaseModel

import types

def modelcfgs(depth: int):
    if depth in valid:
        d = (depth - 2)//6
        w = 16
        return [(w, d), (2*w, d), (4*w, d)]

valid = tuple(2 + 6 * n for n in range(1, 10))
ivalid = (18, 34, 50, 101, 152)


def bn_init(self) -> None:
    self.reset_running_stats()
    if self.affine:
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

def conv_fc_init(self) -> None:
    nn.init.kaiming_normal_(self.weight)

def conv(input_channels, num_filters, kernel_size, stride, padding, custom_init):
    conv_obj = LotteryConv2D(input_channels, num_filters, kernel_size, stride, padding)
    if custom_init: 
        conv_obj.reset_parameters = types.MethodType(conv_fc_init, conv_obj)
        conv_obj.reset_parameters()
    return conv_obj   

def bn(num_filters, custom_init, bn_track):

    norm_obj = nn.BatchNorm2d(num_filters, track_running_stats = bn_track)

    if custom_init: 
        norm_obj.reset_parameters = types.MethodType(bn_init, norm_obj)
        norm_obj.reset_parameters()
    
    return norm_obj

def fc(infeatures, outfeatures, custom_init):

    fc_obj = LotteryDense(infeatures, outfeatures)

    if custom_init: 
        fc_obj.reset_parameters = types.MethodType(conv_fc_init, fc_obj)
        fc_obj.reset_parameters()
    
    return fc_obj

class ResNetCifar(BaseModel):

    class ResBlock(nn.Module):

        def __init__(self, input_channels: int,
                        num_filters:int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1,
                        custom_init = True,
                        downsample: bool = False,
                        bn_track: bool = False):

            super(ResNetCifar.ResBlock, self).__init__()        

            self.downsample = downsample

            if downsample and stride == 1: raise ValueError

            self.register_module("1conv", conv(input_channels, num_filters,
                                                              kernel_size, stride, padding, 
                                                              custom_init))


            self.register_module("1norm", bn(num_filters, custom_init, bn_track))

            self.register_module("1relu", nn.ReLU())

            self.register_module("2conv", conv(num_filters, num_filters,
                                                              kernel_size, 1, padding, 
                                                              custom_init))

            self.register_module("2norm", bn(num_filters, custom_init, bn_track))

            if downsample or input_channels != num_filters:
                
                self.register_module("3conv", conv(input_channels, num_filters,
                                                                kernel_size, stride, padding, 
                                                                custom_init))              

                self.register_module("3norm", bn(num_filters, custom_init, bn_track))

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

        def __init__(self, input_channels: int,
                        num_filters:int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1,
                        custom_init = True,
                        bn_track: bool = False):

            super(ResNetCifar.InBlock, self).__init__()        

            self.register_module("conv", conv(input_channels, num_filters,
                                                            kernel_size, stride, padding, 
                                                            custom_init))


            self.register_module("norm", bn(num_filters, custom_init, bn_track))

            self.register_module("relu", nn.ReLU())

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            x = self.get_submodule("relu")(x)
            return x

    class OutBlock(nn.Module):

        def __init__(self, infeatures: int, outfeatures: int, custom_init = False, dropout = None):

            super(ResNetCifar.OutBlock, self).__init__()

            self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))
            self.register_module("fc", fc(infeatures, outfeatures, custom_init))

            if dropout is not None: 
                self.register_module("dropout", nn.Dropout(dropout))
                self.dropout = True
            else:
                self.dropout = False

        def forward(self, x):
            x = self.get_submodule("gap")(x)
            x = x.squeeze((-1, -2))
            if self.dropout: x = self.get_submodule("dropout")(x)
            x = self.get_submodule("fc")(x)
            return x

    def __init__(self, rank: int, world_size: int, depth: int = 20, outfeatures: int = 10, inchannels: int = 3, custom_init = True, bn_track = False, dropout = None):
        super(ResNetCifar, self).__init__()

        if depth not in valid:
            raise ValueError("ResNet architecture must have depth 2 + 6n")
        
        self.layers = []
        plan = modelcfgs(depth)
        
        current = plan[0][0]
        self.inblock = self.InBlock(inchannels, current, custom_init = custom_init, bn_track = bn_track)


        for idx, (filters, num_blocks) in enumerate(plan):
            for block_idx in range(num_blocks):
                downsample = idx > 0 and block_idx == 0
                self.register_module(f"block{idx}{block_idx}", 
                                     self.ResBlock(current, filters, 
                                                stride = 2 if downsample else 1,
                                                downsample = downsample,
                                                custom_init = custom_init,
                                                bn_track = bn_track))
                self.layers.append(f"block{idx}{block_idx}")
                current = filters

        self.outblock = self.OutBlock(plan[-1][0], outfeatures, custom_init = custom_init, dropout = dropout)

        self.init_base(rank, world_size)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.inblock(x)

        for layer in self.layers:
            x = self.get_submodule(layer)(x)

        x = self.outblock(x)

        return x

class ResNetImagenet(BaseModel):

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, downsample=False, custom_init=True, bn_track=False):
            super(ResNetImagenet.BasicBlock, self).__init__()

            self.register_module("1conv", conv(in_planes, planes, 3, stride, 1, custom_init))
            self.register_module("1bn", bn(planes, custom_init, bn_track))
            self.register_module("1relu", nn.ReLU())

            self.register_module("2conv", conv(planes, planes, 3, 1, 1, custom_init))
            self.register_module("2bn", bn(planes, custom_init, bn_track))
            self.register_module("2relu", nn.ReLU())

            self.downsample = downsample
            
            if downsample:
                self.register_module("3conv", conv(in_planes, planes * self.expansion, 1, stride, 0, custom_init))
                self.register_module("3bn", bn(planes * self.expansion, custom_init, bn_track))

        def forward(self, x):
            identity = x

            out = self.get_submodule("1conv")(x)
            out = self.get_submodule("1bn")(out)
            out = self.get_submodule("1relu")(out)

            out = self.get_submodule("2conv")(out)
            out = self.get_submodule("2bn")(out)

            if self.downsample:
                identity = self.get_submodule("3conv")(identity)
                identity = self.get_submodule("3bn")(identity)

            out += identity
            out = self.get_submodule("2relu")(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4
        
        def __init__(self, in_planes, planes, stride=1, downsample=False, custom_init=True, bn_track=False):
            super(ResNetImagenet.Bottleneck, self).__init__()

            self.register_module("1conv", conv(in_planes, planes, 1, 1, 0, custom_init))
            self.register_module("1bn", bn(planes, custom_init, bn_track))
            self.register_module("1relu", nn.ReLU())

            self.register_module("2conv", conv(planes, planes, 3, stride, 1, custom_init))
            self.register_module("2bn", bn(planes, custom_init, bn_track))
            self.register_module("2relu", nn.ReLU())

            self.register_module("3conv", conv(planes, planes * self.expansion, 1, 1, 0, custom_init))
            self.register_module("3bn", bn(planes * self.expansion, custom_init, bn_track))
            self.register_module("3relu", nn.ReLU())

            self.downsample = downsample
            
            if downsample:
                self.register_module("4conv", conv(in_planes, planes * self.expansion, 1, stride, 0, custom_init))
                self.register_module("4bn", bn(planes * self.expansion, custom_init, bn_track))

        def forward(self, x):
            identity = x

            out = self.get_submodule("1conv")(x)
            out = self.get_submodule("1bn")(out)
            out = self.get_submodule("1relu")(out)

            out = self.get_submodule("2conv")(out)
            out = self.get_submodule("2bn")(out)
            out = self.get_submodule("2relu")(out)

            out = self.get_submodule("3conv")(out)
            out = self.get_submodule("3bn")(out)

            if self.downsample:
                identity = self.get_submodule("4conv")(identity)
                identity = self.get_submodule("4bn")(identity)

            out += identity
            out = self.get_submodule("3relu")(out)
            return out


    class InBlock(nn.Module):

        def __init__(self, input_channels: int,
                        num_filters:int,
                        kernel_size: int = 3,
                        stride: int = 1,
                        padding: int = 1,
                        custom_init = True,
                        bn_track: bool = False):

            super(ResNetImagenet.InBlock, self).__init__()        

            self.register_module("conv", conv(input_channels, num_filters,
                                                            kernel_size, stride, padding, 
                                                            custom_init))


            self.register_module("norm", bn(num_filters, custom_init, bn_track))

            self.register_module("relu", nn.ReLU())

            self.register_module("mp", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        def forward(self, x):
            x = self.get_submodule("conv")(x)
            x = self.get_submodule("norm")(x)
            x = self.get_submodule("relu")(x)
            x = self.get_submodule("mp")(x)
            return x

    class OutBlock(nn.Module):

        def __init__(self, infeatures: int, outfeatures: int, custom_init = False, dropout = None):

            super(ResNetImagenet.OutBlock, self).__init__()

            self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))
            self.register_module("fc", fc(infeatures, outfeatures, custom_init))
            if dropout is not None: 
                self.register_module("dropout", nn.Dropout(dropout))
                self.dropout = True
            else:
                self.dropout = False

        def forward(self, x):
            x = self.get_submodule("gap")(x)
            x = x.squeeze((-1, -2))
            if self.dropout: x = self.get_submodule("dropout")(x)
            x = self.get_submodule("fc")(x)
            return x

    def __init__(self, rank, world_size, depth=18, outfeatures=1000, inchannels=3, custom_init=True, bn_track=False, dropout = None):
        super().__init__()

        self.layers = []

        cfgs = {
            18: (self.BasicBlock, [2, 2, 2, 2]),
            34: (self.BasicBlock, [3, 4, 6, 3]),
            50: (self.Bottleneck, [3, 4, 6, 3]),
            101: (self.Bottleneck, [3, 4, 23, 3]),
            152: (self.Bottleneck, [3, 8, 36, 3]),
        }
        if depth not in cfgs:
            raise ValueError(f"Unsupported ResNet-Imagenet depth: {depth}")

        block, layers_cfg = cfgs[depth]

        self.register_module("inblock", self.InBlock(inchannels, 64, 7, 2, 3, custom_init = custom_init, bn_track = bn_track))

        self.inplanes = 64
        self._make_layer(0, block, 64, layers_cfg[0], stride=1, custom_init=custom_init, bn_track=bn_track)
        self._make_layer(1, block, 128, layers_cfg[1], stride=2, custom_init=custom_init, bn_track=bn_track)
        self._make_layer(2, block, 256, layers_cfg[2], stride=2, custom_init=custom_init, bn_track=bn_track)
        self._make_layer(3, block, 512, layers_cfg[3], stride=2, custom_init=custom_init, bn_track=bn_track)
        
        self.register_module("outblock", self.OutBlock(512 * block.expansion, outfeatures, custom_init, dropout))

        self.init_base(rank, world_size)

    def _make_layer(self, prefix, block, planes, blocks, stride=1, custom_init=True, bn_track=False):
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        block_name = f"block{prefix}{0}"
        self.register_module(block_name, block(self.inplanes, planes, stride, downsample, custom_init, bn_track))
        self.layers.append(block_name)

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_name = f"block{prefix}{i}"
            self.register_module(block_name, block(self.inplanes, planes, 1, False, custom_init, bn_track))
            self.layers.append(block_name)

    def forward(self, x):
        x = self.get_submodule("inblock")(x)
        for layer_name in self.layers:
            x = self.get_submodule(layer_name)(x)
        x = self.get_submodule("outblock")(x)
        return x


def resnet(rank: int, world_size: int, depth: int = 20, outfeatures: int = 10, inchannels: int = 3, custom_init = True, bn_track = False, dropout = None):

    if depth in ivalid: return ResNetImagenet(rank, world_size, depth, outfeatures, inchannels, custom_init, bn_track, dropout)
    elif depth in valid: return ResNetCifar(rank, world_size, depth, outfeatures, inchannels, custom_init, bn_track, dropout)