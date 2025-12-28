import torch
from torch import nn
import types

from .LotteryLayers import LotteryConv2D, LotteryLinear
from .base import MaskedModel


# ============================================================
# Initialization helpers (identical style to ResNet reference)
# ============================================================

def bn_init(self):
    self.reset_running_stats()
    if self.affine:
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

def conv_fc_init(self):
    nn.init.kaiming_normal_(self.weight)

def conv(in_ch, out_ch, k, s, p, custom_init=True):
    m = LotteryConv2D(in_ch, out_ch, k, s, p)
    if custom_init:
        m.reset_parameters = types.MethodType(conv_fc_init, m)
        m.reset_parameters()
    return m

def fc(in_f, out_f, custom_init=True):
    m = LotteryLinear(in_f, out_f)
    if custom_init:
        m.reset_parameters = types.MethodType(conv_fc_init, m)
        m.reset_parameters()
    return m

def bn(ch, custom_init=True, bn_track=False):
    m = nn.BatchNorm2d(ch, track_running_stats=bn_track)
    if custom_init:
        m.reset_parameters = types.MethodType(bn_init, m)
        m.reset_parameters()
    return m


# ============================================================
# Shared Blocks
# ============================================================

class ConvBlock(nn.Module):
    """
    Conv → BN → ReLU
    """
    def __init__(self, in_ch, out_ch, custom_init=True, bn_track=False):
        super().__init__()
        self.register_module("conv", conv(in_ch, out_ch, 3, 1, 1, custom_init))
        self.register_module("norm", bn(out_ch, custom_init, bn_track))
        self.register_module("relu", nn.ReLU())

    def forward(self, x):
        x = self.get_submodule("conv")(x)
        x = self.get_submodule("norm")(x)
        x = self.get_submodule("relu")(x)
        return x


class Conv1x1Block(nn.Module):
    """
    1×1 Conv → BN → ReLU
    """
    def __init__(self, in_ch, out_ch, custom_init=True, bn_track=False):
        super().__init__()
        self.register_module("conv", conv(in_ch, out_ch, 1, 1, 0, custom_init))
        self.register_module("norm", bn(out_ch, custom_init, bn_track))
        self.register_module("relu", nn.ReLU())

    def forward(self, x):
        x = self.get_submodule("conv")(x)
        x = self.get_submodule("norm")(x)
        x = self.get_submodule("relu")(x)
        return x


class OutBlock(nn.Module):
    """
    GAP → Linear
    """
    def __init__(self, in_ch, out_ch, custom_init=True):
        super().__init__()
        self.register_module("gap", nn.AdaptiveAvgPool2d((1, 1)))
        self.register_module("fc", fc(in_ch, out_ch, custom_init))

    def forward(self, x):
        x = self.get_submodule("gap")(x)
        x = x.squeeze((-1, -2))
        x = self.get_submodule("fc")(x)
        return x


# ============================================================
# CNN-A : Deeper, Narrow
# ============================================================

class CNNA(MaskedModel):
    """
    3 conv layers, width 8
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False):
        super().__init__()

        self.register_module("block0", ConvBlock(inchannels, 8, custom_init, bn_track))
        self.register_module("block1", ConvBlock(8, 8, custom_init, bn_track))
        self.register_module("block2", ConvBlock(8, 8, custom_init, bn_track))
        self.register_module("outblock", OutBlock(8, outfeatures, custom_init))

        self.layers = ["block0", "block1", "block2"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


# ============================================================
# CNN-B : Wider, Shallow
# ============================================================

class CNNB(MaskedModel):
    """
    2 conv layers, width 16
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False):
        super().__init__()

        self.register_module("block0", ConvBlock(inchannels, 16, custom_init, bn_track))
        self.register_module("block1", ConvBlock(16, 16, custom_init, bn_track))
        self.register_module("outblock", OutBlock(16, outfeatures, custom_init))

        self.layers = ["block0", "block1"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


# ============================================================
# CNN-D : Very Deep, Ultra-Narrow
# ============================================================

class CNND(MaskedModel):
    """
    6 conv layers, width 4
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False):
        super().__init__()

        self.register_module("block0", ConvBlock(inchannels, 4, custom_init, bn_track))
        for i in range(1, 6):
            self.register_module(f"block{i}", ConvBlock(4, 4, custom_init, bn_track))

        self.register_module("outblock", OutBlock(4, outfeatures, custom_init))
        self.layers = [f"block{i}" for i in range(6)]

        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


# ============================================================
# CNN-W : Extremely Wide, Extremely Shallow
# ============================================================

class CNNW(MaskedModel):
    """
    2 layers, width expands to 64 via 1×1 conv
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False):
        super().__init__()

        self.register_module("block0", ConvBlock(inchannels, 32, custom_init, bn_track))
        self.register_module("block1", Conv1x1Block(32, 64, custom_init, bn_track))
        self.register_module("outblock", OutBlock(64, outfeatures, custom_init))

        self.layers = ["block0", "block1"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


def small_cnn(depth: int, rank: int, world_size: int, outfeatures: int = 10, inchannels: int = 3, custom_init = True, bn_track = False, dropout = None,  **kwargs):
    """
    Depths range from 1-4, Higher depth = deeper model.
    """

    if depth == 1: return CNNW(rank, world_size, outfeatures, inchannels, custom_init, bn_track)
    if depth == 2: return CNNB(rank, world_size, outfeatures, inchannels, custom_init, bn_track)
    if depth == 3: return CNNA(rank, world_size, outfeatures, inchannels, custom_init, bn_track)
    if depth == 4: return CNND(rank, world_size, outfeatures, inchannels, custom_init, bn_track)
    raise ValueError(f"Unknown CNN variant: {depth}")