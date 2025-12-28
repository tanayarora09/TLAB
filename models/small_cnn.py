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
    Conv → (optional BN) → ReLU
    """
    def __init__(self, in_ch, out_ch, custom_init=True, bn_track=False, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.register_module("conv", conv(in_ch, out_ch, 3, 1, 1, custom_init))
        if use_bn:
            self.register_module("norm", bn(out_ch, custom_init, bn_track))
        self.register_module("relu", nn.ReLU())

    def forward(self, x):
        x = self.get_submodule("conv")(x)
        if self.use_bn:
            x = self.get_submodule("norm")(x)
        x = self.get_submodule("relu")(x)
        return x


class Conv1x1Block(nn.Module):
    """
    1×1 Conv → (optional BN) → ReLU
    """
    def __init__(self, in_ch, out_ch, custom_init=True, bn_track=False, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.register_module("conv", conv(in_ch, out_ch, 1, 1, 0, custom_init))
        if use_bn:
            self.register_module("norm", bn(out_ch, custom_init, bn_track))
        self.register_module("relu", nn.ReLU())

    def forward(self, x):
        x = self.get_submodule("conv")(x)
        if self.use_bn:
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


class CNND(MaskedModel):
    """
    Deepest. 3766 | 2114
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False, no_batchnorm=False):
        super().__init__()
        
        use_bn = not no_batchnorm
        if no_batchnorm:
            # Without BN: progressive widths 3->3->7->7->7->7->7
            self.register_module("block0", ConvBlock(inchannels, 3, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(3, 7, custom_init, bn_track, use_bn))
            for i in range(2, 6):
                self.register_module(f"block{i}", ConvBlock(7, 7, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(7, outfeatures, custom_init))
        else:
            # With BN: progressive widths 3->6->9->9->9->9->9
            self.register_module("block0", ConvBlock(inchannels, 6, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(6, 9, custom_init, bn_track, use_bn))
            for i in range(2, 6):
                self.register_module(f"block{i}", ConvBlock(9, 9, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(9, outfeatures, custom_init))
        
        self.layers = [f"block{i}" for i in range(6)]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)

class CNNA(MaskedModel):
    """
    Deep. 3757 | 2180
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False, no_batchnorm=False):
        super().__init__()
        
        use_bn = not no_batchnorm
        if no_batchnorm:
            ch = 10
            self.register_module("block0", ConvBlock(inchannels, ch, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(ch, ch, custom_init, bn_track, use_bn))
            self.register_module("block2", ConvBlock(ch, ch, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(ch, outfeatures, custom_init))
        else:
            self.register_module("block0", ConvBlock(inchannels, 14, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(14, 13, custom_init, bn_track, use_bn))
            self.register_module("block2", ConvBlock(13, 13, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(13, outfeatures, custom_init))

        self.layers = ["block0", "block1", "block2"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


class CNNB(MaskedModel):
    """
    Wide. 3664 | 2139
    """
    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False, no_batchnorm=False):
        super().__init__()
        
        use_bn = not no_batchnorm
        if no_batchnorm:
            self.register_module("block0", ConvBlock(inchannels, 13, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(13, 14, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(14, outfeatures, custom_init))
        else:
            self.register_module("block0", ConvBlock(inchannels, 18, custom_init, bn_track, use_bn))
            self.register_module("block1", ConvBlock(18, 18, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(18, outfeatures, custom_init))

        self.layers = ["block0", "block1"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


class CNNW(MaskedModel):
    """
    Widest. 3733 | 2063
    """

    def __init__(self, rank, world_size, outfeatures=10, inchannels=3,
                 custom_init=True, bn_track=False, no_batchnorm=False):
        super().__init__()
        
        use_bn = not no_batchnorm
        if no_batchnorm:
            self.register_module("block0", ConvBlock(inchannels, 13, custom_init, bn_track, use_bn))
            self.register_module("block1", Conv1x1Block(13, 74, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(74, outfeatures, custom_init))
        else:
            self.register_module("block0", ConvBlock(inchannels, 11, custom_init, bn_track, use_bn))
            self.register_module("block1", Conv1x1Block(11, 148, custom_init, bn_track, use_bn))
            self.register_module("outblock", OutBlock(148, outfeatures, custom_init))

        self.layers = ["block0", "block1"]
        self.init_base(rank, world_size)

    def forward(self, x):
        for l in self.layers:
            x = self.get_submodule(l)(x)
        return self.get_submodule("outblock")(x)


def small_cnn(depth: int, rank: int, world_size: int, outfeatures: int = 10, inchannels: int = 3, custom_init = True, bn_track = False, dropout = None, no_batchnorm = False, **kwargs):
    """
    Depths range from 1-4, Higher depth = deeper model.
    no_batchnorm: If True, models use no batch normalization and target ~2150 params.
    """

    if depth == 1: return CNNW(rank, world_size, outfeatures, inchannels, custom_init, bn_track, no_batchnorm)
    if depth == 2: return CNNB(rank, world_size, outfeatures, inchannels, custom_init, bn_track, no_batchnorm)
    if depth == 3: return CNNA(rank, world_size, outfeatures, inchannels, custom_init, bn_track, no_batchnorm)
    if depth == 4: return CNND(rank, world_size, outfeatures, inchannels, custom_init, bn_track, no_batchnorm)
    raise ValueError(f"Unknown CNN variant: {depth}")