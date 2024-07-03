import torch
from torch import nn
import torch.nn.functional as F

class LotteryDense(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        name = None
        ):
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features)
        self.name = name
        self.register_parameter("kernel", self.weight)
        self.register_parameter("bias", self.bias)        
        self._mask = torch.ones_like(self.weight)
        self.register_buffer("kernel_mask", self._mask)

    def forward(self, inputs):
        masked_kernel = self.weight.mul(self._mask)
        return F.linear(inputs, masked_kernel, self.bias)


class LotteryConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = (3, 3),
        stride = (1, 1),
        padding = 'same',
        name = None
    ):
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding = padding)
        self.name = name
        self.register_parameter("kernel", self.weight)
        self.register_parameter("bias", self.bias)
        self._mask = torch.ones_like(self.weight)
        self.register_buffer("kernel_mask", self._mask)

    def forward(self, inputs):
        return self._conv_forward(inputs, self.weight.mul(self._mask), self.bias)