import torch
from torch import nn
import torch.nn.functional as F

class Lottery:

    def __init__(self):
        return

class LotteryDense(nn.Linear, Lottery):

    def __init__(
        self,
        in_features,
        out_features
        ):
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features)        
        self.register_buffer("weight_mask", torch.ones_like(self.weight, requires_grad = False), persistent = False)
        #nn.init.kaiming_normal_(self.weight)

    def forward(self, inputs):
        kernel = self.weight * self.get_buffer("weight_mask")
        return F.linear(inputs, kernel, self.bias)

class LotteryConv2D(nn.Conv2d, Lottery):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = (3, 3),
        stride = (1, 1),
        padding = 'same'
    ):
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding = padding, bias = False)
        self.register_buffer("weight_mask", torch.ones_like(self.weight, requires_grad = False), persistent = False)
        #nn.init.kaiming_normal_(self.weight)
        
    def forward(self, inputs):
        kernel = self.weight * self.get_buffer("weight_mask")
        return self._conv_forward(inputs, kernel, self.bias)