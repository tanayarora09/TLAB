import torch
from torch import nn
import torch.nn.functional as F

class LotteryDense(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features
        ):
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features)        
        self.register_buffer("weight_mask", torch.ones_like(self.weight), persistent = False)

    def forward(self, inputs):
        masked_kernel = self.weight.mul(self.get_buffer("weight_mask"))
        return F.linear(inputs, masked_kernel, self.bias)

class LotteryConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = (3, 3),
        stride = (1, 1),
        padding = 'same'
    ):
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding = padding, bias = False)
        self.register_buffer("weight_mask", torch.ones_like(self.weight, requires_grad = False), persistent = False)#torch.full_like(self.weight, True, dtype = torch.bool, requires_grad = False), persistent = False)
        
    def forward(self, inputs):
        kernel = self.weight * self.get_buffer("weight_mask")
        return self._conv_forward(inputs, kernel, self.bias)