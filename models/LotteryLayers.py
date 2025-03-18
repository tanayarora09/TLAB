import torch
from torch import nn
import torch.nn.functional as F

class Lottery:

    def __init__(self):
        self.MASK_SHAPE = None
        self.MASK_NUMEL = None
        self.MASKED_NAME = None       
        pass

    def update_mask(self, mask: torch.Tensor, offset: int) -> None:
        pass

class LotteryDense(nn.Linear, Lottery):

    def __init__(
        self,
        in_features,
        out_features
        ):
        self.MASK_SHAPE = (out_features, in_features)
        self.MASK_NUMEL = in_features * out_features
        self.MASKED_NAME = "weight"
        self.weight_mask = torch.empty(*self.MASK_SHAPE, dtype = torch.bool, device = "cuda")
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features)        


    @torch.no_grad()
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)

    def forward(self, inputs):
        if self.training:
            with torch.no_grad():
                self.weight.data *= self.weight_mask
            return F.linear(inputs, self.weight, self.bias)
        else:
            kernel = self.weight * self.weight_mask
            return F.linear(inputs, kernel, self.bias)

class LotteryConv2D(nn.Conv2d, Lottery):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding = 'same'
    ):
        self.MASK_SHAPE = (out_channels, in_channels, kernel_size, kernel_size)
        self.MASK_NUMEL = in_channels * out_channels * kernel_size * kernel_size
        self.MASKED_NAME = "weight"
        self.weight_mask = torch.empty(*self.MASK_SHAPE, dtype = torch.bool, device = "cuda")
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    
    @torch.no_grad()
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)

    def forward(self, inputs):
        if self.training:
            with torch.no_grad(): self.weight.data *= self.weight_mask
            return self._conv_forward(inputs, self.weight, self.bias)   
        else:
            kernel = self.weight * self.weight_mask
            return self._conv_forward(inputs, kernel, self.bias)