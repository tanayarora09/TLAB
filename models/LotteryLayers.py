import torch
from torch import nn
import torch.nn.functional as F

class Lottery:

    def __init__(self):
        self.MASK_SHAPE = None
        self.MASK_NUMEL = None
        self.MASKED_NAME = None   
        self.is_continuous = None       
        self.mask_is_active = True
        pass

    def activate_mask(self):
        self.mask_is_active = True

    def deactivate_mask(self):
        self.mask_is_active = False

    def update_mask(self, mask: torch.Tensor, offset: int) -> None:
        pass

    def update_temperature(self, mask: torch.Tensor) -> None:
        pass

class LotteryDense(nn.Linear, Lottery):

    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        ):
        self.MASK_SHAPE = (out_features, in_features)
        self.MASK_NUMEL = in_features * out_features
        self.MASKED_NAME = "weight"
        self.weight_mask = torch.empty(*self.MASK_SHAPE, dtype = torch.bool, device = "cuda")
        self.mask_is_active = True
        self.is_continuous = False
        self.concrete_temperature = None
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features, bias = bias)        


    #@torch.no_grad()
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)
        self.is_continuous = self.weight_mask.dtype != torch.bool

    @torch.no_grad()
    def update_temperature(self, mask: torch.Tensor) -> None:
        self.concrete_temperature = mask

    def forward(self, inputs):
        if self.training:
            with torch.no_grad():
                self.weight.data *= self.weight_mask
            return F.linear(inputs, self.weight, self.bias)
        else:
            if not self.mask_is_active: kernel = self.weight
            elif not self.is_continuous: kernel = self.weight * self.weight_mask
            else: 
                u = torch.rand_like(self.weight_mask, requires_grad = False)
                u = torch.clamp(u, 1e-10, 1.-1e-10)
                l = torch.log(u) - torch.log(1 - u)
                mask = F.sigmoid((l + self.weight_mask)/self.concrete_temperature)
                kernel = self.weight * mask

            return F.linear(inputs, kernel, self.bias)

class LotteryConv2D(nn.Conv2d, Lottery):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding = 'same',
        bias = False,
    ):
        self.MASK_SHAPE = (out_channels, in_channels, kernel_size, kernel_size)
        self.MASK_NUMEL = in_channels * out_channels * kernel_size * kernel_size
        self.MASKED_NAME = "weight"
        self.weight_mask = torch.empty(*self.MASK_SHAPE, dtype = torch.bool, device = "cuda")
        self.is_continuous = False
        self.mask_is_active = True
        self.concrete_temperature = None
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
    
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)
        self.is_continuous = self.weight_mask.dtype != torch.bool

    @torch.no_grad()
    def update_temperature(self, mask: torch.Tensor) -> None:
        self.concrete_temperature = mask

    def forward(self, inputs):
        if self.training:
            with torch.no_grad(): self.weight.data *= self.weight_mask
            return self._conv_forward(inputs, self.weight, self.bias)   
        else:
            if not self.mask_is_active: kernel = self.weight
            elif not self.is_continuous: kernel = self.weight * self.weight_mask
            else: 
                u = torch.rand_like(self.weight_mask, requires_grad = False)
                u = torch.clamp(u, 1e-10, 1.-1e-10)
                l = torch.log(u) - torch.log(1 - u)
                mask = F.sigmoid((l + self.weight_mask)/self.concrete_temperature)
                kernel = self.weight * mask
                
            return self._conv_forward(inputs, kernel, self.bias)