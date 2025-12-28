import torch
from torch import nn
import torch.nn.functional as F

class Lottery:

    def _lottery_init(self):
        self.MASK_SHAPE: tuple = None
        self.MASK_NUMEL: int = None
        self.MASK_NAME: str = "weight_mask"  
        self.is_soft: bool = None       
        self.is_masked: bool = True
        self.concrete_tau: torch.Tensor = None
        self.weight_mask: torch.Tensor = None # should only ever be a view
        pass

    def activate(self):
        self.is_masked = True

    def deactivate(self):
        self.is_masked = False

    def update_mask(self, mask: torch.Tensor, offset: int) -> None: # must be called before layer use
        setattr(self, self.MASK_NAME, mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE))
        self.is_soft = getattr(self, self.MASK_NAME).dtype != torch.bool

    @torch.no_grad()
    def set_concrete_temperature(self, tau: torch.Tensor) -> None:
        self.concrete_tau = tau

    def sample_concrete(self) -> torch.Tensor:
        u = torch.rand_like(getattr(self, self.MASK_NAME), requires_grad = False)
        u = torch.clamp(u, 1e-10, 1.-1e-10)
        l = torch.log(u) - torch.log(1 - u)
        mask = F.sigmoid((l + getattr(self, self.MASK_NAME) * self.concrete_tau)/self.concrete_tau) # weight * tau for reproducibility
        return mask

    def _get_kernel(self):
        
        if not self.is_masked: 
            return getattr(self, self.MASKED_NAME)
        
        elif not self.is_soft: 
            if self.training: 
                with torch.no_grad(): 
                    getattr(self, self.MASKED_NAME).data *= getattr(self, self.MASK_NAME)
                    return getattr(self, self.MASKED_NAME)
            else: 
                return getattr(self, self.MASKED_NAME) * getattr(self, self.MASK_NAME)
        
        else: 
            mask = self.sample_concrete()
            return getattr(self, self.MASKED_NAME) * mask

class LotteryLinear(nn.Linear, Lottery):

    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        ):
        self._lottery_init()      
        self.MASK_SHAPE = (out_features, in_features)
        self.MASK_NUMEL = in_features * out_features
        self.MASKED_NAME = "weight"
        super(LotteryLinear, self).__init__(in_features = in_features, out_features = out_features, bias = bias)
          
    def forward(self, inputs):
        return F.linear(inputs, self._get_kernel(), self.bias)

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
        self._lottery_init()
        self.MASK_SHAPE = (out_channels, in_channels, kernel_size, kernel_size)
        self.MASK_NUMEL = in_channels * out_channels * kernel_size * kernel_size
        self.MASKED_NAME = "weight"
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
    

    def forward(self, inputs):
        return self._conv_forward(inputs, self._get_kernel(), self.bias)