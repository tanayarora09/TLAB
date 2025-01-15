import torch
from torch import nn
import torch.nn.functional as F

from models.SparseLayers import SparseWeightedConv2D

__all__ = ["Lottery", "LotteryDense", "LotteryConv2D"]

class Lottery:

    def __init__(self):
        self.MASK_SHAPE = None
        self.MASK_NUMEL = None
        self.MASKED_NAME = None    
        self.NAME = None   
        pass

    def migrate_to_sparse(self):
        pass

    def update_mask(self, mask: torch.Tensor, offset: int) -> None:
        pass

class LotteryDense(nn.Linear, Lottery):

    def __init__(
        self,
        in_features,
        out_features, 
        name: str = 'dense'
        ):
        self.__infeatures = in_features
        self.__outfeatures = out_features
        self.NAME = name
        super(LotteryDense, self).__init__(in_features = in_features, out_features = out_features)        
        self.MASK_SHAPE = (out_features, in_features)
        self.MASK_NUMEL = in_features * out_features
        self.MASKED_NAME = "weight"

    @torch.no_grad()
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)

    @torch.no_grad()
    def migrate_to_sparse(self):
        new_weight = (self.weight * self.weight_mask).to_sparse_coo()
        new_layer = nn.Linear(self.__infeatures, self.__outfeatures)
        state = new_layer.state_dict()
        state["weight"] = new_weight
        new_layer.load_state_dict(state)
        return new_layer
        

    def forward(self, inputs):
        kernel = self.weight * self.weight_mask
        return F.linear(inputs, kernel, self.bias)

class LotteryConv2D(nn.Conv2d, Lottery):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        padding = 'same',
        name: str = 'conv'
    ):
        self.__inchannels = in_channels
        self.__outchannels = out_channels
        self.__kernelsize = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.NAME = name
        self.MASK_SHAPE = (out_channels, in_channels, kernel_size, kernel_size)
        self.MASK_NUMEL = in_channels * out_channels * kernel_size * kernel_size
        self.MASKED_NAME = "weight"
        self.weight_mask = torch.empty(*self.MASK_SHAPE, dtype = torch.bool, device = "cuda")
        super(LotteryConv2D, self).__init__(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    
    @torch.no_grad()
    def update_mask(self, mask: torch.Tensor, offset: int):
        self.weight_mask =  mask[offset: offset + self.MASK_NUMEL].view(self.MASK_SHAPE)

    @torch.no_grad()
    def migrate_to_sparse(self):
        new_weight = (self.weight * self.weight_mask)
        new_weight = new_weight.reshape(self.__outchannels, -1)
        new_layer = SparseWeightedConv2D(self.__inchannels, self.__outchannels, 
                              kernel_size = self.__kernelsize, 
                              stride = self.__stride,
                              padding = self.__padding, 
                              bias = False)
        new_layer.weight = nn.Parameter(new_weight.to_sparse_csr())
        return new_layer

    def forward(self, inputs):
        kernel = self.weight * self.weight_mask
        return self._conv_forward(inputs, kernel, self.bias)
    