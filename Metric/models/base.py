import torch
from torch import nn

from models.LotteryLayers import Lottery

class BaseModel(nn.Module):

    """
    Overhead for Models:

    Models MUST use LotteryLayers for all layers that must be pruned.

    Expected Structure:

    - Model

    -- Blocks 

    --- Conv / Linear / Norm / Activation / Other Layers that Data is Passed Through

    Expected Name of Parameters:

    $BLOCK.$LAYER.$PARAM

    """

    def init_prune_info(self):

        """
        Call in __init__ after initializing layers.
        """

        self._prunable_count = 0

        for name, mask in self.named_buffers():
            if not name.endswith("_mask"): continue
            self._prunable_count += mask.numel()


    @property
    def sparsity(self):
        
        nonzero = 0

        for name, mask in self.named_buffers():
            if not name.endswith("_mask"): continue
            nonzero += torch.count_nonzero(mask)

        return (nonzero / self.num_prunable) * 100


    @property
    def num_prunable(self): return self._prunable_count
