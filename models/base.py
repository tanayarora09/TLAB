import torch
from torch import nn
import torch.distributed as dist

import numpy as np
import h5py

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

    def init_base(self):

        """
        Call in __init__ after initializing layers.
        """

        self._prunable_count = 0

        self.lottery_layers = ((name, layer) for name, block in self.named_children() for name, layer in block.named_children() if isinstance(layer, Lottery))

        for layer in self.lottery_layers:
            self._prunable_count += layer.MASK_NUMEL

        self.register_buffer("MASK", torch.ones(self._prunable_count, dtype = torch.bool, 
                                                device = "cuda", requires_grad = False), persistent = False)


    @torch.no_grad()
    def set_ticket(self, mask: torch.Tensor) -> None:
        
        if mask.numel() != self.num_prunable: raise ValueError("Mask must have correct number of parameters.")

        self.get_buffer("MASK").copy_(mask)

        """offset = 0       
        for layer in self.lottery_layers:
            layer.update_mask(self.get_buffer("MASK"), offset)
            offset += layer.MASK_NUMEL"""

    @property
    def sparsity(self):
        return self.sparsity_d * 100

    @property
    def sparsity_d(self):
        return (self.get_buffer("MASK").count_nonzero() / self.num_prunable)


    @property
    def num_prunable(self): return self._prunable_count

    ### ------------------------------------- SERIALIZATION -------------------------

    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket_cpu(self) -> torch.Tensor:

        return self.get_buffer("MASK").detach().cpu()
    
    
    def reset_ticket(self) -> None:

        self.mask.fill_(True)

    
    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket(self, name: str, root: int) -> None:
        if not root: return
        with h5py.File(f"./logs/TICKETS/{name}.h5", 'w') as f:
            f.create_dataset("mask", data = self.export_ticket_cpu().numpy())


    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket(self, name: str, root: int) -> None:

        if root:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f["mask"][:], device = "cuda")
        else:
            data = torch.empty(self.num_prunable) 

        dist.broadcast(data, src=2)

        self.set_mask(data)

    ### --------------------------------------- PRUNING ----------------------------

    @torch.no_grad()
    def prune_by_mg(self, rate: float, iteration: float) -> None:

        all_magnitudes = (torch.cat([(layer.get_parameter(layer.MASKED_NAME)).detach().cpu().reshape([-1]) for name, layer in self.lottery_layers], 
                                 dim = 0) * self.get_buffer("MASK")).abs_()
        
        threshold = np.quantile(all_magnitudes.numpy(), q = 1.0 - rate ** iteration, method = "higher")

        self.set_mask(all_magnitudes.ge(threshold))

        return 
    
    @torch.no_grad()
    def prune_random(self, rate: float) -> None:

        ticket = self.get_buffer("MASK").clone()

        nonzero = ticket.count_nonzero()

        prune = ticket.nonzero()[torch.multinomial(torch.ones(nonzero), int(nonzero * rate), replacement = False)]

        ticket[prune] = False

        self.set_mask(ticket)

        return