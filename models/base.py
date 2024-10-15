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

        self.lottery_layers = tuple([layer for name, block in self.named_children() for name, layer in block.named_children() if isinstance(layer, Lottery)])

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
    def export_ticket(self, name: str, root: int = 2, entry_name: str = "mask") -> None:
        if not (dist.get_rank() == root): return
        with h5py.File(f"./logs/TICKETS/{name}.h5", 'a') as f:
            print(f"Adding {entry_name} to {name}.")
            f.create_dataset(entry_name, data = self.export_ticket_cpu().numpy())

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket(self, name: str, root: int = 0, entry_name: str = "mask") -> torch.Tensor:
        """
        ROOT = 0
        """
        dist.barrier()
        print(f"[rank {dist.get_rank()}] Loading {entry_name} from {name}.")
        if dist.get_rank() == root:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f[entry_name][:], device = "cuda", dtype = torch.bool)
        else:
            data = torch.zeros(self.num_prunable, device = "cuda", dtype = torch.bool) 

        
        print(f"[rank {torch.distributed.get_rank()}] {data.shape}, {data.device} ")
        dist.barrier()

        dist.broadcast(data, src=root)
        
        dist.barrier()
        print(f"[rank {torch.distributed.get_rank()}] {data.shape}, {data.device} ")

        return data

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket_to_model(self, name: str, root: int = 0, entry_name: str = "mask") -> None:
        """
        ROOT = 0
        """
        if dist.get_rank() == root:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f[entry_name][:], device = "cuda")
        else:
            data = torch.empty(self.num_prunable, device = "cuda") 

        dist.broadcast(data, src=root)

        self.set_ticket(data)

    ### --------------------------------------- PRUNING ----------------------------

    @torch.no_grad()
    def prune_by_mg(self, rate: float, iteration: float) -> None:

        all_magnitudes = (torch.cat([(layer.get_parameter(layer.MASKED_NAME)).detach().view(-1) for layer in self.lottery_layers], 
                                 dim = 0) * self.get_buffer("MASK")).abs_().cpu()
        
        threshold = np.quantile(all_magnitudes.numpy(), q = 1.0 - rate ** iteration, method = "higher")

        self.set_ticket(all_magnitudes.ge(threshold))

        return 
    
    @torch.no_grad()
    def prune_random(self, rate: float, distributed: bool, root: bool = True, rootn: int = 0) -> None:
        
        if root: print("Pruning random")

        if not distributed or root:
            
            print(f"[rank {dist.get_rank()}] Entered root random prune.")

            ticket = self.get_buffer("MASK").clone()

            nonzero = ticket.count_nonzero()

            prune = ticket.nonzero(as_tuple = True)[0][torch.multinomial(torch.ones(nonzero), int(nonzero * rate), replacement = False)]

            ticket[prune] = False

        elif distributed: 
            print(f"[rank {dist.get_rank()}] Entered non root random prune")
            ticket = torch.zeros_like(self.get_buffer("MASK"), device = "cuda", dtype = torch.bool)
        

        if distributed: 
            dist.barrier()
            dist.broadcast(ticket, rootn)
            print(f"[rank {dist.get_rank()}] Broadcasted random ticket.")

        self.set_ticket(ticket)
        
        return

    @torch.no_grad()
    def merge_tickets(self, t1: torch.Tensor, t2: torch.Tensor, t1_weight: float, t2_weight: float) -> torch.Tensor:
        """
        Merges two tickets stochastically (genetic breeding)

        O(N) Runtime and Space Complexity
        """

        child = t1 & t2
        
        remaining = t1.sum() - child.sum()

        if remaining == 0: return child
        
        available = torch.bitwise_xor(t1, t2)

        sample_probs = (t1_weight * t1.float() + t2_weight * t2.float())[available]

        sample = torch.multinomial(sample_probs, remaining, replacement = False)

        child[available.nonzero(as_tuple = True)[0][sample]] = True

        return child


    @torch.compile
    @torch.no_grad()
    def mutate_ticket(self, ticket: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """
        Mutates a given ticket by randomly swapping temperature * min(true_values, false_values) values
        """
        ticket = ticket.clone()
        ntrue = ticket.count_nonzero()
        nfalse = ticket.numel() - ntrue
        swaps = int(temperature * min(ntrue, nfalse))

        swap_true = torch.multinomial(torch.ones(ntrue), swaps, replacement = False)
        swap_false = torch.multinomial(torch.ones(nfalse), swaps, replacement = False)

        false = (~ticket).nonzero(as_tuple = True)[0] # So swapping trues does not affect it

        ticket[ticket.nonzero(as_tuple = True)[0][swap_true]] = False 
        ticket[false[swap_false]] = True

        return ticket
    

